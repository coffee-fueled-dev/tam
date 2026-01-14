"""
Viability Experiment: Dimension × Mode Scaling Grid

Tests whether the TAM architecture (tube + commitment z + competitive binding)
trains and organizes into ports as we scale state dimension d and mode count K.

Grid:
- d ∈ {2, 4, 8, 16}
- K ∈ {2, 3, 5}

Key questions:
1. Does training converge across the grid?
2. Does CEM binding outperform RANDOM?
3. Do z-space ports emerge (basins aligned with modes)?
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.cmg_env import CMGEnv, CMGConfig, generate_episode, rollout_with_forced_mode
from actors.knot import KnotActor as Actor, BindingMode


# =============================================================================
# Phase 0: Diagnostics
# =============================================================================

def compute_hedge_viability(
    actor: Actor,
    env: CMGEnv,
    n_episodes: int = 100,
    k_sigma: float = 1.5,
    hedge_scale: float = 2.0,
) -> Dict[str, float]:
    """
    Diagnostic 1: Log hedge viability per episode.
    
    Computes:
    - max_normalized_dev: max_t ||x_t - μ_t|| / (k*σ_t) (worst normalized deviation)
    - fail_soft, fail_hard for actual tubes
    - Same metrics for wide hedge baseline (σ scaled by hedge_scale)
    
    Goal: Confirm hedging is cheap in the current CMG config.
    """
    actor.eval()
    
    metrics = defaultdict(list)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy_mode="goal_seeking")
        obs = episode["obs"][0]
        traj = episode["x"][1:]  # (T, d)
        
        obs_t = torch.tensor(obs, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        with torch.no_grad():
            # Get tube prediction
            z_mu, z_logstd = actor.encode(obs_t)
            z = actor.sample_z(z_mu, z_logstd).squeeze(0)
            result = actor.predict_tube(obs_t, z)
            mu, sigma = result[0], result[1]  # KnotActor returns (mu, sigma, mu_knots, logsig_knots)
            if mu.dim() == 3:
                mu = mu.squeeze(0)
                sigma = sigma.squeeze(0)
            
            # Normalized deviation at each timestep
            deviation = torch.abs(traj_t - mu)  # (T, d)
            normalized_dev = deviation / (k_sigma * sigma + 1e-8)  # (T, d)
            
            # Max normalized deviation (worst case)
            max_norm_dev = normalized_dev.max().item()
            mean_norm_dev = normalized_dev.mean().item()
            
            # Fail metrics for actual tube
            inside = (deviation < k_sigma * sigma).all(dim=-1).float()  # (T,)
            fail_hard = 1.0 - inside.mean().item()
            fail_soft = (1.0 - inside).mean().item()  # same as fail_hard here
            
            # Wide hedge baseline (inflate σ)
            sigma_hedge = sigma * hedge_scale
            inside_hedge = (deviation < k_sigma * sigma_hedge).all(dim=-1).float()
            fail_hard_hedge = 1.0 - inside_hedge.mean().item()
            
            # Volume comparison
            log_vol = torch.log(sigma).mean().item()
            log_vol_hedge = torch.log(sigma_hedge).mean().item()
            
            metrics["max_norm_dev"].append(max_norm_dev)
            metrics["mean_norm_dev"].append(mean_norm_dev)
            metrics["fail_hard"].append(fail_hard)
            metrics["fail_soft"].append(fail_soft)
            metrics["fail_hard_hedge"].append(fail_hard_hedge)
            metrics["log_vol"].append(log_vol)
            metrics["log_vol_hedge"].append(log_vol_hedge)
    
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def compute_mode_separability(
    env: CMGEnv,
    n_episodes: int = 50,
) -> Dict[str, float]:
    """
    Diagnostic 2: Measure mode separability over time.
    
    For each s0, generates both mode trajectories and computes:
    - sep_t = ||x_t(mode0) - x_t(mode1)|| at each timestep
    - min_t, max_t, mean separation
    - sep_early = mean_{t < T/3} sep_t
    
    Goal: Verify if modes diverge late/weakly (makes commitment unnecessary).
    """
    K = env.num_modes
    T = env.config.T
    
    # For K > 2, we compute pairwise separations
    if K == 2:
        mode_pairs = [(0, 1)]
    else:
        mode_pairs = [(i, j) for i in range(K) for j in range(i+1, K)]
    
    metrics = defaultdict(list)
    
    for _ in range(n_episodes):
        # Reset to get initial state
        env.reset()
        x0 = env.x.copy()
        
        # Generate trajectory for each mode
        mode_trajs = {}
        for k in range(K):
            record = rollout_with_forced_mode(env, k, "goal_seeking")
            mode_trajs[k] = record["x"]  # (T+1, d)
        
        # Compute separation for each pair
        for k1, k2 in mode_pairs:
            traj1 = mode_trajs[k1]  # (T+1, d)
            traj2 = mode_trajs[k2]  # (T+1, d)
            
            # Separation at each timestep
            sep_t = np.linalg.norm(traj1 - traj2, axis=1)  # (T+1,)
            
            metrics[f"sep_min_{k1}_{k2}"].append(sep_t.min())
            metrics[f"sep_max_{k1}_{k2}"].append(sep_t.max())
            metrics[f"sep_mean_{k1}_{k2}"].append(sep_t.mean())
            
            # Early separation (first third)
            early_end = max(1, T // 3)
            sep_early = sep_t[:early_end].mean()
            metrics[f"sep_early_{k1}_{k2}"].append(sep_early)
            
            # Late separation (last third)
            late_start = T - T // 3
            sep_late = sep_t[late_start:].mean()
            metrics[f"sep_late_{k1}_{k2}"].append(sep_late)
    
    # Aggregate across pairs
    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    
    # Add summary stats (average across all pairs)
    if K == 2:
        result["sep_min"] = result["sep_min_0_1"]
        result["sep_max"] = result["sep_max_0_1"]
        result["sep_early"] = result["sep_early_0_1"]
        result["sep_late"] = result["sep_late_0_1"]
    else:
        result["sep_min"] = np.mean([result[f"sep_min_{i}_{j}"] for i, j in mode_pairs])
        result["sep_max"] = np.mean([result[f"sep_max_{i}_{j}"] for i, j in mode_pairs])
        result["sep_early"] = np.mean([result[f"sep_early_{i}_{j}"] for i, j in mode_pairs])
        result["sep_late"] = np.mean([result[f"sep_late_{i}_{j}"] for i, j in mode_pairs])
    
    return result


def compute_cem_opportunity(
    actor: Actor,
    env: CMGEnv,
    n_episodes: int = 100,
    n_samples: int = 64,
) -> Dict[str, float]:
    """
    Diagnostic 3: Quantify CEM opportunity.
    
    Samples K z's and records:
    - best_score - mean_score
    - best_bind - mean_bind
    - best_log_vol - mean_log_vol
    
    Goal: If gaps are tiny, CEM can't win even if it's correct.
    """
    actor.eval()
    
    metrics = defaultdict(list)
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy_mode="goal_seeking")
        obs = episode["obs"][0]
        traj = episode["x"][1:]
        
        obs_t = torch.tensor(obs, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Get mode prototypes for scoring
        prototypes = get_mode_prototypes(env)
        
        with torch.no_grad():
            # Sample multiple z's from encoder
            z_mu, z_logstd = actor.encode(obs_t)
            z_mu = z_mu.squeeze(0)
            z_logstd = z_logstd.squeeze(0)
            
            eps = torch.randn(n_samples, actor.z_dim)
            z_samples = z_mu + torch.exp(z_logstd) * eps  # (n_samples, z_dim)
            
            # Score each z
            scores, details = actor.score_commitment(obs_t, z_samples, prototypes)
            
            # Compute bind and volume for each z
            binds = []
            log_vols = []
            for i in range(n_samples):
                mu_i = details["mu"][i]
                sigma_i = details["sigma"][i]
                bind_i = actor.compute_bind_hard(mu_i, sigma_i, traj_t).item()
                vol_i = torch.log(sigma_i).mean().item()
                binds.append(bind_i)
                log_vols.append(vol_i)
            
            scores_np = scores.numpy()
            binds_np = np.array(binds)
            log_vols_np = np.array(log_vols)
            
            # Compute gaps (best - mean)
            metrics["score_gap"].append(scores_np.max() - scores_np.mean())
            metrics["bind_gap"].append(binds_np.max() - binds_np.mean())
            metrics["vol_gap"].append(log_vols_np.max() - log_vols_np.mean())
            
            # Also record absolute ranges
            metrics["score_range"].append(scores_np.max() - scores_np.min())
            metrics["bind_range"].append(binds_np.max() - binds_np.min())
            metrics["vol_range"].append(log_vols_np.max() - log_vols_np.min())
            
            # Best vs worst
            metrics["score_best"].append(scores_np.max())
            metrics["score_worst"].append(scores_np.min())
            metrics["bind_best"].append(binds_np.max())
            metrics["bind_worst"].append(binds_np.min())
    
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def run_phase0_diagnostics(
    actor: Actor,
    env: CMGEnv,
    out_dir: Path = None,
) -> Dict[str, float]:
    """Run all Phase 0 diagnostics and optionally save results."""
    
    print("\n" + "="*60)
    print("Phase 0: Diagnostics")
    print("="*60)
    
    # 1. Hedge viability
    print("\n[1/3] Computing hedge viability...")
    hedge = compute_hedge_viability(actor, env)
    print(f"  Max normalized deviation: {hedge['max_norm_dev']:.3f}")
    print(f"  Mean normalized deviation: {hedge['mean_norm_dev']:.3f}")
    print(f"  Fail rate (actual): {hedge['fail_hard']:.3f}")
    print(f"  Fail rate (hedge 2x): {hedge['fail_hard_hedge']:.3f}")
    print(f"  Log vol (actual): {hedge['log_vol']:.2f}")
    print(f"  Log vol (hedge 2x): {hedge['log_vol_hedge']:.2f}")
    
    # 2. Mode separability
    print("\n[2/3] Computing mode separability...")
    sep = compute_mode_separability(env)
    print(f"  Min separation: {sep['sep_min']:.3f}")
    print(f"  Max separation: {sep['sep_max']:.3f}")
    print(f"  Early separation (t < T/3): {sep['sep_early']:.3f}")
    print(f"  Late separation (t > 2T/3): {sep['sep_late']:.3f}")
    
    # 3. CEM opportunity
    print("\n[3/3] Computing CEM opportunity...")
    cem = compute_cem_opportunity(actor, env)
    print(f"  Score gap (best - mean): {cem['score_gap']:.3f}")
    print(f"  Bind gap (best - mean): {cem['bind_gap']:.3f}")
    print(f"  Volume gap (best - mean): {cem['vol_gap']:.3f}")
    print(f"  Score range: {cem['score_range']:.3f}")
    print(f"  Bind range: {cem['bind_range']:.3f}")
    
    # Interpretation
    print("\n" + "-"*60)
    print("Interpretation:")
    
    if hedge['fail_hard_hedge'] < 0.05:
        print("  ⚠ Hedging is CHEAP - 2x wider tubes still bind well")
    else:
        print("  ✓ Hedging has cost - wider tubes fail more often")
    
    if sep['sep_early'] < sep['sep_late'] * 0.5:
        print("  ⚠ Modes diverge LATE - commitment less necessary")
    else:
        print("  ✓ Modes diverge early - commitment should matter")
    
    if cem['score_range'] < 0.1:
        print("  ⚠ CEM opportunity SMALL - z samples are similar")
    else:
        print("  ✓ CEM has opportunity - z samples vary meaningfully")
    
    # Combine results
    results = {"hedge": hedge, "separability": sep, "cem_opportunity": cem}
    
    if out_dir is not None:
        with open(out_dir / "phase0_diagnostics.json", "w") as f:
            json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# Training
# =============================================================================

def train_actor(
    actor: Actor,
    env: CMGEnv,
    n_steps: int = 10000,
    log_every: int = 1000,
    eval_every: int = 2000,
    n_eval: int = 100,
    n_competitive: int = 8,  # Phase 1: Number of z candidates for competitive sampling
) -> Dict[str, List]:
    """
    Train actor on CMG environment with COMPETITIVE SAMPLING.
    
    Phase 1 approach (no mode labels):
    - Sample K candidate z's from encoder
    - Score each under the same objective CEM uses
    - Train on the BEST z only (winner-take-gradient)
    
    This creates pressure for z to become a decision variable, not noise.
    
    Returns:
        history: Training and evaluation metrics
    """
    history = defaultdict(list)
    
    print(f"Training for {n_steps} steps (competitive sampling K={n_competitive})...")
    
    for step in range(n_steps):
        # Generate episode with goal-seeking (deterministic policy for clean signal)
        episode = generate_episode(env, policy_mode="goal_seeking")
        
        # Get observation and trajectory
        obs = episode["obs"][0]
        traj = episode["x"][1:]  # State trajectory (T, d)
        
        obs_t = torch.tensor(obs, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Phase 1: COMPETITIVE SAMPLING (winner-take-gradient)
        # Sample K candidate z's and pick the best under the scoring function
        with torch.no_grad():
            z_mu, z_logstd = actor.encode(obs_t)
            z_mu = z_mu.squeeze(0)
            z_logstd = z_logstd.squeeze(0)
            
            # Sample K candidates
            eps = torch.randn(n_competitive, actor.z_dim)
            z_candidates = z_mu + torch.exp(z_logstd) * eps  # (K, z_dim)
            
            # Score each candidate using the same objective as training
            # Score = -NLL - α*vol - α_fail*soft_max_fail (lower is better for loss, higher for score)
            scores = []
            for i in range(n_competitive):
                z_i = z_candidates[i]
                result_i = actor.predict_tube(obs_t, z_i)
                mu_i, sigma_i = result_i[0], result_i[1]  # KnotActor returns 4 values
                if mu_i.dim() == 3:
                    mu_i = mu_i.squeeze(0)
                    sigma_i = sigma_i.squeeze(0)
                
                # NLL
                residual_sq = (traj_t - mu_i) ** 2
                nll_i = (residual_sq / (sigma_i ** 2) + 2 * torch.log(sigma_i)).mean()
                
                # Global volume
                log_vol_i = torch.log(sigma_i).sum()
                
                # Soft max-fail
                deviation = torch.abs(traj_t - mu_i)
                normalized_dev = deviation / (actor.k_sigma * sigma_i + 1e-8)
                max_dev_per_t = normalized_dev.max(dim=-1).values
                soft_max_fail_i = torch.logsumexp(actor.max_fail_gamma * max_dev_per_t, dim=0) / actor.max_fail_gamma
                
                # Score = negative loss (higher is better)
                score_i = -(nll_i + actor.alpha_vol * log_vol_i + actor.alpha_fail * soft_max_fail_i)
                scores.append(score_i)
            
            scores = torch.stack(scores)
            best_idx = scores.argmax()
            z_best = z_candidates[best_idx]
        
        # Train step on the BEST z only (winner-take-gradient)
        metrics = actor.train_step(obs_t, traj_t, z_best)
        
        # Log
        history["step"].append(step)
        history["loss"].append(metrics["loss"])
        history["nll"].append(metrics["nll"])
        history["bind_hard"].append(metrics["bind_hard"])
        history["log_vol"].append(metrics["log_vol"])
        history["soft_max_fail"].append(metrics["soft_max_fail"])
        
        # Track score gap for diagnostics
        score_gap = (scores.max() - scores.mean()).item()
        history["score_gap"].append(score_gap)
        
        if step % log_every == 0:
            recent_bind = np.mean(history["bind_hard"][-min(500, step+1):])
            recent_vol = np.mean(history["log_vol"][-min(500, step+1):])
            recent_gap = np.mean(history["score_gap"][-min(500, step+1):])
            print(f"  Step {step}: bind={recent_bind:.3f}, log_vol={recent_vol:.2f}, score_gap={recent_gap:.3f}")
        
        # Periodic evaluation (uses CEM for binding)
        if step > 0 and step % eval_every == 0:
            eval_results = evaluate_actor(actor, env, n_eval)
            for k, v in eval_results.items():
                history[f"eval_{k}"].append(v)
            history["eval_step"].append(step)
            
            print(f"  [Eval] CEM_success={eval_results['cem_success']:.2f}, "
                  f"RAND_success={eval_results['random_success']:.2f}, "
                  f"z_mode_acc={eval_results['z_mode_accuracy']:.2f}, "
                  f"z_silh={eval_results['z_silhouette']:.2f}")
    
    # Final evaluation
    final_eval = evaluate_actor(actor, env, n_eval * 2)
    for k, v in final_eval.items():
        history[f"final_{k}"] = v
    
    return dict(history)


def get_mode_prototypes(env: CMGEnv) -> Tuple[torch.Tensor, ...]:
    """Generate mode prototype trajectories."""
    prototypes = []
    for k in range(env.num_modes):
        record = rollout_with_forced_mode(env, k, "goal_seeking")
        tau = torch.tensor(record["x"][1:], dtype=torch.float32)
        prototypes.append(tau)
    return tuple(prototypes)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_actor(
    actor: Actor,
    env: CMGEnv,
    n_episodes: int = 100,
) -> Dict[str, float]:
    """
    Evaluate actor with both CEM and RANDOM binding.
    
    Returns metrics for:
    - Success rate (goal reached)
    - Tube bind rate
    - Mode accuracy (z predicts mode)
    - Volume (agency)
    """
    actor.eval()
    
    results = {
        "cem": defaultdict(list),
        "random": defaultdict(list),
    }
    
    z_samples = []
    mode_labels = []
    
    for _ in range(n_episodes):
        # Reset env
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        
        # Get prototypes
        prototypes = get_mode_prototypes(env)
        
        # Run episode with goal-seeking and get ground truth trajectory
        episode = generate_episode(env, policy_mode="goal_seeking")
        traj = episode["x"][1:]
        traj_t = torch.tensor(traj, dtype=torch.float32)
        final_mode = episode["k"][-1]
        final_goal = env.params.g[final_mode]
        final_x = episode["x"][-1]
        
        # Phase 1: Tightened success threshold (scales DOWN with sqrt(d))
        # This ensures success is not trivially easy at high dimensions
        success_threshold = 0.5 / np.sqrt(env.state_dim)
        goal_dist = np.linalg.norm(final_x - final_goal)
        success = goal_dist < success_threshold
        
        # Evaluate both binding modes
        for mode_name, bind_fn in [
            ("cem", lambda: actor.select_z_cem(obs_t, mode_prototypes=prototypes)),
            ("random", lambda: actor.select_z_random(obs_t)),
        ]:
            with torch.no_grad():
                z, details = bind_fn()
                mu = details["mu"]
                sigma = details["sigma"]
                
                # Bind rate
                bind = actor.compute_bind_hard(mu, sigma, traj_t).item()
                
                # Volume
                log_vol = torch.log(sigma).mean().item()
                
                # Normalize volume by dimension
                norm_log_vol = log_vol / env.state_dim
                
                results[mode_name]["bind"].append(bind)
                results[mode_name]["log_vol"].append(norm_log_vol)
                results[mode_name]["success"].append(float(success))
                
                # Collect z samples for mode prediction (CEM only)
                if mode_name == "cem":
                    z_samples.append(z.numpy())
                    mode_labels.append(final_mode)
    
    # Compute z→mode accuracy and z-space quality metrics
    z_samples = np.array(z_samples)
    mode_labels = np.array(mode_labels)
    z_mode_accuracy = compute_z_mode_accuracy(z_samples, mode_labels, env.num_modes)
    z_metrics = compute_z_space_metrics(z_samples, mode_labels, env.num_modes)
    
    # Summarize
    summary = {
        "cem_success": np.mean(results["cem"]["success"]),
        "cem_bind": np.mean(results["cem"]["bind"]),
        "cem_log_vol": np.mean(results["cem"]["log_vol"]),
        "random_success": np.mean(results["random"]["success"]),
        "random_bind": np.mean(results["random"]["bind"]),
        "random_log_vol": np.mean(results["random"]["log_vol"]),
        "cem_advantage": np.mean(results["cem"]["success"]) - np.mean(results["random"]["success"]),
        "z_mode_accuracy": z_mode_accuracy,
        # Z-space quality metrics (work in full dimensions, not just 2D PCA)
        "z_silhouette": z_metrics["silhouette"],
        "z_cluster_purity": z_metrics["cluster_purity"],
        "z_inter_intra_ratio": z_metrics["inter_intra_ratio"],
    }
    
    return summary


def compute_z_mode_accuracy(z_samples: np.ndarray, mode_labels: np.ndarray, K: int) -> float:
    """
    Train a linear classifier z → mode and return accuracy.
    This measures how well z encodes mode information.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    if len(np.unique(mode_labels)) < 2:
        return 1.0 / K  # Chance
    
    try:
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, z_samples, mode_labels, cv=min(5, len(z_samples)))
        return float(np.mean(scores))
    except Exception:
        return 1.0 / K


def compute_z_space_metrics(z_samples: np.ndarray, mode_labels: np.ndarray, K: int) -> Dict[str, float]:
    """
    Compute z-space quality metrics that work in full dimensions (no PCA needed).
    
    Returns:
        silhouette: How well clusters are separated (-1 to 1, higher is better)
        cluster_purity: Fraction of samples where nearest cluster centroid matches mode
        inter_intra_ratio: Ratio of inter-mode to intra-mode distances (higher is better)
    """
    from sklearn.metrics import silhouette_score
    
    unique_modes = np.unique(mode_labels)
    if len(unique_modes) < 2:
        return {"silhouette": 0.0, "cluster_purity": 1.0, "inter_intra_ratio": 0.0}
    
    # Silhouette score (works in full z_dim)
    try:
        silhouette = silhouette_score(z_samples, mode_labels)
    except Exception:
        silhouette = 0.0
    
    # Compute mode centroids
    centroids = {}
    for k in unique_modes:
        mask = mode_labels == k
        if mask.sum() > 0:
            centroids[k] = z_samples[mask].mean(axis=0)
    
    # Cluster purity: for each sample, is nearest centroid the correct mode?
    correct = 0
    for i, (z, true_mode) in enumerate(zip(z_samples, mode_labels)):
        if true_mode not in centroids:
            continue
        # Find nearest centroid
        min_dist = float('inf')
        nearest_mode = None
        for k, c in centroids.items():
            dist = np.linalg.norm(z - c)
            if dist < min_dist:
                min_dist = dist
                nearest_mode = k
        if nearest_mode == true_mode:
            correct += 1
    cluster_purity = correct / len(z_samples) if len(z_samples) > 0 else 0.0
    
    # Inter/intra mode distance ratio
    intra_dists = []
    inter_dists = []
    for k in unique_modes:
        mask = mode_labels == k
        if mask.sum() < 2:
            continue
        z_k = z_samples[mask]
        # Intra-mode: distances within this mode
        for i in range(len(z_k)):
            for j in range(i+1, min(i+10, len(z_k))):  # Sample to avoid O(n²)
                intra_dists.append(np.linalg.norm(z_k[i] - z_k[j]))
    
    # Inter-mode: distances between mode centroids
    centroid_list = list(centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i+1, len(centroid_list)):
            inter_dists.append(np.linalg.norm(centroid_list[i] - centroid_list[j]))
    
    if len(intra_dists) > 0 and len(inter_dists) > 0:
        inter_intra_ratio = np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
    else:
        inter_intra_ratio = 0.0
    
    return {
        "silhouette": float(silhouette),
        "cluster_purity": float(cluster_purity),
        "inter_intra_ratio": float(inter_intra_ratio),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_mode_separability(env: CMGEnv, out_path: Path, n_episodes: int = 20):
    """
    Visualize mode separation over time.
    
    Shows how trajectories for different modes diverge over the horizon.
    """
    K = env.num_modes
    T = env.config.T
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Separation curves over time
    ax = axes[0]
    
    sep_curves = defaultdict(list)
    
    for _ in range(n_episodes):
        env.reset()
        
        # Generate trajectory for each mode
        mode_trajs = {}
        for k in range(K):
            record = rollout_with_forced_mode(env, k, "goal_seeking")
            mode_trajs[k] = record["x"]
        
        # Compute pairwise separations
        for k1 in range(K):
            for k2 in range(k1+1, K):
                sep_t = np.linalg.norm(mode_trajs[k1] - mode_trajs[k2], axis=1)
                sep_curves[f"{k1}-{k2}"].append(sep_t)
    
    # Plot mean separation curves
    colors = plt.cm.tab10(np.linspace(0, 1, K*(K-1)//2))
    for i, (pair, curves) in enumerate(sep_curves.items()):
        curves = np.array(curves)
        mean_sep = curves.mean(axis=0)
        std_sep = curves.std(axis=0)
        
        t_axis = np.arange(len(mean_sep))
        ax.plot(t_axis, mean_sep, color=colors[i], label=f"Mode {pair}")
        ax.fill_between(t_axis, mean_sep - std_sep, mean_sep + std_sep, 
                        color=colors[i], alpha=0.2)
    
    # Mark gating window
    ax.axvline(env.config.t_gate, color='red', linestyle='--', alpha=0.7, label='Gating end')
    ax.axvline(T//3, color='orange', linestyle=':', alpha=0.7, label='T/3 (early)')
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mode Separation (||x_k1 - x_k2||)")
    ax.set_title("Mode Divergence Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Example trajectories (2D projection)
    ax = axes[1]
    
    env.reset()
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for k in range(K):
        record = rollout_with_forced_mode(env, k, "goal_seeking")
        traj = record["x"]
        goal = env.params.g[k]
        
        ax.plot(traj[:, 0], traj[:, 1], color=colors[k], linewidth=2, 
                label=f'Mode {k}', alpha=0.8)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color='green', s=100, marker='o', zorder=5)
        ax.scatter([goal[0]], [goal[1]], color=colors[k], s=150, marker='*', zorder=5)
    
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title(f"Example Trajectories (K={K} modes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: Dict, out_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    window = 100
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # Training metrics
    axes[0, 0].plot(smooth(history["bind_hard"]))
    axes[0, 0].set_title("Bind Rate")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(smooth(history["log_vol"]))
    axes[0, 1].set_title("Log Volume")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(smooth(history["nll"]))
    axes[0, 2].set_title("NLL")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].grid(True, alpha=0.3)
    
    # Eval metrics
    if "eval_step" in history and len(history["eval_step"]) > 0:
        steps = history["eval_step"]
        
        axes[1, 0].plot(steps, history["eval_cem_success"], 'b-o', label='CEM')
        axes[1, 0].plot(steps, history["eval_random_success"], 'r-o', label='Random')
        axes[1, 0].set_title("Success Rate")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(steps, history["eval_cem_advantage"], 'g-o')
        axes[1, 1].axhline(0, color='black', linestyle='--')
        axes[1, 1].set_title("CEM Advantage")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(steps, history["eval_z_mode_accuracy"], 'm-o')
        axes[1, 2].axhline(1/2, color='red', linestyle='--', label='Chance (K=2)')
        axes[1, 2].set_title("z→Mode Accuracy")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_z_space(
    actor: Actor,
    env: CMGEnv,
    out_path: Path,
    n_samples: int = 500,
):
    """Visualize z-space colored by mode."""
    from sklearn.decomposition import PCA
    
    actor.eval()
    
    z_samples = []
    mode_labels = []
    
    for _ in range(n_samples):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        
        prototypes = get_mode_prototypes(env)
        episode = generate_episode(env, policy_mode="goal_seeking")
        final_mode = episode["k"][-1]
        
        with torch.no_grad():
            z, _ = actor.select_z_cem(obs_t, mode_prototypes=prototypes)
            z_samples.append(z.numpy())
            mode_labels.append(final_mode)
    
    z_samples = np.array(z_samples)
    mode_labels = np.array(mode_labels)
    
    # PCA projection
    if z_samples.shape[1] > 2:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_samples)
    else:
        z_2d = z_samples
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    K = env.num_modes
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for k in range(K):
        mask = mode_labels == k
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors[k]], 
                   alpha=0.6, label=f'Mode {k}', s=30)
    
    ax.set_xlabel("PC1" if z_samples.shape[1] > 2 else "z[0]")
    ax.set_ylabel("PC2" if z_samples.shape[1] > 2 else "z[1]")
    ax.set_title(f"z-space (d={env.state_dim}, K={K})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tube_projection(
    actor: Actor,
    env: CMGEnv,
    out_path: Path,
    n_examples: int = 4,
):
    """
    Visualize tube predictions vs actual trajectories.
    Projects to 2D using first two coordinates.
    """
    actor.eval()
    
    fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, 8))
    
    for col in range(n_examples):
        # Generate episode first, then use its initial observation for binding
        prototypes = get_mode_prototypes(env)
        episode = generate_episode(env, policy_mode="goal_seeking")
        
        # Use the initial observation from THIS episode
        obs_t = torch.tensor(episode["obs"][0], dtype=torch.float32)
        
        # Trajectory after initial state (what actor predicts)
        traj = episode["x"][1:]  # (T, d) - excludes initial
        start = episode["x"][0]  # Initial state for plotting
        final_mode = episode["k"][-1]
        goal = env.params.g[final_mode]
        
        # CEM binding on the same initial state
        with torch.no_grad():
            z_cem, details_cem = actor.select_z_cem(obs_t, mode_prototypes=prototypes)
            mu_cem = details_cem["mu"].numpy()
            sigma_cem = details_cem["sigma"].numpy()
            # Check for knot data (KnotActor provides this)
            mu_knots_cem = details_cem.get("mu_knots")
            if mu_knots_cem is not None:
                mu_knots_cem = mu_knots_cem.numpy()
            
            z_rand, details_rand = actor.select_z_random(obs_t)
            mu_rand = details_rand["mu"].numpy()
            sigma_rand = details_rand["sigma"].numpy()
            mu_knots_rand = details_rand.get("mu_knots")
            if mu_knots_rand is not None:
                mu_knots_rand = mu_knots_rand.numpy()
        
        # Plot for each binding mode
        for row, (mu, sigma, mu_knots, title) in enumerate([
            (mu_cem, sigma_cem, mu_knots_cem, "CEM"),
            (mu_rand, sigma_rand, mu_knots_rand, "Random"),
        ]):
            ax = axes[row, col]
            
            # Actual trajectory (first 2 dims) - include start for context
            full_traj = np.vstack([start, traj])
            ax.plot(full_traj[:, 0], full_traj[:, 1], 'k-', linewidth=2, label='Actual', alpha=0.8)
            ax.scatter([start[0]], [start[1]], c='green', s=100, marker='o', zorder=5)
            ax.scatter([goal[0]], [goal[1]], c='red', s=100, marker='*', zorder=5)
            
            # Predicted mean (first 2 dims) - starts from t=1
            ax.plot(mu[:, 0], mu[:, 1], 'b--', linewidth=2, label='μ', alpha=0.8)
            ax.scatter([mu[0, 0]], [mu[0, 1]], c='blue', s=50, marker='s', zorder=4)  # μ start
            
            # KNOT POINTS: Draw if available (shows actual port geometry)
            if mu_knots is not None:
                ax.scatter(mu_knots[:, 0], mu_knots[:, 1], 
                          c='magenta', s=120, marker='D', zorder=6, 
                          edgecolors='black', linewidths=1.5, label='knots')
            
            # Tube ellipses at a few timesteps
            for t_idx in [0, len(mu)//2, len(mu)-1]:
                circle = plt.Circle(
                    (mu[t_idx, 0], mu[t_idx, 1]),
                    sigma[t_idx, :2].mean() * 1.5,
                    color='blue', fill=False, linestyle='--', alpha=0.5
                )
                ax.add_patch(circle)
            
            ax.set_title(f"{title} (mode={final_mode})")
            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            ax.legend(loc='upper right', fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Single Run
# =============================================================================

def run_single(
    d: int,
    K: int,
    n_steps: int = 10000,
    z_dim: int = None,
    seed: int = 0,
    out_dir: Path = None,
    run_diagnostics: bool = True,
) -> Dict:
    """Run training for a single (d, K) configuration."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # z_dim = K gives each mode a natural "slot" in z-space
    if z_dim is None:
        z_dim = max(2, K)  # At least 2, but scale with modes
    
    # Environment config with dimension-adjusted scaling
    config = CMGConfig(
        d=d,
        K=K,
        T=16,
        t_gate=2,
        obs_mode="x_goal_time",
        drift_scale=0.1 / np.sqrt(d),  # Scale drift with dimension
        goal_scale=1.0 / np.sqrt(d),   # Scale goals with dimension
        # Phase 1: Early divergence (makes commitment necessary)
        early_divergence=True,
        early_divergence_scale=0.8,    # VERY strong early drift to overpower goal-seeking
        early_divergence_end=0.33,     # First third of horizon
        # Phase 2: Deterministic mode from initial state (inferable without labels)
        # mode = argmax(W_mode @ x0) - creates obs→mode correlation
        deterministic_mode=True,       # Mode = f(x0) - inferable from observation
        uniform_mode=False,            # Turn off random mode
        seed=seed,
    )
    env = CMGEnv(config)
    
    # KnotActor: knot-based ports for globally coherent tubes
    actor = Actor(
        obs_dim=env.obs_dim,
        z_dim=z_dim,
        pred_dim=d,
        T=config.T,
        n_knots=5,           # Knot bottleneck: forces globally coherent tubes
        lr=1e-3,
        # Phase 1: Make hedging expensive
        alpha_vol=0.01,      # Volume scales with T×d now (global sum), so reduce weight
        alpha_fail=1.0,      # P1: Max-fail penalty in loss
        k_sigma=1.0,         # P1: Lowered from 1.5 - tighter tubes required
        # Smoothness
        alpha_smooth=0.01,   # Time-domain σ smoothness
        alpha_knot_smooth=0.01,  # Knot curvature penalty
        use_global_vol=True, # TRUE sum over all (t,d)
        max_fail_gamma=5.0,  # Softmax temp for worst-timestep
        beta_kl=0.001,       # Light KL regularization
    )
    
    print(f"\nRunning d={d}, K={K}")
    print(f"  Env: obs_dim={env.obs_dim}, state_dim={d}, modes={K}")
    print(f"  Actor: z_dim={z_dim}, pred_dim={d}")
    
    # Run Phase 0 diagnostics BEFORE training to understand the environment
    if run_diagnostics:
        print("\n--- Pre-training diagnostics ---")
        sep = compute_mode_separability(env)
        print(f"  Mode sep (early): {sep['sep_early']:.3f}, (late): {sep['sep_late']:.3f}")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_mode_separability(env, out_dir / "mode_separability.png")
    
    # Train with competitive sampling (Phase 1)
    history = train_actor(
        actor, env,
        n_steps=n_steps,
        log_every=n_steps // 10,
        eval_every=n_steps // 5,
        n_competitive=8,  # P1: Winner-take-gradient with K candidates
    )
    
    # Run Phase 0 diagnostics AFTER training
    if run_diagnostics:
        diagnostics = run_phase0_diagnostics(actor, env, out_dir)
        history["diagnostics"] = diagnostics
    
    # Save outputs
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(actor.state_dict(), out_dir / "actor.pt")
        
        # Plots
        plot_training_curves(history, out_dir / "training.png")
        
        try:
            plot_z_space(actor, env, out_dir / "z_space.png")
        except Exception as e:
            print(f"  Warning: z-space plot failed: {e}")
        
        try:
            plot_tube_projection(actor, env, out_dir / "tubes.png")
        except Exception as e:
            print(f"  Warning: tube plot failed: {e}")
        
        # Save summary
        summary = {
            "d": d,
            "K": K,
            "z_dim": z_dim,
            "n_steps": n_steps,
            "seed": seed,
            **{k: v for k, v in history.items() if k.startswith("final_")},
        }
        
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    return history


# =============================================================================
# Grid Sweep
# =============================================================================

def run_grid(
    d_values: List[int] = [2, 4, 8, 16],
    K_values: List[int] = [2, 3, 5],
    n_steps: int = 10000,
    seed: int = 0,
    out_dir: Path = None,
) -> Dict[Tuple[int, int], Dict]:
    """Run the full (d, K) grid."""
    
    results = {}
    
    for d in d_values:
        for K in K_values:
            run_name = f"d{d}_K{K}"
            run_dir = out_dir / run_name if out_dir else None
            
            history = run_single(
                d=d, K=K,
                n_steps=n_steps,
                seed=seed,
                out_dir=run_dir,
            )
            
            results[(d, K)] = history
    
    # Create scaling summary plots
    if out_dir is not None:
        plot_scaling_heatmaps(results, d_values, K_values, out_dir)
    
    return results


def plot_scaling_heatmaps(
    results: Dict[Tuple[int, int], Dict],
    d_values: List[int],
    K_values: List[int],
    out_dir: Path,
):
    """Create heatmaps showing scaling behavior."""
    
    # Extract metrics
    metrics = {
        "CEM Success": "final_cem_success",
        "CEM Advantage": "final_cem_advantage",
        "z→Mode Accuracy": "final_z_mode_accuracy",
        "CEM Bind Rate": "final_cem_bind",
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, (title, key) in zip(axes, metrics.items()):
        data = np.zeros((len(d_values), len(K_values)))
        
        for i, d in enumerate(d_values):
            for j, K in enumerate(K_values):
                if (d, K) in results and key in results[(d, K)]:
                    data[i, j] = results[(d, K)][key]
                else:
                    data[i, j] = np.nan
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(K_values)))
        ax.set_xticklabels(K_values)
        ax.set_yticks(range(len(d_values)))
        ax.set_yticklabels(d_values)
        ax.set_xlabel("K (modes)")
        ax.set_ylabel("d (state dim)")
        ax.set_title(title)
        
        # Add text annotations
        for i in range(len(d_values)):
            for j in range(len(K_values)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                           color='black' if 0.3 < data[i, j] < 0.7 else 'white')
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle("Scaling Grid: Dimension × Modes", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "scaling_heatmaps.png", dpi=150)
    plt.close(fig)
    
    # Line plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success vs d for each K
    ax = axes[0]
    for K in K_values:
        success = [results[(d, K)].get("final_cem_success", np.nan) for d in d_values]
        ax.plot(d_values, success, 'o-', label=f'K={K}')
    ax.set_xlabel("d (state dimension)")
    ax.set_ylabel("CEM Success Rate")
    ax.set_title("Success vs Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # z→mode accuracy vs K for each d
    ax = axes[1]
    for d in d_values:
        acc = [results[(d, K)].get("final_z_mode_accuracy", np.nan) for K in K_values]
        ax.plot(K_values, acc, 'o-', label=f'd={d}')
    ax.set_xlabel("K (number of modes)")
    ax.set_ylabel("z→Mode Accuracy")
    ax.set_title("Port Structure vs Modes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "scaling_lines.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Viability Scaling Experiment")
    parser.add_argument("--mode", choices=["single", "grid"], default="single")
    parser.add_argument("--d", type=int, default=4, help="State dimension (single mode)")
    parser.add_argument("--K", type=int, default=2, help="Number of modes (single mode)")
    parser.add_argument("--d-values", type=str, default="2,4,8", help="Dimensions (grid mode)")
    parser.add_argument("--K-values", type=str, default="2,3", help="Modes (grid mode)")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if args.mode == "single":
        out_dir = Path(__file__).parent / "runs" / f"d{args.d}_K{args.K}_{timestamp}"
        run_single(
            d=args.d,
            K=args.K,
            n_steps=args.steps,
            seed=args.seed,
            out_dir=out_dir,
        )
        print(f"\nResults saved to {out_dir}")
    
    else:  # grid
        d_values = [int(x) for x in args.d_values.split(",")]
        K_values = [int(x) for x in args.K_values.split(",")]
        
        out_dir = Path(__file__).parent / "runs" / f"grid_{timestamp}"
        
        print(f"\n{'='*60}")
        print(f"Viability Scaling Grid")
        print(f"d ∈ {d_values}, K ∈ {K_values}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")
        
        run_grid(
            d_values=d_values,
            K_values=K_values,
            n_steps=args.steps,
            seed=args.seed,
            out_dir=out_dir,
        )
        
        print(f"\nGrid results saved to {out_dir}")


if __name__ == "__main__":
    main()
