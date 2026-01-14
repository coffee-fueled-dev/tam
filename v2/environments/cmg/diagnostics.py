"""
CMG-specific topology diagnostics.
Tests whether the environment REQUIRES distinct commitment ports.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .episode import rollout_with_forced_mode
from .env import CMGEnv, CMGConfig


def fork_separability_test(
    env: CMGEnv,
    config: CMGConfig,
    output_dir: Path,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    Test 1: Fork Separability - Does the environment force disjoint futures?
    
    For each episode, generate K counterfactual rollouts with forced modes
    and measure how quickly trajectories diverge.
    """
    print("\n" + "="*60)
    print("CMG TOPOLOGY TEST 1: Fork Separability")
    print("="*60)
    
    all_D_curves = []  # Pairwise divergence over time
    all_R_curves = []  # Minimum covering radius over time
    all_fork_indices = []
    
    for ep in range(n_episodes):
        # Reset environment to get consistent starting conditions
        env.reset()
        x0 = env.x.copy()
        goals = env.params.g.copy()  # (K, d) array
        
        # Generate K counterfactual trajectories
        mode_trajectories = {}
        for k in range(config.K):
            # Reset to same x0
            env.reset()
            env.x = x0.copy()
            env.params.g = goals.copy()
            
            record = rollout_with_forced_mode(env, k, "goal_seeking")
            mode_trajectories[k] = np.array(record['x'])  # (T+1, d)
        
        T = mode_trajectories[0].shape[0] - 1
        
        # Compute D(t): minimum pairwise divergence at each timestep
        D_curve = []
        for t in range(T + 1):
            min_dist = float('inf')
            for i in range(config.K):
                for j in range(i + 1, config.K):
                    dist = np.linalg.norm(mode_trajectories[i][t] - mode_trajectories[j][t])
                    min_dist = min(min_dist, dist)
            D_curve.append(min_dist)
        all_D_curves.append(D_curve)
        
        # Compute R_all(t): minimum covering radius
        R_curve = []
        for t in range(T + 1):
            points = np.array([mode_trajectories[k][t] for k in range(config.K)])
            center = points.mean(axis=0)
            max_dist = max(np.linalg.norm(p - center) for p in points)
            R_curve.append(max_dist)
        all_R_curves.append(R_curve)
        
        # Compute Fork Index
        sigma_floor = np.array([np.sqrt(t + 1) * config.noise_x for t in range(T + 1)])
        fork_index = np.mean(np.array(R_curve) / (sigma_floor + 1e-6))
        all_fork_indices.append(fork_index)
    
    # Average across episodes
    D_mean = np.mean(all_D_curves, axis=0)
    D_std = np.std(all_D_curves, axis=0)
    R_mean = np.mean(all_R_curves, axis=0)
    R_std = np.std(all_R_curves, axis=0)
    
    fork_index_mean = np.mean(all_fork_indices)
    fork_index_std = np.std(all_fork_indices)
    
    print(f"\n  Fork Index: {fork_index_mean:.2f} ± {fork_index_std:.2f}")
    if fork_index_mean > 5:
        print(f"    → Fork Index >> 1: Environment REQUIRES distinct ports")
    elif fork_index_mean > 1:
        print(f"    → Fork Index > 1: Environment has moderate fork pressure")
    else:
        print(f"    → Fork Index ≈ 1: Environment allows unified commitment")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t_axis = np.arange(len(D_mean))
    
    # Left: D(t) - Pairwise divergence
    ax1 = axes[0]
    ax1.plot(t_axis, D_mean, 'b-', linewidth=2, label='D(t) mean')
    ax1.fill_between(t_axis, D_mean - D_std, D_mean + D_std, alpha=0.3)
    ax1.axvline(config.t_gate, color='r', linestyle='--', label=f't_gate={config.t_gate}')
    if hasattr(config, 'early_divergence') and config.early_divergence:
        t_early_end = int(0.33 * config.T)
        ax1.axvline(t_early_end, color='orange', linestyle='--', label=f'early_div_end')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Min Pairwise Distance D(t)')
    ax1.set_title('Pairwise Divergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle: R_all(t) - Covering radius
    ax2 = axes[1]
    ax2.plot(t_axis, R_mean, 'g-', linewidth=2, label='R_all(t)')
    ax2.fill_between(t_axis, R_mean - R_std, R_mean + R_std, alpha=0.3, color='green')
    sigma_floor = np.array([np.sqrt(t + 1) * config.noise_x for t in t_axis])
    ax2.plot(t_axis, sigma_floor, 'k--', alpha=0.5, label='σ_floor (noise)')
    ax2.axvline(config.t_gate, color='r', linestyle='--')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Covering Radius R_all(t)')
    ax2.set_title('Minimum Covering Radius')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Right: Example trajectories (last episode)
    ax3 = axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, config.K))
    
    # PCA if d > 2
    if config.d > 2:
        all_points = np.vstack([mode_trajectories[k] for k in range(config.K)])
        pca = PCA(n_components=2)
        pca.fit(all_points)
        for k in range(config.K):
            traj_2d = pca.transform(mode_trajectories[k])
            ax3.plot(traj_2d[:, 0], traj_2d[:, 1], color=colors[k], linewidth=2, label=f'Mode {k}')
            ax3.scatter([traj_2d[0, 0]], [traj_2d[0, 1]], color=colors[k], s=100, marker='o', edgecolors='black')
            ax3.scatter([traj_2d[-1, 0]], [traj_2d[-1, 1]], color=colors[k], s=100, marker='*', edgecolors='black')
        ax3.set_title(f'Counterfactual Trajectories (PCA)')
    else:
        for k in range(config.K):
            ax3.plot(mode_trajectories[k][:, 0], mode_trajectories[k][:, 1], 
                    color=colors[k], linewidth=2, label=f'Mode {k}')
            ax3.scatter([mode_trajectories[k][0, 0]], [mode_trajectories[k][0, 1]], 
                       color=colors[k], s=100, marker='o', edgecolors='black')
            ax3.scatter([mode_trajectories[k][-1, 0]], [mode_trajectories[k][-1, 1]], 
                       color=colors[k], s=100, marker='*', edgecolors='black')
        ax3.set_title('Counterfactual Trajectories')
    ax3.set_xlabel('x[0]' if config.d <= 2 else 'PC1')
    ax3.set_ylabel('x[1]' if config.d <= 2 else 'PC2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fork Separability Test (Fork Index: {fork_index_mean:.2f})')
    plt.tight_layout()
    plt.savefig(output_dir / "fork_separability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved fork separability plot to {output_dir / 'fork_separability.png'}")
    
    return {
        'fork_index_mean': float(fork_index_mean),
        'fork_index_std': float(fork_index_std),
        'D_final': float(D_mean[-1]),
        'R_final': float(R_mean[-1]),
        'ports_required': bool(fork_index_mean > 1.5)
    }


def commitment_regret_test(
    actor,
    env: CMGEnv,
    config: CMGConfig,
    mode_centroids: Dict[int, np.ndarray],
    output_dir: Path,
    n_samples: int = 20,
) -> Dict[str, float]:
    """
    Test 2: Commitment Regret Gap - Does using shared z increase volume?
    
    Measures how much extra tube volume is required if you force the actor
    to use the same z across all modes.
    """
    import torch
    import torch.nn.functional as F
    
    print("\n" + "="*60)
    print("CMG TOPOLOGY TEST 2: Commitment Regret Gap")
    print("="*60)
    
    if len(mode_centroids) < 2:
        print("  Need at least 2 mode centroids")
        return None
    
    all_regrets = []
    per_mode_volumes = {k: [] for k in mode_centroids}
    shared_volumes = []
    
    # Use mode 3 centroid (or first available) as shared z
    shared_mode = max(mode_centroids.keys())
    z_shared = torch.tensor(mode_centroids[shared_mode], dtype=torch.float32, device=next(actor.parameters()).device)
    
    for ep in range(n_samples):
        obs = env.reset()
        x0 = env.x.copy()
        goals = env.params.g.copy()
        s0 = torch.tensor(obs, dtype=torch.float32, device=z_shared.device)
        
        # Per-mode optimal volumes
        mode_vols = {}
        for k in mode_centroids:
            env.reset()
            env.x = x0.copy()
            env.params.g = goals.copy()
            
            record = rollout_with_forced_mode(env, k, "goal_seeking")
            traj = torch.tensor(record['x'][1:], dtype=torch.float32, device=z_shared.device)
            traj_delta = traj - s0[:config.d]
            
            with torch.no_grad():
                if hasattr(actor, 'encode'):
                    z_k = actor.encode(traj_delta.reshape(-1))
                else:
                    z_k = torch.tensor(mode_centroids[k], dtype=torch.float32, device=z_shared.device)
                mu, sigma = actor.get_tube(s0, z_k)
            
            vol_k = sigma.mean().item()
            mode_vols[k] = vol_k
            per_mode_volumes[k].append(vol_k)
        
        # Shared z volume
        env.reset()
        env.x = x0.copy()
        env.params.g = goals.copy()
        
        with torch.no_grad():
            mu_shared, sigma_shared = actor.get_tube(s0, z_shared)
        
        vol_shared = sigma_shared.mean().item()
        shared_volumes.append(vol_shared)
        
        # Compute regret
        avg_mode_vol = np.mean(list(mode_vols.values()))
        regret = vol_shared / (avg_mode_vol + 1e-8)
        all_regrets.append(regret)
    
    regret_mean = np.mean(all_regrets)
    regret_std = np.std(all_regrets)
    
    print(f"\n  Volume Statistics:")
    print(f"  Mode | Mean Vol | Std Vol")
    print(f"  -----|----------|--------")
    for k in sorted(per_mode_volumes.keys()):
        vols = per_mode_volumes[k]
        print(f"  {k:4d} | {np.mean(vols):.4f}   | {np.std(vols):.4f}")
    print(f"  Shared | {np.mean(shared_volumes):.4f}   | {np.std(shared_volumes):.4f}")
    
    print(f"\n  Regret Ratio: {regret_mean:.2f} ± {regret_std:.2f}")
    if regret_mean < 1.2:
        print(f"    → Regret ≈ 1: Environment does NOT require ports")
    elif regret_mean < 2.0:
        print(f"    → Regret moderate: Some volume penalty for sharing")
    else:
        print(f"    → Regret >> 1: Environment REQUIRES distinct ports")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Volume comparison
    ax1 = axes[0]
    modes = sorted(per_mode_volumes.keys())
    x_pos = np.arange(len(modes) + 1)
    vol_means = [np.mean(per_mode_volumes[k]) for k in modes] + [np.mean(shared_volumes)]
    vol_stds = [np.std(per_mode_volumes[k]) for k in modes] + [np.std(shared_volumes)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    bar_colors = list(colors) + ['gray']
    
    ax1.bar(x_pos, vol_means, yerr=vol_stds, capsize=5, color=bar_colors, alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Mode {k}' for k in modes] + ['Shared'])
    ax1.set_ylabel('Volume')
    ax1.set_title('Per-Mode vs Shared Volume')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Regret histogram
    ax2 = axes[1]
    ax2.hist(all_regrets, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Regret=1 (no penalty)')
    ax2.axvline(regret_mean, color='red', linestyle='-', linewidth=2, label=f'Mean={regret_mean:.2f}')
    ax2.set_xlabel('Regret (V_shared / V_avg)')
    ax2.set_ylabel('Count')
    ax2.set_title('Commitment Regret Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Commitment Regret Gap (Regret: {regret_mean:.2f})')
    plt.tight_layout()
    plt.savefig(output_dir / "commitment_regret.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved commitment regret plot to {output_dir / 'commitment_regret.png'}")
    
    return {
        'regret_mean': float(regret_mean),
        'regret_std': float(regret_std),
        'shared_mode': int(shared_mode),
        'ports_required': bool(regret_mean > 1.5)
    }


def gating_irreversibility_test(
    env: CMGEnv,
    config: CMGConfig,
    output_dir: Path,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    Test 3: Gating Irreversibility - Can late control undo mode choice?
    
    Tests if switching mode after the gate causes large terminal error.
    """
    print("\n" + "="*60)
    print("CMG TOPOLOGY TEST 3: Gating Irreversibility")
    print("="*60)
    
    # Test intervention at different times
    intervention_times = list(range(0, config.T, max(1, config.T // 10)))
    
    all_results = {t: [] for t in intervention_times}
    
    for ep in range(n_episodes):
        env.reset()
        x0 = env.x.copy()
        goals = env.params.g.copy()
        
        # Pick two modes to test switching between
        k_original = 0
        k_switch = min(1, config.K - 1)
        
        for t_switch in intervention_times:
            env.reset()
            env.x = x0.copy()
            env.params.g = goals.copy()
            
            # Run with original mode until t_switch
            env.k = k_original
            x = x0.copy()
            
            for t in range(config.T):
                if t == t_switch:
                    env.k = k_switch
                
                goal = goals[env.k]
                action = np.clip(goal - env.x, -1.0, 1.0).astype(np.float32)
                obs, _, _, info = env.step(action)
                
                if t >= t_switch:
                    env.k = k_switch
            
            terminal_dist = np.linalg.norm(env.x - goals[k_switch])
            all_results[t_switch].append(terminal_dist)
    
    # Compute statistics
    t_means = [np.mean(all_results[t]) for t in intervention_times]
    t_stds = [np.std(all_results[t]) for t in intervention_times]
    
    print(f"\n  Terminal Distance vs Intervention Time:")
    print(f"  t_switch | Mean Dist | Std Dist")
    print(f"  ---------|-----------|--------")
    for i, t in enumerate(intervention_times):
        print(f"  {t:8d} | {t_means[i]:.4f}    | {t_stds[i]:.4f}")
    
    # Find the "knee" - where distance starts increasing sharply
    knee_idx = 0
    for i in range(1, len(t_means)):
        if t_means[i] > t_means[0] * 1.5:
            knee_idx = i
            break
    
    t_knee = intervention_times[knee_idx] if knee_idx < len(intervention_times) else intervention_times[-1]
    
    print(f"\n  Irreversibility Knee: t ≈ {t_knee}")
    if t_knee <= config.t_gate:
        print(f"    → Irreversibility at/before gate: Environment REQUIRES early commitment")
    elif t_knee < config.T * 0.5:
        print(f"    → Moderate irreversibility: Some commitment pressure")
    else:
        print(f"    → Late irreversibility: Environment allows late switching")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Terminal distance vs intervention time
    ax1 = axes[0]
    ax1.errorbar(intervention_times, t_means, yerr=t_stds, fmt='o-', capsize=5, 
                linewidth=2, markersize=8, color='steelblue')
    ax1.axvline(config.t_gate, color='red', linestyle='--', linewidth=2, label=f't_gate={config.t_gate}')
    ax1.axvline(t_knee, color='green', linestyle=':', linewidth=2, label=f'knee≈{t_knee}')
    if hasattr(config, 'early_divergence') and config.early_divergence:
        t_early_end = int(0.33 * config.T)
        ax1.axvline(t_early_end, color='orange', linestyle='--', label=f'early_div_end')
    ax1.axhline(t_means[0], color='gray', linestyle=':', alpha=0.5, label=f'Best (t=0)')
    ax1.set_xlabel('Intervention Time (t_switch)')
    ax1.set_ylabel('Terminal Distance to New Goal')
    ax1.set_title('Mode Switch Penalty vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Normalized penalty
    ax2 = axes[1]
    normalized = [t_means[i] / (t_means[0] + 1e-8) for i in range(len(t_means))]
    ax2.plot(intervention_times, normalized, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.axhline(1.5, color='red', linestyle=':', alpha=0.5, label='50% penalty threshold')
    ax2.axvline(config.t_gate, color='red', linestyle='--', linewidth=2)
    ax2.axvline(t_knee, color='green', linestyle=':', linewidth=2)
    ax2.set_xlabel('Intervention Time (t_switch)')
    ax2.set_ylabel('Normalized Penalty (dist / dist_best)')
    ax2.set_title('Irreversibility Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Gating Irreversibility Test (Knee at t={t_knee})')
    plt.tight_layout()
    plt.savefig(output_dir / "gating_irreversibility.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved gating irreversibility plot to {output_dir / 'gating_irreversibility.png'}")
    
    return {
        't_knee': int(t_knee),
        'final_dist_early': float(t_means[0]),
        'final_dist_late': float(t_means[-1]),
        'irreversibility_ratio': float(t_means[-1] / (t_means[0] + 1e-8)),
        'requires_early_commitment': bool(t_knee <= config.t_gate)
    }


def run_cmg_topology_suite(
    actor,
    env: CMGEnv,
    config: CMGConfig,
    mode_centroids: Dict[int, np.ndarray],
    output_dir: Path,
) -> Dict[str, Dict]:
    """
    Run all CMG topology diagnostics.
    
    Returns:
        Dict with results from all tests
    """
    print("\n" + "="*60)
    print("CMG TOPOLOGY NECESSITY SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Fork Separability
    results['fork_separability'] = fork_separability_test(env, config, output_dir)
    
    # Test 2: Commitment Regret
    results['commitment_regret'] = commitment_regret_test(actor, env, config, mode_centroids, output_dir)
    
    # Test 3: Gating Irreversibility
    results['gating_irreversibility'] = gating_irreversibility_test(env, config, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("TOPOLOGY NECESSITY SUMMARY")
    print("="*60)
    
    ports_required_votes = 0
    total_tests = 0
    
    if results.get('fork_separability'):
        fork_idx = results['fork_separability']['fork_index_mean']
        print(f"  Fork Index: {fork_idx:.2f}", end=" → ")
        if fork_idx > 1.5:
            print("PORTS REQUIRED")
            ports_required_votes += 1
        else:
            print("Unified OK")
        total_tests += 1
    
    if results.get('commitment_regret'):
        regret = results['commitment_regret']['regret_mean']
        print(f"  Regret Ratio: {regret:.2f}", end=" → ")
        if regret > 1.5:
            print("PORTS REQUIRED")
            ports_required_votes += 1
        else:
            print("Unified OK")
        total_tests += 1
    
    if results.get('gating_irreversibility'):
        t_knee = results['gating_irreversibility']['t_knee']
        print(f"  Irreversibility Knee: t={t_knee}", end=" → ")
        if t_knee <= config.t_gate:
            print("EARLY COMMITMENT REQUIRED")
            ports_required_votes += 1
        else:
            print("Late switching OK")
        total_tests += 1
    
    print(f"\n  Verdict: {ports_required_votes}/{total_tests} tests indicate ports are REQUIRED")
    if ports_required_votes >= 2:
        print("  → Environment topology DEMANDS distinct commitment ports")
    elif ports_required_votes == 1:
        print("  → Environment has MODERATE commitment pressure")
    else:
        print("  → Environment allows UNIFIED commitment (rational collapse OK)")
    
    return results
