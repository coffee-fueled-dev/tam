"""
Competitive Port Binding Experiment: CEM vs Random vs Best-of-K.

This experiment shows that the actor can choose among multiple candidate
commitments by planning in z-space to optimize:
- Forecast intent (accuracy)
- Agency (tight tube)
- Reliability (low failure)

And that this choice adapts across environment difficulty regimes.

Experimental conditions:
1. Random: sample one z ~ q(z|s0)
2. Best-of-K: sample K, pick best score (no refinement)
3. CEM: iterative refinement
4. Oracle: use true outcome (upper bound diagnostic)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from actor import CompetitiveActor, BindingMode


class SimpleEnv:
    """
    Minimal 2D environment with different dynamics per rule.
    
    Improvements over v1:
    - Base noise on ALL rules (so bind is non-saturating)
    - Latent mode (bimodal futures) for rules 1-2 (so intent matters)
    - Configurable difficulty
    """
    
    def __init__(
        self,
        seed: int = 0,
        base_noise: float = 0.03,  # Base noise for all rules
        bimodal: bool = True,      # Enable bimodal trajectories
    ):
        self.rng = np.random.default_rng(seed)
        self.n_rules = 4
        self.base_noise = base_noise
        self.bimodal = bimodal
        self.latent_mode = None  # Hidden mode for bimodal rules
        self.reset()
    
    def reset(self, rule: int = None) -> np.ndarray:
        self.x = self.rng.uniform(0.1, 0.9)
        self.y = self.rng.uniform(0.1, 0.9)
        self.goal_x = self.rng.uniform(0.1, 0.9)
        self.goal_y = self.rng.uniform(0.1, 0.9)
        self.rule = rule if rule is not None else self.rng.integers(0, self.n_rules)
        
        # For bimodal rules (1, 2), sample a hidden mode
        # This creates multiple plausible futures from same s0
        if self.bimodal and self.rule in [1, 2]:
            self.latent_mode = self.rng.choice([-1, 1])  # CCW or CW variation
        else:
            self.latent_mode = 1
        
        return self.observe()
    
    def observe(self) -> np.ndarray:
        """Observation does NOT include latent_mode - that's the hidden stochasticity."""
        rule_oh = np.zeros(self.n_rules)
        rule_oh[self.rule] = 1.0
        return np.array([self.x, self.y, self.goal_x, self.goal_y, *rule_oh], dtype=np.float32)
    
    def generate_trajectory(self, T: int = 16, noise_scale: float = 1.0) -> np.ndarray:
        """Generate trajectory based on rule + latent mode."""
        traj = np.zeros((T, 2), dtype=np.float32)
        
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        
        # Base noise applied to ALL rules
        base = self.base_noise * noise_scale
        
        for t in range(T):
            progress = (t + 1) / T
            
            # Add base noise to all trajectories
            noise_x = self.rng.normal(0, base) if base > 0 else 0
            noise_y = self.rng.normal(0, base) if base > 0 else 0
            
            if self.rule == 0:  # Straight (easy but with base noise)
                traj[t, 0] = self.x + dx * progress + noise_x
                traj[t, 1] = self.y + dy * progress + noise_y
                
            elif self.rule == 1:  # Curved with bimodal amplitude
                angle = progress * np.pi / 2
                # Bimodal: curve direction depends on latent_mode
                amp = 0.1 * self.latent_mode
                traj[t, 0] = self.x + dx * progress + amp * np.sin(angle * 2) + noise_x
                traj[t, 1] = self.y + dy * progress - amp * (1 - np.cos(angle * 2)) + noise_y
                
            elif self.rule == 2:  # Curved with bimodal frequency
                angle = progress * np.pi / 2
                # Bimodal: oscillation pattern varies
                freq = 2 if self.latent_mode > 0 else 3
                amp = 0.08
                traj[t, 0] = self.x + dx * progress + amp * np.sin(angle * freq) + noise_x
                traj[t, 1] = self.y + dy * progress + amp * np.cos(angle * freq) * 0.5 + noise_y
                
            elif self.rule == 3:  # Noisy (hardest)
                rule_noise = 0.15 * noise_scale  # Extra noise for hard rule
                noise_t = (base + rule_noise) * (1 - 0.3 * progress)
                traj[t, 0] = self.x + dx * progress + self.rng.normal(0, noise_t)
                traj[t, 1] = self.y + dy * progress + self.rng.normal(0, noise_t)
        
        return traj


def train(
    actor: CompetitiveActor,
    env: SimpleEnv,
    n_steps: int = 10000,
    log_every: int = 1000,
    mode: BindingMode = BindingMode.CEM,
) -> Dict[str, List[float]]:
    """Train with specified binding mode."""
    
    print(f"Training CompetitiveActor for {n_steps} steps with {mode.value} binding...")
    print(f"  CEM config: iters={actor.cem_iters}, samples={actor.cem_samples}, elites={actor.cem_elites}")
    print(f"  Scoring: intent={actor.intent_weight}, agency={actor.agency_weight}, risk={actor.risk_weight}")
    
    for step in range(n_steps):
        s0 = env.reset()
        traj = env.generate_trajectory(T=actor.T)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Select z using specified mode
        if mode == BindingMode.ORACLE:
            binding_result = actor.select_z(s0_t, mode=mode, trajectory=traj_t)
        else:
            binding_result = actor.select_z(s0_t, mode=mode)
        
        # Train on outcome
        metrics = actor.train_step(s0_t, traj_t, binding_result)
        
        if step % log_every == 0:
            stats = actor.get_equilibrium_stats(window=min(500, step + 1))
            print(f"  Step {step}: bind={metrics['bind_hard']:.3f}, "
                  f"score={metrics['score']:.3f}, risk_loss={metrics['risk_loss']:.4f}")
            if stats and "cem_improvement" in actor.history and len(actor.history["cem_improvement"]) > 0:
                recent_cem = np.mean(actor.history["cem_improvement"][-min(100, step+1):])
                print(f"    CEM improvement: {recent_cem:.4f}")
    
    return actor.history


def evaluate_binding_modes(
    actor: CompetitiveActor,
    env: SimpleEnv,
    n_samples: int = 200,
) -> Dict[str, Dict[str, float]]:
    """Compare binding modes on same initial states."""
    actor.eval()
    
    modes = [BindingMode.RANDOM, BindingMode.BEST_OF_K, BindingMode.CEM, BindingMode.ORACLE]
    results = {mode.value: defaultdict(list) for mode in modes}
    
    print(f"\nEvaluating binding modes on {n_samples} samples...")
    
    for i in range(n_samples):
        # Same initial state for all modes
        s0 = env.reset()
        traj = env.generate_trajectory(T=actor.T)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        for mode in modes:
            with torch.no_grad():
                if mode == BindingMode.ORACLE:
                    result = actor.select_z(s0_t, mode=mode, trajectory=traj_t)
                else:
                    result = actor.select_z(s0_t, mode=mode)
                
                # Evaluate TRUE outcome
                bind = actor.compute_bind_hard(result.mu, result.sigma, traj_t).item()
                log_vol = actor.compute_log_volume(result.sigma).item()
                mse = actor.compute_mse(result.mu, traj_t).item()
            
            results[mode.value]["bind"].append(bind)
            results[mode.value]["log_vol"].append(log_vol)
            results[mode.value]["mse"].append(mse)
            results[mode.value]["score"].append(result.score)
            results[mode.value]["intent_proxy"].append(result.intent_proxy)
            results[mode.value]["agency"].append(result.agency)
            results[mode.value]["risk_pred"].append(result.risk_pred)
            results[mode.value]["rule"].append(env.rule)
            results[mode.value]["z_star"].append(result.z_star.numpy())
    
    # Summarize
    summary = {}
    for mode in modes:
        mode_name = mode.value
        
        binds = np.array(results[mode_name]["bind"])
        risk_preds = np.array(results[mode_name]["risk_pred"])
        fail_actual = 1.0 - binds  # Actual failure rate
        
        # Risk calibration metrics
        risk_corr = np.corrcoef(risk_preds, fail_actual)[0, 1] if len(risk_preds) > 1 else 0.0
        risk_mae = np.abs(risk_preds - fail_actual).mean()
        
        # Expected Calibration Error (ECE) - binned calibration
        n_bins = 5
        ece = 0.0
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (risk_preds >= lo) & (risk_preds < hi)
            if mask.sum() > 0:
                bin_conf = risk_preds[mask].mean()
                bin_acc = fail_actual[mask].mean()
                ece += mask.sum() * np.abs(bin_conf - bin_acc)
        ece /= len(risk_preds)
        
        summary[mode_name] = {
            "bind_mean": np.mean(binds),
            "bind_std": np.std(binds),
            "log_vol_mean": np.mean(results[mode_name]["log_vol"]),
            "log_vol_std": np.std(results[mode_name]["log_vol"]),
            "mse_mean": np.mean(results[mode_name]["mse"]),
            "mse_std": np.std(results[mode_name]["mse"]),
            "score_mean": np.mean(results[mode_name]["score"]),
            "z_stars": np.array(results[mode_name]["z_star"]),
            "rules": np.array(results[mode_name]["rule"]),
            # Risk calibration
            "risk_preds": risk_preds,
            "fail_actual": fail_actual,
            "risk_corr": risk_corr if not np.isnan(risk_corr) else 0.0,
            "risk_mae": risk_mae,
            "risk_ece": ece,
        }
        print(f"  {mode_name}: bind={summary[mode_name]['bind_mean']:.3f}±{summary[mode_name]['bind_std']:.3f}, "
              f"vol={summary[mode_name]['log_vol_mean']:.2f}, mse={summary[mode_name]['mse_mean']:.4f}, "
              f"risk_corr={summary[mode_name]['risk_corr']:.3f}")
    
    return summary


def evaluate_per_rule(
    actor: CompetitiveActor,
    env: SimpleEnv,
    mode: BindingMode,
    n_samples: int = 100,
) -> Dict[int, Dict[str, float]]:
    """Evaluate one binding mode per rule."""
    actor.eval()
    
    results = {r: defaultdict(list) for r in range(4)}
    
    for rule in range(4):
        for _ in range(n_samples):
            s0 = env.reset(rule=rule)
            traj = env.generate_trajectory(T=actor.T)
            
            s0_t = torch.tensor(s0, dtype=torch.float32)
            traj_t = torch.tensor(traj, dtype=torch.float32)
            
            with torch.no_grad():
                if mode == BindingMode.ORACLE:
                    result = actor.select_z(s0_t, mode=mode, trajectory=traj_t)
                else:
                    result = actor.select_z(s0_t, mode=mode)
                
                bind = actor.compute_bind_hard(result.mu, result.sigma, traj_t).item()
                log_vol = actor.compute_log_volume(result.sigma).item()
                sigma_mean = result.sigma.mean().item()
            
            results[rule]["bind"].append(bind)
            results[rule]["log_vol"].append(log_vol)
            results[rule]["sigma"].append(sigma_mean)
            results[rule]["z_star"].append(result.z_star.numpy())
    
    summary = {}
    for rule in range(4):
        summary[rule] = {
            "bind_mean": np.mean(results[rule]["bind"]),
            "bind_std": np.std(results[rule]["bind"]),
            "log_vol_mean": np.mean(results[rule]["log_vol"]),
            "log_vol_std": np.std(results[rule]["log_vol"]),
            "sigma_mean": np.mean(results[rule]["sigma"]),
            "sigma_std": np.std(results[rule]["sigma"]),
            "z_stars": np.array(results[rule]["z_star"]),
        }
    
    return summary


def plot_mode_comparison(summary: Dict[str, Dict], out_path: Path):
    """Plot comparison of binding modes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    modes = ["random", "best_of_k", "cem", "oracle"]
    mode_names = ["Random", "Best-of-K", "CEM", "Oracle"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    
    # 1. Bind rate
    ax = axes[0]
    binds = [summary[m]["bind_mean"] for m in modes]
    bind_stds = [summary[m]["bind_std"] for m in modes]
    bars = ax.bar(mode_names, binds, yerr=bind_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_ylabel("Bind Rate")
    ax.set_title("Bind Rate by Mode\n(higher = more reliable)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Volume (lower = tighter)
    ax = axes[1]
    vols = [summary[m]["log_vol_mean"] for m in modes]
    vol_stds = [summary[m]["log_vol_std"] for m in modes]
    bars = ax.bar(mode_names, vols, yerr=vol_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_ylabel("Log Volume")
    ax.set_title("Tube Volume by Mode\n(lower = more agency)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. MSE (prediction accuracy)
    ax = axes[2]
    mses = [summary[m]["mse_mean"] for m in modes]
    mse_stds = [summary[m]["mse_std"] for m in modes]
    bars = ax.bar(mode_names, mses, yerr=mse_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_ylabel("MSE")
    ax.set_title("Prediction Error by Mode\n(lower = better intent)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Competitive Binding: Mode Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_bind_vs_volume(summary: Dict[str, Dict], out_path: Path):
    """Plot bind rate vs volume tradeoff frontier."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    modes = ["random", "best_of_k", "cem", "oracle"]
    mode_names = ["Random", "Best-of-K", "CEM", "Oracle"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    
    for mode, name, color in zip(modes, mode_names, colors):
        ax.scatter(
            summary[mode]["log_vol_mean"],
            summary[mode]["bind_mean"],
            s=200, c=color, label=name,
            edgecolors='black', linewidths=2
        )
        ax.errorbar(
            summary[mode]["log_vol_mean"],
            summary[mode]["bind_mean"],
            xerr=summary[mode]["log_vol_std"],
            yerr=summary[mode]["bind_std"],
            color=color, fmt='none', capsize=5
        )
    
    ax.set_xlabel("Log Volume (lower = tighter tube)")
    ax.set_ylabel("Bind Rate (higher = more reliable)")
    ax.set_title("Competitive Binding: Bind-Volume Tradeoff\n(CEM should dominate Random)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add arrow showing Pareto direction
    ax.annotate('', xy=(-3, 1.0), xytext=(-2.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(-2.75, 0.92, 'Better', fontsize=10, color='gray', ha='center')
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_rule_comparison(
    cem_summary: Dict[int, Dict],
    random_summary: Dict[int, Dict],
    out_path: Path,
):
    """Compare CEM vs Random per rule."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    rules = [0, 1, 2, 3]
    rule_names = ["Straight\n(easy)", "Curved CW\n(medium)", "Curved CCW\n(medium)", "Noisy\n(hard)"]
    x = np.arange(len(rules))
    width = 0.35
    
    # 1. Bind rate
    ax = axes[0, 0]
    cem_binds = [cem_summary[r]["bind_mean"] for r in rules]
    random_binds = [random_summary[r]["bind_mean"] for r in rules]
    ax.bar(x - width/2, cem_binds, width, label='CEM', color='#2ecc71', edgecolor='black')
    ax.bar(x + width/2, random_binds, width, label='Random', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("Bind Rate")
    ax.set_title("Bind Rate by Rule")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Sigma (tube width)
    ax = axes[0, 1]
    cem_sigmas = [cem_summary[r]["sigma_mean"] for r in rules]
    random_sigmas = [random_summary[r]["sigma_mean"] for r in rules]
    ax.bar(x - width/2, cem_sigmas, width, label='CEM', color='#2ecc71', edgecolor='black')
    ax.bar(x + width/2, random_sigmas, width, label='Random', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("σ (tube width)")
    ax.set_title("Tube Width by Rule\n(CEM should widen for hard rules)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Log volume
    ax = axes[1, 0]
    cem_vols = [cem_summary[r]["log_vol_mean"] for r in rules]
    random_vols = [random_summary[r]["log_vol_mean"] for r in rules]
    ax.bar(x - width/2, cem_vols, width, label='CEM', color='#2ecc71', edgecolor='black')
    ax.bar(x + width/2, random_vols, width, label='Random', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("Log Volume")
    ax.set_title("Log Volume by Rule")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. CEM advantage (bind improvement)
    ax = axes[1, 1]
    bind_advantage = [cem_binds[i] - random_binds[i] for i in range(4)]
    colors = ['#2ecc71' if a > 0 else '#e74c3c' for a in bind_advantage]
    ax.bar(x, bind_advantage, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("Bind Rate Δ (CEM - Random)")
    ax.set_title("CEM Advantage over Random\n(positive = CEM better)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Competitive Binding: Per-Rule Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_z_space(summary: Dict[str, Dict], out_path: Path):
    """Visualize z* distributions per mode, colored by rule."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    modes = ["random", "best_of_k", "cem", "oracle"]
    mode_names = ["Random", "Best-of-K", "CEM", "Oracle"]
    rule_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    
    for ax, mode, name in zip(axes.flat, modes, mode_names):
        z_stars = summary[mode]["z_stars"]
        rules = summary[mode]["rules"]
        
        # PCA to 2D if z_dim > 2
        if z_stars.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_stars)
        else:
            z_2d = z_stars
        
        for rule in range(4):
            mask = rules == rule
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                c=rule_colors[rule], alpha=0.6, s=30,
                label=f"Rule {rule}"
            )
        
        ax.set_xlabel("z* PC1")
        ax.set_ylabel("z* PC2")
        ax.set_title(f"{name} Mode: z* by Rule")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Competitive Binding: Selected z* Distribution", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_risk_calibration(summary: Dict[str, Dict], out_path: Path):
    """
    Plot risk critic calibration:
    - Reliability diagram (predicted vs actual failure)
    - Correlation scatter
    - ECE summary
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Focus on CEM mode (main mode of interest)
    cem = summary.get("cem", {})
    risk_preds = cem.get("risk_preds", np.array([]))
    fail_actual = cem.get("fail_actual", np.array([]))
    
    if len(risk_preds) == 0:
        plt.close(fig)
        return
    
    # 1. Reliability diagram (calibration curve)
    ax = axes[0]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_counts = []
    bin_conf = []
    bin_acc = []
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (risk_preds >= lo) & (risk_preds < hi)
        if mask.sum() > 0:
            bin_counts.append(mask.sum())
            bin_conf.append(risk_preds[mask].mean())
            bin_acc.append(fail_actual[mask].mean())
        else:
            bin_counts.append(0)
            bin_conf.append(bin_centers[i])
            bin_acc.append(0)
    
    bin_conf = np.array(bin_conf)
    bin_acc = np.array(bin_acc)
    bin_counts = np.array(bin_counts)
    
    # Plot bars
    ax.bar(bin_centers, bin_acc, width=0.08, alpha=0.7, label='Actual fail rate', color='#3498db')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax.scatter(bin_conf, bin_acc, c='red', s=50, zorder=5)
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Actual failure rate")
    ax.set_title(f"Risk Calibration (CEM)\nECE = {cem.get('risk_ece', 0):.3f}")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter plot: predicted vs actual
    ax = axes[1]
    ax.scatter(risk_preds, fail_actual, alpha=0.3, c='blue', s=20)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax.set_xlabel("Predicted failure")
    ax.set_ylabel("Actual failure")
    ax.set_title(f"Risk Scatter\nCorr = {cem.get('risk_corr', 0):.3f}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 3. Summary metrics across modes
    ax = axes[2]
    modes = ["random", "best_of_k", "cem", "oracle"]
    mode_names = ["Random", "Best-of-K", "CEM", "Oracle"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    
    corrs = [summary.get(m, {}).get("risk_corr", 0) for m in modes]
    maes = [summary.get(m, {}).get("risk_mae", 0) for m in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, corrs, width, label='Correlation', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, maes, width, label='MAE', color='#e74c3c', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names)
    ax.set_ylabel("Metric value")
    ax.set_title("Risk Critic Quality by Mode")
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Risk Critic Calibration Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training(history: Dict[str, List[float]], out_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    window = 200
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # 1. Bind rate
    ax = axes[0, 0]
    ax.plot(smooth(history["bind_hard"]), color='blue', alpha=0.8)
    ax.set_title("Bind Rate")
    ax.set_ylabel("Bind rate")
    ax.grid(True, alpha=0.3)
    
    # 2. Score
    ax = axes[0, 1]
    ax.plot(smooth(history["score"]), color='green')
    ax.set_title("Selection Score")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    
    # 3. Risk prediction
    ax = axes[0, 2]
    ax.plot(smooth(history["risk_pred"]), color='red', alpha=0.8, label='Predicted')
    ax.plot(smooth(history["fail_soft"]), color='darkred', alpha=0.5, label='Actual')
    ax.set_title("Risk: Predicted vs Actual")
    ax.set_ylabel("Fail rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Risk loss
    ax = axes[0, 3]
    ax.plot(smooth(history["risk_loss"]), color='purple')
    ax.set_title("Risk Critic Loss")
    ax.set_ylabel("BCE Loss")
    ax.grid(True, alpha=0.3)
    
    # 5. Log volume
    ax = axes[1, 0]
    ax.plot(smooth(history["log_vol"]), color='green')
    ax.set_title("Log Volume")
    ax.set_ylabel("Log σ")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    # 6. Lambda
    ax = axes[1, 1]
    ax.plot(history["lambda"], color='orange')
    ax.set_title("λ (failure price)")
    ax.set_ylabel("λ")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    # 7. CEM improvement
    ax = axes[1, 2]
    if "cem_improvement" in history and len(history["cem_improvement"]) > 0:
        ax.plot(smooth(history["cem_improvement"]), color='cyan')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title("CEM Improvement per Episode")
    ax.set_ylabel("Score Δ")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    # 8. MSE
    ax = axes[1, 3]
    ax.plot(smooth(history["mse"]), color='brown')
    ax.set_title("MSE (mean fit)")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Competitive Port Binding Training", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--z-dim", type=int, default=4)
    
    # CEM parameters
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--cem-samples", type=int, default=128)
    parser.add_argument("--cem-elites", type=int, default=16)
    
    # Scoring weights
    parser.add_argument("--intent-weight", type=float, default=1.0)
    parser.add_argument("--agency-weight", type=float, default=0.5)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    
    # Homeostasis parameters
    parser.add_argument("--alpha-vol", type=float, default=0.5)
    parser.add_argument("--eta-lambda", type=float, default=0.05)
    
    # Difficulty parameters (new)
    parser.add_argument("--k-sigma", type=float, default=1.5, 
                        help="K-sigma threshold for binding (lower = harder)")
    parser.add_argument("--base-noise", type=float, default=0.03,
                        help="Base noise for all rules")
    parser.add_argument("--bimodal", action="store_true", default=True,
                        help="Enable bimodal trajectories")
    parser.add_argument("--no-bimodal", action="store_false", dest="bimodal")
    
    # Training phases (new)
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Tube warmup before risk-aware CEM")
    
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"competitive_{timestamp}"
    if args.name:
        run_name = f"competitive_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Competitive Port Binding Experiment")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Save config
    config = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create env with configurable difficulty
    env = SimpleEnv(
        seed=args.seed,
        base_noise=args.base_noise,
        bimodal=args.bimodal,
    )
    
    # Create actor
    actor = CompetitiveActor(
        obs_dim=8,
        pred_dim=2,
        z_dim=args.z_dim,
        T=16,
        k_sigma=args.k_sigma,
        cem_iters=args.cem_iters,
        cem_samples=args.cem_samples,
        cem_elites=args.cem_elites,
        intent_weight=args.intent_weight,
        agency_weight=args.agency_weight,
        risk_weight=args.risk_weight,
        alpha_vol=args.alpha_vol,
        eta_lambda=args.eta_lambda,
    )
    
    # Two-timescale training:
    # Phase 1: Warm up tube without risk-aware selection (Random mode)
    if args.warmup_steps > 0:
        print(f"\n[Phase 1] Tube warmup ({args.warmup_steps} steps, Random mode)")
        warmup_history = train(
            actor, env, 
            n_steps=args.warmup_steps, 
            mode=BindingMode.RANDOM,
            log_every=args.warmup_steps // 5,
        )
        print(f"  Warmup complete: bind={warmup_history['bind_hard'][-1]:.3f}, "
              f"vol={warmup_history['log_vol'][-1]:.2f}")
    
    # Phase 2: Train with CEM binding (risk-aware)
    print(f"\n[Phase 2] CEM training ({args.steps} steps)")
    history = train(actor, env, n_steps=args.steps, mode=BindingMode.CEM)
    
    # Evaluate all binding modes
    print("\n" + "="*60)
    print("EVALUATING BINDING MODES")
    print("="*60)
    mode_summary = evaluate_binding_modes(actor, env)
    
    # Evaluate per-rule for CEM and Random
    print("\nEvaluating per-rule (CEM)...")
    cem_per_rule = evaluate_per_rule(actor, env, BindingMode.CEM)
    print("Evaluating per-rule (Random)...")
    random_per_rule = evaluate_per_rule(actor, env, BindingMode.RANDOM)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training(history, out_dir / "training.png")
    plot_mode_comparison(mode_summary, out_dir / "mode_comparison.png")
    plot_bind_vs_volume(mode_summary, out_dir / "bind_vs_volume.png")
    plot_per_rule_comparison(cem_per_rule, random_per_rule, out_dir / "per_rule.png")
    plot_risk_calibration(mode_summary, out_dir / "risk_calibration.png")
    
    try:
        plot_z_space(mode_summary, out_dir / "z_space.png")
    except ImportError:
        print("  (Skipping z_space plot - sklearn not available)")
    
    # Save results
    torch.save(actor.state_dict(), out_dir / "model.pt")
    
    # Convert numpy arrays for JSON
    def make_serializable(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                continue  # Skip numpy arrays
            elif isinstance(v, dict):
                result[k] = make_serializable(v)
            else:
                result[k] = v
        return result
    
    with open(out_dir / "mode_summary.json", "w") as f:
        json.dump(make_serializable(mode_summary), f, indent=2)
    
    # Final analysis
    print(f"\n{'='*60}")
    print("COMPETITIVE BINDING RESULTS")
    print(f"{'='*60}")
    
    # Success criteria
    cem_bind = mode_summary["cem"]["bind_mean"]
    random_bind = mode_summary["random"]["bind_mean"]
    cem_vol = mode_summary["cem"]["log_vol_mean"]
    random_vol = mode_summary["random"]["log_vol_mean"]
    oracle_bind = mode_summary["oracle"]["bind_mean"]
    
    print(f"\nMode Performance:")
    print(f"  Random:  bind={random_bind:.3f}, vol={random_vol:.2f}")
    print(f"  CEM:     bind={cem_bind:.3f}, vol={cem_vol:.2f}")
    print(f"  Oracle:  bind={oracle_bind:.3f} (upper bound)")
    
    # Check success criteria (updated for v2)
    # CEM should achieve tighter tubes (lower vol) - agency
    cem_tighter = cem_vol < random_vol - 0.3
    
    # CEM trades some bind for tightness - this is the intended tradeoff
    bind_tradeoff_reasonable = cem_bind >= 0.7  # Still reliable enough
    
    # CEM should provide meaningfully better volume efficiency
    volume_improvement = random_vol - cem_vol
    
    # Rule adaptation check
    cem_noisy_sigma = cem_per_rule[3]["sigma_mean"]
    cem_easy_sigma = cem_per_rule[0]["sigma_mean"]
    rule_adapted = cem_noisy_sigma > cem_easy_sigma * 1.1
    
    # Risk calibration
    cem_risk_corr = mode_summary["cem"].get("risk_corr", 0)
    risk_calibrated = cem_risk_corr > 0.3
    
    print(f"\nSuccess Criteria:")
    print(f"  {'✓' if cem_tighter else '✗'} CEM achieves tighter tubes ({cem_vol:.2f} vs {random_vol:.2f}, Δ={volume_improvement:.2f})")
    print(f"  {'✓' if bind_tradeoff_reasonable else '✗'} CEM maintains acceptable bind rate ({cem_bind:.3f} ≥ 0.7)")
    print(f"  {'✓' if rule_adapted else '✗'} CEM adapts to rules (noisy σ={cem_noisy_sigma:.3f} vs easy σ={cem_easy_sigma:.3f})")
    print(f"  {'✓' if risk_calibrated else '✗'} Risk critic calibrated (corr={cem_risk_corr:.3f} > 0.3)")
    
    n_success = sum([cem_tighter, bind_tradeoff_reasonable, rule_adapted, risk_calibrated])
    if n_success >= 3:
        print("\n🎉 SUCCESS: Competitive binding achieved!")
        print(f"   CEM finds z* that optimizes the agency-reliability tradeoff.")
        print(f"   Trade: {(random_bind - cem_bind)*100:.1f}% bind → {volume_improvement:.2f} volume improvement")
    else:
        print(f"\n⚠ Competitive binding incomplete ({n_success}/4 criteria). May need tuning.")
    
    # CEM search quality
    if "cem_improvement" in history and len(history["cem_improvement"]) > 0:
        avg_improvement = np.mean(history["cem_improvement"][-1000:])
        print(f"\nCEM Search Quality:")
        print(f"  Average score improvement: {avg_improvement:.4f}")
        print(f"  {'✓' if avg_improvement > 0 else '✗'} CEM iterations improve score")
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
