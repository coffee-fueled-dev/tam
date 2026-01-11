"""
Train SelfCalibratingActor with emergent bind rate.

Key insight: The optimal bind rate is NOT a hyperparameter - it emerges
from the environment structure via the surprise-based λ update.

Expected behavior:
- Easy rules (deterministic): low residuals → failures are surprising → high λ → tight tubes
- Hard rules (noisy): high residuals → failures are expected → low λ → wider tubes
- Each rule converges to ITS OWN optimal bind rate
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from homeostasis_actor import SelfCalibratingActor


class SimpleEnv:
    """Minimal 2D environment with different dynamics per rule."""
    
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_rules = 4
        self.reset()
    
    def reset(self, rule: int = None) -> np.ndarray:
        self.x = self.rng.uniform(0.1, 0.9)
        self.y = self.rng.uniform(0.1, 0.9)
        self.goal_x = self.rng.uniform(0.1, 0.9)
        self.goal_y = self.rng.uniform(0.1, 0.9)
        self.rule = rule if rule is not None else self.rng.integers(0, self.n_rules)
        return self.observe()
    
    def observe(self) -> np.ndarray:
        rule_oh = np.zeros(self.n_rules)
        rule_oh[self.rule] = 1.0
        return np.array([self.x, self.y, self.goal_x, self.goal_y, *rule_oh], dtype=np.float32)
    
    def generate_trajectory(self, T: int = 16, noise_scale: float = 1.0) -> np.ndarray:
        """Generate trajectory based on rule."""
        traj = np.zeros((T, 2), dtype=np.float32)
        
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        
        for t in range(T):
            progress = (t + 1) / T
            
            if self.rule == 0:  # Straight (easiest)
                traj[t, 0] = self.x + dx * progress
                traj[t, 1] = self.y + dy * progress
            elif self.rule == 1:  # Curved CW (predictable)
                angle = progress * np.pi / 2
                traj[t, 0] = self.x + dx * progress + 0.1 * np.sin(angle * 2)
                traj[t, 1] = self.y + dy * progress - 0.1 * (1 - np.cos(angle * 2))
            elif self.rule == 2:  # Curved CCW (predictable)
                angle = progress * np.pi / 2
                traj[t, 0] = self.x + dx * progress - 0.1 * np.sin(angle * 2)
                traj[t, 1] = self.y + dy * progress + 0.1 * (1 - np.cos(angle * 2))
            elif self.rule == 3:  # Noisy (hardest)
                base_noise = 0.2 * noise_scale
                # More noise early, less later
                noise_t = base_noise * (1 - 0.5 * progress)
                traj[t, 0] = self.x + dx * progress + self.rng.normal(0, noise_t)
                traj[t, 1] = self.y + dy * progress + self.rng.normal(0, noise_t)
        
        return traj


def train(
    actor: SelfCalibratingActor,
    env: SimpleEnv,
    n_steps: int = 8000,
    log_every: int = 1000,
) -> Dict[str, List[float]]:
    """Train with self-calibrating homeostasis."""
    
    print(f"Training SelfCalibratingActor for {n_steps} steps...")
    print(f"  Volume weight (α): {actor.alpha_vol}")
    print(f"  λ learning rate (η): {actor.eta_lambda}")
    print(f"  Initial λ: {actor.lambda_fail}")
    print(f"  NO FIXED BIND TARGET - emergent from environment")
    
    for step in range(n_steps):
        s0 = env.reset()
        traj = env.generate_trajectory(T=actor.T)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        metrics = actor.train_step(s0_t, traj_t)
        
        if step % log_every == 0:
            stats = actor.get_equilibrium_stats(window=min(500, step + 1))
            print(f"  Step {step}: bind={metrics['bind_hard']:.3f}, "
                  f"σ={metrics['sigma_mean']:.3f}, λ={metrics['lambda']:.2f}")
            if stats:
                print(f"    Emergent bind: {stats['emergent_bind']:.3f}, "
                      f"σ: {stats['mean_sigma']:.3f}, λ stable: {'✓' if stats['lambda_stable'] else '✗'}")
    
    return actor.history


def evaluate_per_rule(
    actor: SelfCalibratingActor,
    env: SimpleEnv,
    n_samples: int = 100,
) -> Dict[int, Dict[str, float]]:
    """Evaluate emergent bind rate per rule."""
    actor.eval()
    
    results = {r: {"bind": [], "log_vol": [], "sigma": [], "residual": [], "z": []} for r in range(4)}
    
    for rule in range(4):
        for _ in range(n_samples):
            s0 = env.reset(rule=rule)
            traj = env.generate_trajectory(T=actor.T)
            
            s0_t = torch.tensor(s0, dtype=torch.float32)
            traj_t = torch.tensor(traj, dtype=torch.float32)
            
            with torch.no_grad():
                out = actor.get_commitment(s0_t)
                bind = actor.compute_bind_hard(out.mu, out.sigma, traj_t).item()
                log_vol = actor.compute_log_volume(out.sigma).item()
                sigma_mean = out.sigma.mean().item()
                residual = actor.compute_residual(out.mu, traj_t).item()
            
            results[rule]["bind"].append(bind)
            results[rule]["log_vol"].append(log_vol)
            results[rule]["sigma"].append(sigma_mean)
            results[rule]["residual"].append(residual)
            results[rule]["z"].append(out.z.numpy())
    
    # Summarize
    summary = {}
    for rule in range(4):
        summary[rule] = {
            "bind_mean": np.mean(results[rule]["bind"]),
            "bind_std": np.std(results[rule]["bind"]),
            "log_vol_mean": np.mean(results[rule]["log_vol"]),
            "log_vol_std": np.std(results[rule]["log_vol"]),
            "sigma_mean": np.mean(results[rule]["sigma"]),
            "sigma_std": np.std(results[rule]["sigma"]),
            "residual_mean": np.mean(results[rule]["residual"]),
            "residual_std": np.std(results[rule]["residual"]),
        }
    
    return summary


def plot_training(history: Dict[str, List[float]], out_path: Path):
    """Plot training curves showing emergent behavior."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    window = 200
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # 1. Bind rate (hard vs soft)
    ax = axes[0, 0]
    ax.plot(smooth(history["bind_hard"]), color='blue', alpha=0.8, label='Hard')
    ax.plot(smooth(history["bind_soft"]), color='cyan', alpha=0.6, label='Soft')
    ax.set_title("Emergent Bind Rate\n(no fixed target!)")
    ax.set_ylabel("Bind rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Log volume + sigma
    ax = axes[0, 1]
    ax.plot(smooth(history["log_vol"]), color='green', label='log(σ)')
    ax2 = ax.twinx()
    ax2.plot(smooth(history["sigma_mean"]), color='lightgreen', alpha=0.7, label='σ mean')
    ax2.set_ylabel("σ mean", color='lightgreen')
    ax.set_title("Volume (should NOT floor)")
    ax.set_ylabel("Log σ")
    ax.grid(True, alpha=0.3)
    
    # 3. Lambda (failure price)
    ax = axes[0, 2]
    ax.plot(history["lambda"], color='red')
    ax.set_title("λ (failure price)\n(self-calibrating)")
    ax.set_ylabel("λ")
    ax.grid(True, alpha=0.3)
    
    # 4. Hardness (normalized by sigma)
    ax = axes[1, 0]
    ax.plot(smooth(history["hardness"]), color='purple', alpha=0.8)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.5, label='h=1')
    ax.set_title("Hardness (residual/σ)\n(creates negative feedback)")
    ax.set_ylabel("Hardness")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Fail rate (soft, continuous)
    ax = axes[1, 1]
    ax.plot(smooth(history["fail_soft"]), color='red', alpha=0.8)
    ax.set_title("Soft Fail Rate\n(continuous for gradients)")
    ax.set_ylabel("Fail rate")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    # 6. MSE
    ax = axes[1, 2]
    ax.plot(smooth(history["mse"]), color='orange')
    ax.set_title("MSE (mean fit)")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Self-Calibrating Homeostasis (with Soft Bind)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_rule_emergent(summary: Dict[int, Dict], out_path: Path):
    """Plot per-rule metrics showing emergent differentiation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    rules = list(summary.keys())
    rule_names = ["Straight\n(easy)", "Curved CW\n(medium)", "Curved CCW\n(medium)", "Noisy\n(hard)"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]  # Green to red
    
    # 1. Emergent bind rate
    ax = axes[0, 0]
    binds = [summary[r]["bind_mean"] for r in rules]
    bind_stds = [summary[r]["bind_std"] for r in rules]
    ax.bar(rules, binds, yerr=bind_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(rules)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("Bind Rate")
    ax.set_title("EMERGENT Bind Rate by Rule\n(expected: lower for hard rules)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Sigma (tube width)
    ax = axes[0, 1]
    sigmas = [summary[r]["sigma_mean"] for r in rules]
    sigma_stds = [summary[r]["sigma_std"] for r in rules]
    ax.bar(rules, sigmas, yerr=sigma_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(rules)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("σ (tube width)")
    ax.set_title("Tube Width by Rule\n(expected: LARGER for hard rules)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Log volume
    ax = axes[1, 0]
    vols = [summary[r]["log_vol_mean"] for r in rules]
    vol_stds = [summary[r]["log_vol_std"] for r in rules]
    ax.bar(rules, vols, yerr=vol_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(rules)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("Log Volume")
    ax.set_title("Log Tube Volume by Rule")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Residual (inherent difficulty)
    ax = axes[1, 1]
    residuals = [summary[r]["residual_mean"] for r in rules]
    residual_stds = [summary[r]["residual_std"] for r in rules]
    ax.bar(rules, residuals, yerr=residual_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(rules)
    ax.set_xticklabels(rule_names)
    ax.set_ylabel("Residual |τ - μ|")
    ax.set_title("Prediction Difficulty\n(ground truth for calibration)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Self-Calibration: Volume Should Increase with Difficulty", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_calibration_check(summary: Dict[int, Dict], out_path: Path):
    """Check if bind rate correlates with inverse difficulty."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    rule_names = ["Straight", "Curved CW", "Curved CCW", "Noisy"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    
    for rule in range(4):
        ax.scatter(
            summary[rule]["residual_mean"],
            summary[rule]["bind_mean"],
            s=200, c=colors[rule], label=rule_names[rule],
            edgecolors='black', linewidths=2
        )
        # Error bars
        ax.errorbar(
            summary[rule]["residual_mean"],
            summary[rule]["bind_mean"],
            xerr=summary[rule]["residual_std"],
            yerr=summary[rule]["bind_std"],
            color=colors[rule], fmt='none', capsize=5
        )
    
    # Fit trend line
    residuals = [summary[r]["residual_mean"] for r in range(4)]
    binds = [summary[r]["bind_mean"] for r in range(4)]
    
    if len(set(residuals)) > 1:  # Avoid division by zero
        z = np.polyfit(residuals, binds, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(residuals) * 0.9, max(residuals) * 1.1, 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel("Prediction Difficulty (residual)")
    ax.set_ylabel("Emergent Bind Rate")
    ax.set_title("Self-Calibration Check\n(Expected: negative correlation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--alpha-vol", type=float, default=0.5, help="Volume weight")
    parser.add_argument("--eta-lambda", type=float, default=0.05, help="λ learning rate")
    parser.add_argument("--lambda-init", type=float, default=1.0, help="Initial λ")
    parser.add_argument("--beta-kl", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"selfcal_{timestamp}"
    if args.name:
        run_name = f"selfcal_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Self-Calibrating Homeostasis")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Save config
    config = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create actor
    env = SimpleEnv(seed=args.seed)
    actor = SelfCalibratingActor(
        obs_dim=8,
        pred_dim=2,
        z_dim=args.z_dim,
        T=16,
        alpha_vol=args.alpha_vol,
        eta_lambda=args.eta_lambda,
        lambda_init=args.lambda_init,
        beta_kl=args.beta_kl,
    )
    
    # Train
    history = train(actor, env, n_steps=args.steps)
    
    # Evaluate per-rule
    print("\nEvaluating per-rule performance...")
    per_rule = evaluate_per_rule(actor, env)
    
    print("\nPer-rule results (EMERGENT, not fixed):")
    for rule in range(4):
        print(f"  Rule {rule}: bind={per_rule[rule]['bind_mean']:.3f}±{per_rule[rule]['bind_std']:.3f}, "
              f"vol={per_rule[rule]['log_vol_mean']:.2f}±{per_rule[rule]['log_vol_std']:.2f}, "
              f"residual={per_rule[rule]['residual_mean']:.3f}")
    
    # Equilibrium stats
    stats = actor.get_equilibrium_stats(window=1000)
    print(f"\nEquilibrium stats (last 1000 steps):")
    print(f"  Mean emergent bind: {stats['emergent_bind']:.3f}")
    print(f"  Mean λ: {stats['mean_lambda']:.2f}")
    print(f"  λ stable: {'✓' if stats['lambda_stable'] else '✗'}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training(history, out_dir / "training.png")
    plot_per_rule_emergent(per_rule, out_dir / "per_rule.png")
    plot_calibration_check(per_rule, out_dir / "calibration.png")
    
    # Save results
    torch.save(actor.state_dict(), out_dir / "model.pt")
    
    with open(out_dir / "per_rule.json", "w") as f:
        json.dump({str(k): v for k, v in per_rule.items()}, f, indent=2)
    
    with open(out_dir / "equilibrium.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")
    
    # Check self-calibration success
    noisy_bind = per_rule[3]["bind_mean"]
    easy_bind = per_rule[0]["bind_mean"]
    noisy_vol = per_rule[3]["log_vol_mean"]
    easy_vol = per_rule[0]["log_vol_mean"]
    noisy_sigma = per_rule[3]["sigma_mean"]
    easy_sigma = per_rule[0]["sigma_mean"]
    noisy_res = per_rule[3]["residual_mean"]
    easy_res = per_rule[0]["residual_mean"]
    
    print(f"\n{'='*60}")
    print("SELF-CALIBRATION CHECK")
    print(f"{'='*60}")
    print(f"Easy (rule 0): bind={easy_bind:.3f}, σ={easy_sigma:.3f}, log_vol={easy_vol:.2f}, difficulty={easy_res:.3f}")
    print(f"Hard (rule 3): bind={noisy_bind:.3f}, σ={noisy_sigma:.3f}, log_vol={noisy_vol:.2f}, difficulty={noisy_res:.3f}")
    
    # Key success criteria
    calibrated_bind = noisy_bind < easy_bind - 0.05  # Hard rules should have lower bind
    calibrated_sigma = noisy_sigma > easy_sigma * 1.1  # Hard rules should have wider tubes
    calibrated_vol = noisy_vol > easy_vol + 0.1  # Hard rules should have larger volume
    difficulty_ordering = noisy_res > easy_res  # Sanity: noisy is actually harder
    
    print(f"\n{'✓' if difficulty_ordering else '✗'} Difficulty ordering correct: hard has higher residual")
    print(f"{'✓' if calibrated_bind else '✗'} Bind rate calibrated: hard < easy ({noisy_bind:.3f} < {easy_bind:.3f})")
    print(f"{'✓' if calibrated_sigma else '✗'} Sigma calibrated: hard > easy ({noisy_sigma:.3f} > {easy_sigma:.3f})")
    print(f"{'✓' if calibrated_vol else '✗'} Volume calibrated: hard > easy ({noisy_vol:.2f} > {easy_vol:.2f})")
    
    if calibrated_sigma and (calibrated_bind or calibrated_vol):
        print("\n🎉 SUCCESS: Self-calibration achieved!")
        print("   The agent learned wider tubes for harder rules.")
    else:
        print("\n⚠ Self-calibration not fully achieved. May need more training or tuning.")


if __name__ == "__main__":
    main()
