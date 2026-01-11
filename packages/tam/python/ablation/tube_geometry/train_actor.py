"""
Train and evaluate the Actor on the simple environment.

Tests the core thesis: z can learn to control tube geometry.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from actor import Actor


class SimpleEnv:
    """Minimal 2D environment with different dynamics per rule."""
    
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_rules = 4
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.x = self.rng.uniform(0.1, 0.9)
        self.y = self.rng.uniform(0.1, 0.9)
        self.goal_x = self.rng.uniform(0.1, 0.9)
        self.goal_y = self.rng.uniform(0.1, 0.9)
        self.rule = self.rng.integers(0, self.n_rules)
        return self.observe()
    
    def observe(self) -> np.ndarray:
        rule_oh = np.zeros(self.n_rules)
        rule_oh[self.rule] = 1.0
        return np.array([self.x, self.y, self.goal_x, self.goal_y, *rule_oh], dtype=np.float32)
    
    def generate_trajectory(self, T: int = 16) -> np.ndarray:
        traj = np.zeros((T, 2), dtype=np.float32)
        
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        
        for t in range(T):
            progress = (t + 1) / T
            
            if self.rule == 0:
                traj[t, 0] = self.x + dx * progress
                traj[t, 1] = self.y + dy * progress
            elif self.rule == 1:
                angle = progress * np.pi / 2
                traj[t, 0] = self.x + dx * progress + 0.1 * np.sin(angle * 2)
                traj[t, 1] = self.y + dy * progress - 0.1 * (1 - np.cos(angle * 2))
            elif self.rule == 2:
                angle = progress * np.pi / 2
                traj[t, 0] = self.x + dx * progress - 0.1 * np.sin(angle * 2)
                traj[t, 1] = self.y + dy * progress + 0.1 * (1 - np.cos(angle * 2))
            elif self.rule == 3:
                noise_scale = 0.15
                traj[t, 0] = self.x + dx * progress + self.rng.normal(0, noise_scale * (1 - progress))
                traj[t, 1] = self.y + dy * progress + self.rng.normal(0, noise_scale * (1 - progress))
        
        return traj


def train(
    actor: Actor,
    env: SimpleEnv,
    n_steps: int = 4000,
    T: int = 16,
) -> Dict[str, List[float]]:
    """Train the actor on the environment."""
    
    print(f"Training Actor for {n_steps} steps...")
    print(f"  z_dim: {actor.z_dim}")
    print(f"  T: {T}")
    print(f"  beta_kl: {actor.beta_kl}")
    
    for step in range(n_steps):
        s0 = env.reset()
        traj = env.generate_trajectory(T=T)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        metrics = actor.train_step(s0_t, traj_t)
        
        if step % 500 == 0:
            print(f"  Step {step}: nll={metrics['nll']:.3f}, bind={metrics['bind']:.3f}, "
                  f"kl={metrics['kl']:.3f}, cone={metrics['cone_vol']:.4f}")
    
    return actor.history


def evaluate(actor: Actor, env: SimpleEnv, n_samples: int = 100) -> Dict:
    """Evaluate the trained actor."""
    actor.eval()
    
    z_values = []
    cone_vols = []
    bind_rates = []
    rules = []
    
    for _ in range(n_samples):
        s0 = env.reset()
        traj = env.generate_trajectory(T=actor.T)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        with torch.no_grad():
            out = actor.get_commitment(s0_t)
            bind = actor.compute_bind_rate(out.mu, out.sigma, traj_t)
        
        z_values.append(out.z.numpy())
        cone_vol = np.log(torch.prod(out.sigma.mean(dim=0)).item() + 1e-8)
        cone_vols.append(cone_vol)
        bind_rates.append(bind.item())
        rules.append(env.rule)
    
    z_values = np.array(z_values)
    cone_vols = np.array(cone_vols)
    bind_rates = np.array(bind_rates)
    rules = np.array(rules)
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    print(f"\nOverall bind rate: {bind_rates.mean():.3f}")
    print(f"Overall cone vol: {cone_vols.mean():.3f} ± {cone_vols.std():.3f}")
    
    print(f"\nPer-rule statistics:")
    for r in range(4):
        mask = rules == r
        if mask.sum() > 0:
            z_mean = z_values[mask].mean(axis=0)
            print(f"  Rule {r}: bind={bind_rates[mask].mean():.3f}, "
                  f"cone={cone_vols[mask].mean():.3f}±{cone_vols[mask].std():.3f}, "
                  f"z=[{', '.join(f'{v:.2f}' for v in z_mean)}]")
    
    print(f"\nz-cone correlations:")
    for d in range(z_values.shape[1]):
        corr = np.corrcoef(z_values[:, d], cone_vols)[0, 1]
        marker = "✓" if abs(corr) > 0.3 else ""
        print(f"  z[{d}] ↔ cone: r = {corr:.3f} {marker}")
    
    return {
        "z_values": z_values,
        "cone_vols": cone_vols,
        "bind_rates": bind_rates,
        "rules": rules,
    }


def test_z_manipulation(actor: Actor, env: SimpleEnv, out_dir: Path):
    """Test if z manipulation controls tube geometry."""
    actor.eval()
    
    # Get a Rule 0 situation
    for _ in range(100):
        s0 = env.reset()
        if env.rule == 0:
            break
    
    s0_t = torch.tensor(s0, dtype=torch.float32)
    traj = env.generate_trajectory(T=actor.T)
    traj_t = torch.tensor(traj, dtype=torch.float32)
    
    with torch.no_grad():
        z_mu, z_logstd = actor.tube.encode(s0_t.unsqueeze(0))
    
    print("\n" + "=" * 60)
    print("Z MANIPULATION TEST")
    print("=" * 60)
    print(f"Situation: Rule 0")
    print(f"Natural z: [{', '.join(f'{v:.3f}' for v in z_mu.squeeze().numpy())}]")
    
    # Find which z dimension has highest correlation with cone
    results = actor.evaluate_z_control(s0_t)
    
    # Group by dimension and compute correlation
    dim_effects = {}
    for r in results:
        d = r["dim"]
        if d not in dim_effects:
            dim_effects[d] = {"scales": [], "cones": []}
        dim_effects[d]["scales"].append(r["scale"])
        dim_effects[d]["cones"].append(r["cone_vol"])
    
    print("\nz dimension → cone sensitivity:")
    best_dim = 0
    best_range = 0
    for d, data in dim_effects.items():
        cone_range = max(data["cones"]) - min(data["cones"])
        print(f"  z[{d}]: range = {cone_range:.3f}")
        if cone_range > best_range:
            best_range = cone_range
            best_dim = d
    
    print(f"\nBest control dimension: z[{best_dim}] (range = {best_range:.3f})")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: All dimensions
    ax = axes[0]
    for d, data in dim_effects.items():
        ax.plot(data["scales"], data["cones"], label=f"z[{d}]", linewidth=2)
    ax.set_xlabel("z perturbation")
    ax.set_ylabel("Log cone volume")
    ax.set_title("z → Cone Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Best dimension with tube visualization
    ax = axes[1]
    
    scales_to_show = [-2, -1, 0, 1, 2]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(scales_to_show)))
    
    for i, scale in enumerate(scales_to_show):
        z = z_mu.clone()
        z[0, best_dim] = z[0, best_dim] + scale
        
        with torch.no_grad():
            mu, sigma = actor.tube.predict_tube(s0_t.unsqueeze(0), z)
        
        mu_np = mu.squeeze().numpy()
        sigma_np = sigma.squeeze().numpy()
        
        ax.plot(mu_np[:, 0], mu_np[:, 1], color=colors[i], 
               label=f"z[{best_dim}]+{scale}", linewidth=2)
        
        # Show tube at t=8
        t = 8
        ellipse = plt.matplotlib.patches.Ellipse(
            (mu_np[t, 0], mu_np[t, 1]),
            width=4 * sigma_np[t, 0],
            height=4 * sigma_np[t, 1],
            alpha=0.2,
            color=colors[i],
        )
        ax.add_patch(ellipse)
    
    ax.plot(traj[:, 0], traj[:, 1], 'k--', linewidth=2, label='Actual')
    ax.set_title(f"Tube with varying z[{best_dim}]")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Z Manipulation: Does z control tube geometry?", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "z_manipulation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved: {out_dir / 'z_manipulation.png'}")
    
    if best_range > 1.0:
        print(f"\n✓ SUCCESS: z controls cone geometry (range = {best_range:.2f})")
    else:
        print(f"\n✗ z has weak control (range = {best_range:.2f})")


def plot_training(history: Dict[str, List[float]], out_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(history["nll"])
    axes[0, 0].set_title("NLL")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["bind"])
    axes[0, 1].axhline(0.85, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title("Bind Rate")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history["kl"])
    axes[0, 2].set_title("KL Divergence")
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history["cone_vol"])
    axes[1, 0].set_title("Cone Volume")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history["z_norm"])
    axes[1, 1].set_title("z Norm")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Smoothed bind rate
    window = 100
    if len(history["bind"]) > window:
        smoothed = np.convolve(history["bind"], np.ones(window)/window, mode='valid')
        axes[1, 2].plot(smoothed)
        axes[1, 2].axhline(0.85, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].set_title(f"Bind Rate (smoothed, window={window})")
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle("Actor Training", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "training.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--beta-kl", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"actor_{timestamp}"
    if args.name:
        run_name = f"actor_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Save config
    config = {
        "steps": args.steps,
        "z_dim": args.z_dim,
        "beta_kl": args.beta_kl,
        "seed": args.seed,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create actor and environment
    env = SimpleEnv(seed=args.seed)
    actor = Actor(
        obs_dim=8,
        pred_dim=2,
        action_dim=4,
        z_dim=args.z_dim,
        T=16,
        beta_kl=args.beta_kl,
    )
    
    # Train
    history = train(actor, env, n_steps=args.steps)
    
    # Evaluate
    eval_results = evaluate(actor, env, n_samples=100)
    
    # Test z manipulation
    test_z_manipulation(actor, env, out_dir)
    
    # Plot training
    plot_training(history, out_dir)
    
    # Save model
    torch.save(actor.state_dict(), out_dir / "model.pt")
    
    # Save eval results
    eval_summary = {
        "mean_bind": float(eval_results["bind_rates"].mean()),
        "mean_cone": float(eval_results["cone_vols"].mean()),
        "per_rule_cone": {
            str(r): float(eval_results["cone_vols"][eval_results["rules"] == r].mean())
            for r in range(4)
        },
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
