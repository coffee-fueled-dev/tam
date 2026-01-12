"""
Visualize tubes and uncertainty geometry.

Provides multiple views:
1. Tube overlay: trajectory with 2σ uncertainty ellipses
2. Cone cross-section: tube width over time
3. z-space: how z values map to tube shapes
4. Rule comparison: side-by-side tubes for different rules
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import EllipseCollection
import numpy as np
import torch

from actor import Actor
from train_actor import SimpleEnv


def plot_tube_2d(
    ax,
    mu: np.ndarray,
    sigma: np.ndarray,
    trajectory: Optional[np.ndarray] = None,
    k_sigma: float = 2.0,
    title: str = "",
    show_ellipses_every: int = 2,
    cmap: str = "Blues",
):
    """
    Plot 2D tube with uncertainty ellipses.
    
    Args:
        mu: [T, 2] trajectory mean
        sigma: [T, 2] tube std
        trajectory: [T, 2] actual trajectory (optional)
        k_sigma: number of sigmas to show
    """
    T = mu.shape[0]
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, T))
    
    # Plot uncertainty ellipses
    for t in range(0, T, show_ellipses_every):
        ellipse = patches.Ellipse(
            (mu[t, 0], mu[t, 1]),
            width=2 * k_sigma * sigma[t, 0],
            height=2 * k_sigma * sigma[t, 1],
            alpha=0.15,
            color=colors[t],
            linewidth=0,
        )
        ax.add_patch(ellipse)
    
    # Plot mean trajectory
    ax.plot(mu[:, 0], mu[:, 1], 'b-', linewidth=2, label='Tube mean', zorder=3)
    ax.scatter(mu[::show_ellipses_every, 0], mu[::show_ellipses_every, 1], 
              c=colors[::show_ellipses_every], s=30, zorder=4, edgecolors='white')
    
    # Plot actual trajectory
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', 
               linewidth=1.5, label='Actual', zorder=2, alpha=0.7)
        ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], 
                  c='green', s=80, marker='o', label='Start', zorder=5)
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], 
                  c='red', s=80, marker='*', label='End', zorder=5)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cone_profile(
    ax,
    sigma: np.ndarray,
    title: str = "Tube Width Over Time",
    label: str = None,
    color: str = None,
):
    """
    Plot tube width (sigma) over time - shows the "cone" shape.
    
    Args:
        sigma: [T, D] tube std
    """
    T = sigma.shape[0]
    t = np.arange(T)
    
    # Total "volume" at each timestep
    vol = np.prod(sigma, axis=-1)
    
    kwargs = {"linewidth": 2}
    if label:
        kwargs["label"] = label
    if color:
        kwargs["color"] = color
    
    ax.plot(t, vol, **kwargs)
    ax.fill_between(t, 0, vol, alpha=0.2, color=color)
    
    ax.set_xlabel("Time step")
    ax.set_ylabel("Tube width (σ₁ × σ₂)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()


def plot_sigma_components(
    ax,
    sigma: np.ndarray,
    title: str = "Sigma Components",
):
    """Plot individual sigma dimensions over time."""
    T, D = sigma.shape
    t = np.arange(T)
    
    for d in range(D):
        ax.plot(t, sigma[:, d], label=f"σ[{d}]", linewidth=2)
    
    ax.set_xlabel("Time step")
    ax.set_ylabel("Standard deviation")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def visualize_single_episode(
    actor: Actor,
    env: SimpleEnv,
    out_path: Path,
    rule: Optional[int] = None,
):
    """Visualize a single episode's tube."""
    
    # Get episode with specified rule
    for _ in range(100):
        s0 = env.reset()
        if rule is None or env.rule == rule:
            break
    
    s0_t = torch.tensor(s0, dtype=torch.float32)
    traj = env.generate_trajectory(T=actor.T)
    
    with torch.no_grad():
        out = actor.get_commitment(s0_t)
    
    mu = out.mu.numpy()
    sigma = out.sigma.numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Tube overlay
    plot_tube_2d(
        axes[0, 0], mu, sigma, traj,
        title=f"Tube Overlay (Rule {env.rule})",
    )
    axes[0, 0].set_xlim(-0.1, 1.1)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # 2. Cone profile
    plot_cone_profile(
        axes[0, 1], sigma,
        title="Tube Width Over Time",
    )
    
    # 3. Sigma components
    plot_sigma_components(
        axes[1, 0], sigma,
        title="Sigma Components",
    )
    
    # 4. z values
    ax = axes[1, 1]
    z = out.z.numpy()
    ax.bar(range(len(z)), z, color='steelblue', edgecolor='black')
    ax.set_xlabel("z dimension")
    ax.set_ylabel("Value")
    ax.set_title(f"Latent z (norm={np.linalg.norm(z):.2f})")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    cone_vol = np.log(np.prod(sigma.mean(axis=0)) + 1e-8)
    bind_rate = actor.compute_bind_rate(
        out.mu, out.sigma, torch.tensor(traj, dtype=torch.float32)
    ).item()
    
    plt.suptitle(
        f"Episode Visualization | Rule {env.rule} | "
        f"Cone vol: {cone_vol:.2f} | Bind: {bind_rate:.2%}",
        fontsize=14
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {out_path}")


def visualize_rule_comparison(
    actor: Actor,
    env: SimpleEnv,
    out_path: Path,
):
    """Compare tubes across all rules."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for rule in range(4):
        # Find episode with this rule
        for _ in range(100):
            s0 = env.reset()
            if env.rule == rule:
                break
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj = env.generate_trajectory(T=actor.T)
        
        with torch.no_grad():
            out = actor.get_commitment(s0_t)
        
        mu = out.mu.numpy()
        sigma = out.sigma.numpy()
        
        # Top row: tube overlay
        plot_tube_2d(
            axes[0, rule], mu, sigma, traj,
            title=f"Rule {rule}",
            cmap=["Blues", "Greens", "Oranges", "Reds"][rule],
        )
        axes[0, rule].set_xlim(-0.1, 1.1)
        axes[0, rule].set_ylim(-0.1, 1.1)
        
        # Bottom row: cone profile
        cone_vol = np.log(np.prod(sigma.mean(axis=0)) + 1e-8)
        z_str = ", ".join(f"{v:.1f}" for v in out.z.numpy())
        
        vol = np.prod(sigma, axis=-1)
        axes[1, rule].fill_between(range(len(vol)), 0, vol, alpha=0.4)
        axes[1, rule].plot(vol, linewidth=2)
        axes[1, rule].set_title(f"Cone: {cone_vol:.2f}\nz=[{z_str}]", fontsize=10)
        axes[1, rule].set_xlabel("Time")
        axes[1, rule].set_ylabel("Width")
        axes[1, rule].grid(True, alpha=0.3)
    
    plt.suptitle("Tube Comparison Across Rules", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {out_path}")


def visualize_z_space(
    actor: Actor,
    env: SimpleEnv,
    out_path: Path,
    n_samples: int = 100,
):
    """Visualize how z relates to tube geometry."""
    
    z_values = []
    cone_vols = []
    rules = []
    
    for _ in range(n_samples):
        s0 = env.reset()
        s0_t = torch.tensor(s0, dtype=torch.float32)
        
        with torch.no_grad():
            out = actor.get_commitment(s0_t)
        
        z_values.append(out.z.numpy())
        cone_vol = np.log(torch.prod(out.sigma.mean(dim=0)).item() + 1e-8)
        cone_vols.append(cone_vol)
        rules.append(env.rule)
    
    z_values = np.array(z_values)
    cone_vols = np.array(cone_vols)
    rules = np.array(rules)
    
    z_dim = z_values.shape[1]
    n_plots = z_dim + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    rule_colors = ['blue', 'green', 'orange', 'red']
    
    # z dimension vs cone volume
    for d in range(z_dim):
        ax = axes[d]
        for r in range(4):
            mask = rules == r
            ax.scatter(z_values[mask, d], cone_vols[mask], 
                      c=rule_colors[r], alpha=0.6, label=f"Rule {r}", s=30)
        
        corr = np.corrcoef(z_values[:, d], cone_vols)[0, 1]
        ax.set_xlabel(f"z[{d}]")
        ax.set_ylabel("Log cone volume")
        ax.set_title(f"z[{d}] → Cone (r={corr:.2f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # PCA of z colored by cone volume
    ax = axes[z_dim]
    from sklearn.decomposition import PCA
    if z_dim > 2:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_values)
        explained = pca.explained_variance_ratio_.sum()
    else:
        z_2d = z_values[:, :2]
        explained = 1.0
    
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=cone_vols, 
                        cmap='coolwarm', alpha=0.7, s=40)
    plt.colorbar(scatter, ax=ax, label='Log cone vol')
    ax.set_xlabel("z PC1" if z_dim > 2 else "z[0]")
    ax.set_ylabel("z PC2" if z_dim > 2 else "z[1]")
    ax.set_title(f"z-space colored by cone vol\n({explained:.0%} var explained)")
    ax.grid(True, alpha=0.3)
    
    # Hide extra axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Z-Space → Tube Geometry", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {out_path}")


def visualize_tube_3d(
    actor: Actor,
    env: SimpleEnv,
    out_path: Path,
    rule: Optional[int] = None,
):
    """3D visualization of the tube (x, y, time)."""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Get episode
    for _ in range(100):
        s0 = env.reset()
        if rule is None or env.rule == rule:
            break
    
    s0_t = torch.tensor(s0, dtype=torch.float32)
    traj = env.generate_trajectory(T=actor.T)
    
    with torch.no_grad():
        out = actor.get_commitment(s0_t)
    
    mu = out.mu.numpy()
    sigma = out.sigma.numpy()
    T = mu.shape[0]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    t = np.arange(T)
    
    # Plot tube mean
    ax.plot(mu[:, 0], mu[:, 1], t, 'b-', linewidth=2, label='Tube mean')
    
    # Plot actual trajectory
    ax.plot(traj[:, 0], traj[:, 1], t, 'k--', linewidth=1.5, label='Actual', alpha=0.7)
    
    # Plot tube cross-sections (circles at each timestep)
    for i in range(0, T, 2):
        theta = np.linspace(0, 2*np.pi, 50)
        x = mu[i, 0] + 2 * sigma[i, 0] * np.cos(theta)
        y = mu[i, 1] + 2 * sigma[i, 1] * np.sin(theta)
        z = np.full_like(theta, i)
        ax.plot(x, y, z, color='blue', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    ax.set_title(f"3D Tube Visualization (Rule {env.rule})")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = model_path.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path}")
    print(f"Output directory: {out_dir}")
    
    # Create actor and load weights
    actor = Actor(obs_dim=8, pred_dim=2, action_dim=4, z_dim=4, T=16)
    actor.load_state_dict(torch.load(model_path, weights_only=True))
    actor.eval()
    
    env = SimpleEnv(seed=args.seed)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Single episode for each rule
    for rule in range(4):
        visualize_single_episode(
            actor, env, 
            out_dir / f"episode_rule{rule}.png",
            rule=rule,
        )
    
    # 2. Rule comparison
    visualize_rule_comparison(actor, env, out_dir / "rule_comparison.png")
    
    # 3. Z-space
    visualize_z_space(actor, env, out_dir / "z_space.png", n_samples=150)
    
    # 4. 3D tubes
    for rule in [0, 3]:  # Just show deterministic and noisy
        visualize_tube_3d(actor, env, out_dir / f"tube_3d_rule{rule}.png", rule=rule)
    
    print(f"\n✓ All visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
