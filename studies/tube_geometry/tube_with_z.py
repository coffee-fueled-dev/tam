"""
Ablation: Can z control tube geometry?

Fixed horizon (T=16), add z back to tube network.
Goal: Show that z can learn to encode/control cone geometry.

s0 → encoder → z_mu, z_logstd
z ~ N(z_mu, exp(z_logstd))
(s0, z) → tube_net → (mu_knots, sigma_knots)

Success = different z values produce visibly different cone volumes.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TubeNetWithZ(nn.Module):
    """
    Tube network that takes (s0, z) as input.
    
    z is sampled from a posterior q(z|s0).
    
    Key design choice (sigma_from_z_only):
    - If True: sigma depends ONLY on z (not s0)
    - If False: sigma depends on (s0, z)
    
    When sigma_from_z_only=True, z becomes the SOLE controller of tube geometry.
    """
    
    def __init__(
        self,
        state_dim: int,
        z_dim: int = 4,
        pred_dim: int = 2,
        M: int = 8,
        hidden_dim: int = 64,
        sigma_from_z_only: bool = False,  # NEW: sigma depends only on z
    ):
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.pred_dim = pred_dim
        self.M = M
        self.sigma_from_z_only = sigma_from_z_only
        
        # Encoder: s0 → z posterior
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logstd_head = nn.Linear(hidden_dim, z_dim)
        
        # Mu network: (s0, z) → mu (trajectory mean still uses s0)
        self.mu_encoder = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, pred_dim * M)
        
        # Sigma network: depends on z only OR (s0, z)
        if sigma_from_z_only:
            # Sigma depends ONLY on z - forces z to control geometry
            self.sigma_encoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            # Sigma depends on (s0, z)
            self.sigma_encoder = nn.Sequential(
                nn.Linear(state_dim + z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        self.sigma_head = nn.Linear(hidden_dim, pred_dim * M)
        
        # Initialize sigma to reasonable values
        nn.init.constant_(self.sigma_head.bias, 0.0)
    
    def encode(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode s0 to z posterior parameters."""
        h = self.encoder(s0)
        z_mu = self.z_mu_head(h)
        z_logstd = self.z_logstd_head(h).clamp(-4, 2)
        return z_mu, z_logstd
    
    def sample_z(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample from posterior."""
        std = torch.exp(z_logstd)
        eps = torch.randn_like(std)
        return z_mu + std * eps
    
    def tube_forward(
        self, s0: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute tube parameters from (s0, z).
        
        Returns:
            mu_knots: [B, M, pred_dim]
            logsig_knots: [B, M, pred_dim]
        """
        # Mu always uses (s0, z) - needs to know where trajectory goes
        h_mu = self.mu_encoder(torch.cat([s0, z], dim=-1))
        mu = self.mu_head(h_mu).view(-1, self.M, self.pred_dim)
        
        # Sigma uses z only OR (s0, z) depending on config
        if self.sigma_from_z_only:
            h_sigma = self.sigma_encoder(z)
        else:
            h_sigma = self.sigma_encoder(torch.cat([s0, z], dim=-1))
        
        logsig = self.sigma_head(h_sigma).view(-1, self.M, self.pred_dim)
        
        return mu, logsig
    
    def forward(self, s0: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward: encode, sample, predict tube."""
        z_mu, z_logstd = self.encode(s0)
        z = self.sample_z(z_mu, z_logstd)
        mu_knots, logsig_knots = self.tube_forward(s0, z)
        
        return {
            "z": z,
            "z_mu": z_mu,
            "z_logstd": z_logstd,
            "mu_knots": mu_knots,
            "logsig_knots": logsig_knots,
        }


def interp_knots(knots: torch.Tensor, T: int) -> torch.Tensor:
    """Linearly interpolate M knots to T timesteps."""
    B, M, D = knots.shape
    device = knots.device
    
    knot_times = torch.linspace(0, 1, M, device=device)
    query_times = torch.linspace(0, 1, T, device=device)
    
    idx = torch.searchsorted(knot_times, query_times).clamp(1, M - 1)
    t0 = knot_times[idx - 1]
    t1 = knot_times[idx]
    
    alpha = (query_times - t0) / (t1 - t0 + 1e-8)
    
    v0 = knots[:, idx - 1, :]
    v1 = knots[:, idx, :]
    
    alpha = alpha.view(1, T, 1)
    return v0 + alpha * (v1 - v0)


def kl_divergence(z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
    """KL(q(z|s0) || N(0,1))"""
    var = torch.exp(2 * z_logstd)
    kl = 0.5 * (var + z_mu**2 - 1 - 2 * z_logstd).sum(dim=-1)
    return kl.mean()


def compute_loss(
    net: TubeNetWithZ,
    s0: torch.Tensor,
    trajectory: torch.Tensor,
    k_sigma: float = 2.0,
    beta_kl: float = 0.001,
    w_cone: float = 0.1,
    target_bind: float = 0.85,
    lambda_bind: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute training loss."""
    T = trajectory.shape[0]
    
    # Forward pass
    out = net(s0.unsqueeze(0))
    
    # Interpolate to T steps
    mu_traj = interp_knots(out["mu_knots"], T).squeeze(0)
    logsig_traj = interp_knots(out["logsig_knots"], T).squeeze(0)
    std_traj = torch.exp(logsig_traj).clamp(0.01, 10.0)
    
    # NLL
    var_traj = std_traj ** 2
    sq_err = (trajectory - mu_traj) ** 2
    nll_t = 0.5 * (sq_err / var_traj + torch.log(var_traj)).sum(dim=-1)
    nll = nll_t.mean()  # Average over time (fixed horizon)
    
    # Cone volume
    cone_vol_t = torch.prod(std_traj, dim=-1)
    cone_vol = cone_vol_t.mean()
    
    # Bind rate (fraction of timesteps inside tube)
    inside = (torch.abs(trajectory - mu_traj) < k_sigma * std_traj).all(dim=-1).float()
    bind_rate = inside.mean()
    
    # KL divergence
    kl = kl_divergence(out["z_mu"], out["z_logstd"])
    
    # Loss
    loss = nll
    loss = loss + w_cone * cone_vol
    loss = loss + beta_kl * kl
    loss = loss + lambda_bind * (target_bind - bind_rate)
    
    return {
        "loss": loss,
        "nll": nll,
        "cone_vol": cone_vol,
        "bind_rate": bind_rate,
        "kl": kl,
        "z": out["z"].detach(),
        "z_mu": out["z_mu"].detach(),
        "mu_traj": mu_traj.detach(),
        "std_traj": std_traj.detach(),
    }


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
    n_steps: int = 3000,
    z_dim: int = 4,
    lr: float = 1e-3,
    beta_kl: float = 0.001,
    seed: int = 0,
    sigma_from_z_only: bool = False,
) -> Tuple[TubeNetWithZ, Dict[str, List[float]]]:
    """Train the tube network with z."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = SimpleEnv(seed=seed)
    net = TubeNetWithZ(
        state_dim=8, z_dim=z_dim, pred_dim=2, M=8, hidden_dim=64,
        sigma_from_z_only=sigma_from_z_only,
    )
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    lambda_bind = 1.0
    
    history = {
        "nll": [],
        "cone_vol": [],
        "bind_rate": [],
        "kl": [],
        "z_norm": [],
    }
    
    print(f"Training tube network WITH z (dim={z_dim})...")
    print(f"  beta_kl: {beta_kl}")
    print(f"  sigma_from_z_only: {sigma_from_z_only}")
    
    for step in range(n_steps):
        s0 = env.reset()
        traj = env.generate_trajectory(T=16)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        result = compute_loss(
            net, s0_t, traj_t,
            beta_kl=beta_kl,
            lambda_bind=lambda_bind,
        )
        
        optimizer.zero_grad()
        result["loss"].backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        # Update lambda_bind
        bind_gap = 0.85 - result["bind_rate"].item()
        lambda_bind = float(np.clip(lambda_bind + 0.01 * bind_gap, 0.01, 10.0))
        
        history["nll"].append(float(result["nll"].item()))
        history["cone_vol"].append(float(result["cone_vol"].item()))
        history["bind_rate"].append(float(result["bind_rate"].item()))
        history["kl"].append(float(result["kl"].item()))
        history["z_norm"].append(float(result["z"].norm().item()))
        
        if step % 500 == 0:
            print(f"  Step {step}: NLL={result['nll']:.3f}, bind={result['bind_rate']:.3f}, "
                  f"cone={result['cone_vol']:.3f}, KL={result['kl']:.3f}")
    
    return net, history


def evaluate_z_control(net: TubeNetWithZ, n_samples: int = 100):
    """Check if z controls cone geometry."""
    env = SimpleEnv(seed=42)
    
    z_norms = []
    cone_vols = []
    rules = []
    z_values = []
    
    for _ in range(n_samples):
        s0 = env.reset()
        s0_t = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            out = net(s0_t)
            z = out["z"].squeeze()
            
            # Compute cone volume
            mu = out["mu_knots"]
            logsig = out["logsig_knots"]
            std = torch.exp(logsig)
            cone_vol = torch.prod(std.mean(dim=1)).item()
        
        z_norms.append(float(z.norm().item()))
        cone_vols.append(np.log(cone_vol + 1e-8))
        rules.append(env.rule)
        z_values.append(z.numpy())
    
    z_norms = np.array(z_norms)
    cone_vols = np.array(cone_vols)
    rules = np.array(rules)
    z_values = np.array(z_values)
    
    print("\n" + "=" * 60)
    print("Z CONTROL EVALUATION")
    print("=" * 60)
    
    print(f"\nz statistics:")
    print(f"  Norm - Mean: {z_norms.mean():.3f}, Std: {z_norms.std():.3f}")
    for d in range(z_values.shape[1]):
        print(f"  z[{d}] - Mean: {z_values[:, d].mean():.3f}, Std: {z_values[:, d].std():.3f}")
    
    print(f"\nCone volume by rule:")
    for r in range(4):
        mask = rules == r
        print(f"  Rule {r}: cone={cone_vols[mask].mean():.3f}±{cone_vols[mask].std():.3f}")
    
    # Check correlation between z and cone volume
    print(f"\nCorrelation between z dimensions and cone volume:")
    for d in range(z_values.shape[1]):
        corr = np.corrcoef(z_values[:, d], cone_vols)[0, 1]
        marker = "✓" if abs(corr) > 0.3 else ""
        print(f"  z[{d}] ↔ cone_vol: r = {corr:.3f} {marker}")
    
    # Check if z separates rules
    print(f"\nz by rule (mean):")
    for r in range(4):
        mask = rules == r
        z_mean = z_values[mask].mean(axis=0)
        print(f"  Rule {r}: z = [{', '.join(f'{v:.2f}' for v in z_mean)}]")
    
    return z_values, cone_vols, rules


def visualize_z_effect(net: TubeNetWithZ, save_path: Path):
    """Visualize how changing z affects the tube."""
    env = SimpleEnv(seed=0)
    
    # Get a fixed s0
    s0 = env.reset()
    while env.rule != 0:  # Use rule 0 for visualization
        s0 = env.reset()
    
    s0_t = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
    traj = env.generate_trajectory(T=16)
    
    # Get baseline z
    with torch.no_grad():
        z_mu, z_logstd = net.encode(s0_t)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Vary each z dimension
    z_dim = z_mu.shape[1]
    
    for d in range(min(z_dim, 4)):
        ax = axes[d // 2, d % 2]
        
        for scale in [-2, -1, 0, 1, 2]:
            z = z_mu.clone()
            z[0, d] = z[0, d] + scale * 1.0
            
            with torch.no_grad():
                mu_knots, logsig_knots = net.tube_forward(s0_t, z)
            
            mu_traj = interp_knots(mu_knots, 16).squeeze(0).numpy()
            std_traj = torch.exp(interp_knots(logsig_knots, 16)).squeeze(0).numpy()
            
            cone_vol = np.prod(std_traj.mean(axis=0))
            
            alpha = 0.3 + 0.15 * (scale + 2)
            ax.plot(mu_traj[:, 0], mu_traj[:, 1], 
                   label=f"z[{d}]+{scale}: cone={np.log(cone_vol):.2f}",
                   alpha=alpha)
            
            # Show tube at a few points
            for t in [0, 8, 15]:
                ellipse = plt.matplotlib.patches.Ellipse(
                    (mu_traj[t, 0], mu_traj[t, 1]),
                    width=4 * std_traj[t, 0],
                    height=4 * std_traj[t, 1],
                    alpha=0.1,
                    color='blue',
                )
                ax.add_patch(ellipse)
        
        ax.plot(traj[:, 0], traj[:, 1], 'k--', linewidth=2, label='Actual traj')
        ax.set_title(f"Varying z[{d}]")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Summary plot: z_dim effect on cone volume
    ax = axes[1, 2]
    
    for d in range(min(z_dim, 4)):
        scales = np.linspace(-3, 3, 21)
        cone_vols = []
        
        for scale in scales:
            z = z_mu.clone()
            z[0, d] = z[0, d] + scale
            
            with torch.no_grad():
                mu_knots, logsig_knots = net.tube_forward(s0_t, z)
            
            std = torch.exp(logsig_knots)
            cone_vol = np.log(torch.prod(std.mean(dim=1)).item() + 1e-8)
            cone_vols.append(cone_vol)
        
        ax.plot(scales, cone_vols, label=f"z[{d}]")
    
    ax.set_xlabel("z perturbation")
    ax.set_ylabel("Log cone volume")
    ax.set_title("z dimension → Cone volume sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Effect of z on Tube Geometry", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")


def test_z_manipulation(net: TubeNetWithZ, out_dir: Path):
    """
    Test if manipulating z actually controls tube geometry.
    
    Key test: Take a deterministic situation (Rule 0), 
    override z[2] to be lower → does the tube widen?
    """
    print("\n" + "=" * 60)
    print("Z MANIPULATION TEST")
    print("=" * 60)
    print("Can we override the natural z to change tube geometry?")
    
    env = SimpleEnv(seed=123)
    
    # Find a Rule 0 situation (deterministic)
    for _ in range(100):
        s0 = env.reset()
        if env.rule == 0:
            break
    
    s0_t = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
    traj = env.generate_trajectory(T=16)
    
    # Get natural z for this situation
    with torch.no_grad():
        z_mu, z_logstd = net.encode(s0_t)
        z_natural = z_mu.clone()
    
    print(f"\nSituation: Rule 0 (deterministic straight line)")
    print(f"Natural z: [{', '.join(f'{v:.3f}' for v in z_natural.squeeze().numpy())}]")
    
    # Test different z[2] values (the cone-control dimension)
    z2_values = [-1.0, -0.5, -0.29, z_natural[0, 2].item(), 0.0, 0.5, 1.0]
    
    results = []
    print(f"\nManipulating z[2] (cone-control dimension):")
    print(f"{'z[2]':>8} | {'Cone Vol':>10} | {'Bind Rate':>10} | {'Effect':>20}")
    print("-" * 55)
    
    for z2 in z2_values:
        z_test = z_natural.clone()
        z_test[0, 2] = z2
        
        with torch.no_grad():
            mu_knots, logsig_knots = net.tube_forward(s0_t, z_test)
            
            mu_traj = interp_knots(mu_knots, 16).squeeze(0)
            std_traj = torch.exp(interp_knots(logsig_knots, 16)).squeeze(0)
            
            cone_vol = np.log(torch.prod(std_traj.mean(dim=0)).item() + 1e-8)
            
            # Check bind rate
            traj_t = torch.tensor(traj, dtype=torch.float32)
            inside = (torch.abs(traj_t - mu_traj) < 2.0 * std_traj).all(dim=-1).float()
            bind_rate = inside.mean().item()
        
        is_natural = abs(z2 - z_natural[0, 2].item()) < 0.01
        effect = "← natural" if is_natural else ("WIDER" if z2 < z_natural[0, 2].item() else "tighter")
        
        results.append({
            "z2": z2,
            "cone_vol": cone_vol,
            "bind_rate": bind_rate,
        })
        
        print(f"{z2:>8.2f} | {cone_vol:>10.3f} | {bind_rate:>10.3f} | {effect:>20}")
    
    # Check if manipulation works
    cone_at_low_z2 = [r["cone_vol"] for r in results if r["z2"] < -0.5]
    cone_at_high_z2 = [r["cone_vol"] for r in results if r["z2"] > 0.3]
    
    if cone_at_low_z2 and cone_at_high_z2:
        avg_low = np.mean(cone_at_low_z2)
        avg_high = np.mean(cone_at_high_z2)
        delta = avg_low - avg_high
        
        print(f"\nSummary:")
        print(f"  Low z[2] avg cone: {avg_low:.3f}")
        print(f"  High z[2] avg cone: {avg_high:.3f}")
        print(f"  Difference: {delta:.3f}")
        
        if delta > 0.5:
            print(f"\n✓ SUCCESS: z[2] CONTROLS cone geometry!")
            print(f"  Lower z[2] → wider tubes (as expected)")
        else:
            print(f"\n✗ z[2] doesn't control cone geometry")
    
    # Visualize the manipulation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(z2_values)))
    
    # Plot 1: Trajectories with different z[2]
    ax = axes[0]
    for i, z2 in enumerate(z2_values):
        z_test = z_natural.clone()
        z_test[0, 2] = z2
        
        with torch.no_grad():
            mu_knots, logsig_knots = net.tube_forward(s0_t, z_test)
            mu_traj = interp_knots(mu_knots, 16).squeeze(0).numpy()
        
        ax.plot(mu_traj[:, 0], mu_traj[:, 1], color=colors[i], 
               label=f"z[2]={z2:.2f}", linewidth=2)
    
    ax.plot(traj[:, 0], traj[:, 1], 'k--', linewidth=2, label='Actual')
    ax.set_title("Tube mean with different z[2]")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Tube width at t=8 for different z[2]
    ax = axes[1]
    for i, z2 in enumerate(z2_values):
        z_test = z_natural.clone()
        z_test[0, 2] = z2
        
        with torch.no_grad():
            mu_knots, logsig_knots = net.tube_forward(s0_t, z_test)
            mu_traj = interp_knots(mu_knots, 16).squeeze(0).numpy()
            std_traj = torch.exp(interp_knots(logsig_knots, 16)).squeeze(0).numpy()
        
        # Draw tube at t=8
        t = 8
        ellipse = plt.matplotlib.patches.Ellipse(
            (mu_traj[t, 0], mu_traj[t, 1]),
            width=4 * std_traj[t, 0],
            height=4 * std_traj[t, 1],
            alpha=0.3,
            color=colors[i],
            label=f"z[2]={z2:.2f}",
        )
        ax.add_patch(ellipse)
        ax.plot(mu_traj[t, 0], mu_traj[t, 1], 'o', color=colors[i], markersize=5)
    
    ax.plot(traj[8, 0], traj[8, 1], 'k*', markersize=15, label='Actual t=8')
    ax.set_title("Tube width at t=8 (2σ ellipse)")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: z[2] vs cone volume
    ax = axes[2]
    z2s = [r["z2"] for r in results]
    cones = [r["cone_vol"] for r in results]
    binds = [r["bind_rate"] for r in results]
    
    ax.scatter(z2s, cones, c=binds, cmap='RdYlGn', s=100, edgecolors='black')
    ax.plot(z2s, cones, 'b-', alpha=0.5)
    
    # Mark natural z[2]
    natural_z2 = z_natural[0, 2].item()
    ax.axvline(natural_z2, color='gray', linestyle='--', label=f'Natural z[2]={natural_z2:.2f}')
    
    ax.set_xlabel("z[2] value")
    ax.set_ylabel("Log cone volume")
    ax.set_title("z[2] → Cone Volume\n(color = bind rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=ax, label='Bind rate')
    
    plt.suptitle("Z Manipulation Test: Can we control tube width?", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "z_manipulation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved: {out_dir / 'z_manipulation.png'}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--beta-kl", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-from-z-only", action="store_true",
                       help="Sigma depends ONLY on z, not s0")
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"tube_with_z_{timestamp}"
    if args.name:
        run_name = f"tube_with_z_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    config = {
        "steps": args.steps,
        "z_dim": args.z_dim,
        "beta_kl": args.beta_kl,
        "seed": args.seed,
        "sigma_from_z_only": args.sigma_from_z_only,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Train
    net, history = train(
        n_steps=args.steps,
        z_dim=args.z_dim,
        lr=1e-3,
        beta_kl=args.beta_kl,
        seed=args.seed,
        sigma_from_z_only=args.sigma_from_z_only,
    )
    
    # Evaluate
    z_values, cone_vols, rules = evaluate_z_control(net, n_samples=100)
    
    # Visualize z effect
    visualize_z_effect(net, save_path=out_dir / "z_effect.png")
    
    # Test z manipulation
    manipulation_results = test_z_manipulation(net, out_dir)
    
    # Training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(history["nll"])
    axes[0, 0].set_title("NLL")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["bind_rate"])
    axes[0, 1].axhline(0.85, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title("Bind Rate")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history["cone_vol"])
    axes[0, 2].set_title("Cone Volume")
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history["kl"])
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history["z_norm"])
    axes[1, 1].set_title("z Norm")
    axes[1, 1].grid(True, alpha=0.3)
    
    # z-cone correlation scatter
    for d in range(min(args.z_dim, 3)):
        axes[1, 2].scatter(z_values[:, d], cone_vols, alpha=0.5, label=f"z[{d}]", s=20)
    axes[1, 2].set_xlabel("z dimension value")
    axes[1, 2].set_ylabel("Log cone volume")
    axes[1, 2].set_title("z → Cone Volume")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Tube with z (dim={args.z_dim})", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "training.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save results
    eval_results = {
        "cone_vol_by_rule": {r: float(cone_vols[rules == r].mean()) for r in range(4)},
        "z_cone_correlations": [
            float(np.corrcoef(z_values[:, d], cone_vols)[0, 1])
            for d in range(args.z_dim)
        ],
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
