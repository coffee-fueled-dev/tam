"""
Alien Sensors Experiment: Empirical Sufficiency Test

Tests whether a learned functor F can map commitments A→B even when
B's observations have ~zero mutual information with A's observations,
as long as the viability basins (left vs right) are isomorphic.

Protocol:
1. Train Actor A on normal observations
2. Train Actor B on alien observations (same latent dynamics)
3. Train functor F: zA → zB on paired commitments
4. Evaluate: Transfer should match B-CEM despite alien obs
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
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Also add competitive-port-binding for BimodalActor
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "competitive-port-binding"))

from train_bimodal import BimodalActor, BimodalEnv as CPBBimodalEnv
from alien_obs import AlienObsWrapper, AlienLevel


# =============================================================================
# 1) Bimodal Environment
# =============================================================================

class BimodalEnv:
    """
    Bimodal environment with two incompatible futures from same s0.
    
    Observation format: [x, y, goal_x, goal_y] matching CompetitiveActor's
    extract_goal() which expects goal at indices 2:4.
    
    CRITICAL: Mode is DETERMINISTIC from s0, not random.
    This ensures both actors learn consistent z→mode mappings.
    """
    
    def __init__(
        self,
        seed: int = 0,
        T: int = 16,
        bend_amp: float = 0.15,
        base_noise: float = 0.02,
        deterministic_mode: bool = True,  # Mode determined by s0
    ):
        self.rng = np.random.default_rng(seed)
        self.T = T
        self.bend_amp = bend_amp
        self.base_noise = base_noise
        self.deterministic_mode = deterministic_mode
        
        self.s0_latent = None
        self.mode = None
        self.reset()
    
    def _compute_deterministic_mode(self, s0: np.ndarray) -> int:
        """
        Deterministic mode from s0.
        
        Mode depends on the perpendicular direction from start to goal:
        - If goal is "above" the x=y diagonal relative to start: mode +1
        - Otherwise: mode -1
        
        This creates a consistent mode labeling that both actors can learn.
        """
        x, y, goal_x, goal_y = s0
        # Direction from start to goal
        dx, dy = goal_x - x, goal_y - y
        # Perpendicular signed: positive if goal is "left" of forward direction
        # Using cross product with vertical: dx * 0 - dy * 1 = -dy
        # Actually simpler: use the angle or a hash of the state
        # Most stable: mode = sign of (goal_y - y) - (goal_x - x)
        # This creates two regions based on whether goal is above or below the x=y line through start
        perp_sign = (goal_y - y) - (goal_x - x)
        return 1 if perp_sign > 0 else -1
    
    def reset(self, force_mode: int = None) -> np.ndarray:
        """Reset and return latent state [x, y, goal_x, goal_y]."""
        self.s0_latent = np.array([
            self.rng.uniform(0.2, 0.8),  # x
            self.rng.uniform(0.2, 0.8),  # y
            self.rng.uniform(0.2, 0.8),  # goal_x
            self.rng.uniform(0.2, 0.8),  # goal_y
        ], dtype=np.float32)
        
        if force_mode is not None:
            self.mode = force_mode
        elif self.deterministic_mode:
            self.mode = self._compute_deterministic_mode(self.s0_latent)
        else:
            self.mode = self.rng.choice([-1, 1])
        
        return self.s0_latent.copy()
    
    def get_mode_prototypes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get deterministic mode trajectories (no noise)."""
        return self._generate_trajectory(mode=1, noise=False), \
               self._generate_trajectory(mode=-1, noise=False)
    
    def _generate_trajectory(self, mode: int, noise: bool = True) -> np.ndarray:
        """Generate trajectory for given mode."""
        traj = np.zeros((self.T, 2), dtype=np.float32)
        
        x, y, goal_x, goal_y = self.s0_latent
        dx = goal_x - x
        dy = goal_y - y
        
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        d_unit = np.array([dx / dist, dy / dist])
        d_perp = np.array([-d_unit[1], d_unit[0]])
        
        for t in range(self.T):
            progress = (t + 1) / self.T
            base_x = x + dx * progress
            base_y = y + dy * progress
            bend = np.sin(np.pi * progress) * self.bend_amp * mode
            
            traj[t, 0] = base_x + bend * d_perp[0]
            traj[t, 1] = base_y + bend * d_perp[1]
            
            if noise:
                traj[t, 0] += self.rng.normal(0, self.base_noise)
                traj[t, 1] += self.rng.normal(0, self.base_noise)
        
        return traj
    
    def generate_trajectory(self) -> np.ndarray:
        """Generate trajectory for current mode (with noise)."""
        return self._generate_trajectory(self.mode, noise=True)
    
    def get_realized_mode(self, trajectory: np.ndarray) -> int:
        """Determine which mode a trajectory belongs to based on deviation."""
        mode_plus, mode_minus = self.get_mode_prototypes()
        d_plus = ((trajectory - mode_plus) ** 2).mean()
        d_minus = ((trajectory - mode_minus) ** 2).mean()
        return 1 if d_plus < d_minus else -1


# =============================================================================
# 2) Functor Network
# =============================================================================

class FunctorNet(nn.Module):
    """Small MLP functor: zA → zB_hat."""
    
    def __init__(self, z_dim: int, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, z_A: torch.Tensor) -> torch.Tensor:
        return self.net(z_A)


# =============================================================================
# 3) Training Functions
# =============================================================================

def train_actor(
    actor: BimodalActor,
    env: BimodalEnv,
    obs_transform,  # callable: latent -> obs
    bend_amplitude: float,
    n_steps: int = 5000,
    log_every: int = 1000,
    name: str = "Actor",
    use_latent_for_deviation: bool = False,  # If True, use original latent for d computation
) -> Dict[str, List[float]]:
    """Train a BimodalActor with mode commitment."""
    print(f"Training {name} for {n_steps} steps...")
    
    for step in range(n_steps):
        latent = env.reset()
        traj = env.generate_trajectory()
        
        obs = obs_transform(latent)
        obs_t = torch.tensor(obs, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        latent_t = torch.tensor(latent, dtype=torch.float32)  # Original s0 for deviation
        
        # BimodalActor uses CEM with mode commitment scoring
        # For alien obs, we use obs for prediction but latent for deviation computation
        if use_latent_for_deviation:
            z, details = actor.select_z_cem(obs_t, bend_amplitude)
            # Recompute d using original latent
            d_signed = actor.compute_signed_deviation(details["mu"].unsqueeze(0), latent_t).item()
        else:
            z, details = actor.select_z_cem(obs_t, bend_amplitude)
        
        mu = details["mu"]
        sigma = details["sigma"]
        
        metrics = actor.train_step(obs_t, traj_t, z, mu, sigma, bend_amplitude)
        
        if step % log_every == 0:
            recent_bind = np.mean(actor.history["bind_hard"][-min(500, step+1):]) if actor.history["bind_hard"] else 0
            recent_mci = np.mean(actor.history["mci"][-min(500, step+1):]) if actor.history["mci"] else 0
            print(f"  Step {step}: bind={recent_bind:.3f}, MCI={recent_mci:.3f}")
    
    return actor.history


def collect_paired_data(
    actor_A: BimodalActor,
    actor_B: BimodalActor,
    env: BimodalEnv,
    obs_A_fn,
    obs_B_fn,
    bend_amplitude: float,
    n_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """Collect paired (zA*, zB*, mode) from CEM binding."""
    actor_A.eval()
    actor_B.eval()
    
    Z_A, Z_B, d_A, d_B = [], [], [], []
    
    print(f"Collecting {n_samples} paired samples...")
    
    for _ in range(n_samples):
        latent = env.reset()
        latent_t = torch.tensor(latent, dtype=torch.float32)  # Original s0 for deviation
        
        obs_A = torch.tensor(obs_A_fn(latent), dtype=torch.float32)
        obs_B = torch.tensor(obs_B_fn(latent), dtype=torch.float32)
        
        with torch.no_grad():
            z_A, details_A = actor_A.select_z_cem(obs_A, bend_amplitude)
            z_B, details_B = actor_B.select_z_cem(obs_B, bend_amplitude)
            
            # Compute signed deviation using ORIGINAL LATENT (not alien obs)
            # This gives consistent mode measurement across A and B
            d_signed_A = actor_A.compute_signed_deviation(details_A["mu"].unsqueeze(0), latent_t).item()
            d_signed_B = actor_B.compute_signed_deviation(details_B["mu"].unsqueeze(0), latent_t).item()
        
        Z_A.append(z_A.numpy())
        Z_B.append(z_B.numpy())
        d_A.append(d_signed_A)
        d_B.append(d_signed_B)
    
    # Derive modes from signed deviation
    modes_A = np.sign(np.array(d_A))
    modes_B = np.sign(np.array(d_B))
    
    return {
        "Z_A": np.array(Z_A),
        "Z_B": np.array(Z_B),
        "d_A": np.array(d_A),
        "d_B": np.array(d_B),
        "modes_A": modes_A,
        "modes_B": modes_B,
    }


def train_functor(
    functor: FunctorNet,
    pairs: Dict[str, np.ndarray],
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_struct: float = 0.1,
    filter_by_mode: bool = True,
) -> Dict[str, List[float]]:
    """Train functor F: zA → zB."""
    optimizer = optim.Adam(functor.parameters(), lr=lr)
    
    # Filter to pairs where A and B committed to the same mode
    if filter_by_mode and "modes_A" in pairs and "modes_B" in pairs:
        mask = pairs["modes_A"] == pairs["modes_B"]
        Z_A = torch.tensor(pairs["Z_A"][mask], dtype=torch.float32)
        Z_B = torch.tensor(pairs["Z_B"][mask], dtype=torch.float32)
        print(f"  Filtered to {mask.sum()}/{len(mask)} mode-aligned pairs ({mask.mean():.1%})")
    else:
        Z_A = torch.tensor(pairs["Z_A"], dtype=torch.float32)
        Z_B = torch.tensor(pairs["Z_B"], dtype=torch.float32)
    
    N = Z_A.shape[0]
    
    history = {"loss": [], "align_loss": [], "struct_loss": []}
    
    print(f"Training functor for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_losses = []
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_A = Z_A[idx]
            z_B = Z_B[idx]
            
            z_B_hat = functor(z_A)
            
            align_loss = ((z_B_hat - z_B) ** 2).mean()
            
            B = z_A.shape[0]
            if B > 1:
                D_B = torch.cdist(z_B, z_B)
                D_F = torch.cdist(z_B_hat, z_B_hat)
                struct_loss = ((D_F - D_B) ** 2).mean()
            else:
                struct_loss = torch.tensor(0.0)
            
            loss = align_loss + lambda_struct * struct_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        history["loss"].append(np.mean(epoch_losses))
        history["align_loss"].append(float(align_loss.item()))
        history["struct_loss"].append(float(struct_loss.item()))
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={history['loss'][-1]:.4f}")
    
    return history


# =============================================================================
# 4) Evaluation
# =============================================================================

def evaluate_transfer(
    actor_A: BimodalActor,
    actor_B: BimodalActor,
    functor: FunctorNet,
    env: BimodalEnv,
    obs_A_fn,
    obs_B_fn,
    bend_amplitude: float,
    n_samples: int = 500,
) -> Dict[str, Dict]:
    """Evaluate transfer with mode agreement metric."""
    actor_A.eval()
    actor_B.eval()
    functor.eval()
    
    results = {
        "transfer": defaultdict(list),
        "B_CEM": defaultdict(list),
        "B_random": defaultdict(list),
    }
    mode_agreements = []
    d_A_list, d_transfer_list = [], []
    
    print(f"Evaluating on {n_samples} samples...")
    
    for _ in range(n_samples):
        latent = env.reset()
        traj = env.generate_trajectory()
        latent_t = torch.tensor(latent, dtype=torch.float32)  # Original s0 for deviation
        
        obs_A = torch.tensor(obs_A_fn(latent), dtype=torch.float32)
        obs_B = torch.tensor(obs_B_fn(latent), dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        with torch.no_grad():
            # 1. Transfer: zA* from A, map through F
            z_A, details_A = actor_A.select_z_cem(obs_A, bend_amplitude)
            z_B_transfer = functor(z_A.unsqueeze(0)).squeeze(0)
            mu_trans, sigma_trans = actor_B.predict_tube(obs_B, z_B_transfer)
            mu_trans = mu_trans.squeeze(0)
            sigma_trans = sigma_trans.squeeze(0)
            
            # 2. B-CEM
            z_B_cem, details_B_CEM = actor_B.select_z_cem(obs_B, bend_amplitude)
            
            # 3. B-random
            z_B_rand, details_B_rand = actor_B.select_z_random(obs_B)
            
            # Mode agreement: compare signed deviations using ORIGINAL LATENT
            d_A = actor_A.compute_signed_deviation(details_A["mu"].unsqueeze(0), latent_t).item()
            d_transfer = actor_B.compute_signed_deviation(mu_trans.unsqueeze(0), latent_t).item()
            
            d_A_list.append(d_A)
            d_transfer_list.append(d_transfer)
            
            # Mode agreement: same sign = same mode commitment
            mode_agree = (np.sign(d_A) == np.sign(d_transfer)) if abs(d_A) > 0.02 and abs(d_transfer) > 0.02 else False
            mode_agreements.append(1.0 if mode_agree else 0.0)
            
            # Evaluate all methods
            for name, mu, sigma in [
                ("transfer", mu_trans, sigma_trans),
                ("B_CEM", details_B_CEM["mu"], details_B_CEM["sigma"]),
                ("B_random", details_B_rand["mu"], details_B_rand["sigma"]),
            ]:
                bind = actor_B.compute_bind_hard(mu, sigma, traj_t).item()
                log_vol = torch.log(sigma).mean().item()
                mse = ((traj_t - mu) ** 2).mean().item()
                
                results[name]["bind"].append(bind)
                results[name]["log_vol"].append(log_vol)
                results[name]["mse"].append(mse)
    
    # Summarize
    summary = {}
    for name in results:
        summary[name] = {
            "bind_mean": np.mean(results[name]["bind"]),
            "bind_std": np.std(results[name]["bind"]),
            "log_vol_mean": np.mean(results[name]["log_vol"]),
            "log_vol_std": np.std(results[name]["log_vol"]),
            "mse_mean": np.mean(results[name]["mse"]),
        }
    
    summary["mode_agreement"] = {
        "mean": np.mean(mode_agreements),
        "std": np.std(mode_agreements),
        "n": len(mode_agreements),
    }
    
    # Correlation between A's d and transfer's d
    d_corr = np.corrcoef(d_A_list, d_transfer_list)[0, 1] if len(d_A_list) > 1 else 0
    summary["d_correlation"] = float(d_corr) if not np.isnan(d_corr) else 0
    
    print(f"\nResults:")
    for name in ["B_random", "transfer", "B_CEM"]:
        print(f"  {name}: bind={summary[name]['bind_mean']:.3f}, vol={summary[name]['log_vol_mean']:.2f}")
    print(f"  Mode Agreement: {summary['mode_agreement']['mean']:.1%}")
    print(f"  d Correlation (A vs Transfer): {summary['d_correlation']:.3f}")
    
    return summary


# =============================================================================
# 5) Plotting
# =============================================================================

def plot_training(hist_A: Dict, hist_B: Dict, out_path: Path):
    """Plot training curves for BimodalActor."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    window = 100
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # Row 1: Core metrics
    axes[0, 0].plot(smooth(hist_A["bind_hard"]), label='Actor A', color='blue')
    axes[0, 0].plot(smooth(hist_B["bind_hard"]), label='Actor B (Alien)', color='red')
    axes[0, 0].set_title("Bind Rate")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(smooth(hist_A["log_vol"]), label='Actor A', color='blue')
    axes[0, 1].plot(smooth(hist_B["log_vol"]), label='Actor B (Alien)', color='red')
    axes[0, 1].set_title("Log Volume")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(smooth(hist_A["mse"]), label='Actor A', color='blue')
    axes[0, 2].plot(smooth(hist_B["mse"]), label='Actor B (Alien)', color='red')
    axes[0, 2].set_title("MSE")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Mode commitment metrics
    axes[1, 0].plot(smooth(hist_A["mci"]), label='Actor A', color='blue')
    axes[1, 0].plot(smooth(hist_B["mci"]), label='Actor B (Alien)', color='red')
    axes[1, 0].set_title("Mode Commitment Index")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(smooth(hist_A["d_signed"]), label='Actor A', color='blue', alpha=0.7)
    axes[1, 1].plot(smooth(hist_B["d_signed"]), label='Actor B (Alien)', color='red', alpha=0.7)
    axes[1, 1].axhline(0, color='black', linestyle='--')
    axes[1, 1].set_title("Signed Deviation d")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(hist_A["lambda"], label='Actor A', color='blue')
    axes[1, 2].plot(hist_B["lambda"], label='Actor B (Alien)', color='red')
    axes[1, 2].set_title("Lambda")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_basin_structure(
    actor: BimodalActor,
    env: BimodalEnv,
    obs_fn,
    bend_amplitude: float,
    n_samples: int = 500,
    name: str = "Actor",
) -> Dict[str, float]:
    """Analyze whether actor's z-space has basin structure."""
    actor.eval()
    
    z_plus, z_minus = [], []
    d_values = []
    
    for _ in range(n_samples):
        latent = env.reset()
        latent_t = torch.tensor(latent, dtype=torch.float32)  # Original s0 for deviation
        obs = torch.tensor(obs_fn(latent), dtype=torch.float32)
        
        with torch.no_grad():
            z, details = actor.select_z_cem(obs, bend_amplitude)
            # Use original latent for deviation computation
            d_signed = actor.compute_signed_deviation(details["mu"].unsqueeze(0), latent_t).item()
            d_values.append(d_signed)
            
            if d_signed > 0:
                z_plus.append(z.numpy())
            else:
                z_minus.append(z.numpy())
    
    z_plus = np.array(z_plus)
    z_minus = np.array(z_minus)
    
    # Compute separation metrics
    if len(z_plus) > 10 and len(z_minus) > 10:
        mean_plus = z_plus.mean(axis=0)
        mean_minus = z_minus.mean(axis=0)
        
        # Distance between basin centers
        basin_distance = np.linalg.norm(mean_plus - mean_minus)
        
        # Average within-basin spread
        spread_plus = np.sqrt(((z_plus - mean_plus) ** 2).sum(axis=1)).mean()
        spread_minus = np.sqrt(((z_minus - mean_minus) ** 2).sum(axis=1)).mean()
        avg_spread = (spread_plus + spread_minus) / 2
        
        # Separation ratio (higher = more distinct basins)
        separation_ratio = basin_distance / (avg_spread + 1e-6)
        
        print(f"  {name} basin analysis:")
        print(f"    Mode+ samples: {len(z_plus)}, Mode- samples: {len(z_minus)}")
        print(f"    Basin distance: {basin_distance:.3f}")
        print(f"    Avg spread: {avg_spread:.3f}")
        print(f"    Separation ratio: {separation_ratio:.2f} (>2 = good separation)")
        
        # Also compute MCI (Mode Commitment Index)
        mci_mean = np.abs(np.array(d_values)).mean() / bend_amplitude if 'd_values' in dir() else 0
        
        return {
            "basin_distance": float(basin_distance),
            "avg_spread": float(avg_spread),
            "separation_ratio": float(separation_ratio),
            "n_plus": len(z_plus),
            "n_minus": len(z_minus),
            "mci_mean": float(mci_mean) if 'mci_mean' in dir() else 0,
        }
    else:
        print(f"  {name}: Not enough samples in both modes")
        return {}


def plot_z_space(
    actor_A: BimodalActor,
    actor_B: BimodalActor,
    env: BimodalEnv,
    obs_A_fn,
    obs_B_fn,
    out_path: Path,
    bend_amplitude: float = 0.15,
    n_samples: int = 300,
):
    """Visualize z-space basin structure for both actors."""
    from sklearn.decomposition import PCA
    
    z_A_plus, z_A_minus = [], []
    z_B_plus, z_B_minus = [], []
    
    for _ in range(n_samples):
        latent = env.reset()
        latent_t = torch.tensor(latent, dtype=torch.float32)  # Original s0 for deviation
        obs_A = torch.tensor(obs_A_fn(latent), dtype=torch.float32)
        obs_B = torch.tensor(obs_B_fn(latent), dtype=torch.float32)
        
        with torch.no_grad():
            z_A, details_A = actor_A.select_z_cem(obs_A, bend_amplitude)
            z_B, details_B = actor_B.select_z_cem(obs_B, bend_amplitude)
            
            # Use original latent for deviation computation
            d_A = actor_A.compute_signed_deviation(details_A["mu"].unsqueeze(0), latent_t).item()
            d_B = actor_B.compute_signed_deviation(details_B["mu"].unsqueeze(0), latent_t).item()
            
            if d_A > 0:
                z_A_plus.append(z_A.numpy())
            else:
                z_A_minus.append(z_A.numpy())
            
            if d_B > 0:
                z_B_plus.append(z_B.numpy())
            else:
                z_B_minus.append(z_B.numpy())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, z_plus, z_minus, title in [
        (axes[0], z_A_plus, z_A_minus, "Actor A z-space"),
        (axes[1], z_B_plus, z_B_minus, "Actor B z-space"),
    ]:
        if len(z_plus) > 5 and len(z_minus) > 5:
            z_all = np.vstack([z_plus, z_minus])
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_all)
            
            n_plus = len(z_plus)
            ax.scatter(z_2d[:n_plus, 0], z_2d[:n_plus, 1], c='blue', alpha=0.5, label='Mode+')
            ax.scatter(z_2d[n_plus:, 0], z_2d[n_plus:, 1], c='red', alpha=0.5, label='Mode-')
            ax.set_title(title)
            ax.legend()
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(title)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_results(summary: Dict, alien_level: str, out_path: Path):
    """Plot transfer comparison with mode agreement."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ["B_random", "transfer", "B_CEM"]
    labels = ["B-Random", "Transfer", "B-CEM"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    
    # Bar: Bind rate
    ax = axes[0]
    binds = [summary[m]["bind_mean"] for m in methods]
    bind_stds = [summary[m]["bind_std"] for m in methods]
    ax.bar(range(len(methods)), binds, yerr=bind_stds, color=colors, capsize=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Bind Rate")
    ax.set_title("Reliability")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Scatter: Pareto
    ax = axes[1]
    for m, label, color in zip(methods, labels, colors):
        ax.scatter(summary[m]["log_vol_mean"], summary[m]["bind_mean"],
                   s=200, c=color, label=label, edgecolors='black', linewidths=2)
    ax.set_xlabel("Log Volume")
    ax.set_ylabel("Bind Rate")
    ax.set_title("Bind vs Volume")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Mode agreement
    ax = axes[2]
    ma = summary["mode_agreement"]
    ax.bar([0], [ma["mean"]], yerr=[ma["std"]], color='#2ecc71', capsize=10, width=0.5)
    ax.axhline(0.5, color='red', linestyle='--', label='Chance')
    ax.axhline(0.85, color='blue', linestyle='--', label='Target (85%)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(['Transfer'])
    ax.set_ylabel("Mode Agreement")
    ax.set_title(f"Intent Transfer ({alien_level})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Alien Sensors: {alien_level}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 6) Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alien-level", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Alien level: 0=none, 1=tanh, 2=RFF, 3=bitstream")
    parser.add_argument("--actor-steps", type=int, default=5000)
    parser.add_argument("--functor-epochs", type=int, default=100)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--alien-dim", type=int, default=64)
    parser.add_argument("--bend-amplitude", type=float, default=0.15,
                        help="Mode separation amplitude")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    alien_level = AlienLevel(args.alien_level)
    
    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "runs" / f"{alien_level.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Alien Sensors Experiment")
    print(f"Level: {alien_level.name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Environment
    env = BimodalEnv(seed=args.seed)
    
    # Observation transforms
    obs_A_fn = lambda x: x  # Identity for A
    
    if alien_level == AlienLevel.NONE:
        obs_B_fn = lambda x: x
        obs_B_dim = 4
    else:
        alien_wrapper = AlienObsWrapper(
            input_dim=4,
            level=alien_level,
            output_dim=args.alien_dim,
            seed=args.seed + 1000,
        )
        obs_B_fn = alien_wrapper
        obs_B_dim = alien_wrapper.obs_dim
    
    # Actors (using BimodalActor with mode commitment)
    actor_A = BimodalActor(obs_dim=4, z_dim=args.z_dim, T=16)
    actor_B = BimodalActor(obs_dim=obs_B_dim, z_dim=args.z_dim, T=16)
    
    bend_amplitude = args.bend_amplitude
    
    # === Stage 0: Train Actor A ===
    print("\n[Stage 0] Training Actor A (normal obs)")
    hist_A = train_actor(actor_A, env, obs_A_fn, bend_amplitude, n_steps=args.actor_steps, name="A")
    torch.save(actor_A.state_dict(), out_dir / "actor_A.pt")
    
    # === Stage 1: Train Actor B (alien obs) ===
    print(f"\n[Stage 1] Training Actor B ({alien_level.name} obs)")
    hist_B = train_actor(actor_B, env, obs_B_fn, bend_amplitude, n_steps=args.actor_steps, name="B")
    torch.save(actor_B.state_dict(), out_dir / "actor_B.pt")
    
    plot_training(hist_A, hist_B, out_dir / "training.png")
    
    # === Basin Analysis ===
    print("\n[Basin Analysis] Checking z-space structure")
    basin_A = analyze_basin_structure(actor_A, env, obs_A_fn, bend_amplitude, n_samples=500, name="A")
    basin_B = analyze_basin_structure(actor_B, env, obs_B_fn, bend_amplitude, n_samples=500, name="B")
    
    try:
        plot_z_space(actor_A, actor_B, env, obs_A_fn, obs_B_fn, out_dir / "z_space.png", bend_amplitude)
    except ImportError:
        print("  (sklearn not available for PCA visualization)")
    
    # === Stage 2: Train Functor ===
    print("\n[Stage 2] Collecting paired data and training functor")
    pairs = collect_paired_data(actor_A, actor_B, env, obs_A_fn, obs_B_fn, bend_amplitude, n_samples=args.pair_samples)
    np.savez(out_dir / "pairs.npz", **pairs)
    
    # Check mode agreement in paired data
    pair_agreement = (pairs["modes_A"] == pairs["modes_B"]).mean()
    print(f"  Pair mode agreement (A vs B native): {pair_agreement:.1%}")
    
    # d correlation shows how well commitment directions align
    d_corr = np.corrcoef(pairs["d_A"], pairs["d_B"])[0, 1]
    print(f"  d correlation (A vs B): {d_corr:.3f}")
    
    functor = FunctorNet(z_dim=args.z_dim)
    # Don't filter - let functor learn any correspondence
    functor_hist = train_functor(functor, pairs, n_epochs=args.functor_epochs, filter_by_mode=False)
    torch.save(functor.state_dict(), out_dir / "functor.pt")
    
    # === Stage 3: Evaluate ===
    print("\n[Stage 3] Evaluating transfer")
    summary = evaluate_transfer(actor_A, actor_B, functor, env, obs_A_fn, obs_B_fn, bend_amplitude, n_samples=args.eval_samples)
    
    summary["pair_mode_agreement"] = float(pair_agreement)
    summary["basin_A"] = basin_A
    summary["basin_B"] = basin_B
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    plot_results(summary, alien_level.name, out_dir / "results.png")
    
    # === Final Report ===
    print(f"\n{'='*60}")
    print("ALIEN SENSORS RESULTS")
    print(f"{'='*60}")
    
    ma = summary["mode_agreement"]["mean"]
    trans_bind = summary["transfer"]["bind_mean"]
    cem_bind = summary["B_CEM"]["bind_mean"]
    
    print(f"\nBasin Structure:")
    if basin_A:
        print(f"  A separation ratio: {basin_A.get('separation_ratio', 0):.2f}")
    if basin_B:
        print(f"  B separation ratio: {basin_B.get('separation_ratio', 0):.2f}")
    
    print(f"\nPair Mode Agreement (A vs B): {pair_agreement:.1%}")
    print(f"Transfer Mode Agreement: {ma:.1%}")
    print(f"Transfer Bind: {trans_bind:.3f}")
    print(f"B-CEM Bind: {cem_bind:.3f}")
    
    # Success criteria
    agreement_ok = ma > 0.85
    bind_ok = abs(trans_bind - cem_bind) < 0.05
    
    print(f"\nSuccess Criteria:")
    print(f"  {'✓' if agreement_ok else '✗'} Mode agreement > 85%")
    print(f"  {'✓' if bind_ok else '✗'} Transfer matches B-CEM bind")
    
    if agreement_ok and bind_ok:
        print(f"\n🎉 SUCCESS: Topological bridge works at {alien_level.name}!")
    else:
        print(f"\n⚠ Transfer not fully achieved at {alien_level.name}")
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
