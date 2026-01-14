"""
Alien Sensors Experiment: Empirical Sufficiency Test

Tests whether a learned functor F can map commitments A→B even when
B's observations have ~zero mutual information with A's observations,
as long as the viability basins (left vs right) are isomorphic.

Protocol:
1. Train Actor A on normal observations (learns basins from geometry)
2. Train Actor B on alien observations (same latent dynamics, different obs)
3. Train functor F: zA → zB on paired commitments
4. Evaluate: Transfer should match B-CEM despite alien obs

Fairness Controls (per feedback):
- Bayes ceiling: What mode accuracy can B achieve from obs alone?
- Lift = Transfer - Ceiling: The actual value added by A's commitment
- Shuffle ablation: Functor trained on shuffled z_A→z_B (should collapse to ceiling)

Uses MinimalCompetitiveActor with:
- Gaussian NLL + volume loss (clean, interpretable)
- Mode-prototype max-fit + agency scoring (forces mode commitment)
- No RiskNet, no homeostasis (minimal confounders)
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

from minimal_actor import MinimalCompetitiveActor, BindingMode
from alien_obs import AlienObsWrapper, AlienLevel


# =============================================================================
# 1) Bimodal Environment
# =============================================================================

class BimodalEnv:
    """
    Bimodal environment with two incompatible futures from same s0.
    
    Observation format: [x, y, goal_x, goal_y]
    
    IMPORTANT: Mode can be RANDOM or DETERMINISTIC.
    - Random mode: Neither A nor B can decode mode from obs (Bayes ceiling ~50%)
    - Deterministic mode: Both can decode (ceiling ~100%), tests different claim
    
    For the "topological bridge" claim, use random mode so coordination
    MUST flow through A's commitment geometry.
    """
    
    def __init__(
        self,
        seed: int = 0,
        T: int = 16,
        bend_amp: float = 0.15,
        base_noise: float = 0.02,
        deterministic_mode: bool = False,  # FALSE = random mode (fair test)
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
        """Deterministic mode from s0 (for baseline comparison)."""
        x, y, goal_x, goal_y = s0
        perp_sign = (goal_y - y) - (goal_x - x)
        return 1 if perp_sign > 0 else -1
    
    def reset(self, force_mode: int = None) -> np.ndarray:
        """Reset and return latent state."""
        self.s0_latent = np.array([
            self.rng.uniform(0.2, 0.8),
            self.rng.uniform(0.2, 0.8),
            self.rng.uniform(0.2, 0.8),
            self.rng.uniform(0.2, 0.8),
        ], dtype=np.float32)
        
        if force_mode is not None:
            self.mode = force_mode
        elif self.deterministic_mode:
            self.mode = self._compute_deterministic_mode(self.s0_latent)
        else:
            # Random mode - neither A nor B can decode from observations
            self.mode = self.rng.choice([-1, 1])
        
        return self.s0_latent.copy()
    
    def get_mode_prototypes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get clean mode trajectories (no noise)."""
        return (
            self._generate_trajectory(mode=1, noise=False),
            self._generate_trajectory(mode=-1, noise=False),
        )
    
    def _generate_trajectory(self, mode: int, noise: bool = True) -> np.ndarray:
        """Generate trajectory for given mode."""
        traj = np.zeros((self.T, 2), dtype=np.float32)
        
        x, y, goal_x, goal_y = self.s0_latent
        dx, dy = goal_x - x, goal_y - y
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
        """Generate trajectory for current mode."""
        return self._generate_trajectory(self.mode, noise=True)


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


class BayesCeilingClassifier(nn.Module):
    """
    Simple classifier: B_obs → mode prediction.
    
    Used to compute the Bayes ceiling: what mode accuracy can B achieve
    from its observations alone, without any information from A?
    
    If ceiling is high, B's sensors carry mode information.
    If ceiling is ~50%, B's sensors are truly "alien" (mode-blind).
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logit for mode=+1."""
        return self.net(obs).squeeze(-1)
    
    def predict_mode(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns predicted mode sign (+1 or -1)."""
        with torch.no_grad():
            logit = self.forward(obs)
            return torch.sign(torch.tanh(logit))


def train_bayes_ceiling(
    obs_fn,
    input_dim: int = 4,
    n_samples: int = 5000,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 999,
) -> Tuple[BayesCeilingClassifier, float]:
    """
    Train a classifier to predict mode from B's observations alone.
    
    IMPORTANT: Uses a fresh environment with RANDOM mode to measure
    the true Bayes ceiling (what B can decode from obs alone).
    
    Returns:
        classifier: Trained BayesCeilingClassifier
        accuracy: Best validation accuracy (the "Bayes ceiling")
    """
    # Use a separate env with RANDOM mode for ceiling computation
    ceiling_env = BimodalEnv(seed=seed, deterministic_mode=False)
    
    # Collect training data
    obs_list, mode_list = [], []
    
    for _ in range(n_samples):
        latent = ceiling_env.reset()
        obs = obs_fn(latent)
        mode = ceiling_env.mode
        
        obs_list.append(obs)
        mode_list.append(1.0 if mode > 0 else 0.0)  # Binary label
    
    obs_data = torch.tensor(np.array(obs_list), dtype=torch.float32)
    mode_data = torch.tensor(np.array(mode_list), dtype=torch.float32)
    
    # Split train/val
    n_train = int(0.8 * n_samples)
    obs_train, obs_val = obs_data[:n_train], obs_data[n_train:]
    mode_train, mode_val = mode_data[:n_train], mode_data[n_train:]
    
    # Train classifier
    obs_dim = obs_data.shape[1]
    classifier = BayesCeilingClassifier(obs_dim)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.5
    
    for epoch in range(n_epochs):
        classifier.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            obs_batch = obs_train[idx]
            mode_batch = mode_train[idx]
            
            logits = classifier(obs_batch)
            loss = criterion(logits, mode_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation accuracy
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(obs_val)
            val_preds = (val_logits > 0).float()
            val_acc = (val_preds == mode_val).float().mean().item()
            best_acc = max(best_acc, val_acc)
    
    return classifier, best_acc


# =============================================================================
# 3) Training Functions
# =============================================================================

def train_actor(
    actor: MinimalCompetitiveActor,
    env: BimodalEnv,
    obs_transform,
    n_steps: int = 5000,
    log_every: int = 1000,
    name: str = "Actor",
) -> Dict[str, List[float]]:
    """Train a MinimalCompetitiveActor with mode commitment."""
    print(f"Training {name} for {n_steps} steps...")
    
    for step in range(n_steps):
        latent = env.reset()
        traj = env.generate_trajectory()
        
        # Get mode prototypes for scoring
        proto_plus, proto_minus = env.get_mode_prototypes()
        prototypes = (
            torch.tensor(proto_plus, dtype=torch.float32),
            torch.tensor(proto_minus, dtype=torch.float32),
        )
        
        obs = obs_transform(latent)
        obs_t = torch.tensor(obs, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Select z via CEM with mode-prototype scoring
        z, details = actor.select_z_cem(obs_t, prototypes, env.bend_amp)
        
        # Train
        metrics = actor.train_step(obs_t, traj_t, z, env.bend_amp)
        
        if step % log_every == 0:
            recent_bind = np.mean(actor.history["bind_hard"][-min(500, step+1):]) if actor.history["bind_hard"] else 0
            recent_mci = np.mean(actor.history["mci"][-min(500, step+1):]) if actor.history["mci"] else 0
            print(f"  Step {step}: bind={recent_bind:.3f}, MCI={recent_mci:.3f}")
    
    return dict(actor.history)


def collect_paired_data(
    actor_A: MinimalCompetitiveActor,
    actor_B: MinimalCompetitiveActor,
    env: BimodalEnv,
    obs_A_fn,
    obs_B_fn,
    n_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """Collect paired (zA*, zB*, mode) from CEM binding."""
    actor_A.eval()
    actor_B.eval()
    
    Z_A, Z_B, d_A, d_B = [], [], [], []
    
    print(f"Collecting {n_samples} paired samples...")
    
    for _ in range(n_samples):
        latent = env.reset()
        latent_t = torch.tensor(latent, dtype=torch.float32)
        
        # Mode prototypes (same for both since same underlying dynamics)
        proto_plus, proto_minus = env.get_mode_prototypes()
        prototypes = (
            torch.tensor(proto_plus, dtype=torch.float32),
            torch.tensor(proto_minus, dtype=torch.float32),
        )
        
        obs_A = torch.tensor(obs_A_fn(latent), dtype=torch.float32)
        obs_B = torch.tensor(obs_B_fn(latent), dtype=torch.float32)
        
        with torch.no_grad():
            z_A, details_A = actor_A.select_z_cem(obs_A, prototypes, env.bend_amp)
            z_B, details_B = actor_B.select_z_cem(obs_B, prototypes, env.bend_amp)
            
            # Compute signed deviation using ORIGINAL LATENT for consistent mode measurement
            d_signed_A = actor_A.compute_signed_deviation(details_A["mu"].unsqueeze(0), latent_t).item()
            d_signed_B = actor_B.compute_signed_deviation(details_B["mu"].unsqueeze(0), latent_t).item()
        
        Z_A.append(z_A.numpy())
        Z_B.append(z_B.numpy())
        d_A.append(d_signed_A)
        d_B.append(d_signed_B)
    
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
    shuffle_pairs: bool = False,
) -> Dict[str, List[float]]:
    """
    Train functor F: zA → zB.
    
    Args:
        shuffle_pairs: If True, shuffle Z_A independently of Z_B (ablation).
                       This breaks the correspondence and should collapse 
                       transfer to Bayes ceiling if A's commitment is the bridge.
    """
    optimizer = optim.Adam(functor.parameters(), lr=lr)
    
    Z_A = torch.tensor(pairs["Z_A"], dtype=torch.float32)
    Z_B = torch.tensor(pairs["Z_B"], dtype=torch.float32)
    N = Z_A.shape[0]
    
    if shuffle_pairs:
        # Shuffle Z_A to break correspondence (ablation)
        shuffle_idx = torch.randperm(N)
        Z_A = Z_A[shuffle_idx]
        print(f"  [SHUFFLE ABLATION] Breaking z_A ↔ z_B correspondence")
    
    history = {"loss": [], "align_loss": [], "struct_loss": []}
    
    label = "shuffled functor" if shuffle_pairs else "functor"
    print(f"Training {label} for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_losses = []
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            z_A = Z_A[idx]
            z_B = Z_B[idx]
            
            z_B_hat = functor(z_A)
            
            # Alignment loss
            align_loss = ((z_B_hat - z_B) ** 2).mean()
            
            # Structure preservation (pairwise distances)
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
    actor_A: MinimalCompetitiveActor,
    actor_B: MinimalCompetitiveActor,
    functor: FunctorNet,
    env: BimodalEnv,
    obs_A_fn,
    obs_B_fn,
    n_samples: int = 500,
    functor_label: str = "transfer",
    use_random_mode: bool = True,  # Force random mode during eval
) -> Dict[str, Dict]:
    """
    Evaluate transfer with mode agreement metric.
    
    IMPORTANT: use_random_mode=True ensures B can't exploit learned obs→mode
    mappings during evaluation. Mode agreement must come from the functor.
    """
    actor_A.eval()
    actor_B.eval()
    functor.eval()
    
    # Temporarily override mode setting if needed
    original_deterministic = env.deterministic_mode
    if use_random_mode:
        env.deterministic_mode = False
    
    results = {
        functor_label: defaultdict(list),
        "B_CEM": defaultdict(list),
        "B_random": defaultdict(list),
    }
    mode_agreements = []
    A_correct_list = []  # How often A commits to correct mode (upper bound)
    d_A_list, d_transfer_list = [], []
    true_modes = []
    
    print(f"Evaluating {functor_label} on {n_samples} samples (random_mode={use_random_mode})...")
    
    for _ in range(n_samples):
        latent = env.reset()
        traj = env.generate_trajectory()
        latent_t = torch.tensor(latent, dtype=torch.float32)
        true_mode = env.mode
        true_modes.append(true_mode)
        
        proto_plus, proto_minus = env.get_mode_prototypes()
        prototypes = (
            torch.tensor(proto_plus, dtype=torch.float32),
            torch.tensor(proto_minus, dtype=torch.float32),
        )
        
        obs_A = torch.tensor(obs_A_fn(latent), dtype=torch.float32)
        obs_B = torch.tensor(obs_B_fn(latent), dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        with torch.no_grad():
            # 1. Transfer: zA* from A, map through F
            z_A, details_A = actor_A.select_z_cem(obs_A, prototypes, env.bend_amp)
            z_B_transfer = functor(z_A.unsqueeze(0)).squeeze(0)
            mu_trans, sigma_trans = actor_B.predict_tube(obs_B, z_B_transfer)
            mu_trans = mu_trans.squeeze(0)
            sigma_trans = sigma_trans.squeeze(0)
            
            # 2. B-CEM
            z_B_cem, details_B_CEM = actor_B.select_z_cem(obs_B, prototypes, env.bend_amp)
            
            # 3. B-random
            z_B_rand, details_B_rand = actor_B.select_z_random(obs_B)
            
            # Mode agreement: compare A's commitment against TRUE MODE
            # This is the fair test: does the functor map A→B preserving mode intent?
            d_A = actor_A.compute_signed_deviation(details_A["mu"].unsqueeze(0), latent_t).item()
            d_transfer = actor_B.compute_signed_deviation(mu_trans.unsqueeze(0), latent_t).item()
            
            d_A_list.append(d_A)
            d_transfer_list.append(d_transfer)
            
            # Mode agreement: A's commitment matches TRUE mode
            # (d > 0 means committed to mode +1, d < 0 means committed to mode -1)
            A_mode_commit = np.sign(d_A) if abs(d_A) > 0.02 else 0
            transfer_mode_commit = np.sign(d_transfer) if abs(d_transfer) > 0.02 else 0
            
            # Does A commit to correct mode? (upper bound - A can see normal obs)
            A_correct = (A_mode_commit == true_mode) if A_mode_commit != 0 else False
            A_correct_list.append(1.0 if A_correct else 0.0)
            
            # Does transfer commit to correct mode?
            transfer_correct = (transfer_mode_commit == true_mode) if transfer_mode_commit != 0 else False
            mode_agreements.append(1.0 if transfer_correct else 0.0)
            
            # Evaluate all methods
            for name, mu, sigma in [
                (functor_label, mu_trans, sigma_trans),
                ("B_CEM", details_B_CEM["mu"], details_B_CEM["sigma"]),
                ("B_random", details_B_rand["mu"], details_B_rand["sigma"]),
            ]:
                bind = actor_B.compute_bind_hard(mu, sigma, traj_t).item()
                log_vol = torch.log(sigma).mean().item()
                mse = ((traj_t - mu) ** 2).mean().item()
                
                results[name]["bind"].append(bind)
                results[name]["log_vol"].append(log_vol)
                results[name]["mse"].append(mse)
    
    # Restore original mode setting
    env.deterministic_mode = original_deterministic
    
    # Summarize
    summary = {}
    for name in results:
        summary[name] = {
            "bind_mean": float(np.mean(results[name]["bind"])),
            "bind_std": float(np.std(results[name]["bind"])),
            "log_vol_mean": float(np.mean(results[name]["log_vol"])),
            "log_vol_std": float(np.std(results[name]["log_vol"])),
            "mse_mean": float(np.mean(results[name]["mse"])),
        }
    
    summary["mode_agreement"] = {
        "mean": float(np.mean(mode_agreements)),
        "std": float(np.std(mode_agreements)),
        "n": len(mode_agreements),
    }
    
    # A's accuracy (upper bound for transfer)
    summary["A_mode_accuracy"] = float(np.mean(A_correct_list))
    
    d_corr = np.corrcoef(d_A_list, d_transfer_list)[0, 1] if len(d_A_list) > 1 else 0
    summary["d_correlation"] = float(d_corr) if not np.isnan(d_corr) else 0
    
    print(f"  {functor_label}: bind={summary[functor_label]['bind_mean']:.3f}, mode_agr={summary['mode_agreement']['mean']:.1%}, A_acc={summary['A_mode_accuracy']:.1%}")
    
    return summary


# =============================================================================
# 5) Plotting
# =============================================================================

def plot_training(hist_A: Dict, hist_B: Dict, out_path: Path):
    """Plot training curves."""
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
    
    # Row 2: Mode commitment
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
    
    axes[1, 2].plot(smooth(hist_A["nll"]), label='Actor A', color='blue')
    axes[1, 2].plot(smooth(hist_B["nll"]), label='Actor B (Alien)', color='red')
    axes[1, 2].set_title("NLL Loss")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_basin_structure(
    actor: MinimalCompetitiveActor,
    env: BimodalEnv,
    obs_fn,
    n_samples: int = 500,
    name: str = "Actor",
) -> Dict[str, float]:
    """Analyze whether actor's z-space has basin structure."""
    actor.eval()
    
    z_plus, z_minus = [], []
    d_values = []
    
    for _ in range(n_samples):
        latent = env.reset()
        latent_t = torch.tensor(latent, dtype=torch.float32)
        obs = torch.tensor(obs_fn(latent), dtype=torch.float32)
        
        proto_plus, proto_minus = env.get_mode_prototypes()
        prototypes = (
            torch.tensor(proto_plus, dtype=torch.float32),
            torch.tensor(proto_minus, dtype=torch.float32),
        )
        
        with torch.no_grad():
            z, details = actor.select_z_cem(obs, prototypes, env.bend_amp)
            d_signed = actor.compute_signed_deviation(details["mu"].unsqueeze(0), latent_t).item()
            d_values.append(d_signed)
            
            if d_signed > 0:
                z_plus.append(z.numpy())
            else:
                z_minus.append(z.numpy())
    
    z_plus = np.array(z_plus) if z_plus else np.array([]).reshape(0, actor.z_dim)
    z_minus = np.array(z_minus) if z_minus else np.array([]).reshape(0, actor.z_dim)
    
    if len(z_plus) > 10 and len(z_minus) > 10:
        mean_plus = z_plus.mean(axis=0)
        mean_minus = z_minus.mean(axis=0)
        
        basin_distance = np.linalg.norm(mean_plus - mean_minus)
        spread_plus = np.sqrt(((z_plus - mean_plus) ** 2).sum(axis=1)).mean()
        spread_minus = np.sqrt(((z_minus - mean_minus) ** 2).sum(axis=1)).mean()
        avg_spread = (spread_plus + spread_minus) / 2
        separation_ratio = basin_distance / (avg_spread + 1e-6)
        mci_mean = np.abs(np.array(d_values)).mean() / env.bend_amp
        
        print(f"  {name} basin analysis:")
        print(f"    Mode+ samples: {len(z_plus)}, Mode- samples: {len(z_minus)}")
        print(f"    Basin distance: {basin_distance:.3f}")
        print(f"    Avg spread: {avg_spread:.3f}")
        print(f"    Separation ratio: {separation_ratio:.2f} (>2 = good separation)")
        
        return {
            "basin_distance": float(basin_distance),
            "avg_spread": float(avg_spread),
            "separation_ratio": float(separation_ratio),
            "n_plus": len(z_plus),
            "n_minus": len(z_minus),
            "mci_mean": float(mci_mean),
        }
    else:
        print(f"  {name}: Not enough samples in both modes")
        return {}


def plot_z_space(
    actor_A: MinimalCompetitiveActor,
    actor_B: MinimalCompetitiveActor,
    env: BimodalEnv,
    obs_A_fn,
    obs_B_fn,
    out_path: Path,
    n_samples: int = 300,
):
    """Visualize z-space basin structure for both actors."""
    from sklearn.decomposition import PCA
    
    z_A_plus, z_A_minus = [], []
    z_B_plus, z_B_minus = [], []
    
    for _ in range(n_samples):
        latent = env.reset()
        latent_t = torch.tensor(latent, dtype=torch.float32)
        obs_A = torch.tensor(obs_A_fn(latent), dtype=torch.float32)
        obs_B = torch.tensor(obs_B_fn(latent), dtype=torch.float32)
        
        proto_plus, proto_minus = env.get_mode_prototypes()
        prototypes = (
            torch.tensor(proto_plus, dtype=torch.float32),
            torch.tensor(proto_minus, dtype=torch.float32),
        )
        
        with torch.no_grad():
            z_A, details_A = actor_A.select_z_cem(obs_A, prototypes, env.bend_amp)
            z_B, details_B = actor_B.select_z_cem(obs_B, prototypes, env.bend_amp)
            
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
    parser.add_argument("--bend-amplitude", type=float, default=0.15)
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
    print(f"Alien Sensors Experiment (Minimal Actor + Fairness Controls)")
    print(f"Level: {alien_level.name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Environment: DETERMINISTIC mode for consistent A↔B mapping
    env = BimodalEnv(seed=args.seed, bend_amp=args.bend_amplitude, deterministic_mode=True)
    
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
    
    # =========================================================================
    # FAIRNESS CONTROL 1: Bayes Ceiling
    # What mode accuracy can B achieve from observations alone?
    # Uses RANDOM mode to measure true information content of B's obs.
    # =========================================================================
    print("\n[Bayes Ceiling] Training classifier: B_obs → mode (random mode)")
    ceiling_classifier, bayes_ceiling = train_bayes_ceiling(
        obs_B_fn, input_dim=4, n_samples=args.pair_samples, seed=args.seed + 5000
    )
    print(f"  Bayes Ceiling (B_obs → mode): {bayes_ceiling:.1%}")
    print(f"  (If ceiling ≈ 50%, B's sensors are truly mode-blind)")
    print(f"  (Training/pairing use deterministic mode for consistent mappings)")
    
    # Actors (using MinimalCompetitiveActor)
    actor_A = MinimalCompetitiveActor(obs_dim=4, z_dim=args.z_dim, T=16)
    actor_B = MinimalCompetitiveActor(obs_dim=obs_B_dim, z_dim=args.z_dim, T=16)
    
    # === Stage 0: Train Actor A ===
    print("\n[Stage 0] Training Actor A (normal obs)")
    hist_A = train_actor(actor_A, env, obs_A_fn, n_steps=args.actor_steps, name="A")
    torch.save(actor_A.state_dict(), out_dir / "actor_A.pt")
    
    # === Stage 1: Train Actor B (alien obs) ===
    print(f"\n[Stage 1] Training Actor B ({alien_level.name} obs)")
    hist_B = train_actor(actor_B, env, obs_B_fn, n_steps=args.actor_steps, name="B")
    torch.save(actor_B.state_dict(), out_dir / "actor_B.pt")
    
    plot_training(hist_A, hist_B, out_dir / "training.png")
    
    # === Basin Analysis ===
    print("\n[Basin Analysis] Checking z-space structure")
    basin_A = analyze_basin_structure(actor_A, env, obs_A_fn, n_samples=500, name="A")
    basin_B = analyze_basin_structure(actor_B, env, obs_B_fn, n_samples=500, name="B")
    
    try:
        plot_z_space(actor_A, actor_B, env, obs_A_fn, obs_B_fn, out_dir / "z_space.png")
    except ImportError:
        print("  (sklearn not available for PCA visualization)")
    
    # === Stage 2: Collect Paired Data ===
    print("\n[Stage 2] Collecting paired data")
    pairs = collect_paired_data(actor_A, actor_B, env, obs_A_fn, obs_B_fn, n_samples=args.pair_samples)
    np.savez(out_dir / "pairs.npz", **pairs)
    
    pair_agreement = (pairs["modes_A"] == pairs["modes_B"]).mean()
    print(f"  Pair mode agreement (A vs B native): {pair_agreement:.1%}")
    
    d_corr = np.corrcoef(pairs["d_A"], pairs["d_B"])[0, 1]
    print(f"  d correlation (A vs B): {d_corr:.3f}")
    
    # === Stage 2a: Train Normal Functor ===
    print("\n[Stage 2a] Training NORMAL functor (z_A → z_B)")
    functor = FunctorNet(z_dim=args.z_dim)
    functor_hist = train_functor(functor, pairs, n_epochs=args.functor_epochs, shuffle_pairs=False)
    torch.save(functor.state_dict(), out_dir / "functor.pt")
    
    # =========================================================================
    # FAIRNESS CONTROL 2: Shuffle Ablation
    # Train functor with shuffled z_A to break correspondence.
    # If transfer still works, there's leakage; if it collapses to ceiling, A is the bridge.
    # =========================================================================
    print("\n[Stage 2b] Training SHUFFLED functor (ablation)")
    functor_shuffled = FunctorNet(z_dim=args.z_dim)
    _ = train_functor(functor_shuffled, pairs, n_epochs=args.functor_epochs, shuffle_pairs=True)
    torch.save(functor_shuffled.state_dict(), out_dir / "functor_shuffled.pt")
    
    # === Stage 3: Evaluate Normal Transfer ===
    print("\n[Stage 3a] Evaluating NORMAL transfer")
    summary = evaluate_transfer(
        actor_A, actor_B, functor, env, obs_A_fn, obs_B_fn,
        n_samples=args.eval_samples, functor_label="transfer"
    )
    
    # === Stage 3b: Evaluate Shuffled Transfer (Ablation) ===
    print("\n[Stage 3b] Evaluating SHUFFLED transfer (ablation)")
    summary_shuffled = evaluate_transfer(
        actor_A, actor_B, functor_shuffled, env, obs_A_fn, obs_B_fn,
        n_samples=args.eval_samples, functor_label="transfer_shuffled"
    )
    
    # =========================================================================
    # Compute Lift and Compile Results
    # =========================================================================
    transfer_acc = summary["mode_agreement"]["mean"]
    shuffle_acc = summary_shuffled["mode_agreement"]["mean"]
    lift = transfer_acc - bayes_ceiling
    shuffle_lift = shuffle_acc - bayes_ceiling
    
    summary["bayes_ceiling"] = float(bayes_ceiling)
    summary["lift"] = float(lift)
    summary["shuffle_ablation"] = {
        "mode_agreement": summary_shuffled["mode_agreement"],
        "lift": float(shuffle_lift),
    }
    summary["pair_mode_agreement"] = float(pair_agreement)
    summary["basin_A"] = basin_A
    summary["basin_B"] = basin_B
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    plot_results(summary, alien_level.name, out_dir / "results.png")
    
    # === Final Report ===
    print(f"\n{'='*60}")
    print("ALIEN SENSORS RESULTS (with Fairness Controls)")
    print(f"{'='*60}")
    
    print(f"\n┌─────────────────────────────────────────────────────────┐")
    print(f"│  FAIRNESS METRICS                                       │")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│  Bayes Ceiling (B_obs → mode):  {bayes_ceiling:>6.1%}                 │")
    print(f"│  Transfer Accuracy:             {transfer_acc:>6.1%}                 │")
    print(f"│  Lift (Transfer - Ceiling):     {lift:>+6.1%}                 │")
    print(f"├─────────────────────────────────────────────────────────┤")
    print(f"│  Shuffle Ablation Accuracy:     {shuffle_acc:>6.1%}                 │")
    print(f"│  Shuffle Lift:                  {shuffle_lift:>+6.1%}                 │")
    print(f"└─────────────────────────────────────────────────────────┘")
    
    print(f"\nBasin Structure:")
    if basin_A:
        print(f"  A separation ratio: {basin_A.get('separation_ratio', 0):.2f}")
    if basin_B:
        print(f"  B separation ratio: {basin_B.get('separation_ratio', 0):.2f}")
    
    trans_bind = summary["transfer"]["bind_mean"]
    cem_bind = summary["B_CEM"]["bind_mean"]
    print(f"\nTransfer Bind: {trans_bind:.3f}")
    print(f"B-CEM Bind: {cem_bind:.3f}")
    
    # Success criteria (updated with fairness)
    lift_positive = lift > 0.05  # Transfer adds >5% over B's own ceiling
    shuffle_collapsed = shuffle_acc < bayes_ceiling + 0.10  # Shuffle near ceiling
    agreement_ok = transfer_acc > 0.85
    bind_ok = abs(trans_bind - cem_bind) < 0.05
    
    print(f"\nSuccess Criteria:")
    print(f"  {'✓' if lift_positive else '✗'} Positive lift (Transfer > Ceiling + 5%)")
    print(f"  {'✓' if shuffle_collapsed else '✗'} Shuffle ablation collapses (~ceiling)")
    print(f"  {'✓' if agreement_ok else '✗'} Mode agreement > 85%")
    print(f"  {'✓' if bind_ok else '✗'} Transfer matches B-CEM bind")
    
    # Interpretation
    print(f"\nInterpretation:")
    if lift_positive and shuffle_collapsed:
        print(f"  ✓ A's commitment IS the bridge (not B's sensor leakage)")
    elif not lift_positive:
        print(f"  ⚠ B can already infer mode from sensors (low lift)")
    elif not shuffle_collapsed:
        print(f"  ⚠ Possible leakage: shuffle should collapse to ceiling")
    
    if agreement_ok and bind_ok and lift_positive and shuffle_collapsed:
        print(f"\n🎉 STRONG RESULT: Topological bridge verified at {alien_level.name}!")
    elif agreement_ok and bind_ok:
        print(f"\n✓ Transfer works at {alien_level.name} (but see fairness caveats)")
    else:
        print(f"\n⚠ Transfer not fully achieved at {alien_level.name}")
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
