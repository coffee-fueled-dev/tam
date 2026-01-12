"""
Minimal Composition / Transfer Experiment: A → F → B

Tests whether commitments (z*) can be transferred between agents
that observe the same latent dynamics through different observation maps.

Core claim: Transfer (1 candidate) beats B-random/B-best-of-K,
approaching B-CEM performance at far lower compute.
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

from actor import CompetitiveActor, BindingMode


# =============================================================================
# 1) Multi-View Bimodal Environment
# =============================================================================

class LatentBimodalEnv:
    """
    Bimodal environment with factored latent state and observation maps.
    
    Same latent dynamics, two different observation transforms g_A and g_B.
    """
    
    def __init__(
        self,
        seed: int = 0,
        T: int = 16,
        bend_amp: float = 0.15,
        base_noise: float = 0.02,
        latent_dim: int = 4,  # x, y, goal_x, goal_y
    ):
        self.rng = np.random.default_rng(seed)
        self.T = T
        self.bend_amp = bend_amp
        self.base_noise = base_noise
        self.latent_dim = latent_dim
        
        # Fixed random transform for obs_B (rotation + scale)
        angle = np.pi / 4  # 45 degree rotation
        scale = 1.2
        self.M_B = scale * np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, np.cos(angle), -np.sin(angle)],
            [0, 0, np.sin(angle), np.cos(angle)],
        ], dtype=np.float32)
        self.b_B = np.array([0.1, -0.05, 0.05, 0.1], dtype=np.float32)
        
        self.s0_latent = None
        self.mode = None
        self.reset()
    
    def reset(self, force_mode: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset and return (obs_A, obs_B, s0_latent)."""
        self.s0_latent = np.array([
            self.rng.uniform(0.2, 0.8),  # x
            self.rng.uniform(0.2, 0.8),  # y
            self.rng.uniform(0.2, 0.8),  # goal_x
            self.rng.uniform(0.2, 0.8),  # goal_y
        ], dtype=np.float32)
        
        self.mode = force_mode if force_mode is not None else self.rng.choice([-1, 1])
        
        obs_A = self.g_A(self.s0_latent)
        obs_B = self.g_B(self.s0_latent)
        
        return obs_A, obs_B, self.s0_latent.copy()
    
    def g_A(self, s0_latent: np.ndarray) -> np.ndarray:
        """Observation transform A: identity."""
        return s0_latent.copy()
    
    def g_B(self, s0_latent: np.ndarray) -> np.ndarray:
        """Observation transform B: rotation + scale + offset."""
        return (self.M_B @ s0_latent + self.b_B).astype(np.float32)
    
    def generate_trajectory(self) -> np.ndarray:
        """Generate trajectory in latent space based on mode."""
        traj = np.zeros((self.T, 2), dtype=np.float32)
        
        x, y, goal_x, goal_y = self.s0_latent
        dx = goal_x - x
        dy = goal_y - y
        
        # Baseline direction and perpendicular
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        d_unit = np.array([dx / dist, dy / dist])
        d_perp = np.array([-d_unit[1], d_unit[0]])
        
        for t in range(self.T):
            progress = (t + 1) / self.T
            
            # Baseline straight-line
            base_x = x + dx * progress
            base_y = y + dy * progress
            
            # Mode-dependent bend (sinusoidal, peaks at mid-horizon)
            bend = np.sin(np.pi * progress) * self.bend_amp * self.mode
            
            traj[t, 0] = base_x + bend * d_perp[0] + self.rng.normal(0, self.base_noise)
            traj[t, 1] = base_y + bend * d_perp[1] + self.rng.normal(0, self.base_noise)
        
        return traj


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
# 3) Training Loop
# =============================================================================

def train_actors(
    actor_A: CompetitiveActor,
    actor_B: CompetitiveActor,
    env: LatentBimodalEnv,
    n_steps: int = 5000,
    log_every: int = 1000,
) -> Tuple[Dict, Dict]:
    """Train both actors on their respective observations."""
    print(f"Training Actor A and B for {n_steps} steps...")
    
    for step in range(n_steps):
        obs_A, obs_B, _ = env.reset()
        traj = env.generate_trajectory()
        
        obs_A_t = torch.tensor(obs_A, dtype=torch.float32)
        obs_B_t = torch.tensor(obs_B, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Train A
        result_A = actor_A.select_z(obs_A_t, mode=BindingMode.CEM)
        actor_A.train_step(obs_A_t, traj_t, result_A)
        
        # Train B
        result_B = actor_B.select_z(obs_B_t, mode=BindingMode.CEM)
        actor_B.train_step(obs_B_t, traj_t, result_B)
        
        if step % log_every == 0:
            stats_A = actor_A.get_equilibrium_stats(window=min(500, step + 1))
            stats_B = actor_B.get_equilibrium_stats(window=min(500, step + 1))
            bind_A = stats_A.get("emergent_bind", 0)
            bind_B = stats_B.get("emergent_bind", 0)
            print(f"  Step {step}: A bind={bind_A:.3f}, B bind={bind_B:.3f}")
    
    return actor_A.history, actor_B.history


def collect_paired_data(
    actor_A: CompetitiveActor,
    actor_B: CompetitiveActor,
    env: LatentBimodalEnv,
    n_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """Collect paired (zA*, zB*) from CEM binding on same latent situations."""
    actor_A.eval()
    actor_B.eval()
    
    Z_A = []
    Z_B = []
    modes = []
    
    print(f"Collecting {n_samples} paired commitment samples...")
    
    for i in range(n_samples):
        obs_A, obs_B, _ = env.reset()
        
        obs_A_t = torch.tensor(obs_A, dtype=torch.float32)
        obs_B_t = torch.tensor(obs_B, dtype=torch.float32)
        
        with torch.no_grad():
            result_A = actor_A.select_z(obs_A_t, mode=BindingMode.CEM)
            result_B = actor_B.select_z(obs_B_t, mode=BindingMode.CEM)
        
        Z_A.append(result_A.z_star.numpy())
        Z_B.append(result_B.z_star.numpy())
        modes.append(env.mode)
    
    return {
        "Z_A": np.array(Z_A),
        "Z_B": np.array(Z_B),
        "modes": np.array(modes),
    }


def train_functor(
    functor: FunctorNet,
    pairs: Dict[str, np.ndarray],
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_struct: float = 0.1,
) -> Dict[str, List[float]]:
    """Train functor F: zA → zB with alignment + structure preservation loss."""
    optimizer = optim.Adam(functor.parameters(), lr=lr)
    
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
            
            # Forward
            z_B_hat = functor(z_A)
            
            # Alignment loss: MSE(F(zA), zB)
            align_loss = ((z_B_hat - z_B) ** 2).mean()
            
            # Structure preservation loss: pairwise distance matching
            B = z_A.shape[0]
            if B > 1:
                # Compute pairwise distances
                D_A = torch.cdist(z_A, z_A)  # [B, B]
                D_B = torch.cdist(z_B, z_B)
                D_F = torch.cdist(z_B_hat, z_B_hat)
                
                # Match F's distances to B's distances
                struct_loss = ((D_F - D_B) ** 2).mean()
            else:
                struct_loss = torch.tensor(0.0)
            
            loss = align_loss + lambda_struct * struct_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        mean_loss = np.mean(epoch_losses)
        history["loss"].append(mean_loss)
        history["align_loss"].append(float(align_loss.item()))
        history["struct_loss"].append(float(struct_loss.item()))
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={mean_loss:.4f}")
    
    return history


# =============================================================================
# 4) Transfer Evaluation
# =============================================================================

def evaluate_transfer(
    actor_A: CompetitiveActor,
    actor_B: CompetitiveActor,
    functor: FunctorNet,
    env: LatentBimodalEnv,
    n_samples: int = 500,
) -> Dict[str, Dict]:
    """Evaluate transfer: A → F → B vs native B methods."""
    actor_A.eval()
    actor_B.eval()
    functor.eval()
    
    results = {
        "transfer": defaultdict(list),
        "B_CEM": defaultdict(list),
        "B_random": defaultdict(list),
        "B_best_of_16": defaultdict(list),
    }
    
    print(f"Evaluating transfer on {n_samples} samples...")
    
    for i in range(n_samples):
        obs_A, obs_B, _ = env.reset()
        traj = env.generate_trajectory()
        
        obs_A_t = torch.tensor(obs_A, dtype=torch.float32)
        obs_B_t = torch.tensor(obs_B, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        with torch.no_grad():
            # 1. Transfer: zA* from A, map through F, evaluate in B
            result_A = actor_A.select_z(obs_A_t, mode=BindingMode.CEM)
            z_B_transfer = functor(result_A.z_star.unsqueeze(0)).squeeze(0)
            mu_trans, sigma_trans = actor_B.tube.predict_tube(obs_B_t, z_B_transfer)
            mu_trans = mu_trans.squeeze(0)
            sigma_trans = sigma_trans.squeeze(0)
            
            # 2. B-CEM: native CEM in B
            result_B_CEM = actor_B.select_z(obs_B_t, mode=BindingMode.CEM)
            
            # 3. B-random: sample from posterior
            result_B_random = actor_B.select_z(obs_B_t, mode=BindingMode.RANDOM)
            
            # 4. B-best-of-16
            result_B_bok = actor_B.select_z_best_of_k(obs_B_t, K=16)
            
            # Evaluate all
            for name, mu, sigma in [
                ("transfer", mu_trans, sigma_trans),
                ("B_CEM", result_B_CEM.mu, result_B_CEM.sigma),
                ("B_random", result_B_random.mu, result_B_random.sigma),
                ("B_best_of_16", result_B_bok.mu, result_B_bok.sigma),
            ]:
                bind = actor_B.compute_bind_hard(mu, sigma, traj_t).item()
                log_vol = actor_B.compute_log_volume(sigma).item()
                mse = actor_B.compute_mse(mu, traj_t).item()
                
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
        print(f"  {name}: bind={summary[name]['bind_mean']:.3f}±{summary[name]['bind_std']:.3f}, "
              f"vol={summary[name]['log_vol_mean']:.2f}")
    
    return summary


# =============================================================================
# 5) Plotting
# =============================================================================

def plot_training(hist_A: Dict, hist_B: Dict, out_path: Path):
    """Plot training curves for both actors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    window = 100
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # Bind rate
    ax = axes[0]
    ax.plot(smooth(hist_A["bind_hard"]), label='Actor A', color='blue')
    ax.plot(smooth(hist_B["bind_hard"]), label='Actor B', color='red')
    ax.set_title("Bind Rate")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log volume
    ax = axes[1]
    ax.plot(smooth(hist_A["log_vol"]), label='Actor A', color='blue')
    ax.plot(smooth(hist_B["log_vol"]), label='Actor B', color='red')
    ax.set_title("Log Volume")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MSE
    ax = axes[2]
    ax.plot(smooth(hist_A["mse"]), label='Actor A', color='blue')
    ax.plot(smooth(hist_B["mse"]), label='Actor B', color='red')
    ax.set_title("MSE")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Actor Training")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_functor_training(history: Dict, out_path: Path):
    """Plot functor training loss."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.plot(history["loss"], label='Total Loss', color='black')
    ax.plot(history["align_loss"], label='Align Loss', color='blue', alpha=0.7)
    ax.plot(history["struct_loss"], label='Struct Loss', color='red', alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Functor Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_transfer_comparison(summary: Dict, out_path: Path):
    """Plot bind vs volume Pareto comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ["B_random", "B_best_of_16", "transfer", "B_CEM"]
    labels = ["B-Random", "B-Best-of-16", "Transfer (A→F→B)", "B-CEM"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    
    # Bar chart
    ax = axes[0]
    x = np.arange(len(methods))
    binds = [summary[m]["bind_mean"] for m in methods]
    bind_stds = [summary[m]["bind_std"] for m in methods]
    ax.bar(x, binds, yerr=bind_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Bind Rate")
    ax.set_title("Reliability")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Pareto scatter
    ax = axes[1]
    for m, label, color in zip(methods, labels, colors):
        ax.scatter(
            summary[m]["log_vol_mean"],
            summary[m]["bind_mean"],
            s=200, c=color, label=label,
            edgecolors='black', linewidths=2
        )
        ax.errorbar(
            summary[m]["log_vol_mean"],
            summary[m]["bind_mean"],
            xerr=summary[m]["log_vol_std"],
            yerr=summary[m]["bind_std"],
            color=color, fmt='none', capsize=5
        )
    
    ax.set_xlabel("Log Volume (lower = tighter)")
    ax.set_ylabel("Bind Rate (higher = reliable)")
    ax.set_title("Bind vs Volume Tradeoff")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Transfer Evaluation: A → F → B")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 6) Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actor-steps", type=int, default=5000)
    parser.add_argument("--functor-epochs", type=int, default=100)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--lambda-struct", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"transfer_{timestamp}"
    if args.name:
        run_name = f"transfer_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Composition / Transfer Experiment")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create environment
    env = LatentBimodalEnv(seed=args.seed)
    
    # Create actors for A and B (same architecture, different obs spaces)
    actor_A = CompetitiveActor(
        obs_dim=4,  # g_A is identity
        pred_dim=2,
        z_dim=args.z_dim,
        T=16,
    )
    actor_B = CompetitiveActor(
        obs_dim=4,  # g_B has same dim (transformed)
        pred_dim=2,
        z_dim=args.z_dim,
        T=16,
    )
    
    # === Step 1: Train both actors ===
    print("\n[Step 1] Training Actor A and Actor B")
    hist_A, hist_B = train_actors(actor_A, actor_B, env, n_steps=args.actor_steps)
    plot_training(hist_A, hist_B, out_dir / "actor_training.png")
    
    # Save actor checkpoints
    torch.save(actor_A.state_dict(), out_dir / "actor_A.pt")
    torch.save(actor_B.state_dict(), out_dir / "actor_B.pt")
    
    # === Step 2: Collect paired commitment data ===
    print("\n[Step 2] Collecting paired commitment data")
    pairs = collect_paired_data(actor_A, actor_B, env, n_samples=args.pair_samples)
    np.savez(out_dir / "pairs.npz", **pairs)
    
    # === Step 3: Train functor ===
    print("\n[Step 3] Training functor F: zA → zB")
    functor = FunctorNet(z_dim=args.z_dim)
    functor_hist = train_functor(
        functor, pairs,
        n_epochs=args.functor_epochs,
        lambda_struct=args.lambda_struct,
    )
    plot_functor_training(functor_hist, out_dir / "functor_training.png")
    torch.save(functor.state_dict(), out_dir / "functor.pt")
    
    # === Step 4: Evaluate transfer ===
    print("\n[Step 4] Evaluating transfer")
    summary = evaluate_transfer(actor_A, actor_B, functor, env, n_samples=args.eval_samples)
    plot_transfer_comparison(summary, out_dir / "transfer_comparison.png")
    
    # Save summary
    with open(out_dir / "transfer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # === Final Analysis ===
    print(f"\n{'='*60}")
    print("TRANSFER RESULTS")
    print(f"{'='*60}")
    
    trans_bind = summary["transfer"]["bind_mean"]
    random_bind = summary["B_random"]["bind_mean"]
    bok_bind = summary["B_best_of_16"]["bind_mean"]
    cem_bind = summary["B_CEM"]["bind_mean"]
    
    trans_vol = summary["transfer"]["log_vol_mean"]
    random_vol = summary["B_random"]["log_vol_mean"]
    cem_vol = summary["B_CEM"]["log_vol_mean"]
    
    print(f"\nPerformance:")
    print(f"  B-Random:     bind={random_bind:.3f}, vol={random_vol:.2f}")
    print(f"  B-Best-of-16: bind={bok_bind:.3f}")
    print(f"  Transfer:     bind={trans_bind:.3f}, vol={trans_vol:.2f}")
    print(f"  B-CEM:        bind={cem_bind:.3f}, vol={cem_vol:.2f}")
    
    # Success criteria (Pareto-aware)
    # Transfer should match CEM's tradeoff, not Random's loose tubes
    matches_cem_bind = abs(trans_bind - cem_bind) < 0.05
    matches_cem_vol = abs(trans_vol - cem_vol) < 0.3
    tighter_than_random = trans_vol < random_vol - 0.3
    tighter_than_bok = trans_vol < summary["B_best_of_16"]["log_vol_mean"] - 0.1
    
    print(f"\nSuccess Criteria (Pareto-aware):")
    print(f"  {'✓' if matches_cem_bind else '✗'} Transfer matches B-CEM bind rate (within 5%)")
    print(f"  {'✓' if matches_cem_vol else '✗'} Transfer matches B-CEM volume (within 0.3)")
    print(f"  {'✓' if tighter_than_random else '✗'} Transfer uses tighter tubes than Random")
    print(f"  {'✓' if tighter_than_bok else '✗'} Transfer uses tighter tubes than Best-of-K")
    
    if matches_cem_bind and matches_cem_vol:
        print("\n🎉 SUCCESS: Decision-level transfer works!")
        print("   Transfer (1 candidate) matches B-CEM (512 candidates).")
        print("   Commitments learned in A transfer meaningfully to B.")
    else:
        print("\n⚠ Transfer not fully achieved. May need tuning.")
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
