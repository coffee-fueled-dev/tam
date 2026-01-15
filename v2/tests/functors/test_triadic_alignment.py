"""
Triadic Coordination-Induced Gauge Alignment.

Tests whether interaction pressure alone can induce globally consistent
functor mappings between three independently trained actors.

The key insight: three agents with arbitrary internal conventions must
coordinate on a shared task (agreeing on which basin is "correct" for
a hidden token). This coordination pressure forces the functor maps
to become globally consistent, fixing the gauge without pinning any basin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .zmap import ZMap
from .utils_energy import sample_z_sphere, compute_energy, get_basin_centers


@dataclass
class TriadicConfig:
    """Configuration for triadic alignment experiment."""
    z_dim: int = 2
    K: int = 3  # Number of basins/modes
    n_basin_samples: int = 200  # Samples for basin detection
    tau: float = 0.05  # Temperature for basin softmax
    lambda_cyc: float = 1.0
    lambda_comp: float = 1.0
    lr: float = 1e-3


class TriadicFunctorTrainer:
    """
    Trains 6 functor maps (F_AB, F_BA, F_AC, F_CA, F_BC, F_CB) to become
    globally consistent through triadic coordination.
    
    Loss = L_task + λ_cyc * L_cyc + λ_comp * L_comp
    """
    
    def __init__(
        self,
        actor_A,
        actor_B,
        actor_C,
        env_A,
        env_B,
        env_C,
        config: TriadicConfig,
        device: torch.device,
    ):
        self.actor_A = actor_A
        self.actor_B = actor_B
        self.actor_C = actor_C
        self.env_A = env_A
        self.env_B = env_B
        self.env_C = env_C
        self.config = config
        self.device = device
        
        # Freeze actors
        for actor in [actor_A, actor_B, actor_C]:
            for param in actor.parameters():
                param.requires_grad = False
        
        # 6 functor maps
        self.F_AB = ZMap(config.z_dim).to(device)
        self.F_BA = ZMap(config.z_dim).to(device)
        self.F_AC = ZMap(config.z_dim).to(device)
        self.F_CA = ZMap(config.z_dim).to(device)
        self.F_BC = ZMap(config.z_dim).to(device)
        self.F_CB = ZMap(config.z_dim).to(device)
        
        all_params = (
            list(self.F_AB.parameters()) +
            list(self.F_BA.parameters()) +
            list(self.F_AC.parameters()) +
            list(self.F_CA.parameters()) +
            list(self.F_BC.parameters()) +
            list(self.F_CB.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=config.lr)
        
        self.history = []
    
    def _get_basin_representatives(
        self,
        actor,
        s0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get K basin representative z vectors for an actor at s0.
        
        Returns:
            (K, z_dim) tensor of basin centers
        """
        result = get_basin_centers(
            actor, s0, 
            n_samples=self.config.n_basin_samples,
            n_basins_max=self.config.K
        )
        centers = result['centers']
        
        # If we found fewer than K basins, pad with random z
        n_found = centers.shape[0]
        if n_found < self.config.K:
            padding = sample_z_sphere(
                self.config.K - n_found, 
                self.config.z_dim, 
                self.device
            )
            centers = torch.cat([centers, padding], dim=0)
        elif n_found > self.config.K:
            centers = centers[:self.config.K]
        
        return centers
    
    def _basin_logits(
        self,
        z: torch.Tensor,
        basin_reps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute softmax logits for basin assignment.
        
        p(k | z) = softmax_k(cos(z, z^k) / tau)
        
        Args:
            z: (N, z_dim) candidate z values
            basin_reps: (K, z_dim) basin representative vectors
        
        Returns:
            (N, K) logits
        """
        # Cosine similarity
        z_norm = F.normalize(z, p=2, dim=-1)  # (N, z_dim)
        reps_norm = F.normalize(basin_reps, p=2, dim=-1)  # (K, z_dim)
        
        sim = z_norm @ reps_norm.T  # (N, K)
        logits = sim / self.config.tau
        
        return logits
    
    def _sample_message(
        self,
        actor,
        basin_reps: torch.Tensor,
        target_basin: int,
        n_samples: int = 10,
        noise_scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Sample message z_A near the target basin representative.
        
        Returns:
            (n_samples, z_dim) tensor of z samples near target basin
        """
        target_z = basin_reps[target_basin]  # (z_dim,)
        
        # Add noise and renormalize
        noise = torch.randn(n_samples, self.config.z_dim, device=self.device)
        z = target_z.unsqueeze(0) + noise_scale * noise
        z = F.normalize(z, p=2, dim=-1)
        
        return z
    
    def train_step(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Single training step with triadic coordination.
        
        Returns:
            Dict of metrics
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 1. Sample hidden token c
        c = np.random.randint(0, self.config.K)
        
        # 2. Get corresponding situations (same random seed)
        episode_seed = np.random.randint(0, 1000000)
        np.random.seed(episode_seed)
        obs_A = self.env_A.reset()
        np.random.seed(episode_seed)
        obs_B = self.env_B.reset()
        np.random.seed(episode_seed)
        obs_C = self.env_C.reset()
        
        s0_A = torch.tensor(obs_A, dtype=torch.float32, device=self.device)
        s0_B = torch.tensor(obs_B, dtype=torch.float32, device=self.device)
        s0_C = torch.tensor(obs_C, dtype=torch.float32, device=self.device)
        
        # 3. Get basin representatives for each actor
        with torch.no_grad():
            basins_A = self._get_basin_representatives(self.actor_A, s0_A)
            basins_B = self._get_basin_representatives(self.actor_B, s0_B)
            basins_C = self._get_basin_representatives(self.actor_C, s0_C)
        
        # 4. Sample message z_A from A near its target basin
        z_A = self._sample_message(self.actor_A, basins_A, c, n_samples=32)
        
        # 5. Map to B and C
        z_B = self.F_AB(z_A)
        z_C = self.F_AC(z_A)
        
        # Also sample in B and C for cycle losses
        z_B_native = sample_z_sphere(32, self.config.z_dim, self.device)
        z_C_native = sample_z_sphere(32, self.config.z_dim, self.device)
        
        # ========== TASK LOSS ==========
        # Cross-entropy for B and C matching basin c
        logits_A = self._basin_logits(z_A, basins_A)
        logits_B = self._basin_logits(z_B, basins_B)
        logits_C = self._basin_logits(z_C, basins_C)
        
        target = torch.full((z_A.shape[0],), c, device=self.device, dtype=torch.long)
        
        loss_task_A = F.cross_entropy(logits_A, target)
        loss_task_B = F.cross_entropy(logits_B, target)
        loss_task_C = F.cross_entropy(logits_C, target)
        
        L_task = loss_task_A + loss_task_B + loss_task_C
        
        # Accuracy
        pred_A = logits_A.argmax(dim=-1)
        pred_B = logits_B.argmax(dim=-1)
        pred_C = logits_C.argmax(dim=-1)
        
        acc_A = (pred_A == target).float().mean()
        acc_B = (pred_B == target).float().mean()
        acc_C = (pred_C == target).float().mean()
        
        # ========== CYCLE CONSISTENCY LOSS ==========
        # A -> B -> A
        z_ABA = self.F_BA(self.F_AB(z_A))
        # A -> C -> A
        z_ACA = self.F_CA(self.F_AC(z_A))
        # B -> A -> B
        z_BAB = self.F_AB(self.F_BA(z_B_native))
        # B -> C -> B
        z_BCB = self.F_CB(self.F_BC(z_B_native))
        # C -> A -> C
        z_CAC = self.F_AC(self.F_CA(z_C_native))
        # C -> B -> C
        z_CBC = self.F_BC(self.F_CB(z_C_native))
        
        def cycle_loss(z, z_cycle):
            sim = (F.normalize(z, dim=-1) * F.normalize(z_cycle, dim=-1)).sum(dim=-1)
            return (1.0 - sim).mean()
        
        L_cyc = (
            cycle_loss(z_A, z_ABA) +
            cycle_loss(z_A, z_ACA) +
            cycle_loss(z_B_native, z_BAB) +
            cycle_loss(z_B_native, z_BCB) +
            cycle_loss(z_C_native, z_CAC) +
            cycle_loss(z_C_native, z_CBC)
        ) / 6.0
        
        # ========== COMPOSITION CONSISTENCY LOSS ==========
        # F_AC(z) ≈ F_BC(F_AB(z))
        z_AC = self.F_AC(z_A)
        z_ABC = self.F_BC(self.F_AB(z_A))
        
        # F_AB(z) ≈ F_CB(F_AC(z))  (using inverse direction)
        z_AB = self.F_AB(z_A)
        z_ACB = self.F_CB(self.F_AC(z_A))
        
        # F_BC(z) ≈ F_AC(F_BA(z))
        z_BC = self.F_BC(z_B_native)
        z_BAC = self.F_AC(self.F_BA(z_B_native))
        
        def comp_loss(z1, z2):
            sim = (F.normalize(z1, dim=-1) * F.normalize(z2, dim=-1)).sum(dim=-1)
            return (1.0 - sim).mean()
        
        L_comp = (
            comp_loss(z_AC, z_ABC) +
            comp_loss(z_AB, z_ACB) +
            comp_loss(z_BC, z_BAC)
        ) / 3.0
        
        # ========== TOTAL LOSS ==========
        loss = L_task + self.config.lambda_cyc * L_cyc + self.config.lambda_comp * L_comp
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Composition metric (for monitoring)
        with torch.no_grad():
            z_test = sample_z_sphere(100, self.config.z_dim, self.device)
            z_AC_test = self.F_AC(z_test)
            z_ABC_test = self.F_BC(self.F_AB(z_test))
            comp_sim = (F.normalize(z_AC_test, dim=-1) * F.normalize(z_ABC_test, dim=-1)).sum(dim=-1).mean()
        
        metrics = {
            'loss': loss.item(),
            'L_task': L_task.item(),
            'L_cyc': L_cyc.item(),
            'L_comp': L_comp.item(),
            'acc_A': acc_A.item(),
            'acc_B': acc_B.item(),
            'acc_C': acc_C.item(),
            'comp_sim': comp_sim.item(),
            'target_basin': c,
        }
        self.history.append(metrics)
        
        return metrics
    
    def train(
        self,
        n_epochs: int = 5000,
        print_every: int = 500,
    ) -> List[Dict[str, float]]:
        """Run full training loop."""
        for epoch in range(n_epochs):
            metrics = self.train_step()
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"loss={metrics['loss']:.3f}, "
                      f"task={metrics['L_task']:.3f}, "
                      f"comp_sim={metrics['comp_sim']:.3f}, "
                      f"acc_B={metrics['acc_B']:.2f}")
        
        return self.history
    
    def get_functors(self) -> Dict[str, ZMap]:
        """Return all trained functors."""
        return {
            'F_AB': self.F_AB,
            'F_BA': self.F_BA,
            'F_AC': self.F_AC,
            'F_CA': self.F_CA,
            'F_BC': self.F_BC,
            'F_CB': self.F_CB,
        }


def plot_triadic_results(
    history: List[Dict[str, float]],
    save_path: Optional[Path] = None,
    K: int = 3,
) -> None:
    """
    Plot training results for triadic alignment.
    
    Creates a 2x3 panel:
    - Top left: Task loss components
    - Top middle: Cycle and composition losses
    - Top right: Composition similarity over time
    - Bottom left: Accuracy per agent
    - Bottom middle: Mean accuracy over time
    - Bottom right: Basin usage entropy (non-triviality)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    epochs = np.arange(len(history))
    
    # Smooth for visualization
    def smooth(x, window=50):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # Top left: Loss components
    ax = axes[0, 0]
    ax.plot(epochs, [h['L_task'] for h in history], alpha=0.3, label='Task (raw)')
    ax.plot(smooth([h['L_task'] for h in history]), label='Task (smoothed)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Task Loss')
    ax.legend()
    ax.set_yscale('log')
    
    # Top middle: Cycle and composition losses
    ax = axes[0, 1]
    ax.plot(smooth([h['L_cyc'] for h in history]), label='Cycle')
    ax.plot(smooth([h['L_comp'] for h in history]), label='Composition')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Consistency Losses')
    ax.legend()
    
    # Top right: Composition similarity
    ax = axes[0, 2]
    comp_sim = [h['comp_sim'] for h in history]
    ax.plot(epochs, comp_sim, alpha=0.3, color='blue')
    ax.plot(smooth(comp_sim), color='blue', linewidth=2, label='cos(F_AC, F_BC∘F_AB)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.9)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Composition Similarity')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Bottom left: Per-agent accuracy
    ax = axes[1, 0]
    ax.plot(smooth([h['acc_A'] for h in history]), label='A')
    ax.plot(smooth([h['acc_B'] for h in history]), label='B')
    ax.plot(smooth([h['acc_C'] for h in history]), label='C')
    ax.axhline(1.0 / K, color='gray', linestyle='--', alpha=0.5, label=f'Chance (1/{K})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Basin Agreement Accuracy')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Bottom middle: Mean accuracy
    ax = axes[1, 1]
    mean_acc = [(h['acc_A'] + h['acc_B'] + h['acc_C']) / 3 for h in history]
    ax.plot(epochs, mean_acc, alpha=0.3)
    ax.plot(smooth(mean_acc), linewidth=2)
    ax.axhline(1.0 / K, color='red', linestyle='--', label=f'Chance (1/{K})')
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Overall Task Performance')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Bottom right: Basin usage (check for mode collapse)
    ax = axes[1, 2]
    basin_counts = {}
    window = 100
    for i in range(0, len(history), window):
        chunk = history[i:i+window]
        for k in range(K):
            if k not in basin_counts:
                basin_counts[k] = []
            count = sum(1 for h in chunk if h['target_basin'] == k)
            basin_counts[k].append(count / len(chunk))
    
    for k in range(K):
        ax.plot(np.arange(len(basin_counts[k])) * window, basin_counts[k], label=f'Basin {k}')
    ax.axhline(1.0 / K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Proportion')
    ax.set_title('Basin Usage (should be uniform)')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def run_triadic_alignment_test(
    output_dir: Path,
    n_actor_epochs: int = 500,
    n_functor_epochs: int = 5000,
    K: int = 3,
    z_dim: int = 2,
) -> Dict:
    """
    Run the full triadic alignment experiment.
    
    Args:
        output_dir: Directory to save results
        n_actor_epochs: Epochs for training each actor
        n_functor_epochs: Epochs for training functors
        K: Number of basins/modes
        z_dim: Latent dimension
    
    Returns:
        Results dict
    """
    from v2.actors.actor import Actor
    from .world_variants import create_world_A, create_world_B, create_world_C, CMGWorldVariant
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("============================================================")
    print("TRIADIC COORDINATION-INDUCED GAUGE ALIGNMENT")
    print("============================================================")
    print(f"K={K} basins, z_dim={z_dim}")
    print(f"Actor epochs: {n_actor_epochs}, Functor epochs: {n_functor_epochs}")
    print()
    
    # Phase 0: Create environments
    print("Phase 0: Creating environments...")
    config_A = create_world_A(K=K)
    config_B = create_world_B(K=K)
    config_C = create_world_C(K=K)
    
    env_A = CMGWorldVariant(config_A)
    env_B = CMGWorldVariant(config_B)
    env_C = CMGWorldVariant(config_C)
    
    # Phase 0: Train actors independently
    print("Phase 0: Training actors...")
    
    def train_actor(env, name):
        print(f"  Training actor {name}...")
        actor = Actor(
            obs_dim=env.obs_dim,
            pred_dim=env.state_dim,
            T=env.T,
            z_dim=z_dim,
            n_knots=5,
        ).to(device)
        
        optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        
        for epoch in range(n_actor_epochs):
            obs = env.reset()
            traj = [obs]
            done = False
            while not done:
                action = np.random.randn(env.state_dim).astype(np.float32) * 0.3
                obs, _, done, _ = env.step(action)
                traj.append(obs)
            
            traj_array = np.array(traj)
            traj_delta = torch.tensor(
                traj_array[1:, :env.state_dim] - traj_array[0, :env.state_dim],
                dtype=torch.float32, device=device
            )
            s0 = torch.tensor(traj[0], dtype=torch.float32, device=device)
            
            actor.train_step_wta(s0, traj_delta, optimizer)
            
            if (epoch + 1) % max(1, n_actor_epochs // 5) == 0:
                print(f"    Epoch {epoch+1}/{n_actor_epochs}")
        
        return actor
    
    actor_A = train_actor(env_A, "A")
    actor_B = train_actor(env_B, "B")
    actor_C = train_actor(env_C, "C")
    
    # Phase 1 & 2: Train functors with triadic coordination
    print()
    print("Phase 1-2: Training functors with triadic coordination...")
    
    config = TriadicConfig(
        z_dim=z_dim,
        K=K,
        tau=0.05,
        lambda_cyc=1.0,
        lambda_comp=1.0,
        lr=1e-3,
    )
    
    trainer = TriadicFunctorTrainer(
        actor_A, actor_B, actor_C,
        env_A, env_B, env_C,
        config, device,
    )
    
    history = trainer.train(n_epochs=n_functor_epochs, print_every=500)
    
    # Evaluate final results
    print()
    print("Evaluating final results...")
    
    # Final composition similarity
    z_test = sample_z_sphere(500, z_dim, device)
    with torch.no_grad():
        functors = trainer.get_functors()
        z_AC = functors['F_AC'](z_test)
        z_ABC = functors['F_BC'](functors['F_AB'](z_test))
        final_comp_sim = (F.normalize(z_AC, dim=-1) * F.normalize(z_ABC, dim=-1)).sum(dim=-1).mean().item()
    
    # Final accuracies (average over last 100 epochs)
    final_acc_A = np.mean([h['acc_A'] for h in history[-100:]])
    final_acc_B = np.mean([h['acc_B'] for h in history[-100:]])
    final_acc_C = np.mean([h['acc_C'] for h in history[-100:]])
    
    results = {
        'final_comp_sim': final_comp_sim,
        'final_acc_A': final_acc_A,
        'final_acc_B': final_acc_B,
        'final_acc_C': final_acc_C,
        'final_mean_acc': (final_acc_A + final_acc_B + final_acc_C) / 3,
        'gauge_fixed': final_comp_sim > 0.7 and final_acc_B > 1.0/K + 0.1,
        'K': K,
        'z_dim': z_dim,
        'n_actor_epochs': n_actor_epochs,
        'n_functor_epochs': n_functor_epochs,
    }
    
    print()
    print("============================================================")
    print("RESULTS")
    print("============================================================")
    print(f"Composition similarity: {final_comp_sim:.3f}")
    print(f"Task accuracy: A={final_acc_A:.3f}, B={final_acc_B:.3f}, C={final_acc_C:.3f}")
    print(f"Gauge fixed by interaction: {results['gauge_fixed']}")
    
    # Plot results
    plot_path = output_dir / "triadic_alignment.png"
    plot_triadic_results(history, plot_path, K=K)
    
    return results


# Standalone entry point
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-epochs", type=int, default=500)
    parser.add_argument("--functor-epochs", type=int, default=5000)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--z-dim", type=int, default=2)
    args = parser.parse_args()
    
    output_dir = Path("artifacts/functor_tests") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_triadic_alignment_test(
        output_dir,
        n_actor_epochs=args.actor_epochs,
        n_functor_epochs=args.functor_epochs,
        K=args.K,
        z_dim=args.z_dim,
    )
