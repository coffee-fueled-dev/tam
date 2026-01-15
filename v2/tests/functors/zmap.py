"""
ZMap: Structure-preserving commitment mapping between actors.

A functor F: Z_A -> Z_B that preserves viability structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import numpy as np


class ZMap(nn.Module):
    """
    Learnable structure-preserving map between commitment spaces.
    
    Maps z_A (from actor A) to z_B (for actor B) such that
    viability structure (energy basins) is preserved.
    """
    
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.z_dim = z_dim
        
        layers = []
        in_dim = z_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, z_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map z from source to target space.
        
        Args:
            z: (N, z_dim) or (z_dim,) on unit sphere
        
        Returns:
            Mapped z on unit sphere
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)
        
        z_out = self.net(z)
        z_out = F.normalize(z_out, p=2, dim=-1)
        
        if squeeze:
            z_out = z_out.squeeze(0)
        
        return z_out


def rank_preserving_loss(
    z_A: torch.Tensor,
    energy_A: torch.Tensor,
    z_mapped: torch.Tensor,
    energy_B_of_mapped: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Loss that enforces energy rank preservation under mapping.
    
    If E_A(z_i) < E_A(z_j), then E_B(F(z_i)) should be < E_B(F(z_j)).
    
    Args:
        z_A: (N, z_dim) samples from A
        energy_A: (N,) energies in A
        z_mapped: (N, z_dim) mapped samples
        energy_B_of_mapped: (N,) energies of mapped samples in B
        margin: Ranking margin
    
    Returns:
        Scalar loss
    """
    N = z_A.shape[0]
    
    # Sample pairs
    n_pairs = min(N * (N - 1) // 2, 500)
    
    # Random pairs
    idx_i = torch.randint(0, N, (n_pairs,), device=z_A.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=z_A.device)
    
    # Filter same indices
    valid = idx_i != idx_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    
    # Energy differences
    diff_A = energy_A[idx_i] - energy_A[idx_j]  # Positive if i worse than j
    diff_B = energy_B_of_mapped[idx_i] - energy_B_of_mapped[idx_j]
    
    # Sign should be preserved
    sign_A = torch.sign(diff_A)
    
    # Hinge loss: penalize if sign is violated
    # If sign_A * diff_B < margin, penalize
    loss = torch.relu(margin - sign_A * diff_B).mean()
    
    return loss


def basin_preservation_loss(
    labels_A: torch.Tensor,
    z_mapped: torch.Tensor,
    centers_B: torch.Tensor,
    perm_A_to_B: torch.Tensor,
) -> torch.Tensor:
    """
    Loss that encourages mapped z to land in corresponding basin.
    
    Args:
        labels_A: (N,) basin labels in A
        z_mapped: (N, z_dim) mapped samples
        centers_B: (K_B, z_dim) basin centers in B
        perm_A_to_B: (K_A,) permutation from A basins to B basins
    
    Returns:
        Scalar loss
    """
    N = z_mapped.shape[0]
    
    # Target basin in B for each sample
    target_basin = perm_A_to_B[labels_A]
    
    # Filter invalid mappings
    valid = target_basin >= 0
    if not valid.any():
        return torch.tensor(0.0, device=z_mapped.device)
    
    z_valid = z_mapped[valid]
    target_valid = target_basin[valid]
    
    # Get target centers
    target_centers = centers_B[target_valid]
    
    # Cosine similarity loss (maximize similarity to target)
    z_norm = F.normalize(z_valid, p=2, dim=-1)
    c_norm = F.normalize(target_centers, p=2, dim=-1)
    sim = (z_norm * c_norm).sum(dim=-1)
    
    loss = (1.0 - sim).mean()
    
    return loss


def cycle_consistency_loss(
    z: torch.Tensor,
    z_cycle: torch.Tensor,
) -> torch.Tensor:
    """
    Loss for G(F(z)) ≈ z (cycle consistency).
    
    Args:
        z: (N, z_dim) original samples
        z_cycle: (N, z_dim) samples after F then G
    
    Returns:
        Scalar loss
    """
    # Cosine distance
    z_norm = F.normalize(z, p=2, dim=-1)
    z_cycle_norm = F.normalize(z_cycle, p=2, dim=-1)
    
    sim = (z_norm * z_cycle_norm).sum(dim=-1)
    loss = (1.0 - sim).mean()
    
    return loss


class FunctorTrainer:
    """
    Trainer for learning structure-preserving maps between commitment spaces.
    """
    
    def __init__(
        self,
        actor_A,
        actor_B,
        z_dim: int,
        device: torch.device,
        lr: float = 1e-3,
        cycle_weight: float = 0.5,
        rank_weight: float = 1.0,
        basin_weight: float = 0.5,
    ):
        self.actor_A = actor_A
        self.actor_B = actor_B
        self.z_dim = z_dim
        self.device = device
        
        # Forward and backward maps
        self.F = ZMap(z_dim).to(device)  # A -> B
        self.G = ZMap(z_dim).to(device)  # B -> A
        
        self.optimizer = optim.Adam(
            list(self.F.parameters()) + list(self.G.parameters()),
            lr=lr
        )
        
        self.cycle_weight = cycle_weight
        self.rank_weight = rank_weight
        self.basin_weight = basin_weight
        
        self.history = []
    
    def train_step(
        self,
        s0_A: torch.Tensor,
        s0_B: torch.Tensor,
        n_samples: int = 100,
        compute_energy_fn=None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            s0_A: Initial observation for actor A
            s0_B: Initial observation for actor B (paired situation)
            n_samples: Number of z samples
            compute_energy_fn: Function to compute energy
        
        Returns:
            Dict with loss components
        """
        from .utils_energy import sample_z_sphere, compute_energy
        
        if compute_energy_fn is None:
            compute_energy_fn = compute_energy
        
        # Sample z in both spaces
        z_A = sample_z_sphere(n_samples, self.z_dim, self.device)
        z_B = sample_z_sphere(n_samples, self.z_dim, self.device)
        
        # Compute energies
        with torch.no_grad():
            energy_A = compute_energy_fn(self.actor_A, s0_A, z_A)
            energy_B = compute_energy_fn(self.actor_B, s0_B, z_B)
        
        # Map forward and backward
        z_A_to_B = self.F(z_A)
        z_B_to_A = self.G(z_B)
        
        # Cycle: A -> B -> A and B -> A -> B
        z_A_cycle = self.G(z_A_to_B)
        z_B_cycle = self.F(z_B_to_A)
        
        # Compute energies of mapped points
        with torch.no_grad():
            energy_A_to_B = compute_energy_fn(self.actor_B, s0_B, z_A_to_B)
            energy_B_to_A = compute_energy_fn(self.actor_A, s0_A, z_B_to_A)
        
        # Losses
        # Rank preservation: A -> B
        loss_rank_AB = rank_preserving_loss(z_A, energy_A, z_A_to_B, energy_A_to_B)
        # Rank preservation: B -> A
        loss_rank_BA = rank_preserving_loss(z_B, energy_B, z_B_to_A, energy_B_to_A)
        
        # Cycle consistency
        loss_cycle_A = cycle_consistency_loss(z_A, z_A_cycle)
        loss_cycle_B = cycle_consistency_loss(z_B, z_B_cycle)
        
        # Total loss
        loss = (
            self.rank_weight * (loss_rank_AB + loss_rank_BA) +
            self.cycle_weight * (loss_cycle_A + loss_cycle_B)
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        metrics = {
            'loss': loss.item(),
            'rank_AB': loss_rank_AB.item(),
            'rank_BA': loss_rank_BA.item(),
            'cycle_A': loss_cycle_A.item(),
            'cycle_B': loss_cycle_B.item(),
        }
        self.history.append(metrics)
        
        return metrics
    
    def train(
        self,
        env_A,
        env_B,
        n_epochs: int = 500,
        n_samples_per_epoch: int = 100,
        print_every: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Train the functor maps.
        
        Args:
            env_A: Environment for actor A
            env_B: Environment for actor B
            n_epochs: Number of training epochs
            n_samples_per_epoch: Samples per epoch
            print_every: Print frequency
        
        Returns:
            Training history
        """
        from .utils_energy import compute_energy
        
        for epoch in range(n_epochs):
            # Get paired situations
            obs_A = env_A.reset()
            obs_B = env_B.reset()
            
            s0_A = torch.tensor(obs_A, dtype=torch.float32, device=self.device)
            s0_B = torch.tensor(obs_B, dtype=torch.float32, device=self.device)
            
            metrics = self.train_step(s0_A, s0_B, n_samples_per_epoch, compute_energy)
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"loss={metrics['loss']:.4f}, "
                      f"rank_AB={metrics['rank_AB']:.4f}, "
                      f"cycle_A={metrics['cycle_A']:.4f}")
        
        return self.history


class CompositionChecker:
    """
    Check functor composition law: F_BC(F_AB(z)) ≈ F_AC(z)
    """
    
    def __init__(
        self,
        F_AB: ZMap,
        F_BC: ZMap,
        F_AC: ZMap,
        device: torch.device,
    ):
        self.F_AB = F_AB
        self.F_BC = F_BC
        self.F_AC = F_AC
        self.device = device
    
    def check(
        self,
        n_samples: int = 500,
        z_dim: int = 2,
    ) -> Dict[str, float]:
        """
        Check composition law.
        
        Returns:
            Dict with composition error metrics
        """
        from .utils_energy import sample_z_sphere
        
        z_A = sample_z_sphere(n_samples, z_dim, self.device)
        
        # Direct: A -> C
        z_AC = self.F_AC(z_A)
        
        # Composed: A -> B -> C
        z_AB = self.F_AB(z_A)
        z_ABC = self.F_BC(z_AB)
        
        # Compare
        z_AC_norm = F.normalize(z_AC, p=2, dim=-1)
        z_ABC_norm = F.normalize(z_ABC, p=2, dim=-1)
        
        # Cosine similarity
        sim = (z_AC_norm * z_ABC_norm).sum(dim=-1)
        
        # Angular error
        angles = torch.acos(sim.clamp(-1 + 1e-6, 1 - 1e-6))
        
        return {
            'mean_cosine_sim': sim.mean().item(),
            'std_cosine_sim': sim.std().item(),
            'mean_angular_error_deg': (angles * 180 / np.pi).mean().item(),
            'max_angular_error_deg': (angles * 180 / np.pi).max().item(),
            'composition_holds': sim.mean().item() > 0.9,  # Threshold
        }
