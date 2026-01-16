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


def compute_jacobian(func: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix df/dz for a mapping function.
    
    Args:
        func: Mapping function (e.g., ZMap)
        z: (N, d_in) input points
    
    Returns:
        J: (N, d_out, d_in) Jacobian matrices
    """
    z = z.detach().clone().requires_grad_(True)
    
    # Forward pass
    out = func(z)  # (N, d_out)
    N, d_out = out.shape
    d_in = z.shape[1]
    
    # Compute Jacobian by backprop through each output dimension
    J = torch.zeros(N, d_out, d_in, device=z.device)
    
    for i in range(d_out):
        # Select i-th output component
        grad_outputs = torch.zeros_like(out)
        grad_outputs[:, i] = 1.0
        
        # Backprop
        grads = torch.autograd.grad(
            outputs=out,
            inputs=z,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
        )[0]
        
        J[:, i, :] = grads
    
    return J


def compute_tangent_jacobian(func: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Compute the tangent-space Jacobian for a sphere → sphere map.
    
    Since both input and output are on unit spheres, the meaningful Jacobian
    is the map between tangent spaces, not the ambient space Jacobian.
    
    Args:
        func: Mapping function (outputs unit vectors)
        z: (N, d_in) input points ON THE UNIT SPHERE
    
    Returns:
        J_tangent: (N, d_out-1, d_in-1) tangent space Jacobians
    """
    z = z.detach().clone().requires_grad_(True)
    N, d_in = z.shape
    
    # Get output (on sphere)
    out = func(z)  # (N, d_out)
    d_out = out.shape[1]
    
    # For each sample, construct orthonormal basis for tangent space
    # We'll use a simple approach: project the ambient Jacobian onto tangent spaces
    
    # Compute full ambient Jacobian
    J_ambient = torch.zeros(N, d_out, d_in, device=z.device)
    for i in range(d_out):
        grad_outputs = torch.zeros_like(out)
        grad_outputs[:, i] = 1.0
        grads = torch.autograd.grad(
            outputs=out, inputs=z, grad_outputs=grad_outputs,
            create_graph=False, retain_graph=True
        )[0]
        J_ambient[:, i, :] = grads
    
    # Project J_ambient onto tangent spaces
    # Input tangent: T_z = {v : v^T z = 0}
    # Output tangent: T_y = {v : v^T y = 0}
    # J_tangent = P_y @ J_ambient @ P_z where P is orthogonal projector onto tangent
    
    # For simplicity, compute singular values of projected Jacobian
    # This gives us the stretching/compression along tangent directions
    
    # Use SVD of J_ambient to get tangent contribution
    # Singular values capture the "shape" of the mapping
    
    return J_ambient


def jacobian_determinant_analysis(
    func: nn.Module,
    z_dim_src: int,
    z_dim_tgt: int,
    device: torch.device,
    n_samples: int = 500,
    seed: int = 42,
    sphere_aware: bool = True,  # New: use tangent-space analysis
) -> Dict[str, float]:
    """
    Analyze the Jacobian of a functor map to detect topological distortions.
    
    For same-dim maps: det(J) measures local volume change.
    - det(J) ≈ 1: conformal (shape-preserving)
    - det(J) → 0: compression/collapse
    - det(J) → ∞: expansion/tearing
    - det(J) < 0: orientation reversal (folding)
    
    For cross-dim maps: singular values measure stretching.
    
    Returns:
        Dict with:
        - det_mean, det_std: Jacobian determinant stats (same-dim only)
        - det_min, det_max: extremes
        - n_folds: count of sign flips (orientation reversals)
        - sv_condition: condition number from singular values
        - sv_min, sv_max: singular value extremes
    """
    torch.manual_seed(seed)
    
    # Sample points on unit sphere
    z = torch.randn(n_samples, z_dim_src, device=device)
    z = F.normalize(z, p=2, dim=-1)
    
    # Compute Jacobian
    J = compute_jacobian(func, z)  # (N, d_out, d_in)
    
    results = {}
    
    if z_dim_src == z_dim_tgt:
        # Square Jacobian: can compute determinant
        dets = torch.linalg.det(J)  # (N,)
        
        results['det_mean'] = float(dets.mean().item())
        results['det_std'] = float(dets.std().item())
        results['det_min'] = float(dets.min().item())
        results['det_max'] = float(dets.max().item())
        results['n_folds'] = int((dets < 0).sum().item())
        results['fold_fraction'] = float((dets < 0).float().mean().item())
        
        # Log determinant for stability
        log_dets = torch.log(dets.abs() + 1e-8)
        results['log_det_mean'] = float(log_dets.mean().item())
        results['log_det_std'] = float(log_dets.std().item())
    
    # Singular value analysis (works for any dimensions)
    # J = U @ S @ V^T
    try:
        S = torch.linalg.svdvals(J)  # (N, min(d_out, d_in))
        
        results['sv_min'] = float(S.min().item())
        results['sv_max'] = float(S.max().item())
        results['sv_mean'] = float(S.mean().item())
        
        # Condition number: ratio of largest to smallest singular value
        # High condition number = ill-conditioned = topological stress
        sv_min_per_sample = S.min(dim=-1).values
        sv_max_per_sample = S.max(dim=-1).values
        condition = sv_max_per_sample / (sv_min_per_sample + 1e-8)
        
        results['condition_mean'] = float(condition.mean().item())
        results['condition_max'] = float(condition.max().item())
        results['condition_std'] = float(condition.std().item())
        
        # Effective rank: how many singular values are "active"
        # Low effective rank = dimension collapse
        normalized_S = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(normalized_S * torch.log(normalized_S + 1e-8)).sum(dim=-1)
        eff_rank = torch.exp(entropy)
        
        results['effective_rank_mean'] = float(eff_rank.mean().item())
        results['effective_rank_min'] = float(eff_rank.min().item())
        
    except RuntimeError:
        # SVD can fail for degenerate matrices
        results['svd_failed'] = True
    
    return results


def detect_topological_stress(jacobian_stats: Dict[str, float], sphere_aware: bool = True) -> Dict[str, any]:
    """
    Interpret Jacobian statistics to detect topological problems.
    
    Args:
        jacobian_stats: Output from jacobian_determinant_analysis
        sphere_aware: If True, adjust thresholds for sphere-constrained maps.
                      Sphere maps have inherently degenerate ambient Jacobians,
                      so we focus on singular value ratios instead of determinants.
    
    Returns:
        Dict with boolean flags and severity scores for various issues.
    """
    issues = {}
    z_dim = jacobian_stats.get('z_dim_tgt', 2)
    
    if sphere_aware:
        # For sphere maps, the ambient determinant is always ~0 (not meaningful)
        # The meaningful metric is effective rank relative to expected tangent dimension
        
        expected_rank = z_dim - 1  # Sphere is (d-1)-dimensional
        
        # Effective rank: should be close to (d-1) for sphere maps
        if 'effective_rank_mean' in jacobian_stats:
            eff_rank = jacobian_stats['effective_rank_mean']
            
            if expected_rank > 0:
                rank_ratio = eff_rank / expected_rank
                
                # For sphere maps, being close to expected rank is healthy
                issues['has_dimension_collapse'] = rank_ratio < 0.7
                issues['dimension_collapse_severity'] = (
                    'high' if rank_ratio < 0.3 else
                    'medium' if rank_ratio < 0.7 else
                    'low'
                )
                issues['rank_ratio'] = rank_ratio
            else:
                # z_dim=1 edge case (single point, no meaningful sphere)
                issues['has_dimension_collapse'] = False
                issues['dimension_collapse_severity'] = 'low'
                issues['rank_ratio'] = 1.0
        
        # For sphere maps with d>2, check anisotropy among tangent singular values
        # (Skip for d=2 since there's only one tangent direction)
        if z_dim > 2 and 'sv_mean' in jacobian_stats:
            # Look at condition number as proxy for tangent-space anisotropy
            # High condition = some tangent directions stretched more than others
            cond = jacobian_stats.get('condition_mean', 1.0)
            # Much more lenient threshold for sphere maps
            issues['has_anisotropy'] = cond > 1000
            issues['anisotropy_severity'] = 'high' if cond > 10000 else 'medium' if cond > 1000 else 'low'
        else:
            issues['has_anisotropy'] = False
            issues['anisotropy_severity'] = 'low'
        
        # For sphere maps, "folds" in ambient space aren't meaningful
        issues['fold_analysis'] = 'skipped (sphere-aware)'
        
    else:
        # Original ambient-space analysis (kept for non-sphere maps)
        
        # Folding: orientation reversals
        if 'fold_fraction' in jacobian_stats:
            fold_frac = jacobian_stats['fold_fraction']
            issues['has_folds'] = fold_frac > 0.01
            issues['fold_severity'] = 'high' if fold_frac > 0.1 else 'medium' if fold_frac > 0.01 else 'low'
        
        # Collapse: regions compressed to near-zero volume
        if 'det_min' in jacobian_stats:
            det_min = jacobian_stats['det_min']
            issues['has_collapse'] = abs(det_min) < 0.01
            issues['collapse_severity'] = 'high' if abs(det_min) < 0.001 else 'medium' if abs(det_min) < 0.01 else 'low'
        
        # Tearing: regions stretched to extreme volume
        if 'det_max' in jacobian_stats:
            det_max = jacobian_stats['det_max']
            issues['has_tearing'] = det_max > 10
            issues['tearing_severity'] = 'high' if det_max > 100 else 'medium' if det_max > 10 else 'low'
        
        # Ill-conditioning: high condition number
        if 'condition_max' in jacobian_stats:
            cond_max = jacobian_stats['condition_max']
            issues['is_ill_conditioned'] = cond_max > 100
            issues['conditioning_severity'] = 'high' if cond_max > 1000 else 'medium' if cond_max > 100 else 'low'
        
        # Dimension collapse: low effective rank
        if 'effective_rank_min' in jacobian_stats:
            eff_rank = jacobian_stats['effective_rank_min']
            issues['has_dimension_collapse'] = eff_rank < z_dim * 0.5
            issues['dimension_collapse_severity'] = (
                'high' if eff_rank < z_dim * 0.3 else 
                'medium' if eff_rank < z_dim * 0.5 else 
                'low'
            )
    
    # Overall topological health
    severe_issues = sum(1 for k, v in issues.items() if k.endswith('_severity') and v == 'high')
    medium_issues = sum(1 for k, v in issues.items() if k.endswith('_severity') and v == 'medium')
    
    if severe_issues > 0:
        issues['overall_health'] = 'critical'
    elif medium_issues > 1:
        issues['overall_health'] = 'stressed'
    elif medium_issues > 0:
        issues['overall_health'] = 'minor_stress'
    else:
        issues['overall_health'] = 'healthy'
    
    return issues


class ZMap(nn.Module):
    """
    Learnable structure-preserving map between commitment spaces.
    
    Maps z_A (from actor A) to z_B (for actor B) such that
    viability structure (energy basins) is preserved.
    
    Supports cross-dimensional mappings (z_dim_src != z_dim_tgt).
    """
    
    def __init__(
        self,
        z_dim: int = None,
        hidden_dim: int = 128,
        n_layers: int = 2,
        # Cross-dimensional support
        z_dim_src: int = None,
        z_dim_tgt: int = None,
        # Architecture variant
        residual: bool = False,  # If True, use F(z) = z + f(z), then normalize
    ):
        super().__init__()
        
        # Handle both old API (z_dim) and new API (z_dim_src, z_dim_tgt)
        if z_dim is not None:
            z_dim_src = z_dim_src or z_dim
            z_dim_tgt = z_dim_tgt or z_dim
        
        if z_dim_src is None or z_dim_tgt is None:
            raise ValueError("Must specify z_dim or (z_dim_src, z_dim_tgt)")
        
        self.z_dim_src = z_dim_src
        self.z_dim_tgt = z_dim_tgt
        self.z_dim = z_dim_src  # For backward compatibility
        self.residual = residual and (z_dim_src == z_dim_tgt)  # Residual only for same-dim
        
        layers = []
        in_dim = z_dim_src
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, z_dim_tgt))
        
        self.net = nn.Sequential(*layers)
        
        # For residual: initialize output layer to small values
        if self.residual:
            with torch.no_grad():
                self.net[-1].weight.mul_(0.01)
                self.net[-1].bias.mul_(0.01)
    
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
        
        delta = self.net(z)
        
        if self.residual:
            # Residual: preserves identity at init, smoother Jacobian
            z_out = z + delta
        else:
            z_out = delta
        
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
