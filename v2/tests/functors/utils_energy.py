"""
Energy landscape utilities for functor tests.

Provides functions to sample the commitment space and compute
energy (viability) scores for different z values.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


def sample_z_sphere(n_samples: int, z_dim: int, device: torch.device) -> torch.Tensor:
    """
    Sample z uniformly on the unit hypersphere.
    
    Args:
        n_samples: Number of samples
        z_dim: Dimension of latent space
        device: Torch device
    
    Returns:
        (n_samples, z_dim) tensor of L2-normalized vectors
    """
    z = torch.randn(n_samples, z_dim, device=device)
    return F.normalize(z, p=2, dim=-1)


def sample_z_angles(n_samples: int, device: torch.device) -> torch.Tensor:
    """
    Sample z uniformly by angle for z_dim=2 (dense angles on unit circle).
    
    Args:
        n_samples: Number of samples (evenly spaced angles)
        device: Torch device
    
    Returns:
        (n_samples, 2) tensor on unit circle
    """
    angles = torch.linspace(0, 2 * np.pi, n_samples + 1, device=device)[:-1]
    z = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return z


@torch.no_grad()
def compute_energy(
    actor,
    s0: torch.Tensor,
    z: torch.Tensor,
    alpha_vol: float = 0.1,
    alpha_leak: float = 10.0,
    alpha_progress: float = 3.0,
    alpha_start: float = 100.0,
) -> torch.Tensor:
    """
    Compute CEM-style energy for commitment z given situation s0.
    
    Lower energy = better commitment (more viable).
    
    Args:
        actor: Actor with get_tube method
        s0: Initial observation (obs_dim,)
        z: Commitment(s) to evaluate (N, z_dim) or (z_dim,)
        alpha_*: Weighting coefficients
    
    Returns:
        (N,) tensor of energy values (lower = better)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    N = z.shape[0]
    
    # Get tubes for all z
    mu, sigma = actor.get_tube(s0.unsqueeze(0), z)  # (N, T, d)
    
    # Extract position dimensions
    pred_dim = actor.pred_dim
    s0_pos = s0[:pred_dim]
    
    # Start penalty: tube must start at s0
    start_error = (mu[:, 0, :] - s0_pos).pow(2).sum(dim=-1)
    start_penalty = alpha_start * start_error
    
    # Volume: mean sigma (agency = inverse of volume)
    volume = sigma.mean(dim=(1, 2))
    volume_cost = alpha_vol * volume
    
    # Progress reward: displacement from start to end
    displacement = (mu[:, -1, :] - mu[:, 0, :]).pow(2).sum(dim=-1).sqrt()
    progress_reward = alpha_progress * torch.log(displacement + 1e-6)
    
    # Sigma consistency: penalize high variance in sigma
    sigma_consistency = sigma.std(dim=(1, 2))
    
    # Energy: lower is better
    # Note: we negate progress_reward because it's a reward, not a cost
    energy = start_penalty + volume_cost - progress_reward + sigma_consistency
    
    return energy


@torch.no_grad()
def compute_energy_landscape(
    actor,
    s0: torch.Tensor,
    n_samples: int = 500,
    z_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute full energy landscape for a situation.
    
    Args:
        actor: Actor with get_tube method
        s0: Initial observation
        n_samples: Number of z samples
        z_dim: Latent dimension (default: actor.z_dim)
    
    Returns:
        Dict with 'z', 'energy', 'angles' (if z_dim=2)
    """
    device = s0.device
    z_dim = z_dim or actor.z_dim
    
    if z_dim == 2:
        z = sample_z_angles(n_samples, device)
        angles = torch.atan2(z[:, 1], z[:, 0])
    else:
        z = sample_z_sphere(n_samples, z_dim, device)
        angles = None
    
    energy = compute_energy(actor, s0, z)
    
    result = {
        'z': z,
        'energy': energy,
    }
    if angles is not None:
        result['angles'] = angles
    
    return result


@torch.no_grad()
def find_local_minima(
    energy: torch.Tensor,
    angles: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    window_size: int = 11,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find local minima in the energy landscape.
    
    For z_dim=2, uses angle-based detection.
    For higher dimensions, uses k-nearest neighbors.
    
    Args:
        energy: (N,) energy values
        angles: Optional (N,) angles (for z_dim=2)
        z: Optional (N, z_dim) z values
        window_size: Window for local minima detection
    
    Returns:
        (minima_indices, minima_energies)
    """
    N = energy.shape[0]
    
    if angles is not None:
        # Sort by angle and find local minima
        sorted_idx = torch.argsort(angles)
        sorted_energy = energy[sorted_idx]
        
        # Circular local minima detection
        minima = []
        half_window = window_size // 2
        for i in range(N):
            is_min = True
            center_e = sorted_energy[i]
            for j in range(-half_window, half_window + 1):
                if j == 0:
                    continue
                neighbor_idx = (i + j) % N
                if sorted_energy[neighbor_idx] < center_e:
                    is_min = False
                    break
            if is_min:
                minima.append(sorted_idx[i].item())
        
        minima_indices = torch.tensor(minima, device=energy.device)
        minima_energies = energy[minima_indices]
    else:
        # For higher dimensions, use top-k lowest energy
        # More sophisticated: cluster and find cluster centers
        k = min(10, N // 10)
        minima_indices = torch.topk(energy, k, largest=False).indices
        minima_energies = energy[minima_indices]
    
    return minima_indices, minima_energies


@torch.no_grad()
def get_basin_centers(
    actor,
    s0: torch.Tensor,
    n_samples: int = 500,
    n_basins_max: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Find basin centers in the energy landscape.
    
    Args:
        actor: Actor with get_tube method
        s0: Initial observation
        n_samples: Number of z samples
        n_basins_max: Maximum number of basins to detect
    
    Returns:
        Dict with 'centers', 'energies', 'n_basins'
    """
    landscape = compute_energy_landscape(actor, s0, n_samples)
    z = landscape['z']
    energy = landscape['energy']
    angles = landscape.get('angles')
    
    minima_idx, minima_e = find_local_minima(energy, angles, z)
    
    # Keep only the strongest minima (lowest energy)
    if len(minima_idx) > n_basins_max:
        top_k = torch.topk(minima_e, n_basins_max, largest=False)
        minima_idx = minima_idx[top_k.indices]
        minima_e = top_k.values
    
    return {
        'centers': z[minima_idx],
        'energies': minima_e,
        'n_basins': len(minima_idx),
        'all_z': z,
        'all_energy': energy,
    }
