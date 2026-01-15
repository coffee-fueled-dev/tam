"""
Basin clustering and matching utilities for functor tests.

Provides functions to cluster energy minima into basins
and match basins between different actors.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def cluster_by_energy(
    z: torch.Tensor,
    energy: torch.Tensor,
    n_clusters: int,
    top_fraction: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Cluster top-energy (lowest energy = best) points.
    
    Args:
        z: (N, z_dim) latent samples
        energy: (N,) energy values
        n_clusters: Number of clusters (basins)
        top_fraction: Fraction of lowest-energy points to use
    
    Returns:
        Dict with 'labels', 'centers', 'top_indices'
    """
    N = z.shape[0]
    top_k = max(n_clusters * 5, int(N * top_fraction))
    
    # Get top (lowest energy) points
    top_indices = torch.topk(energy, top_k, largest=False).indices
    top_z = z[top_indices].cpu().numpy()
    
    # Cluster
    if len(top_z) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(top_z)
        centers = torch.tensor(kmeans.cluster_centers_, device=z.device, dtype=z.dtype)
        centers = F.normalize(centers, p=2, dim=-1)  # Project back to sphere
    else:
        # Not enough points
        cluster_labels = np.arange(len(top_z)) % n_clusters
        centers = z[top_indices[:n_clusters]]
    
    # Assign all points to nearest center
    all_labels = assign_to_nearest(z, centers)
    
    return {
        'labels': all_labels,
        'centers': centers,
        'top_indices': top_indices,
        'top_labels': torch.tensor(cluster_labels, device=z.device),
    }


def assign_to_nearest(z: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Assign each z to the nearest center (by cosine similarity).
    
    Args:
        z: (N, z_dim) samples
        centers: (K, z_dim) cluster centers
    
    Returns:
        (N,) labels
    """
    # Cosine similarity
    z_norm = F.normalize(z, p=2, dim=-1)
    c_norm = F.normalize(centers, p=2, dim=-1)
    sim = z_norm @ c_norm.t()  # (N, K)
    return sim.argmax(dim=-1)


def match_basins_hungarian(
    centers_A: torch.Tensor,
    centers_B: torch.Tensor,
    energies_A: Optional[torch.Tensor] = None,
    energies_B: Optional[torch.Tensor] = None,
    use_energy_matching: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match basins between two actors using Hungarian algorithm.
    
    Matching can be based on:
    1. Energy similarity (lower energy basins match)
    2. Position similarity (if actors share observation structure)
    
    Args:
        centers_A: (K_A, z_dim) basin centers from actor A
        centers_B: (K_B, z_dim) basin centers from actor B
        energies_A: Optional (K_A,) basin energies
        energies_B: Optional (K_B,) basin energies
        use_energy_matching: If True, match by energy rank
    
    Returns:
        (perm_A_to_B, perm_B_to_A): permutation tensors
    """
    K_A, K_B = centers_A.shape[0], centers_B.shape[0]
    K = min(K_A, K_B)
    
    if use_energy_matching and energies_A is not None and energies_B is not None:
        # Match by energy rank
        rank_A = torch.argsort(torch.argsort(energies_A))
        rank_B = torch.argsort(torch.argsort(energies_B))
        
        # Cost matrix: difference in energy ranks
        cost = torch.zeros(K_A, K_B, device=centers_A.device)
        for i in range(K_A):
            for j in range(K_B):
                cost[i, j] = abs(rank_A[i] - rank_B[j]).float()
    else:
        # Match by position (cosine distance)
        centers_A_norm = F.normalize(centers_A, p=2, dim=-1)
        centers_B_norm = F.normalize(centers_B, p=2, dim=-1)
        sim = centers_A_norm @ centers_B_norm.t()  # (K_A, K_B)
        cost = 1.0 - sim  # Convert similarity to distance
    
    # Hungarian algorithm
    cost_np = cost.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    
    # Build permutation tensors
    perm_A_to_B = torch.full((K_A,), -1, dtype=torch.long, device=centers_A.device)
    perm_B_to_A = torch.full((K_B,), -1, dtype=torch.long, device=centers_A.device)
    
    for i, j in zip(row_ind, col_ind):
        if i < K_A and j < K_B:
            perm_A_to_B[i] = j
            perm_B_to_A[j] = i
    
    return perm_A_to_B, perm_B_to_A


def compute_basin_confusion_matrix(
    z_A: torch.Tensor,
    labels_A: torch.Tensor,
    z_mapped: torch.Tensor,
    centers_B: torch.Tensor,
    n_basins_A: int,
    n_basins_B: int,
) -> torch.Tensor:
    """
    Compute confusion matrix: rows=A basins, cols=B basins.
    
    Args:
        z_A: (N, z_dim) samples from A
        labels_A: (N,) basin labels in A
        z_mapped: (N, z_dim) samples mapped to B's space
        centers_B: (K_B, z_dim) basin centers in B
        n_basins_A: Number of basins in A
        n_basins_B: Number of basins in B
    
    Returns:
        (n_basins_A, n_basins_B) confusion matrix
    """
    labels_B = assign_to_nearest(z_mapped, centers_B)
    
    confusion = torch.zeros(n_basins_A, n_basins_B, device=z_A.device)
    for i in range(len(z_A)):
        a = labels_A[i].item()
        b = labels_B[i].item()
        if 0 <= a < n_basins_A and 0 <= b < n_basins_B:
            confusion[a, b] += 1
    
    return confusion


def is_permutation_matrix(matrix: torch.Tensor, threshold: float = 0.7) -> bool:
    """
    Check if a confusion matrix is close to a permutation matrix.
    
    A structure-preserving functor should produce a near-permutation.
    
    Args:
        matrix: (K, K) confusion matrix (normalized)
        threshold: Minimum fraction of mass on diagonal after optimal matching
    
    Returns:
        True if matrix is close to permutation
    """
    K = min(matrix.shape)
    
    # Normalize rows
    row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
    matrix_norm = matrix / row_sums
    
    # Hungarian matching
    cost = 1.0 - matrix_norm.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute matched fraction
    matched_mass = 0.0
    total_mass = 0.0
    for i, j in zip(row_ind, col_ind):
        if i < matrix.shape[0] and j < matrix.shape[1]:
            matched_mass += matrix[i, j].item()
            total_mass += matrix[i, :].sum().item()
    
    fraction = matched_mass / max(total_mass, 1e-8)
    return fraction >= threshold


def compute_basin_graph(
    z: torch.Tensor,
    energy: torch.Tensor,
    centers: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Build basin adjacency graph with energy barriers.
    
    Nodes = basins
    Edge weight = energy barrier between basins
    
    Args:
        z: (N, z_dim) samples
        energy: (N,) energies
        centers: (K, z_dim) basin centers
        labels: (N,) basin labels
    
    Returns:
        Dict with 'adjacency', 'barriers', 'distances'
    """
    K = centers.shape[0]
    device = z.device
    
    # Compute pairwise distances between centers
    centers_norm = F.normalize(centers, p=2, dim=-1)
    distances = 1.0 - (centers_norm @ centers_norm.t())  # Cosine distance
    
    # Estimate energy barriers between basins
    # For each pair of basins, find the minimum-energy path
    barriers = torch.full((K, K), float('inf'), device=device)
    
    for i in range(K):
        for j in range(i + 1, K):
            # Find points between basins i and j
            mask_i = (labels == i)
            mask_j = (labels == j)
            
            if mask_i.any() and mask_j.any():
                # Find maximum energy along the "ridge" between basins
                # Approximate: find points with mixed affinity
                z_i = z[mask_i]
                z_j = z[mask_j]
                
                # Sample interpolation points
                n_interp = 10
                alphas = torch.linspace(0, 1, n_interp, device=device)
                
                max_barrier = energy[mask_i].min()  # Start at basin minimum
                for alpha in alphas[1:-1]:
                    # Interpolated z
                    interp_z = alpha * centers[j] + (1 - alpha) * centers[i]
                    interp_z = F.normalize(interp_z, p=2, dim=-1)
                    
                    # Find nearest sample to interpolated point
                    dists = (z - interp_z).pow(2).sum(dim=-1)
                    nearest_idx = dists.argmin()
                    max_barrier = max(max_barrier, energy[nearest_idx])
                
                barriers[i, j] = max_barrier - min(energy[mask_i].min(), energy[mask_j].min())
                barriers[j, i] = barriers[i, j]
    
    # Adjacency (inverse of barrier)
    adjacency = 1.0 / (barriers + 1e-6)
    adjacency[torch.isinf(barriers)] = 0.0
    
    return {
        'adjacency': adjacency,
        'barriers': barriers,
        'distances': distances,
    }


def compare_basin_graphs(
    graph_A: Dict[str, torch.Tensor],
    graph_B: Dict[str, torch.Tensor],
    perm: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare basin graphs under a permutation.
    
    Args:
        graph_A: Basin graph from actor A
        graph_B: Basin graph from actor B
        perm: Permutation mapping A basins to B basins
    
    Returns:
        Dict with correlation metrics
    """
    barriers_A = graph_A['barriers']
    barriers_B = graph_B['barriers']
    
    K_A = barriers_A.shape[0]
    K_B = barriers_B.shape[0]
    K = min(K_A, K_B)
    
    # Apply permutation to A's barriers
    valid = (perm >= 0) & (perm < K_B)
    if not valid.any():
        return {'barrier_correlation': 0.0, 'distance_correlation': 0.0}
    
    # Extract matched barriers
    matched_A = []
    matched_B = []
    
    for i in range(K_A):
        if not valid[i]:
            continue
        for j in range(i + 1, K_A):
            if not valid[j]:
                continue
            
            bi = perm[i].item()
            bj = perm[j].item()
            
            if not torch.isinf(barriers_A[i, j]) and not torch.isinf(barriers_B[bi, bj]):
                matched_A.append(barriers_A[i, j].item())
                matched_B.append(barriers_B[bi, bj].item())
    
    if len(matched_A) < 2:
        return {'barrier_correlation': 0.0, 'distance_correlation': 0.0}
    
    # Compute rank correlation
    from scipy.stats import spearmanr
    
    barrier_corr, _ = spearmanr(matched_A, matched_B)
    
    return {
        'barrier_correlation': float(barrier_corr) if not np.isnan(barrier_corr) else 0.0,
        'n_matched_pairs': len(matched_A),
    }
