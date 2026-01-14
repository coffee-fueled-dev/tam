"""
Universal metrics for latent space and contract evaluation.
Label-free where possible, optional labels for supervised metrics.
"""

import numpy as np
from typing import Dict, Optional, List
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def compute_latent_metrics(
    z_array: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute latent space health metrics (label-free + optional supervised).
    
    Args:
        z_array: (N, z_dim) latent vectors (should be L2-normalized)
        labels: Optional (N,) labels for supervised metrics
    
    Returns:
        Dict of metrics
    """
    N, z_dim = z_array.shape
    
    # Always-available metrics (label-free)
    metrics = {}
    
    # 1. Pairwise cosine similarity (for unit vectors)
    z_norm = z_array / (np.linalg.norm(z_array, axis=1, keepdims=True) + 1e-8)
    cos_sim = z_norm @ z_norm.T
    # Remove diagonal
    mask = ~np.eye(N, dtype=bool)
    pairwise_cos = cos_sim[mask]
    
    metrics["mean_pairwise_cosine"] = float(np.mean(pairwise_cos))
    metrics["std_pairwise_cosine"] = float(np.std(pairwise_cos))
    metrics["min_pairwise_cosine"] = float(np.min(pairwise_cos))
    metrics["max_pairwise_cosine"] = float(np.max(pairwise_cos))
    
    # 2. Effective rank (from covariance eigenvalues)
    cov = np.cov(z_array.T)
    eigenvals = np.linalg.eigvalsh(cov)
    eigenvals = eigenvals[eigenvals > 1e-8]
    eigenvals_norm = eigenvals / (eigenvals.sum() + 1e-8)
    # Shannon entropy of eigenvalue distribution
    entropy = -np.sum(eigenvals_norm * np.log(eigenvals_norm + 1e-8))
    effective_rank = np.exp(entropy)
    metrics["effective_rank"] = float(effective_rank)
    metrics["condition_number"] = float(eigenvals.max() / (eigenvals.min() + 1e-8)) if len(eigenvals) > 1 else 1.0
    
    # 3. Variance statistics
    metrics["z_variance"] = float(z_array.var(axis=0).mean())
    metrics["z_std"] = float(np.sqrt(metrics["z_variance"]))
    
    # 4. Norm statistics (should be ~1.0 if L2-normalized)
    norms = np.linalg.norm(z_array, axis=1)
    metrics["mean_norm"] = float(norms.mean())
    metrics["std_norm"] = float(norms.std())
    
    # Supervised metrics (only if labels provided)
    if labels is not None and len(set(labels)) >= 2:
        labels_arr = np.array(labels)
        unique_labels = len(set(labels))
        
        if unique_labels >= 2 and N >= unique_labels:
            try:
                metrics["silhouette_score"] = float(silhouette_score(z_array, labels_arr))
            except:
                metrics["silhouette_score"] = None
            
            try:
                metrics["calinski_harabasz_index"] = float(calinski_harabasz_score(z_array, labels_arr))
            except:
                metrics["calinski_harabasz_index"] = None
            
            try:
                metrics["davies_bouldin_index"] = float(davies_bouldin_score(z_array, labels_arr))
            except:
                metrics["davies_bouldin_index"] = None
            
            # Custom separability ratio
            unique_labels_list = sorted(set(labels))
            mode_centers = []
            intra_distances = []
            
            for label in unique_labels_list:
                mask = labels_arr == label
                cluster_points = z_array[mask]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    mode_centers.append(center)
                    if len(cluster_points) > 1:
                        distances = np.linalg.norm(cluster_points - center, axis=1)
                        intra_distances.append(distances.mean())
                    else:
                        intra_distances.append(0.0)
            
            if len(mode_centers) >= 2:
                mode_centers = np.array(mode_centers)
                inter_distances = cdist(mode_centers, mode_centers, metric='euclidean')
                inter_distances = inter_distances[~np.eye(inter_distances.shape[0], dtype=bool)]
                avg_inter = inter_distances.mean()
                avg_intra = np.mean(intra_distances) if intra_distances else 0.0
                metrics["separability_ratio"] = float(avg_inter / (avg_intra + 1e-8))
                metrics["inter_cluster_distance"] = float(avg_inter)
                metrics["intra_cluster_distance"] = float(avg_intra)
    
    return metrics


def compute_contract_metrics(
    leakages: List[float],
    volumes: List[float],
    start_errors: List[float],
    direction_losses: Optional[List[float]] = None,
    endpoint_errors: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute contract fulfillment metrics.
    
    Args:
        leakages: List of leakage values
        volumes: List of volume values
        start_errors: List of start anchor errors
        direction_losses: Optional list of direction losses
        endpoint_errors: Optional list of endpoint errors
    
    Returns:
        Dict of aggregated metrics
    """
    metrics = {}
    
    leakages_arr = np.array(leakages)
    volumes_arr = np.array(volumes)
    start_errors_arr = np.array(start_errors)
    
    metrics["leak_mean"] = float(leakages_arr.mean())
    metrics["leak_std"] = float(leakages_arr.std())
    metrics["leak_max"] = float(leakages_arr.max())
    metrics["leak_p95"] = float(np.percentile(leakages_arr, 95))
    
    metrics["volume_mean"] = float(volumes_arr.mean())
    metrics["volume_std"] = float(volumes_arr.std())
    metrics["volume_min"] = float(volumes_arr.min())
    
    metrics["start_error_mean"] = float(start_errors_arr.mean())
    metrics["start_error_std"] = float(start_errors_arr.std())
    metrics["start_error_max"] = float(start_errors_arr.max())
    
    if direction_losses is not None:
        dir_arr = np.array(direction_losses)
        metrics["direction_loss_mean"] = float(dir_arr.mean())
        metrics["direction_loss_std"] = float(dir_arr.std())
    
    if endpoint_errors is not None:
        end_arr = np.array(endpoint_errors)
        metrics["endpoint_error_mean"] = float(end_arr.mean())
        metrics["endpoint_error_std"] = float(end_arr.std())
    
    return metrics
