"""
Universal geometry plotting (tubes, energy landscape, trajectory PCA).
Label-free, actor-only where possible.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import torch
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition import PCA


def plot_tube_overlay(
    mu: np.ndarray,
    sigma: np.ndarray,
    output_path: Path,
    actual_trajectory: Optional[np.ndarray] = None,
    dims_to_plot: Optional[List[int]] = None,
    title: Optional[str] = None,
):
    """
    Plot tube overlay with optional actual trajectory.
    
    Args:
        mu: (T, d) tube centerline
        sigma: (T, d) tube width
        actual_trajectory: Optional (T, d) actual trajectory
        output_path: Path to save plot
        dims_to_plot: Which dimensions to plot (default: first 3)
        title: Optional plot title
    """
    T, d = mu.shape
    if dims_to_plot is None:
        dims_to_plot = list(range(min(3, d)))
    
    n_dims = len(dims_to_plot)
    fig, axes = plt.subplots(1, n_dims, figsize=(5*n_dims, 5))
    if n_dims == 1:
        axes = [axes]
    
    for idx, dim in enumerate(dims_to_plot):
        ax = axes[idx]
        t = np.arange(T)
        
        # Plot tube
        ax.plot(t, mu[:, dim], 'b-', linewidth=2, label='Tube center μ')
        ax.fill_between(t, mu[:, dim] - sigma[:, dim], mu[:, dim] + sigma[:, dim],
                       color='blue', alpha=0.2, label='Tube ±σ')
        
        # Plot actual trajectory if provided
        if actual_trajectory is not None:
            ax.plot(t, actual_trajectory[:, dim], 'r--', linewidth=2, alpha=0.7, label='Actual')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'x[{dim}]')
        ax.set_title(f'Dimension {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if title is None:
        title = 'Tube Overlay'
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tube overlay to {output_path}")


def plot_multi_start_tubes(
    tubes: List[Tuple[np.ndarray, np.ndarray]],  # List of (mu, sigma) tuples
    output_path: Path,
    dims_to_plot: Optional[List[int]] = None,
    title: Optional[str] = None,
):
    """
    Plot multiple tubes from different z starts (for measuring implicit ports).
    
    Args:
        tubes: List of (mu, sigma) tuples, each (T, d)
        output_path: Path to save plot
        dims_to_plot: Which dimensions to plot
        title: Optional plot title
    """
    if len(tubes) == 0:
        return
    
    T, d = tubes[0][0].shape
    if dims_to_plot is None:
        dims_to_plot = list(range(min(3, d)))
    
    n_dims = len(dims_to_plot)
    fig, axes = plt.subplots(1, n_dims, figsize=(5*n_dims, 5))
    if n_dims == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(tubes)))
    
    for idx, dim in enumerate(dims_to_plot):
        ax = axes[idx]
        t = np.arange(T)
        
        for i, (mu, sigma) in enumerate(tubes):
            ax.plot(t, mu[:, dim], color=colors[i], linewidth=1.5, alpha=0.7, label=f'Tube {i}')
            ax.fill_between(t, mu[:, dim] - sigma[:, dim], mu[:, dim] + sigma[:, dim],
                           color=colors[i], alpha=0.1)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'x[{dim}]')
        ax.set_title(f'Dimension {dim}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    if title is None:
        title = f'Multi-Start Tubes (n={len(tubes)})'
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-start tubes to {output_path}")


def plot_energy_landscape(
    actor,
    s0: torch.Tensor,
    z_dim: int,
    pred_dim: int,
    output_path: Path,
    n_samples: int = 500,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Plot CEM energy landscape for fixed s0 (actor-only, universal).
    
    Args:
        actor: Actor with get_tube method
        s0: Initial observation (obs_dim,)
        z_dim: Latent dimension
        pred_dim: Prediction dimension
        output_path: Path to save plot
        n_samples: Number of z samples
        device: Torch device
    
    Returns:
        Dict with basin statistics
    """
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter1d
    
    if device is None:
        device = s0.device
    
    # Sample z on unit sphere
    if z_dim == 2:
        angles = np.linspace(0, 2*np.pi, n_samples)
        z_samples = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        z_samples = np.random.randn(n_samples, z_dim)
        z_samples = z_samples / (np.linalg.norm(z_samples, axis=1, keepdims=True) + 1e-8)
    
    z_tensor = torch.tensor(z_samples, dtype=torch.float32, device=device)
    
    # Compute energy
    volumes = []
    start_errors = []
    displacements = []
    
    with torch.no_grad():
        for i in range(0, n_samples, 50):
            batch_z = z_tensor[i:i+50]
            mu, sigma = actor.get_tube(s0, batch_z)
            
            vol = sigma.mean(dim=(1, 2)).cpu().numpy()
            volumes.extend(vol)
            
            s0_pos = s0[:pred_dim].cpu().numpy()
            start_err = np.linalg.norm(mu[:, 0, :].cpu().numpy() - s0_pos, axis=1)
            start_errors.extend(start_err)
            
            disp = np.linalg.norm(mu[:, -1, :].cpu().numpy() - mu[:, 0, :].cpu().numpy(), axis=1)
            displacements.extend(disp)
    
    volumes = np.array(volumes)
    start_errors = np.array(start_errors)
    displacements = np.array(displacements)
    
    # Energy: reward low volume, penalize bad start, reward progress
    energy = -volumes * 0.3 - start_errors * 10.0 + np.log(displacements + 1e-6) * 3.0
    
    # Find basins
    n_basins = None
    if z_dim == 2:
        energy_smooth = gaussian_filter1d(energy, sigma=5, mode='wrap')
        peaks = []
        for i in range(len(energy_smooth)):
            prev_i = (i - 1) % len(energy_smooth)
            next_i = (i + 1) % len(energy_smooth)
            if energy_smooth[i] > energy_smooth[prev_i] and energy_smooth[i] > energy_smooth[next_i]:
                peaks.append(i)
        n_basins = len(peaks)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(angles * 180 / np.pi, energy, 'b-', alpha=0.5, label='Raw')
        ax1.plot(angles * 180 / np.pi, energy_smooth, 'r-', linewidth=2, label='Smoothed')
        for p in peaks:
            ax1.axvline(angles[p] * 180 / np.pi, color='g', linestyle='--', alpha=0.5)
            ax1.scatter([angles[p] * 180 / np.pi], [energy_smooth[p]], color='g', s=100, zorder=5)
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Energy E(z)')
        ax1.set_title(f'CEM Energy vs Angle ({n_basins} basins)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        scatter = ax2.scatter(z_samples[:, 0], z_samples[:, 1], c=energy, cmap='viridis', s=20)
        for p in peaks:
            ax2.scatter([z_samples[p, 0]], [z_samples[p, 1]], color='red', s=200, marker='*',
                       edgecolors='white', linewidths=2, zorder=5, label='Basin' if p == peaks[0] else None)
        circle = plt.Circle((0, 0), 1, color='gray', linestyle='--', fill=False, alpha=0.5)
        ax2.add_artist(circle)
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-1.3, 1.3)
        ax2.set_aspect('equal')
        ax2.set_xlabel('z[0]')
        ax2.set_ylabel('z[1]')
        ax2.set_title('Energy Landscape on Unit Circle')
        plt.colorbar(scatter, ax=ax2, label='Energy')
        ax2.legend()
    else:
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_samples)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=energy, cmap='viridis', s=20)
        top_k = 5
        top_indices = np.argsort(energy)[-top_k:]
        ax.scatter(z_pca[top_indices, 0], z_pca[top_indices, 1], color='red', s=200, marker='*',
                  edgecolors='white', linewidths=2, zorder=5, label=f'Top {top_k} basins')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'CEM Energy Landscape (PCA, var={pca.explained_variance_ratio_.sum()*100:.1f}%)')
        plt.colorbar(scatter, ax=ax, label='Energy')
        ax.legend()
    
    plt.suptitle(f'CEM Energy Landscape (n={n_samples} samples)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved energy landscape to {output_path}")
    
    return {
        'n_basins': n_basins,
        'energy_range': float(energy.max() - energy.min()),
        'energy_std': float(energy.std()),
    }


def plot_trajectory_pca(
    spines: np.ndarray,  # (N, T*d) flattened tube centerlines
    output_path: Path,
    labels: Optional[np.ndarray] = None,
    color_values: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> dict:
    """
    Plot PCA of tube spines (label-free, optional coloring).
    
    Args:
        spines: (N, T*d) flattened tube centerlines
        output_path: Path to save plot
        labels: Optional (N,) labels for coloring
        color_values: Optional (N,) values for continuous coloring
        title: Optional plot title
    
    Returns:
        Dict with PCA statistics
    """
    pca = PCA(n_components=min(3, spines.shape[1]))
    spines_pca = pca.fit_transform(spines)
    
    variance_explained = pca.explained_variance_ratio_
    
    # Determine coloring
    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        color_array = [label_colors.get(l, 'gray') for l in labels]
        legend_labels = {f'Label {l}': label_colors[l] for l in unique_labels}
    elif color_values is not None:
        color_array = color_values
        legend_labels = None
    else:
        color_array = 'steelblue'
        legend_labels = None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PC1 vs PC2
    ax1 = axes[0]
    if labels is not None:
        for label in unique_labels:
            mask = np.array(labels) == label
            if np.any(mask):
                ax1.scatter(spines_pca[mask, 0], spines_pca[mask, 1],
                           c=[label_colors[label]], s=50, alpha=0.7, label=f'Label {label}',
                           edgecolors='black', linewidths=0.5)
    else:
        ax1.scatter(spines_pca[:, 0], spines_pca[:, 1], c=color_array, s=50, alpha=0.7,
                   edgecolors='black', linewidths=0.5)
    
    ax1.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax1.set_title('Trajectory Space PCA (PC1 vs PC2)')
    if legend_labels:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Centroids with ellipses
    ax2 = axes[1]
    if labels is not None:
        from matplotlib.patches import Ellipse
        for label in unique_labels:
            mask = np.array(labels) == label
            points = spines_pca[mask, :2]
            centroid = points.mean(axis=0)
            
            if len(points) > 2:
                cov = np.cov(points.T)
                eigenvals, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
                width, height = 2 * 2 * np.sqrt(eigenvals)
                ellipse = Ellipse(centroid, width, height, angle=angle,
                                facecolor=label_colors[label], alpha=0.3,
                                edgecolor=label_colors[label], linewidth=2)
                ax2.add_patch(ellipse)
            
            ax2.scatter([centroid[0]], [centroid[1]], c=[label_colors[label]], s=200, marker='*',
                       edgecolors='black', linewidths=2, label=f'Label {label}', zorder=5)
        ax2.legend()
    else:
        centroid = spines_pca[:, :2].mean(axis=0)
        ax2.scatter([centroid[0]], [centroid[1]], c='red', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=5)
    
    ax2.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax2.set_title('Trajectory Centroids (2σ ellipses)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='datalim')
    
    if title is None:
        title = 'Trajectory-Space PCA'
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory PCA to {output_path}")
    
    # Compute silhouette if labels provided
    spine_silhouette = None
    if labels is not None and len(set(labels)) >= 2:
        from sklearn.metrics import silhouette_score
        try:
            spine_silhouette = float(silhouette_score(spines_pca[:, :2], labels))
        except:
            pass
    
    return {
        'spine_silhouette': spine_silhouette,
        'variance_explained': [float(v) for v in variance_explained],
    }
