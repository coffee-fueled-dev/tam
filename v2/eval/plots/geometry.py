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


def plot_tube_heatmap(
    mu: np.ndarray,
    sigma: np.ndarray,
    output_path: Path,
    sort_by: str = "mu_std",
    title: str = "Tube heatmap",
):
    """
    High-D tube visualization: heatmaps of mu and sigma over (time x dim).
    
    Args:
        mu: (T, d) tube centerline
        sigma: (T, d) tube width
        output_path: Path to save plot
        sort_by: "mu_std" | "mu_abs_mean" | "sigma_mean" - how to sort dimensions
        title: Plot title
    """
    T, d = mu.shape

    if sort_by == "mu_std":
        order = np.argsort(mu.std(axis=0))[::-1]
    elif sort_by == "mu_abs_mean":
        order = np.argsort(np.abs(mu).mean(axis=0))[::-1]
    elif sort_by == "sigma_mean":
        order = np.argsort(sigma.mean(axis=0))[::-1]
    else:
        order = np.arange(d)

    mu_s = mu[:, order].T        # (d, T)
    sig_s = sigma[:, order].T    # (d, T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    im0 = axes[0].imshow(mu_s, aspect="auto", cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title("μ (Tube Centerline)\nBlue=negative, Red=positive, White=zero", fontsize=11)
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Dimension (sorted by variability)")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("μ value", rotation=270, labelpad=15)

    im1 = axes[1].imshow(sig_s, aspect="auto", cmap='viridis', interpolation='nearest')
    axes[1].set_title("σ (Tube Width)\nDark=tight, Yellow=wide (uncertainty)", fontsize=11)
    axes[1].set_xlabel("Time step")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("σ value", rotation=270, labelpad=15)
    
    # Add interpretation hints
    fig.text(0.5, 0.02, 
             "Look for: Clear patterns (not noise) | Different dims show different patterns | Smooth temporal transitions",
             ha='center', fontsize=9, style='italic', color='gray')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved tube heatmap to {output_path}")


def plot_sliced_tube(
    mu: np.ndarray,
    sigma: np.ndarray,
    output_path: Path,
    K: int = 12,
    seed: int = 0,
    title: str = "Sliced tube projections",
):
    """
    High-D tube visualization: random projection slices.
    
    Projects the tube onto K random unit vectors to see "how it looks from many viewpoints".
    Scales cleanly to d=256+.
    
    Args:
        mu: (T, d) tube centerline
        sigma: (T, d) tube width (assumed diagonal)
        output_path: Path to save plot
        K: Number of random projection directions
        seed: Random seed for projections
        title: Plot title
    """
    rng = np.random.default_rng(seed)
    T, d = mu.shape

    U = rng.normal(size=(K, d))
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)  # (K,d)

    # projections
    m = mu @ U.T                         # (T,K)
    w = np.sqrt((sigma**2) @ (U**2).T)   # (T,K)

    fig, axes = plt.subplots(K, 1, figsize=(12, 2.2*K), sharex=True)
    if K == 1:
        axes = [axes]
    t = np.arange(T)

    for k in range(K):
        ax = axes[k]
        ax.plot(t, m[:, k], linewidth=1.8, color='steelblue')
        ax.fill_between(t, m[:, k] - w[:, k], m[:, k] + w[:, k], alpha=0.2, color='steelblue')
        ax.set_ylabel(f"slice {k}")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("time")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sliced tube to {output_path}")


def plot_knot_structure(
    mu_knots: np.ndarray,
    output_path: Path,
    title: str = "Knot structure analysis",
):
    """
    High-D knot visualization: heatmap + SVD analysis.
    
    Shows knot commitments as (n_knots × d) matrix and low-rank structure.
    
    Args:
        mu_knots: (n_knots, d) knot positions
        output_path: Path to save plot
        title: Plot title
    """
    n_knots, d = mu_knots.shape
    
    # Sort dimensions by variability
    order = np.argsort(mu_knots.std(axis=0))[::-1]
    mu_sorted = mu_knots[:, order]
    
    # SVD analysis
    U, S, Vt = np.linalg.svd(mu_knots, full_matrices=False)
    n_components = min(3, n_knots, d)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Knot heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(mu_sorted.T, aspect="auto", cmap='RdBu_r', interpolation='nearest')
    ax1.set_title(f"Knot matrix (sorted dims)")
    ax1.set_xlabel("knot index")
    ax1.set_ylabel("dimension (sorted)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Top right: Singular values
    ax2 = axes[0, 1]
    ax2.plot(S, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Singular value")
    ax2.set_title(f"Singular values (rank={np.sum(S > 1e-6)})")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Bottom left: First 3 components over knots
    ax3 = axes[1, 0]
    knot_indices = np.arange(n_knots)
    for i in range(n_components):
        ax3.plot(knot_indices, U[:, i] * S[i], 'o-', linewidth=2, label=f'Component {i+1} (σ={S[i]:.3f})')
    ax3.set_xlabel("Knot index")
    ax3.set_ylabel("Component value")
    ax3.set_title("Low-rank structure (U @ diag(S))")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Variance explained
    ax4 = axes[1, 1]
    variance_explained = S**2 / (S**2).sum()
    cumsum = np.cumsum(variance_explained)
    ax4.plot(range(1, min(10, len(S)) + 1), variance_explained[:10], 'o-', linewidth=2, label='Per component')
    ax4.plot(range(1, min(10, len(S)) + 1), cumsum[:10], 's--', linewidth=2, label='Cumulative')
    ax4.set_xlabel("Component")
    ax4.set_ylabel("Variance explained")
    ax4.set_title("Variance explained by components")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved knot structure to {output_path}")


def plot_commitment_summaries(
    mu: np.ndarray,
    sigma: np.ndarray,
    output_path: Path,
    title: str = "Commitment summaries",
):
    """
    Scalar "commitment summaries" over time (stable and comparable in high-D).
    
    Args:
        mu: (T, d) tube centerline
        sigma: (T, d) tube width
        output_path: Path to save plot
        title: Plot title
    """
    T, d = mu.shape
    t = np.arange(T)
    
    # Compute scalar summaries
    center_speed = np.array([np.linalg.norm(mu[t+1] - mu[t]) if t < T-1 else 0 for t in range(T)])
    tube_thickness_mean = sigma.mean(axis=1)
    tube_thickness_norm = np.linalg.norm(sigma, axis=1)
    mu_norm = np.linalg.norm(mu, axis=1)
    relative_thickness = tube_thickness_norm / (mu_norm + 1e-8)
    volume_proxy = np.mean(np.log(sigma + 1e-8), axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Center speed
    ax = axes[0]
    ax.plot(t[:-1], center_speed[:-1], linewidth=2, color='steelblue')
    ax.set_xlabel("time")
    ax.set_ylabel("||μ[t+1] - μ[t]||₂")
    ax.set_title("Center Speed")
    ax.grid(True, alpha=0.3)
    
    # Tube thickness (mean)
    ax = axes[1]
    ax.plot(t, tube_thickness_mean, linewidth=2, color='green')
    ax.set_xlabel("time")
    ax.set_ylabel("mean(σ)")
    ax.set_title("Tube Thickness (mean)")
    ax.grid(True, alpha=0.3)
    
    # Tube thickness (norm)
    ax = axes[2]
    ax.plot(t, tube_thickness_norm, linewidth=2, color='orange')
    ax.set_xlabel("time")
    ax.set_ylabel("||σ||₂")
    ax.set_title("Tube Thickness (norm)")
    ax.grid(True, alpha=0.3)
    
    # Relative thickness
    ax = axes[3]
    ax.plot(t, relative_thickness, linewidth=2, color='purple')
    ax.set_xlabel("time")
    ax.set_ylabel("||σ||₂ / ||μ||₂")
    ax.set_title("Relative Thickness")
    ax.grid(True, alpha=0.3)
    
    # Volume proxy
    ax = axes[4]
    ax.plot(t, volume_proxy, linewidth=2, color='red')
    ax.set_xlabel("time")
    ax.set_ylabel("mean(log(σ + ε))")
    ax.set_title("Volume Proxy (log-space)")
    ax.grid(True, alpha=0.3)
    
    # Combined view
    ax = axes[5]
    ax2_twin = ax.twinx()
    line1 = ax.plot(t, center_speed, linewidth=2, color='steelblue', label='Speed')
    line2 = ax2_twin.plot(t, tube_thickness_mean, linewidth=2, color='green', label='Thickness')
    ax.set_xlabel("time")
    ax.set_ylabel("Speed", color='steelblue')
    ax2_twin.set_ylabel("Thickness", color='green')
    ax.set_title("Speed vs Thickness")
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved commitment summaries to {output_path}")


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
