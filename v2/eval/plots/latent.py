"""
Universal latent space plotting (label-free, optional labels for coloring).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from sklearn.decomposition import PCA


def plot_latent_scatter(
    z_array: np.ndarray,
    output_path: Path,
    labels: Optional[np.ndarray] = None,
    color_by: Optional[str] = None,
    color_values: Optional[np.ndarray] = None,
    title: Optional[str] = None,
):
    """
    Plot latent space scatter (label-free, with optional coloring).
    
    Args:
        z_array: (N, z_dim) latent vectors (should be L2-normalized)
        output_path: Path to save plot
        labels: Optional (N,) labels for coloring
        color_by: Optional "labels", "reward", "volume", "head" for default coloring
        color_values: Optional (N,) values for continuous coloring
        title: Optional plot title
    """
    if len(z_array) == 0:
        print("No z values to plot")
        return
    
    z_dim = z_array.shape[1]
    n_samples = len(z_array)
    
    # Determine coloring
    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
        label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        color_array = [label_colors.get(l, 'gray') for l in labels]
        legend_labels = {f'Label {l}': label_colors[l] for l in unique_labels}
    elif color_values is not None:
        color_array = color_values
        legend_labels = None
    else:
        # Default: color by index or uniform
        color_array = 'steelblue'
        legend_labels = None
    
    if z_dim == 2:
        # 2D: Cartesian + Polar views
        fig = plt.figure(figsize=(14, 6))
        
        # Left: Cartesian
        ax1 = fig.add_subplot(1, 2, 1)
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1, label='Unit circle')
        
        if labels is not None:
            for label in unique_labels:
                mask = np.array(labels) == label
                if np.any(mask):
                    ax1.scatter(
                        z_array[mask, 0], z_array[mask, 1],
                        c=[label_colors[label]], marker='o', s=80, alpha=0.7,
                        label=f'Label {label}', edgecolors='black', linewidths=0.5
                    )
        else:
            ax1.scatter(z_array[:, 0], z_array[:, 1], c=color_array, s=80, alpha=0.7,
                       edgecolors='black', linewidths=0.5)
        
        ax1.set_xlabel('z[0]', fontsize=12)
        ax1.set_ylabel('z[1]', fontsize=12)
        ax1.set_title('Latent Space (Cartesian)', fontsize=14)
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-1.3, 1.3)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        if legend_labels:
            ax1.legend(loc='upper right', fontsize=9)
        
        # Right: Polar
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        angles = np.arctan2(z_array[:, 1], z_array[:, 0])
        
        if labels is not None:
            for label in unique_labels:
                mask = np.array(labels) == label
                if np.any(mask):
                    ax2.scatter(
                        angles[mask], np.ones(np.sum(mask)),
                        c=[label_colors[label]], marker='o', s=60, alpha=0.7,
                        label=f'Label {label}', edgecolors='black', linewidths=0.5
                    )
        else:
            ax2.scatter(angles, np.ones(n_samples), c=color_array, s=60, alpha=0.7,
                       edgecolors='black', linewidths=0.5)
        
        ax2.set_title('Angular Distribution (Polar)', fontsize=14, pad=15)
        ax2.set_ylim(0, 1.3)
        ax2.set_rticks([])
        if legend_labels:
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    else:
        # Higher dimensions: PCA + coordinate projections
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_array)
        variance_explained = pca.explained_variance_ratio_.sum()
        
        if z_dim == 3:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
            projections = [(0, 1), (0, 2), (1, 2)]
            use_pca = [False, False, False, True]
        else:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
            projections = [(0, 1), (0, 2), (1, 2), (2, 3) if z_dim > 3 else (0, 1)]
            use_pca = [False, False, False, False, True, True]
        
        for idx, ax in enumerate(axes):
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=1)
            
            if idx < len(projections) and (idx >= len(use_pca) or not use_pca[idx]):
                dim1, dim2 = projections[idx]
                if labels is not None:
                    for label in unique_labels:
                        mask = np.array(labels) == label
                        if np.any(mask):
                            ax.scatter(
                                z_array[mask, dim1], z_array[mask, dim2],
                                c=[label_colors[label]], marker='o', s=60, alpha=0.7,
                                label=f'Label {label}', edgecolors='black', linewidths=0.5
                            )
                else:
                    ax.scatter(z_array[:, dim1], z_array[:, dim2], c=color_array, s=60, alpha=0.7,
                              edgecolors='black', linewidths=0.5)
                
                proj_radius = np.sqrt(z_array[:, dim1]**2 + z_array[:, dim2]**2)
                ax.set_title(f'z[{dim1}] vs z[{dim2}] (r={proj_radius.mean():.2f}±{proj_radius.std():.2f})', fontsize=11)
                ax.set_xlabel(f'z[{dim1}]', fontsize=10)
                ax.set_ylabel(f'z[{dim2}]', fontsize=10)
            else:
                # PCA projection
                if labels is not None:
                    for label in unique_labels:
                        mask = np.array(labels) == label
                        if np.any(mask):
                            ax.scatter(
                                z_pca[mask, 0], z_pca[mask, 1],
                                c=[label_colors[label]], marker='o', s=60, alpha=0.7,
                                label=f'Label {label}', edgecolors='black', linewidths=0.5
                            )
                else:
                    ax.scatter(z_pca[:, 0], z_pca[:, 1], c=color_array, s=60, alpha=0.7,
                              edgecolors='black', linewidths=0.5)
                ax.set_title(f'PCA ({variance_explained*100:.1f}% var)', fontsize=11)
                ax.set_xlabel('PC1', fontsize=10)
                ax.set_ylabel('PC2', fontsize=10)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            if legend_labels:
                ax.legend(loc='best', fontsize=8)
        
        # Hide unused axes
        for idx in range(len(projections) + (2 if z_dim > 3 else 1), len(axes)):
            axes[idx].set_visible(False)
    
    if title is None:
        title = f'Latent Space (L2-normalized, n={n_samples})'
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent scatter to {output_path}")


def plot_pairwise_cosine_histogram(
    z_array: np.ndarray,
    output_path: Path,
    bins: int = 50,
):
    """
    Plot histogram of pairwise cosine similarities (label-free).
    
    Args:
        z_array: (N, z_dim) latent vectors (should be L2-normalized)
        output_path: Path to save plot
        bins: Number of histogram bins
    """
    z_norm = z_array / (np.linalg.norm(z_array, axis=1, keepdims=True) + 1e-8)
    cos_sim = z_norm @ z_norm.T
    mask = ~np.eye(len(z_array), dtype=bool)
    pairwise_cos = cos_sim[mask]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(pairwise_cos, bins=bins, alpha=0.7, edgecolor='black')
    ax.axvline(pairwise_cos.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={pairwise_cos.mean():.3f}')
    ax.set_xlabel('Pairwise Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise Cosine Similarity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pairwise cosine histogram to {output_path}")


def plot_cosine_similarity_matrix(
    z_array: np.ndarray,
    output_path: Path,
    labels: Optional[np.ndarray] = None,
    cluster: bool = True,
    title: Optional[str] = None,
):
    """
    Plot cosine similarity matrix with hierarchical clustering (high-D friendly).
    
    Shows block-diagonal structure for distinct basins, uniform blocks for collapse.
    
    Args:
        z_array: (N, z_dim) latent vectors (should be L2-normalized)
        output_path: Path to save plot
        labels: Optional (N,) labels for comparison
        cluster: If True, reorder by hierarchical clustering
        title: Optional plot title
    """
    z_norm = z_array / (np.linalg.norm(z_array, axis=1, keepdims=True) + 1e-8)
    cos_sim = z_norm @ z_norm.T  # (N, N)
    
    # Reorder by clustering if requested
    order = np.arange(len(z_array))
    if cluster and len(z_array) > 1:
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            # Convert similarity to distance (ensure non-negative)
            dist_matrix = 1 - cos_sim
            dist_matrix = np.clip(dist_matrix, 0, 2)  # Clamp to [0, 2]
            np.fill_diagonal(dist_matrix, 0)
            
            # Hierarchical clustering (use 'average' instead of 'ward' for more robustness)
            condensed_dist = squareform(dist_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='average')
            order = leaves_list(linkage_matrix)
            
            cos_sim_ordered = cos_sim[np.ix_(order, order)]
        except (ImportError, ValueError) as e:
            print(f"Clustering failed ({e}), using original order")
            cos_sim_ordered = cos_sim
    else:
        cos_sim_ordered = cos_sim
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(cos_sim_ordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    ax1.set_xlabel("Sample index (clustered)" if cluster else "Sample index")
    ax1.set_ylabel("Sample index (clustered)" if cluster else "Sample index")
    title_str = "Cosine Similarity Matrix\nRed=similar (1.0), White=orthogonal (0.0), Blue=opposite (-1.0)"
    if cluster:
        title_str += "\n(Clustered: similar samples grouped together)"
    ax1.set_title(title_str, fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Cosine Similarity', rotation=270, labelpad=15)
    
    # Add interpretation hint
    ax1.text(0.02, 0.98, 
             "Look for: Red blocks = modes\nDark between blocks = separation",
             transform=ax1.transAxes, fontsize=9, style='italic', color='yellow',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Add label boundaries if provided
    if labels is not None:
        unique_labels = sorted(set(labels))
        label_positions = {}
        for label in unique_labels:
            mask = np.array(labels)[order] == label
            if np.any(mask):
                indices = np.where(mask)[0]
                label_positions[label] = (indices.min(), indices.max())
        
        for label, (start, end) in label_positions.items():
            ax1.axhline(start - 0.5, color='yellow', linewidth=2, alpha=0.7)
            ax1.axhline(end + 0.5, color='yellow', linewidth=2, alpha=0.7)
            ax1.axvline(start - 0.5, color='yellow', linewidth=2, alpha=0.7)
            ax1.axvline(end + 0.5, color='yellow', linewidth=2, alpha=0.7)
    
    # Right: Statistics
    ax2 = axes[1]
    mask = ~np.eye(len(cos_sim), dtype=bool)
    pairwise_cos = cos_sim[mask]
    
    # Histogram
    ax2.hist(pairwise_cos, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax2.axvline(pairwise_cos.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean={pairwise_cos.mean():.3f}')
    ax2.axvline(np.median(pairwise_cos), color='orange', linestyle='--', linewidth=2,
                label=f'Median={np.median(pairwise_cos):.3f}')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('Pairwise Similarity Distribution\nBimodal = good separation, Single peak = collapse')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add interpretation hint
    mean_sim = pairwise_cos.mean()
    if mean_sim > 0.7:
        hint = "⚠️ High mean: Possible mode collapse"
        color = 'red'
    elif mean_sim < 0.3:
        hint = "⚠️ Low mean: May be over-dispersed"
        color = 'orange'
    else:
        hint = "✓ Mean in good range"
        color = 'green'
    
    ax2.text(0.02, 0.98, hint, transform=ax2.transAxes, fontsize=9, 
             style='italic', color=color, weight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add text stats
    stats_text = (
        f"Min: {pairwise_cos.min():.3f}\n"
        f"Max: {pairwise_cos.max():.3f}\n"
        f"Std: {pairwise_cos.std():.3f}\n"
        f"Q25: {np.percentile(pairwise_cos, 25):.3f}\n"
        f"Q75: {np.percentile(pairwise_cos, 75):.3f}"
    )
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, family='monospace')
    
    if title is None:
        title = f'Cosine Similarity Matrix (n={len(z_array)})'
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cosine similarity matrix to {output_path}")
