import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from actors.knotv2 import GeometricKnotActor
from environments.cmg_env import CMGConfig, CMGEnv, generate_episode
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

def plot_results(actor, s0, trajectory, z_star, all_z_stars, all_modes, output_dir):
    """Plot latent space clustering colored by mode - optimized for L2-normalized (spherical) data."""
    if len(all_z_stars) == 0:
        print("No z_stars to plot")
        return
    
    z_array = np.array([z.detach().cpu().numpy() for z in all_z_stars])
    z_dim = z_array.shape[1]
    
    # Color map for modes
    unique_modes = sorted(set(all_modes))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_modes), 10)))
    mode_colors = {mode: colors[i] for i, mode in enumerate(unique_modes)}
    
    if z_dim == 2:
        # Special layout for 2D spherical data: Cartesian + Polar views
        fig = plt.figure(figsize=(14, 6))
        
        # Left: Cartesian view with unit circle
        ax1 = fig.add_subplot(1, 2, 1)
        
        # Draw unit circle reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1, label='Unit circle')
        
        # Plot points colored by mode
        for mode in unique_modes:
            mask = np.array(all_modes) == mode
            if np.any(mask):
                ax1.scatter(
                    z_array[mask, 0], z_array[mask, 1],
                    c=[mode_colors[mode]], marker='o', s=80, alpha=0.7,
                    label=f'Mode {mode}', edgecolors='black', linewidths=0.5
                )
        
        ax1.set_xlabel('z[0]', fontsize=12)
        ax1.set_ylabel('z[1]', fontsize=12)
        ax1.set_title('Latent Space (Cartesian)', fontsize=14)
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-1.3, 1.3)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        
        # Right: Polar view (angle histogram)
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        
        # Compute angles for each point
        angles = np.arctan2(z_array[:, 1], z_array[:, 0])
        
        # Plot points in polar coordinates (radius = 1 for L2-normalized)
        for mode in unique_modes:
            mask = np.array(all_modes) == mode
            if np.any(mask):
                mode_angles = angles[mask]
                # Plot as scatter on unit circle
                ax2.scatter(
                    mode_angles, np.ones_like(mode_angles),
                    c=[mode_colors[mode]], marker='o', s=60, alpha=0.7,
                    label=f'Mode {mode}', edgecolors='black', linewidths=0.5
                )
        
        ax2.set_title('Angular Distribution (Polar)', fontsize=14, pad=15)
        ax2.set_ylim(0, 1.3)
        ax2.set_rticks([])  # Hide radial ticks (all points at r=1)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        
    else:
        # For higher dimensions: PCA projection + coordinate projections
        from sklearn.decomposition import PCA
        
        # Compute PCA to find best 2D view
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
            use_pca = [False, False, False, False, True, True]  # Last 2 for PCA
        
        for idx, ax in enumerate(axes):
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=1, label='Unit circle')
            
            if idx < len(projections) and (idx >= len(use_pca) or not use_pca[idx]):
                # Coordinate projection
                dim1, dim2 = projections[idx]
                for mode in unique_modes:
                    mask = np.array(all_modes) == mode
                    if np.any(mask):
                        ax.scatter(
                            z_array[mask, dim1], z_array[mask, dim2],
                            c=[mode_colors[mode]], marker='o', s=60, alpha=0.7,
                            label=f'Mode {mode}', edgecolors='black', linewidths=0.5
                        )
                
                # Show projected radius stats
                proj_radius = np.sqrt(z_array[:, dim1]**2 + z_array[:, dim2]**2)
                ax.set_title(f'z[{dim1}] vs z[{dim2}] (r={proj_radius.mean():.2f}±{proj_radius.std():.2f})', fontsize=11)
                ax.set_xlabel(f'z[{dim1}]', fontsize=10)
                ax.set_ylabel(f'z[{dim2}]', fontsize=10)
            else:
                # PCA projection
                for mode in unique_modes:
                    mask = np.array(all_modes) == mode
                    if np.any(mask):
                        ax.scatter(
                            z_pca[mask, 0], z_pca[mask, 1],
                            c=[mode_colors[mode]], marker='o', s=60, alpha=0.7,
                            label=f'Mode {mode}', edgecolors='black', linewidths=0.5
                        )
                ax.set_title(f'PCA ({variance_explained*100:.1f}% var)', fontsize=11)
                ax.set_xlabel('PC1', fontsize=10)
                ax.set_ylabel('PC2', fontsize=10)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused axes
        for idx in range(len(projections) + (2 if z_dim > 3 else 1), len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle(f'Latent Space Clustering (L2-normalized, n={len(all_z_stars)} episodes)', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plot_path = output_dir / "latent_clustering.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent clustering plot to {plot_path}")
    
    # Print mode statistics
    print("\nMode distribution:")
    for mode in unique_modes:
        count = sum(1 for m in all_modes if m == mode)
        print(f"  Mode {mode}: {count}/{len(all_modes)} ({100*count/len(all_modes):.1f}%)")


def compute_clustering_metrics(z_array, all_modes):
    """
    Compute quantitative clustering metrics for latent space.
    
    Returns:
        dict with various clustering metrics
    """
    if len(set(all_modes)) < 2:
        # Need at least 2 clusters for most metrics
        return {
            "silhouette_score": None,
            "adjusted_rand_index": None,
            "calinski_harabasz_index": None,
            "davies_bouldin_index": None,
            "inter_cluster_distance": None,
            "intra_cluster_distance": None,
            "separability_ratio": None,
        }
    
    labels = np.array(all_modes)
    
    # Standard clustering metrics
    silhouette = silhouette_score(z_array, labels)
    ari = adjusted_rand_score(labels, labels)  # Perfect since we're using true labels
    calinski_harabasz = calinski_harabasz_score(z_array, labels)
    davies_bouldin = davies_bouldin_score(z_array, labels)
    
    # Custom separability metrics
    unique_modes = sorted(set(all_modes))
    mode_centers = []
    intra_distances = []
    
    for mode in unique_modes:
        mask = labels == mode
        cluster_points = z_array[mask]
        center = cluster_points.mean(axis=0)
        mode_centers.append(center)
        
        # Intra-cluster distances (average distance from center)
        if len(cluster_points) > 1:
            distances = np.linalg.norm(cluster_points - center, axis=1)
            intra_distances.append(distances.mean())
        else:
            intra_distances.append(0.0)
    
    mode_centers = np.array(mode_centers)
    
    # Inter-cluster distances (distances between cluster centers)
    inter_distances = cdist(mode_centers, mode_centers, metric='euclidean')
    # Remove diagonal (self-distances)
    inter_distances = inter_distances[~np.eye(inter_distances.shape[0], dtype=bool)]
    avg_inter_cluster = inter_distances.mean()
    avg_intra_cluster = np.mean(intra_distances) if intra_distances else 0.0
    
    # Separability ratio: higher is better (more separation relative to cluster spread)
    separability_ratio = avg_inter_cluster / (avg_intra_cluster + 1e-8)
    
    return {
        "silhouette_score": float(silhouette),
        "adjusted_rand_index": float(ari),
        "calinski_harabasz_index": float(calinski_harabasz),
        "davies_bouldin_index": float(davies_bouldin),
        "inter_cluster_distance": float(avg_inter_cluster),
        "intra_cluster_distance": float(avg_intra_cluster),
        "separability_ratio": float(separability_ratio),
    }


# =============================================================================
# GEOMETRIC DIAGNOSTIC TESTS
# =============================================================================

def tube_intersection_test(actor, s0, mode_centroids, config, output_dir):
    """
    Test 1: Do tubes from different intended modes geometrically intersect?
    
    Measures:
    - % time overlap between tube pairs
    - Hausdorff distance between tube surfaces
    """
    print("\n" + "="*60)
    print("TEST 1: Tube Intersection Test")
    print("="*60)
    
    if len(mode_centroids) < 2:
        print("  Need at least 2 modes for intersection test")
        return None
    
    # Generate tubes for each mode centroid
    tubes = {}
    for mode, z_centroid in mode_centroids.items():
        with torch.no_grad():
            mu, sigma = actor.get_tube(s0, z_centroid)
        tubes[mode] = {
            'mu': mu.squeeze(0).cpu().numpy(),  # (T, d)
            'sigma': sigma.squeeze(0).cpu().numpy()  # (T, d)
        }
    
    # Compute pairwise overlap and Hausdorff distances
    modes = sorted(tubes.keys())
    n_modes = len(modes)
    overlap_matrix = np.zeros((n_modes, n_modes))
    hausdorff_matrix = np.zeros((n_modes, n_modes))
    
    for i, mode_i in enumerate(modes):
        for j, mode_j in enumerate(modes):
            if i >= j:
                continue
            
            mu_i, sigma_i = tubes[mode_i]['mu'], tubes[mode_i]['sigma']
            mu_j, sigma_j = tubes[mode_j]['mu'], tubes[mode_j]['sigma']
            
            # Compute overlap: timesteps where tubes intersect
            # Tubes intersect at time t if |mu_i[t] - mu_j[t]| < sigma_i[t] + sigma_j[t]
            dist = np.abs(mu_i - mu_j)
            combined_sigma = sigma_i + sigma_j
            overlap_per_t = (dist < combined_sigma).all(axis=1)  # All dims must overlap
            overlap_pct = overlap_per_t.mean() * 100
            overlap_matrix[i, j] = overlap_matrix[j, i] = overlap_pct
            
            # Hausdorff distance between tube centerlines
            from scipy.spatial.distance import directed_hausdorff
            h1 = directed_hausdorff(mu_i, mu_j)[0]
            h2 = directed_hausdorff(mu_j, mu_i)[0]
            hausdorff = max(h1, h2)
            hausdorff_matrix[i, j] = hausdorff_matrix[j, i] = hausdorff
    
    # Print results
    print(f"\n  Tube Overlap Matrix (% timesteps with intersection):")
    print(f"  Mode  | " + " | ".join([f"{m:5d}" for m in modes]))
    print(f"  ------|-" + "-|".join(["------" for _ in modes]))
    for i, mode_i in enumerate(modes):
        row = [f"{overlap_matrix[i,j]:5.1f}" for j in range(n_modes)]
        print(f"  {mode_i:5d} | " + " | ".join(row))
    
    print(f"\n  Hausdorff Distance Matrix (centerline separation):")
    print(f"  Mode  | " + " | ".join([f"{m:5d}" for m in modes]))
    print(f"  ------|-" + "-|".join(["------" for _ in modes]))
    for i, mode_i in enumerate(modes):
        row = [f"{hausdorff_matrix[i,j]:5.2f}" for j in range(n_modes)]
        print(f"  {mode_i:5d} | " + " | ".join(row))
    
    avg_overlap = overlap_matrix[np.triu_indices(n_modes, k=1)].mean()
    avg_hausdorff = hausdorff_matrix[np.triu_indices(n_modes, k=1)].mean()
    
    print(f"\n  Summary:")
    print(f"    Average Tube Overlap: {avg_overlap:.1f}%")
    print(f"    Average Hausdorff Distance: {avg_hausdorff:.3f}")
    
    if avg_overlap > 80:
        print(f"    → High overlap: Modes are topologically UNIFIED")
    elif avg_overlap > 40:
        print(f"    → Moderate overlap: Partial port separation")
    else:
        print(f"    → Low overlap: Genuine port separation achieved")
    
    # Visualize tubes
    fig, axes = plt.subplots(1, min(3, config.d), figsize=(5*min(3, config.d), 5))
    if config.d == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, n_modes))
    
    for dim in range(min(3, config.d)):
        ax = axes[dim]
        for i, mode in enumerate(modes):
            mu = tubes[mode]['mu'][:, dim]
            sigma = tubes[mode]['sigma'][:, dim]
            t = np.arange(len(mu))
            ax.plot(t, mu, color=colors[i], linewidth=2, label=f'Mode {mode}')
            ax.fill_between(t, mu - sigma, mu + sigma, color=colors[i], alpha=0.2)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'x[{dim}]')
        ax.set_title(f'Dimension {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Tube Intersection Test: Tubes from Mode Centroids')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tube_intersection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved tube intersection plot to {output_dir}/tube_intersection.png")
    
    return {
        'overlap_matrix': overlap_matrix.tolist(),
        'hausdorff_matrix': hausdorff_matrix.tolist(),
        'avg_overlap': float(avg_overlap),
        'avg_hausdorff': float(avg_hausdorff)
    }


def cem_energy_landscape(actor, s0, config, output_dir, n_samples=500):
    """
    Test 3: CEM Energy Landscape Visualization
    
    For a fixed s0, sample dense z grid on unit sphere and plot:
    E(z) = α·Volume(z) − β·Binding(z)
    """
    print("\n" + "="*60)
    print("TEST 3: CEM Energy Landscape")
    print("="*60)
    
    z_dim = actor.z_dim
    
    # Sample z values on unit sphere
    if z_dim == 2:
        # Dense sampling on circle
        angles = np.linspace(0, 2*np.pi, n_samples)
        z_samples = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        # Random sampling on hypersphere
        z_samples = np.random.randn(n_samples, z_dim)
        z_samples = z_samples / (np.linalg.norm(z_samples, axis=1, keepdims=True) + 1e-8)
    
    z_tensor = torch.tensor(z_samples, dtype=torch.float32)
    
    # Compute energy for each z
    volumes = []
    start_errors = []
    displacements = []
    
    with torch.no_grad():
        for i in range(0, n_samples, 50):  # Batch processing
            batch_z = z_tensor[i:i+50]
            mu, sigma = actor.get_tube(s0, batch_z)
            
            # Volume (lower is better for agency)
            vol = sigma.mean(dim=(1, 2)).cpu().numpy()
            volumes.extend(vol)
            
            # Start error (should be near s0)
            s0_pos = s0[:config.d].cpu().numpy()
            start_err = np.linalg.norm(mu[:, 0, :].cpu().numpy() - s0_pos, axis=1)
            start_errors.extend(start_err)
            
            # Displacement (progress)
            disp = np.linalg.norm(mu[:, -1, :].cpu().numpy() - mu[:, 0, :].cpu().numpy(), axis=1)
            displacements.extend(disp)
    
    volumes = np.array(volumes)
    start_errors = np.array(start_errors)
    displacements = np.array(displacements)
    
    # Compute energy: reward low volume, penalize bad start, reward progress
    energy = -volumes * 0.3 - start_errors * 10.0 + np.log(displacements + 1e-6) * 3.0
    
    print(f"  Energy Statistics:")
    print(f"    Min Energy: {energy.min():.3f}")
    print(f"    Max Energy: {energy.max():.3f}")
    print(f"    Energy Range: {energy.max() - energy.min():.3f}")
    
    # Find basins (local maxima)
    if z_dim == 2:
        # Smooth and find peaks
        from scipy.ndimage import gaussian_filter1d
        energy_smooth = gaussian_filter1d(energy, sigma=5, mode='wrap')
        
        # Find local maxima
        peaks = []
        for i in range(len(energy_smooth)):
            prev_i = (i - 1) % len(energy_smooth)
            next_i = (i + 1) % len(energy_smooth)
            if energy_smooth[i] > energy_smooth[prev_i] and energy_smooth[i] > energy_smooth[next_i]:
                peaks.append(i)
        
        print(f"    Number of Energy Basins: {len(peaks)}")
        print(f"    Basin Angles: {[f'{angles[p]*180/np.pi:.1f}°' for p in peaks]}")
        
        # Plot energy landscape
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Polar plot
        ax1 = axes[0]
        ax1.plot(angles * 180 / np.pi, energy, 'b-', alpha=0.5, label='Raw')
        ax1.plot(angles * 180 / np.pi, energy_smooth, 'r-', linewidth=2, label='Smoothed')
        for p in peaks:
            ax1.axvline(angles[p] * 180 / np.pi, color='g', linestyle='--', alpha=0.5)
            ax1.scatter([angles[p] * 180 / np.pi], [energy_smooth[p]], color='g', s=100, zorder=5)
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Energy E(z)')
        ax1.set_title(f'CEM Energy vs Angle ({len(peaks)} basins)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2D scatter colored by energy
        ax2 = axes[1]
        scatter = ax2.scatter(z_samples[:, 0], z_samples[:, 1], c=energy, cmap='viridis', s=20)
        for p in peaks:
            ax2.scatter([z_samples[p, 0]], [z_samples[p, 1]], color='red', s=200, marker='*', 
                       edgecolors='white', linewidths=2, zorder=5, label=f'Basin' if p == peaks[0] else None)
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
        # For higher dimensions, use PCA to visualize
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_samples)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=energy, cmap='viridis', s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'CEM Energy Landscape (PCA projection, var={pca.explained_variance_ratio_.sum()*100:.1f}%)')
        plt.colorbar(scatter, ax=ax, label='Energy')
        
        # Find top energy points
        top_k = 5
        top_indices = np.argsort(energy)[-top_k:]
        ax.scatter(z_pca[top_indices, 0], z_pca[top_indices, 1], color='red', s=200, marker='*',
                  edgecolors='white', linewidths=2, zorder=5, label=f'Top {top_k} basins')
        ax.legend()
    
    plt.suptitle(f'CEM Energy Landscape (n={n_samples} samples)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cem_energy_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved energy landscape to {output_dir}/cem_energy_landscape.png")
    
    return {
        'n_basins': len(peaks) if z_dim == 2 else None,
        'energy_range': float(energy.max() - energy.min()),
        'energy_std': float(energy.std())
    }


def mode_conditional_volume_curves(actor, env, mode_centroids, config, output_dir, n_samples=20):
    """
    Test 4: Mode-Conditional Volume Curves
    
    Plot mean σ(t) per mode and volume integral per mode.
    """
    print("\n" + "="*60)
    print("TEST 4: Mode-Conditional Volume Curves")
    print("="*60)
    
    if len(mode_centroids) < 2:
        print("  Need at least 2 modes")
        return None
    
    # Collect volume profiles for each mode
    mode_volumes = {mode: [] for mode in mode_centroids}
    mode_sigmas = {mode: [] for mode in mode_centroids}
    
    for mode, z_centroid in mode_centroids.items():
        for _ in range(n_samples):
            obs = env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32)
            
            # Add small noise to z
            z = z_centroid + torch.randn_like(z_centroid) * 0.05
            z = torch.nn.functional.normalize(z.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            with torch.no_grad():
                mu, sigma = actor.get_tube(s0, z)
            
            sigma_np = sigma.squeeze(0).cpu().numpy()  # (T, d)
            mode_sigmas[mode].append(sigma_np)
            mode_volumes[mode].append(sigma_np.mean())
    
    # Compute statistics
    modes = sorted(mode_centroids.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: σ(t) curves per mode
    ax1 = axes[0]
    for i, mode in enumerate(modes):
        sigmas = np.array(mode_sigmas[mode])  # (n_samples, T, d)
        mean_sigma_t = sigmas.mean(axis=(0, 2))  # Average over samples and dims -> (T,)
        std_sigma_t = sigmas.std(axis=(0, 2))
        t = np.arange(len(mean_sigma_t))
        
        ax1.plot(t, mean_sigma_t, color=colors[i], linewidth=2, label=f'Mode {mode}')
        ax1.fill_between(t, mean_sigma_t - std_sigma_t, mean_sigma_t + std_sigma_t, 
                        color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Mean σ(t)')
    ax1.set_title('Tube Width Over Time by Mode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Volume distribution per mode
    ax2 = axes[1]
    volume_data = [mode_volumes[mode] for mode in modes]
    bp = ax2.boxplot(volume_data, labels=[f'Mode {m}' for m in modes], patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Volume (mean σ)')
    ax2.set_title('Volume Distribution by Mode')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Mode-Conditional Volume Analysis')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mode_volume_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n  Volume Statistics by Mode:")
    print(f"  Mode | Mean Vol | Std Vol")
    print(f"  -----|----------|--------")
    for mode in modes:
        vols = mode_volumes[mode]
        print(f"  {mode:4d} | {np.mean(vols):.4f}   | {np.std(vols):.4f}")
    
    # Check for convergence
    all_vols = [np.mean(mode_volumes[m]) for m in modes]
    vol_range = max(all_vols) - min(all_vols)
    print(f"\n  Volume Range Across Modes: {vol_range:.4f}")
    if vol_range < 0.1:
        print(f"    → Modes have CONVERGED to similar volume profiles")
    else:
        print(f"    → Modes have DISTINCT volume profiles")
    
    print(f"  Saved volume curves to {output_dir}/mode_volume_curves.png")
    
    return {
        'mode_volumes': {m: float(np.mean(mode_volumes[m])) for m in modes},
        'volume_range': float(vol_range)
    }


def trajectory_space_pca(actor, env, mode_centroids, config, output_dir, n_samples=30):
    """
    Test 5: Trajectory-space PCA (Not z-space)
    
    PCA the spines μ(z, s₀) across modes to see if they diverge.
    """
    print("\n" + "="*60)
    print("TEST 5: Trajectory-Space PCA")
    print("="*60)
    
    if len(mode_centroids) < 2:
        print("  Need at least 2 modes")
        return None
    
    from sklearn.decomposition import PCA
    
    # Collect tube spines for each mode
    all_spines = []
    all_labels = []
    
    for mode, z_centroid in mode_centroids.items():
        for _ in range(n_samples):
            obs = env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32)
            
            # Add small noise to z
            z = z_centroid + torch.randn_like(z_centroid) * 0.05
            z = torch.nn.functional.normalize(z.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            with torch.no_grad():
                mu, sigma = actor.get_tube(s0, z)
            
            spine = mu.squeeze(0).cpu().numpy().flatten()  # (T * d,)
            all_spines.append(spine)
            all_labels.append(mode)
    
    all_spines = np.array(all_spines)
    all_labels = np.array(all_labels)
    
    # PCA on trajectory space
    pca = PCA(n_components=min(3, all_spines.shape[1]))
    spines_pca = pca.fit_transform(all_spines)
    
    variance_explained = pca.explained_variance_ratio_
    print(f"\n  PCA Variance Explained:")
    for i, var in enumerate(variance_explained):
        print(f"    PC{i+1}: {var*100:.1f}%")
    print(f"    Total (PC1-2): {sum(variance_explained[:2])*100:.1f}%")
    
    # Compute spine clustering metrics
    modes = sorted(set(all_labels))
    spine_silhouette = silhouette_score(spines_pca[:, :2], all_labels) if len(modes) >= 2 else 0
    print(f"\n  Spine Clustering (Silhouette): {spine_silhouette:.4f}")
    
    if spine_silhouette > 0.3:
        print(f"    → Spines DIVERGE: z separation is functional")
    elif spine_silhouette > 0:
        print(f"    → Spines show PARTIAL separation")
    else:
        print(f"    → Spines OVERLAP: z separation is cosmetic")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    
    # PC1 vs PC2
    ax1 = axes[0]
    for i, mode in enumerate(modes):
        mask = all_labels == mode
        ax1.scatter(spines_pca[mask, 0], spines_pca[mask, 1], 
                   c=[colors[i]], s=50, alpha=0.7, label=f'Mode {mode}',
                   edgecolors='black', linewidths=0.5)
    ax1.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax1.set_title('Trajectory Space PCA (PC1 vs PC2)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mode centroids in spine space
    ax2 = axes[1]
    mode_spine_centroids = {}
    for mode in modes:
        mask = all_labels == mode
        centroid = spines_pca[mask].mean(axis=0)
        mode_spine_centroids[mode] = centroid
    
    # Plot convex hulls or ellipses for each mode
    from matplotlib.patches import Ellipse
    for i, mode in enumerate(modes):
        mask = all_labels == mode
        points = spines_pca[mask, :2]
        centroid = points.mean(axis=0)
        
        # Compute covariance for ellipse
        if len(points) > 2:
            cov = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2 std
            ellipse = Ellipse(centroid, width, height, angle=angle, 
                            facecolor=colors[i], alpha=0.3, edgecolor=colors[i], linewidth=2)
            ax2.add_patch(ellipse)
        
        ax2.scatter([centroid[0]], [centroid[1]], c=[colors[i]], s=200, marker='*',
                   edgecolors='black', linewidths=2, label=f'Mode {mode}', zorder=5)
    
    ax2.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax2.set_title('Mode Centroids in Trajectory Space (2σ ellipses)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='datalim')
    
    plt.suptitle(f'Trajectory-Space PCA (Silhouette: {spine_silhouette:.3f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved trajectory PCA to {output_dir}/trajectory_pca.png")
    
    return {
        'spine_silhouette': float(spine_silhouette),
        'variance_explained': [float(v) for v in variance_explained],
        'spine_overlap': bool(spine_silhouette < 0)
    }


def run_geometric_diagnostics(actor, env, config, mode_centroids, s0, output_dir):
    """Run all geometric diagnostic tests."""
    
    print("\n" + "="*60)
    print("GEOMETRIC DIAGNOSTIC SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Tube Intersection
    results['tube_intersection'] = tube_intersection_test(actor, s0, mode_centroids, config, output_dir)
    
    # Test 3: CEM Energy Landscape
    results['cem_energy'] = cem_energy_landscape(actor, s0, config, output_dir)
    
    # Test 4: Mode-Conditional Volume Curves
    results['volume_curves'] = mode_conditional_volume_curves(actor, env, mode_centroids, config, output_dir)
    
    # Test 5: Trajectory-Space PCA
    results['trajectory_pca'] = trajectory_space_pca(actor, env, mode_centroids, config, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if results['tube_intersection']:
        avg_overlap = results['tube_intersection']['avg_overlap']
        print(f"  Tube Overlap: {avg_overlap:.1f}%", end=" → ")
        if avg_overlap > 80:
            print("Modes are UNIFIED")
        elif avg_overlap > 40:
            print("Partial separation")
        else:
            print("Genuine PORT SEPARATION")
    
    if results['cem_energy'] and results['cem_energy'].get('n_basins'):
        n_basins = results['cem_energy']['n_basins']
        print(f"  Energy Basins: {n_basins}", end=" → ")
        if n_basins >= config.K:
            print(f"Multiple basins detected (≥{config.K} modes)")
        elif n_basins > 1:
            print(f"Fewer basins than modes ({n_basins} < {config.K})")
        else:
            print("Single basin (rational unification)")
    
    if results['volume_curves']:
        vol_range = results['volume_curves']['volume_range']
        print(f"  Volume Range: {vol_range:.4f}", end=" → ")
        if vol_range < 0.1:
            print("Modes CONVERGED to same volume")
        else:
            print("Distinct volume profiles")
    
    if results['trajectory_pca']:
        spine_sil = results['trajectory_pca']['spine_silhouette']
        print(f"  Spine Silhouette: {spine_sil:.3f}", end=" → ")
        if spine_sil > 0.3:
            print("Spines DIVERGE (z is functional)")
        elif spine_sil > 0:
            print("Partial spine separation")
        else:
            print("Spines OVERLAP (z is cosmetic)")
    
    return results


# =============================================================================
# TOPOLOGICAL NECESSITY TESTS
# =============================================================================

def fork_separability_test(env, config, output_dir, n_episodes=10):
    """
    Test 1: Fork Separability - Does the environment force disjoint futures?
    
    For each episode, generate K counterfactual rollouts with forced modes
    and measure how quickly trajectories diverge.
    """
    from environments.cmg_env import rollout_with_forced_mode
    
    print("\n" + "="*60)
    print("TOPOLOGICAL TEST 1: Fork Separability")
    print("="*60)
    
    all_D_curves = []  # Pairwise divergence over time
    all_R_curves = []  # Minimum covering radius over time
    all_fork_indices = []
    
    for ep in range(n_episodes):
        # Reset environment to get consistent starting conditions
        env.reset()
        x0 = env.x.copy()
        goals = env.params.g.copy()  # (K, d) array
        
        # Generate K counterfactual trajectories
        mode_trajectories = {}
        for k in range(config.K):
            # Reset to same x0
            env.reset()
            env.x = x0.copy()
            env.params.g = goals.copy()
            
            record = rollout_with_forced_mode(env, k, "goal_seeking")
            mode_trajectories[k] = np.array(record['x'])  # (T+1, d)
        
        T = mode_trajectories[0].shape[0] - 1
        
        # Compute D(t): minimum pairwise divergence at each timestep
        D_curve = []
        for t in range(T + 1):
            min_dist = float('inf')
            for i in range(config.K):
                for j in range(i + 1, config.K):
                    dist = np.linalg.norm(mode_trajectories[i][t] - mode_trajectories[j][t])
                    min_dist = min(min_dist, dist)
            D_curve.append(min_dist)
        all_D_curves.append(D_curve)
        
        # Compute R_all(t): minimum covering radius
        R_curve = []
        for t in range(T + 1):
            points = np.array([mode_trajectories[k][t] for k in range(config.K)])
            center = points.mean(axis=0)
            max_dist = max(np.linalg.norm(p - center) for p in points)
            R_curve.append(max_dist)
        all_R_curves.append(R_curve)
        
        # Compute Fork Index
        sigma_floor = np.array([np.sqrt(t + 1) * config.noise_x for t in range(T + 1)])
        fork_index = np.mean(np.array(R_curve) / (sigma_floor + 1e-6))
        all_fork_indices.append(fork_index)
    
    # Average across episodes
    D_mean = np.mean(all_D_curves, axis=0)
    D_std = np.std(all_D_curves, axis=0)
    R_mean = np.mean(all_R_curves, axis=0)
    R_std = np.std(all_R_curves, axis=0)
    
    fork_index_mean = np.mean(all_fork_indices)
    fork_index_std = np.std(all_fork_indices)
    
    print(f"\n  Fork Index: {fork_index_mean:.2f} ± {fork_index_std:.2f}")
    if fork_index_mean > 5:
        print(f"    → Fork Index >> 1: Environment REQUIRES distinct ports")
    elif fork_index_mean > 1:
        print(f"    → Fork Index > 1: Environment has moderate fork pressure")
    else:
        print(f"    → Fork Index ≈ 1: Environment allows unified commitment")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t_axis = np.arange(len(D_mean))
    
    # Left: D(t) - Pairwise divergence
    ax1 = axes[0]
    ax1.plot(t_axis, D_mean, 'b-', linewidth=2, label='D(t) mean')
    ax1.fill_between(t_axis, D_mean - D_std, D_mean + D_std, alpha=0.3)
    ax1.axvline(config.t_gate, color='r', linestyle='--', label=f't_gate={config.t_gate}')
    if hasattr(config, 'early_divergence') and config.early_divergence:
        t_early_end = int(0.33 * config.T)
        ax1.axvline(t_early_end, color='orange', linestyle='--', label=f'early_div_end')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Min Pairwise Distance D(t)')
    ax1.set_title('Pairwise Divergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle: R_all(t) - Covering radius
    ax2 = axes[1]
    ax2.plot(t_axis, R_mean, 'g-', linewidth=2, label='R_all(t)')
    ax2.fill_between(t_axis, R_mean - R_std, R_mean + R_std, alpha=0.3, color='green')
    sigma_floor = np.array([np.sqrt(t + 1) * config.noise_x for t in t_axis])
    ax2.plot(t_axis, sigma_floor, 'k--', alpha=0.5, label='σ_floor (noise)')
    ax2.axvline(config.t_gate, color='r', linestyle='--')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Covering Radius R_all(t)')
    ax2.set_title('Minimum Covering Radius')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Right: Example trajectories (last episode)
    ax3 = axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, config.K))
    
    # PCA if d > 2
    if config.d > 2:
        from sklearn.decomposition import PCA
        all_points = np.vstack([mode_trajectories[k] for k in range(config.K)])
        pca = PCA(n_components=2)
        pca.fit(all_points)
        for k in range(config.K):
            traj_2d = pca.transform(mode_trajectories[k])
            ax3.plot(traj_2d[:, 0], traj_2d[:, 1], color=colors[k], linewidth=2, label=f'Mode {k}')
            ax3.scatter([traj_2d[0, 0]], [traj_2d[0, 1]], color=colors[k], s=100, marker='o', edgecolors='black')
            ax3.scatter([traj_2d[-1, 0]], [traj_2d[-1, 1]], color=colors[k], s=100, marker='*', edgecolors='black')
        ax3.set_title(f'Counterfactual Trajectories (PCA)')
    else:
        for k in range(config.K):
            ax3.plot(mode_trajectories[k][:, 0], mode_trajectories[k][:, 1], 
                    color=colors[k], linewidth=2, label=f'Mode {k}')
            ax3.scatter([mode_trajectories[k][0, 0]], [mode_trajectories[k][0, 1]], 
                       color=colors[k], s=100, marker='o', edgecolors='black')
            ax3.scatter([mode_trajectories[k][-1, 0]], [mode_trajectories[k][-1, 1]], 
                       color=colors[k], s=100, marker='*', edgecolors='black')
        ax3.set_title('Counterfactual Trajectories')
    ax3.set_xlabel('x[0]' if config.d <= 2 else 'PC1')
    ax3.set_ylabel('x[1]' if config.d <= 2 else 'PC2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fork Separability Test (Fork Index: {fork_index_mean:.2f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fork_separability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved fork separability plot to {output_dir}/fork_separability.png")
    
    return {
        'fork_index_mean': float(fork_index_mean),
        'fork_index_std': float(fork_index_std),
        'D_final': float(D_mean[-1]),
        'R_final': float(R_mean[-1]),
        'ports_required': bool(fork_index_mean > 1.5)
    }


def commitment_regret_test(actor, env, config, mode_centroids, output_dir, n_episodes=10):
    """
    Test 2: Commitment Regret Gap - Does using shared z increase volume?
    
    Measures how much extra tube volume is required if you force the actor
    to use the same z across all modes.
    """
    from environments.cmg_env import rollout_with_forced_mode
    
    print("\n" + "="*60)
    print("TOPOLOGICAL TEST 2: Commitment Regret Gap")
    print("="*60)
    
    if len(mode_centroids) < 2:
        print("  Need at least 2 mode centroids")
        return None
    
    all_regrets = []
    per_mode_volumes = {k: [] for k in mode_centroids}
    shared_volumes = []
    
    # Use mode 3 centroid (or first available) as shared z
    shared_mode = max(mode_centroids.keys())  # Usually dominant mode
    z_shared = mode_centroids[shared_mode].clone()
    
    for ep in range(n_episodes):
        env.reset()
        x0 = env.x.copy()
        goals = env.params.g.copy()  # (K, d) array
        s0 = torch.tensor(env._get_obs(), dtype=torch.float32)
        
        # Per-mode optimal volumes
        mode_vols = {}
        for k in mode_centroids:
            # Reset environment
            env.reset()
            env.x = x0.copy()
            env.params.g = goals.copy()
            
            # Get trajectory for this mode
            record = rollout_with_forced_mode(env, k, "goal_seeking")
            traj = torch.tensor(record['x'][1:], dtype=torch.float32)
            traj_delta = traj - s0[:config.d]
            
            # Encode to get mode-specific z
            with torch.no_grad():
                z_k = actor.encode(traj_delta.reshape(-1))
                mu, sigma = actor.get_tube(s0, z_k)
            
            vol_k = sigma.mean().item()
            mode_vols[k] = vol_k
            per_mode_volumes[k].append(vol_k)
        
        # Shared z volume (evaluate on random mode's trajectory)
        env.reset()
        env.x = x0.copy()
        env.params.g = goals.copy()
        
        with torch.no_grad():
            mu_shared, sigma_shared = actor.get_tube(s0, z_shared)
        
        vol_shared = sigma_shared.mean().item()
        shared_volumes.append(vol_shared)
        
        # Compute regret
        avg_mode_vol = np.mean(list(mode_vols.values()))
        regret = vol_shared / (avg_mode_vol + 1e-8)
        all_regrets.append(regret)
    
    regret_mean = np.mean(all_regrets)
    regret_std = np.std(all_regrets)
    
    print(f"\n  Volume Statistics:")
    print(f"  Mode | Mean Vol | Std Vol")
    print(f"  -----|----------|--------")
    for k in sorted(per_mode_volumes.keys()):
        vols = per_mode_volumes[k]
        print(f"  {k:4d} | {np.mean(vols):.4f}   | {np.std(vols):.4f}")
    print(f"  Shared | {np.mean(shared_volumes):.4f}   | {np.std(shared_volumes):.4f}")
    
    print(f"\n  Regret Ratio: {regret_mean:.2f} ± {regret_std:.2f}")
    if regret_mean < 1.2:
        print(f"    → Regret ≈ 1: Environment does NOT require ports")
    elif regret_mean < 2.0:
        print(f"    → Regret moderate: Some volume penalty for sharing")
    else:
        print(f"    → Regret >> 1: Environment REQUIRES distinct ports")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Volume comparison bar chart
    ax1 = axes[0]
    modes = sorted(per_mode_volumes.keys())
    x_pos = np.arange(len(modes) + 1)
    vol_means = [np.mean(per_mode_volumes[k]) for k in modes] + [np.mean(shared_volumes)]
    vol_stds = [np.std(per_mode_volumes[k]) for k in modes] + [np.std(shared_volumes)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    bar_colors = list(colors) + ['gray']
    
    bars = ax1.bar(x_pos, vol_means, yerr=vol_stds, capsize=5, color=bar_colors, alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Mode {k}' for k in modes] + ['Shared'])
    ax1.set_ylabel('Volume')
    ax1.set_title('Per-Mode vs Shared Volume')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Regret histogram
    ax2 = axes[1]
    ax2.hist(all_regrets, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Regret=1 (no penalty)')
    ax2.axvline(regret_mean, color='red', linestyle='-', linewidth=2, label=f'Mean={regret_mean:.2f}')
    ax2.set_xlabel('Regret (V_shared / V_avg)')
    ax2.set_ylabel('Count')
    ax2.set_title('Commitment Regret Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Commitment Regret Gap (Regret: {regret_mean:.2f})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/commitment_regret.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved commitment regret plot to {output_dir}/commitment_regret.png")
    
    return {
        'regret_mean': float(regret_mean),
        'regret_std': float(regret_std),
        'shared_mode': int(shared_mode),
        'ports_required': bool(regret_mean > 1.5)
    }


def gating_irreversibility_test(env, config, output_dir, n_episodes=10):
    """
    Test 3: Gating Irreversibility - Can late control undo mode choice?
    
    Tests if switching mode after the gate causes large terminal error.
    """
    from environments.cmg_env import rollout_with_forced_mode
    
    print("\n" + "="*60)
    print("TOPOLOGICAL TEST 3: Gating Irreversibility")
    print("="*60)
    
    # Test intervention at different times
    intervention_times = list(range(0, config.T, max(1, config.T // 10)))
    
    all_results = {t: [] for t in intervention_times}
    
    for ep in range(n_episodes):
        env.reset()
        x0 = env.x.copy()
        goals = env.params.g.copy()  # (K, d) array
        
        # Pick two modes to test switching between
        k_original = 0
        k_switch = min(1, config.K - 1)
        
        for t_switch in intervention_times:
            # Reset environment
            env.reset()
            env.x = x0.copy()
            env.params.g = goals.copy()
            
            # Run with original mode until t_switch
            env.k = k_original
            x = x0.copy()
            
            for t in range(config.T):
                if t == t_switch:
                    # Switch mode
                    env.k = k_switch
                
                # Goal-seeking action
                goal = goals[env.k]
                action = np.clip(goal - env.x, -1.0, 1.0).astype(np.float32)
                obs, _, _, info = env.step(action)
                
                # Keep the switched mode
                if t >= t_switch:
                    env.k = k_switch
            
            # Terminal distance to new goal
            terminal_dist = np.linalg.norm(env.x - goals[k_switch])  # goals is (K, d)
            all_results[t_switch].append(terminal_dist)
    
    # Compute statistics
    t_means = [np.mean(all_results[t]) for t in intervention_times]
    t_stds = [np.std(all_results[t]) for t in intervention_times]
    
    print(f"\n  Terminal Distance vs Intervention Time:")
    print(f"  t_switch | Mean Dist | Std Dist")
    print(f"  ---------|-----------|--------")
    for i, t in enumerate(intervention_times):
        print(f"  {t:8d} | {t_means[i]:.4f}    | {t_stds[i]:.4f}")
    
    # Find the "knee" - where distance starts increasing sharply
    knee_idx = 0
    for i in range(1, len(t_means)):
        if t_means[i] > t_means[0] * 1.5:  # 50% increase threshold
            knee_idx = i
            break
    
    t_knee = intervention_times[knee_idx] if knee_idx < len(intervention_times) else intervention_times[-1]
    
    print(f"\n  Irreversibility Knee: t ≈ {t_knee}")
    if t_knee <= config.t_gate:
        print(f"    → Irreversibility at/before gate: Environment REQUIRES early commitment")
    elif t_knee < config.T * 0.5:
        print(f"    → Moderate irreversibility: Some commitment pressure")
    else:
        print(f"    → Late irreversibility: Environment allows late switching")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Terminal distance vs intervention time
    ax1 = axes[0]
    ax1.errorbar(intervention_times, t_means, yerr=t_stds, fmt='o-', capsize=5, 
                linewidth=2, markersize=8, color='steelblue')
    ax1.axvline(config.t_gate, color='red', linestyle='--', linewidth=2, label=f't_gate={config.t_gate}')
    ax1.axvline(t_knee, color='green', linestyle=':', linewidth=2, label=f'knee≈{t_knee}')
    if hasattr(config, 'early_divergence') and config.early_divergence:
        t_early_end = int(0.33 * config.T)
        ax1.axvline(t_early_end, color='orange', linestyle='--', label=f'early_div_end')
    ax1.axhline(t_means[0], color='gray', linestyle=':', alpha=0.5, label=f'Best (t=0)')
    ax1.set_xlabel('Intervention Time (t_switch)')
    ax1.set_ylabel('Terminal Distance to New Goal')
    ax1.set_title('Mode Switch Penalty vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Normalized penalty
    ax2 = axes[1]
    normalized = [t_means[i] / (t_means[0] + 1e-8) for i in range(len(t_means))]
    ax2.plot(intervention_times, normalized, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.axhline(1.5, color='red', linestyle=':', alpha=0.5, label='50% penalty threshold')
    ax2.axvline(config.t_gate, color='red', linestyle='--', linewidth=2)
    ax2.axvline(t_knee, color='green', linestyle=':', linewidth=2)
    ax2.set_xlabel('Intervention Time (t_switch)')
    ax2.set_ylabel('Normalized Penalty (dist / dist_best)')
    ax2.set_title('Irreversibility Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Gating Irreversibility Test (Knee at t={t_knee})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gating_irreversibility.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved gating irreversibility plot to {output_dir}/gating_irreversibility.png")
    
    return {
        't_knee': int(t_knee),
        'final_dist_early': float(t_means[0]),
        'final_dist_late': float(t_means[-1]),
        'irreversibility_ratio': float(t_means[-1] / (t_means[0] + 1e-8)),
        'requires_early_commitment': bool(t_knee <= config.t_gate)
    }


def run_topological_necessity_tests(actor, env, config, mode_centroids, output_dir):
    """Run all topological necessity tests."""
    
    print("\n" + "="*60)
    print("TOPOLOGICAL NECESSITY TEST SUITE")
    print("="*60)
    print("Testing if the environment REQUIRES distinct commitment ports")
    
    results = {}
    
    # Test 1: Fork Separability
    results['fork_separability'] = fork_separability_test(env, config, output_dir)
    
    # Test 2: Commitment Regret Gap
    results['commitment_regret'] = commitment_regret_test(actor, env, config, mode_centroids, output_dir)
    
    # Test 3: Gating Irreversibility
    results['gating_irreversibility'] = gating_irreversibility_test(env, config, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("TOPOLOGICAL NECESSITY SUMMARY")
    print("="*60)
    
    ports_required_votes = 0
    total_tests = 0
    
    if results['fork_separability']:
        fork_idx = results['fork_separability']['fork_index_mean']
        print(f"  Fork Index: {fork_idx:.2f}", end=" → ")
        if fork_idx > 1.5:
            print("PORTS REQUIRED")
            ports_required_votes += 1
        else:
            print("Unified OK")
        total_tests += 1
    
    if results['commitment_regret']:
        regret = results['commitment_regret']['regret_mean']
        print(f"  Regret Ratio: {regret:.2f}", end=" → ")
        if regret > 1.5:
            print("PORTS REQUIRED")
            ports_required_votes += 1
        else:
            print("Unified OK")
        total_tests += 1
    
    if results['gating_irreversibility']:
        t_knee = results['gating_irreversibility']['t_knee']
        print(f"  Irreversibility Knee: t={t_knee}", end=" → ")
        if t_knee <= config.t_gate:
            print("EARLY COMMITMENT REQUIRED")
            ports_required_votes += 1
        else:
            print("Late switching OK")
        total_tests += 1
    
    print(f"\n  Verdict: {ports_required_votes}/{total_tests} tests indicate ports are REQUIRED")
    if ports_required_votes >= 2:
        print("  → Environment topology DEMANDS distinct commitment ports")
    elif ports_required_votes == 1:
        print("  → Environment has MODERATE commitment pressure")
    else:
        print("  → Environment allows UNIFIED commitment (rational collapse OK)")
    
    return results


def run_test(d=4, K=2, n_train_epochs=1000, n_test_episodes=20, z_dim=None, T=16, t_gate=2, warmup_epochs=1000, s0_emb_dim=8, test_zeroing=False):
    """
    Run the knotv2 test with configurable parameters.
    
    Args:
        d: State dimensionality
        K: Number of modes/modalities
        n_train_epochs: Number of training iterations (total, including warmup)
        n_test_episodes: Number of test episodes
        z_dim: Latent dimension (defaults to d if not specified)
        T: Trajectory length
        t_gate: Gate time
        warmup_epochs: Number of encoder-only warmup epochs (default: 1000)
        s0_emb_dim: Observation embedding dimension (architectural bottleneck, default: 8)
        test_zeroing: Whether to run zeroing diagnostic test (default: False)
    """
    if z_dim is None:
        z_dim = d
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "runs" / f"knotv2_test_d{d}_K{K}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Configuration: d={d}, K={K}, z_dim={z_dim}, train_epochs={n_train_epochs}, warmup_epochs={warmup_epochs}, test_episodes={n_test_episodes}")
    
    # 1. Setup Environment
    config = CMGConfig(d=d, K=K, T=T, t_gate=t_gate, early_divergence=True)
    env = CMGEnv(config)
    
    # 2. Setup Actor (The Geometric Knot variant)
    actor = GeometricKnotActor(
        obs_dim=env.obs_dim, 
        z_dim=z_dim, 
        pred_dim=env.state_dim, # We predict the full state trajectory
        T=config.T,
        alpha_vol=0.01,  # Gentle pressure to shrink
        s0_emb_dim=s0_emb_dim  # Architectural bottleneck dimension
    )
    
    # Log parameter ratios
    param_info = actor.count_parameters()
    print(f"\nArchitectural Bottleneck:")
    print(f"  Obs embedding params: {param_info['obs_params']}")
    print(f"  Z embedding params: {param_info['z_params']}")
    print(f"  Z/Obs ratio: {param_info['z_to_obs_ratio']:.2f}x")
    
    # Stage 1: Z-Warmup - Encoder Only (with Smart Gate)
    print(f"\nStage 1: Z-Warmup - Encoder Only Training (target: {warmup_epochs} epochs, min silhouette: 0.1)...")
    print(f"  Using SupCon loss with lower learning rate (1e-4) and dynamic loss weighting")
    print(f"  Changes Applied:")
    print(f"    - Encoder uses FLATTENED trajectory (T*d) to preserve temporal signature")
    print(f"    - Tanh activation bounds z to [-1, 1] (prevents Z_std explosion)")
    print(f"    - Removed diversity loss (was causing instability)")
    print(f"  Dynamic Weighting Strategy:")
    print(f"    Epochs 0-1000:  recon_weight=0.05, intent_weight=1.0 (force clustering)")
    print(f"    Epochs 1000+:   recon_weight anneals 0.05->0.5, intent_weight=1.0->0.7")
    encoder_optimizer = optim.Adam(actor.encoder_mlp.parameters(), lr=1e-4)  # Lower LR
    
    # Track warmup metrics
    warmup_z_samples = []
    warmup_modes = []
    
    # Batch buffer for contrastive loss
    traj_delta_buffer = []
    labels_buffer = []
    batch_size = 64  # Larger batch for better contrastive learning
    
    batch_idx = 0
    max_warmup_batches = warmup_epochs  # Each "epoch" is now a batch
    max_extension = 2000  # Allow extension up to 2000 more batches
    warmup_complete = False
    first_batch = True
    
    while batch_idx < (max_warmup_batches + max_extension) and not warmup_complete:
        # Collect a full batch of samples
        while len(traj_delta_buffer) < batch_size:
            data = generate_episode(env, policy_mode="goal_seeking")
            s0 = torch.tensor(data["obs"][0], dtype=torch.float32)
            traj = torch.tensor(data["x"][1:], dtype=torch.float32)
            traj_delta = traj - s0[:config.d]  # (T, d)
            
            # Sanity check on first sample
            if first_batch and len(traj_delta_buffer) == 0:
                traj_delta_max = traj_delta.abs().max().item()
                print(f"  Delta-Traj Sanity Check: max(|traj_delta|) = {traj_delta_max:.4f} (should be small, < 10.0)")
                if traj_delta_max > 20.0:
                    print(f"  ⚠ WARNING: Large traj_delta values detected!")
            
            # Get mode
            k_array = data.get('k', np.array([0]))
            mode = int(k_array[-1]) if isinstance(k_array, np.ndarray) else int(k_array)
            
            traj_delta_buffer.append(traj_delta)
            labels_buffer.append(mode)
        
        # Stack batch for training
        traj_batch = torch.stack(traj_delta_buffer[:batch_size])  # (batch_size, T, d)
        labels_batch = torch.tensor(labels_buffer[:batch_size], dtype=torch.long)
        traj_flat_batch = traj_batch.reshape(batch_size, -1)  # (batch_size, T*d)
        
        # Train with batch contrastive loss
        metrics = actor.train_encoder_batch(
            traj_flat_batch, None, labels_batch, encoder_optimizer, epoch=batch_idx
        )
        first_batch = False
        
        # Collect z samples for monitoring
        with torch.no_grad():
            z_batch_monitor = actor.encode(traj_flat_batch)
        for i in range(batch_size):
            warmup_z_samples.append(z_batch_monitor[i].cpu().numpy())
            warmup_modes.append(labels_buffer[i])
        
        # Clear buffer
        traj_delta_buffer = []
        labels_buffer = []
        
        # Print progress
        if batch_idx % max(1, warmup_epochs // 5) == 0 or batch_idx == warmup_epochs - 1:
            recon_w = metrics.get('recon_weight', 0.0)
            intent_w = metrics.get('intent_weight', 0.0)
            z_std_val = metrics.get('z_std', 0.0)
            intent_val = metrics.get('intent', 0.0)
            print(f"Warmup Batch {batch_idx} | Loss: {metrics['loss']:.4f} | Recon: {metrics['recon']:.4f} (w={recon_w:.2f}) | Intent: {intent_val:.4f} (w={intent_w:.2f}) | Z_std: {z_std_val:.4f}")
            if batch_idx < 1000:
                print(f"    → Clustering phase: Very low recon ({recon_w:.2f}), High intent ({intent_w:.2f})")
            else:
                print(f"    → Transition phase: Recon increasing ({recon_w:.2f}), Intent stable ({intent_w:.2f})")
        
        # Smart Warmup Gate: Check clustering quality after target epochs
        if batch_idx >= warmup_epochs and batch_idx % 100 == 0:
            if len(warmup_z_samples) > 0 and len(set(warmup_modes)) >= 2:
                recent_z = np.array(warmup_z_samples[-batch_size*10:])  # Last 10 batches
                recent_modes = warmup_modes[-batch_size*10:]
                recent_clustering = compute_clustering_metrics(recent_z, recent_modes)
                if recent_clustering["silhouette_score"] is not None:
                    if recent_clustering['silhouette_score'] >= 0.1:
                        print(f"\n✓ Warmup complete! Silhouette: {recent_clustering['silhouette_score']:.4f} >= 0.1")
                        warmup_complete = True
                    elif batch_idx % 500 == 0:
                        print(f"  Warmup extended: Silhouette {recent_clustering['silhouette_score']:.4f} < 0.1")
        
        batch_idx += 1
    
    # Final warmup clustering check
    warmup_clustering = None
    if len(warmup_z_samples) > 0 and len(set(warmup_modes)) >= 2:
        warmup_z_array = np.array(warmup_z_samples)
        warmup_clustering = compute_clustering_metrics(warmup_z_array, warmup_modes)
        if warmup_clustering["silhouette_score"] is not None:
            print(f"\nFinal Warmup Silhouette Score: {warmup_clustering['silhouette_score']:.4f}")
            if warmup_clustering['silhouette_score'] < 0.1:
                print("  WARNING: Silhouette score < 0.1 - representation may be poor")
            else:
                print("  ✓ Encoder warmup successful - silhouette score >= 0.1")
    
    actual_warmup_epochs = batch_idx  # Count batches, not samples
    
    # Stage 2: Freeze encoder and train actor/tube networks
    stage2_epochs = n_train_epochs - actual_warmup_epochs
    print(f"\nStage 2: Actor/Tube Training - Encoder Frozen ({stage2_epochs} epochs)...")
    print(f"  Volume annealing: alpha_vol starts at 0.01, increases to 0.1 after 1000 epochs")
    print(f"  Mode-balanced sampling: Ensuring all {config.K} modes are represented")
    actor.encoder_mlp.requires_grad_(False)  # Freeze encoder
    actor_optimizer = optim.Adam(
        list(actor.mu_net.parameters()) + 
        list(actor.sigma_net.parameters()) + 
        list(actor.obs_proj.parameters()) +
        list(actor.z_encoder.parameters()),
        lr=1e-3
    )
    
    # TRUE STRATIFIED SAMPLING: Collect trajectories by mode
    mode_buffers = {k: [] for k in range(config.K)}
    buffer_size_per_mode = 100  # Keep 100 examples per mode for diversity
    min_per_mode = 20  # Minimum required before training
    
    # Pre-fill buffers with required minimum from each mode
    print("  Pre-filling mode buffers (stratified sampling)...")
    attempts = 0
    max_attempts = 2000  # Allow more attempts to find rare modes
    while min(len(buf) for buf in mode_buffers.values()) < min_per_mode and attempts < max_attempts:
        data = generate_episode(env, policy_mode="goal_seeking")
        k_array = data.get('k', np.array([0]))
        mode = int(k_array[-1]) if isinstance(k_array, np.ndarray) else int(k_array)
        if len(mode_buffers[mode]) < buffer_size_per_mode:
            mode_buffers[mode].append(data)
        attempts += 1
    print(f"  Buffer sizes: {[len(mode_buffers[k]) for k in range(config.K)]}")
    
    # Check if all modes have minimum examples
    modes_with_min = sum(1 for k in range(config.K) if len(mode_buffers[k]) >= min_per_mode)
    if modes_with_min < config.K:
        print(f"  ⚠ WARNING: Only {modes_with_min}/{config.K} modes have >={min_per_mode} examples")
    
    for epoch in range(actual_warmup_epochs, n_train_epochs):
        stage2_epoch = epoch - actual_warmup_epochs
        
        # Volume annealing: Start low, increase after 1000 epochs
        if stage2_epoch < 1000:
            actor.alpha_vol = 0.01  # Very gentle volume pressure
        else:
            # Anneal from 0.01 to 0.1 over next 500 epochs
            progress = min(1.0, (stage2_epoch - 1000) / 500.0)
            actor.alpha_vol = 0.01 + 0.09 * progress
        
        # Mode-balanced sampling: Pick random mode, then sample from its buffer
        mode = epoch % config.K  # Cycle through modes
        if mode_buffers[mode]:
            data = mode_buffers[mode][np.random.randint(len(mode_buffers[mode]))]
        else:
            # Fallback: generate new episode
            data = generate_episode(env, policy_mode="goal_seeking")
        
        # Also add new episodes to buffers
        if epoch % 10 == 0:
            new_data = generate_episode(env, policy_mode="goal_seeking")
            k_array = new_data.get('k', np.array([0]))
            new_mode = int(k_array[-1]) if isinstance(k_array, np.ndarray) else int(k_array)
            if len(mode_buffers[new_mode]) < buffer_size_per_mode:
                mode_buffers[new_mode].append(new_data)
            elif np.random.random() < 0.1:  # Replace with 10% probability
                idx = np.random.randint(len(mode_buffers[new_mode]))
                mode_buffers[new_mode][idx] = new_data
        
        s0 = torch.tensor(data["obs"][0], dtype=torch.float32)
        traj = torch.tensor(data["x"][1:], dtype=torch.float32)
        
        # Use frozen encoder with FLATTENED trajectory delta
        traj_delta = traj - s0[:config.d]
        with torch.no_grad():
            traj_flat = traj_delta.reshape(-1)
            z = actor.encode(traj_flat)
        
        metrics = actor.train_step(s0, traj, z, actor_optimizer)
        if epoch % max(1, stage2_epochs // 5) == 0 or epoch == n_train_epochs - 1:
            dir_loss = metrics.get('dir_loss', 0)
            end_err = metrics.get('end_err', 0)
            print(f"Actor Epoch {epoch} | Leak: {metrics['leak']:.4f} | Vol: {metrics['vol']:.3f} | Dir: {dir_loss:.3f} | End: {end_err:.3f} | Mode: {mode}")

    print(f"\nStep 2: Testing Rational Commitment (Zero-Shot Intent) ({n_test_episodes} episodes)...")
    # Run multiple episodes to see distribution of commitments
    all_z_stars = []
    all_rewards = []
    all_modes = []  # Track which mode each episode reached
    
    for episode in range(n_test_episodes):
        obs = env.reset()
        s0 = torch.tensor(obs, dtype=torch.float32)

        # At test time, don't provide trajectory hint - let CEM explore freely
        # This forces the actor to discover intent through the volume minimization objective
        z_star = actor.select_z_geometric(s0, trajectory_delta=None)
        all_z_stars.append(z_star.clone())
        
        # Visualize the commitment
        mu, sigma = actor.get_tube(s0, z_star)
        mu_np = mu.detach().numpy()[0]
        
        # Step 3: Execution (Binding)
        curr_obs = obs
        total_reward = 0
        
        for t in range(config.T):
            action = mu_np[t] - curr_obs[:config.d]
            curr_obs, reward, done, info = env.step(action)
            total_reward += reward
        
        all_rewards.append(total_reward)
        all_modes.append(info['k'])  # Store the mode reached
        if episode == 0 or (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{n_test_episodes} | Reward: {total_reward:.2f} | Mode: {info['k']}")
    
    # Zeroing Diagnostic Test
    if test_zeroing:
        print(f"\nStep 2.5: Zeroing Diagnostic Test (Testing z-dependency)...")
        zeroing_rewards = []
        zeroing_modes = []
        
        for episode in range(min(10, n_test_episodes)):
            obs = env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32)
            
            # Select z with obs embedding zeroed out (no trajectory hint)
            z_star = actor.select_z_geometric(s0, trajectory_delta=None, force_z_only=True)
            mu, sigma = actor.get_tube(s0, z_star, force_z_only=True)
            mu_np = mu.detach().numpy()[0]
            
            curr_obs = obs
            total_reward = 0
            
            for t in range(config.T):
                action = mu_np[t] - curr_obs[:config.d]
                curr_obs, reward, done, info = env.step(action)
                total_reward += reward
            
            zeroing_rewards.append(total_reward)
            zeroing_modes.append(info['k'])
        
        avg_zeroing_reward = np.mean(zeroing_rewards)
        print(f"  Zeroing Test Results:")
        print(f"    Average Reward (z-only): {avg_zeroing_reward:.2f}")
        print(f"    Normal Average Reward: {np.mean(all_rewards[:len(zeroing_rewards)]):.2f}")
        if avg_zeroing_reward > np.mean(all_rewards[:len(zeroing_rewards)]) * 0.5:
            print(f"    ✓ Agent successfully uses z for intent (z-only reward > 50% of normal)")
        else:
            print(f"    ⚠ Agent may be relying on obs leakage (z-only reward < 50% of normal)")
    
    # ============================================================
    # COMMITMENT TOPOLOGY DIAGNOSTIC
    # Tests if the latent space actually captures mode structure
    # ============================================================
    print(f"\n" + "="*60)
    print("Commitment Topology Diagnostic")
    print("="*60)
    
    # 1. ENCODER TEST: Generate trajectories and check if encoder clusters by mode
    print("\n1. Encoder Mode Prediction Test:")
    encoder_z_samples = []
    encoder_modes = []
    n_samples_per_mode = 20
    
    for target_mode in range(config.K):
        samples_for_mode = 0
        attempts = 0
        while samples_for_mode < n_samples_per_mode and attempts < 200:
            data = generate_episode(env, policy_mode="goal_seeking")
            k_array = data.get('k', np.array([0]))
            mode = int(k_array[-1]) if isinstance(k_array, np.ndarray) else int(k_array)
            if mode == target_mode:
                traj = torch.tensor(data["x"][1:], dtype=torch.float32)
                s0_sample = torch.tensor(data["obs"][0], dtype=torch.float32)
                traj_delta = traj - s0_sample[:config.d]
                with torch.no_grad():
                    z = actor.encode(traj_delta.reshape(-1))
                encoder_z_samples.append(z.cpu().numpy())
                encoder_modes.append(mode)
                samples_for_mode += 1
            attempts += 1
    
    encoder_z_array = np.array(encoder_z_samples)
    encoder_clustering = compute_clustering_metrics(encoder_z_array, encoder_modes)
    print(f"   Encoder Silhouette Score: {encoder_clustering['silhouette_score']:.4f}")
    print(f"   (This measures if the ENCODER learned mode separation)")
    
    # 2. CAUSAL COMMITMENT TEST: Pick z from each mode's cluster, execute, check outcome
    print("\n2. Causal Commitment Test (z → mode fidelity):")
    
    # Compute mode centroids from encoder samples
    mode_centroids = {}
    for mode in range(config.K):
        mask = np.array(encoder_modes) == mode
        if np.sum(mask) > 0:
            centroid = encoder_z_array[mask].mean(axis=0)
            # L2 normalize to stay on unit sphere
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            mode_centroids[mode] = torch.tensor(centroid, dtype=torch.float32)
    
    commitment_results = {mode: {"intended": 0, "reached": []} for mode in mode_centroids}
    n_trials_per_mode = 10
    
    for intended_mode, z_centroid in mode_centroids.items():
        for _ in range(n_trials_per_mode):
            obs = env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32)
            
            # Force use of the mode's centroid z (with small noise)
            z_test = z_centroid + torch.randn_like(z_centroid) * 0.1
            z_test = torch.nn.functional.normalize(z_test.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            mu, sigma = actor.get_tube(s0, z_test)
            mu_np = mu.detach().numpy()[0]
            
            curr_obs = obs
            for t in range(config.T):
                action = mu_np[t] - curr_obs[:config.d]
                curr_obs, _, done, info = env.step(action)
            
            commitment_results[intended_mode]["intended"] += 1
            commitment_results[intended_mode]["reached"].append(info['k'])
    
    # Compute commitment fidelity per mode
    total_correct = 0
    total_trials = 0
    print(f"   Mode | Intended | Reached Distribution | Fidelity")
    print(f"   -----|----------|---------------------|----------")
    for mode in sorted(commitment_results.keys()):
        reached = commitment_results[mode]["reached"]
        intended = commitment_results[mode]["intended"]
        correct = sum(1 for r in reached if r == mode)
        fidelity = correct / intended if intended > 0 else 0
        total_correct += correct
        total_trials += intended
        
        # Count distribution of reached modes
        from collections import Counter
        reached_counts = Counter(reached)
        dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(reached_counts.items())])
        print(f"   {mode:4d} | {intended:8d} | {dist_str:19s} | {fidelity:.1%}")
    
    overall_fidelity = total_correct / total_trials if total_trials > 0 else 0
    print(f"\n   Overall Commitment Fidelity: {overall_fidelity:.1%}")
    print(f"   (This measures if picking z from mode cluster → reaches that mode)")
    
    if overall_fidelity > 0.5:
        print(f"   ✓ Actor has learned commitment topology (fidelity > 50%)")
    elif overall_fidelity > 1.0 / config.K:
        print(f"   ~ Actor shows some commitment structure (fidelity > random {1.0/config.K:.1%})")
    else:
        print(f"   ⚠ Actor has NOT learned commitment topology (fidelity ≈ random)")
    
    # Use the last episode for detailed plotting
    obs = env.reset()
    s0 = torch.tensor(obs, dtype=torch.float32)
    z_star = actor.select_z_geometric(s0, trajectory_delta=None)
    mu, sigma = actor.get_tube(s0, z_star)
    mu_np = mu.detach().numpy()[0]
    
    curr_obs = obs
    total_reward = 0
    actual_traj = []
    
    for t in range(config.T):
        action = mu_np[t] - curr_obs[:config.d]
        curr_obs, reward, done, info = env.step(action)
        actual_traj.append(info['x'])
        total_reward += reward

    print(f"\nFinal Episode Reward: {total_reward:.2f}")
    print(f"Reached Goal: {info['k']} (Dist to goal: {np.linalg.norm(info['x'] - info['goal']):.4f})")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")

    # Compute clustering metrics
    z_array = np.array([z.detach().cpu().numpy() for z in all_z_stars])
    clustering_metrics = compute_clustering_metrics(z_array, all_modes)
    
    print("\n" + "="*60)
    print("Clustering Metrics (Latent Space)")
    print("="*60)
    print(f"Mode Coverage: {len(set(all_modes))}/{config.K} modes reached")
    if clustering_metrics["silhouette_score"] is not None:
        print(f"Silhouette Score:        {clustering_metrics['silhouette_score']:.4f} (range: -1 to 1, higher is better)")
        print(f"Calinski-Harabasz Index: {clustering_metrics['calinski_harabasz_index']:.2f} (higher is better)")
        print(f"Davies-Bouldin Index:    {clustering_metrics['davies_bouldin_index']:.4f} (lower is better)")
        print(f"Separability Ratio:      {clustering_metrics['separability_ratio']:.4f} (inter/intra cluster distance)")
        print(f"  Inter-cluster dist:    {clustering_metrics['inter_cluster_distance']:.4f}")
        print(f"  Intra-cluster dist:    {clustering_metrics['intra_cluster_distance']:.4f}")
    else:
        print("  (Need at least 2 modes for clustering metrics)")

    # Save results - focus on latent space clustering
    plot_results(actor, s0, actual_traj, z_star, all_z_stars, all_modes, output_dir)
    
    # Run Geometric Diagnostic Suite
    # Convert mode_centroids dict values to proper format
    mode_centroids_torch = {k: v.clone() for k, v in mode_centroids.items()}
    diagnostic_results = run_geometric_diagnostics(actor, env, config, mode_centroids_torch, s0, output_dir)
    
    # Run Topological Necessity Tests
    topological_results = run_topological_necessity_tests(actor, env, config, mode_centroids_torch, output_dir)
    diagnostic_results['topological'] = topological_results
    
    # Save model
    model_path = output_dir / "model.pt"
    torch.save(actor.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Save config and summary
    summary = {
        "config": {
            "d": config.d,
            "K": config.K,
            "T": config.T,
            "t_gate": config.t_gate,
            "early_divergence": config.early_divergence,
        },
        "actor": {
            "obs_dim": env.obs_dim,
            "z_dim": z_dim,
            "pred_dim": env.state_dim,
            "alpha_vol": 0.5,
            "s0_emb_dim": s0_emb_dim,
            "parameter_ratios": param_info,
        },
        "training": {
            "n_train_epochs": n_train_epochs,
            "warmup_epochs": warmup_epochs,
            "actual_warmup_epochs": actual_warmup_epochs,
            "n_test_episodes": n_test_episodes,
        },
        "warmup_metrics": warmup_clustering,
        "results": {
            "final_reward": float(total_reward),
            "reached_goal": int(info['k']),
            "goal_distance": float(np.linalg.norm(info['x'] - info['goal'])),
            "committed_path_end": mu_np[-1][:2].tolist(),
            "n_test_episodes": n_test_episodes,
            "avg_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mode_distribution": {int(mode): int(sum(1 for m in all_modes if m == mode)) for mode in set(all_modes)},
            "mode_coverage": len(set(all_modes)),  # Number of unique modes reached
            "clustering_metrics": clustering_metrics,
        },
        "geometric_diagnostics": diagnostic_results
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GeometricKnotActor with configurable parameters")
    parser.add_argument("--d", type=int, default=4, help="State dimensionality (default: 4)")
    parser.add_argument("--K", type=int, default=2, help="Number of modes/modalities (default: 2)")
    parser.add_argument("--z-dim", type=int, default=None, dest="z_dim", 
                       help="Latent dimension (defaults to d if not specified)")
    parser.add_argument("--train-epochs", type=int, default=500, dest="n_train_epochs",
                       help="Number of training epochs (default: 500)")
    parser.add_argument("--test-episodes", type=int, default=20, dest="n_test_episodes",
                       help="Number of test episodes (default: 20)")
    parser.add_argument("--T", type=int, default=16, help="Trajectory length (default: 16)")
    parser.add_argument("--t-gate", type=int, default=2, dest="t_gate",
                       help="Gate time (default: 2)")
    parser.add_argument("--warmup-epochs", type=int, default=500, dest="warmup_epochs",
                       help="Number of encoder-only warmup epochs (default: 500)")
    parser.add_argument("--s0-emb-dim", type=int, default=8, dest="s0_emb_dim",
                       help="Observation embedding dimension (architectural bottleneck, default: 8)")
    parser.add_argument("--test-zeroing", action="store_true", dest="test_zeroing",
                       help="Run zeroing diagnostic test to verify z-dependency")
    
    args = parser.parse_args()
    
    run_test(
        d=args.d,
        K=args.K,
        n_train_epochs=args.n_train_epochs,
        n_test_episodes=args.n_test_episodes,
        z_dim=args.z_dim,
        T=args.T,
        t_gate=args.t_gate,
        warmup_epochs=args.warmup_epochs,
        s0_emb_dim=args.s0_emb_dim,
        test_zeroing=args.test_zeroing
    )

