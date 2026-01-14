"""
Universal evaluation suite (actor-agnostic, env-agnostic).
"""

from pathlib import Path
from typing import Dict, Optional, List, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..plots.latent import plot_latent_scatter, plot_pairwise_cosine_histogram
from ..plots.geometry import (
    plot_tube_overlay,
    plot_multi_start_tubes,
    plot_energy_landscape,
    plot_trajectory_pca,
)
from ..metrics import compute_latent_metrics, compute_contract_metrics


class UniversalSuite:
    """
    Universal evaluation suite that works with any actor and environment.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_learning_curves(
        self,
        training_history: Dict[str, List[Dict]],
        output_path: Optional[Path] = None,
    ):
        """
        Plot learning curves from training history.
        
        Args:
            training_history: Dict with 'warmup' and 'actor' lists of metric dicts
            output_path: Optional output path (default: output_dir/learning_curves.png)
        """
        if output_path is None:
            output_path = self.output_dir / "learning_curves.png"
        
        # Extract metrics
        warmup_metrics = training_history.get('warmup', [])
        actor_metrics = training_history.get('actor', [])
        
        if not warmup_metrics and not actor_metrics:
            print("  No training history to plot")
            return
        
        # Determine which metrics to plot
        if warmup_metrics:
            warmup_keys = [k for k in warmup_metrics[0].keys() if k not in ['recon_weight', 'intent_weight']]
        else:
            warmup_keys = []
        
        if actor_metrics:
            actor_keys = [k for k in actor_metrics[0].keys() if k != 'loss']
        else:
            actor_keys = []
        
        n_plots = len(warmup_keys) + len(actor_keys)
        if n_plots == 0:
            return
        
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Warmup plots
        for key in warmup_keys[:6]:  # Limit to 6 plots
            ax = axes[plot_idx]
            values = [m.get(key, 0) for m in warmup_metrics]
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Batch')
            ax.set_ylabel(key)
            ax.set_title(f'Warmup: {key}')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Actor plots
        for key in actor_keys[:6]:  # Limit to 6 plots
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]
            values = [m.get(key, 0) for m in actor_metrics]
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(key)
            ax.set_title(f'Actor: {key}')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Learning Curves')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved learning curves to {output_path}")
    
    def run_episode_rollouts(
        self,
        actor,
        env,
        n_episodes: int = 8,
        output_path: Optional[Path] = None,
    ):
        """
        Plot episode rollouts with tube overlays.
        
        Args:
            actor: Actor instance
            env: Environment instance
            n_episodes: Number of episodes to plot
            output_path: Optional output path
        """
        if output_path is None:
            output_path = self.output_dir / "episode_rollouts.png"
        
        device = next(actor.parameters()).device
        
        # Collect episodes
        episodes = []
        for _ in range(n_episodes):
            obs = env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32, device=device)
            
            # Select z
            if hasattr(actor, 'select_z_geometric'):
                z_star = actor.select_z_geometric(s0, trajectory_delta=None)
            elif hasattr(actor, 'select_z_geometric_multimodal'):
                z_star = actor.select_z_geometric_multimodal(s0)
            else:
                # Fallback: random z
                z_star = torch.randn(actor.z_dim, device=device)
                z_star = torch.nn.functional.normalize(z_star, p=2, dim=0)
            
            # Get tube
            mu, sigma = actor.get_tube(s0, z_star)
            mu_np = mu.detach().cpu().numpy()[0]
            sigma_np = sigma.detach().cpu().numpy()[0]
            
            # Execute
            actual_traj = []
            curr_obs = obs
            for t in range(actor.T):
                action = mu_np[t] - curr_obs[:actor.pred_dim]
                curr_obs, _, _, info = env.step(action)
                actual_traj.append(info.get('x', curr_obs))
            
            episodes.append({
                'mu': mu_np,
                'sigma': sigma_np,
                'actual': np.array(actual_traj),
            })
        
        # Plot
        n_cols = min(4, n_episodes)
        n_rows = (n_episodes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        pred_dim = actor.pred_dim
        dims_to_plot = list(range(min(2, pred_dim)))
        
        for i, ep in enumerate(episodes):
            ax = axes[i]
            t = np.arange(len(ep['mu']))
            
            for dim in dims_to_plot:
                ax.plot(t, ep['mu'][:, dim], 'b-', linewidth=2, alpha=0.7, label=f'μ[{dim}]' if dim == dims_to_plot[0] else None)
                ax.fill_between(t, ep['mu'][:, dim] - ep['sigma'][:, dim], 
                               ep['mu'][:, dim] + ep['sigma'][:, dim], 
                               color='blue', alpha=0.2)
                if ep['actual'].shape[1] > dim:
                    ax.plot(t, ep['actual'][:, dim], 'r--', linewidth=1.5, alpha=0.7, label='Actual' if dim == dims_to_plot[0] else None)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('State')
            ax.set_title(f'Episode {i+1}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for i in range(len(episodes), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Episode Rollouts (n={n_episodes})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved episode rollouts to {output_path}")
    
    def run_tube_overlap_matrix(
        self,
        actor,
        env,
        n_starts: int = 5,
        output_path: Optional[Path] = None,
    ):
        """
        Compute tube overlap matrix for multiple z starts at same s0.
        Measures "implicit ports" without assuming mode labels.
        
        Args:
            actor: Actor instance
            env: Environment instance
            n_starts: Number of different z starts to test
            output_path: Optional output path
        """
        if output_path is None:
            output_path = self.output_dir / "tube_overlap_matrix.png"
        
        device = next(actor.parameters()).device
        obs = env.reset()
        s0 = torch.tensor(obs, dtype=torch.float32, device=device)
        
        # Generate multiple z starts
        tubes = []
        for _ in range(n_starts):
            # Random z on unit sphere
            z = torch.randn(actor.z_dim, device=device)
            z = torch.nn.functional.normalize(z, p=2, dim=0)
            
            with torch.no_grad():
                mu, sigma = actor.get_tube(s0, z)
            
            tubes.append({
                'mu': mu.squeeze(0).cpu().numpy(),
                'sigma': sigma.squeeze(0).cpu().numpy(),
            })
        
        # Compute overlap matrix
        n_tubes = len(tubes)
        overlap_matrix = np.zeros((n_tubes, n_tubes))
        
        for i in range(n_tubes):
            for j in range(i + 1, n_tubes):
                mu_i, sigma_i = tubes[i]['mu'], tubes[i]['sigma']
                mu_j, sigma_j = tubes[j]['mu'], tubes[j]['sigma']
                
                dist = np.abs(mu_i - mu_j)
                combined_sigma = sigma_i + sigma_j
                overlap_per_t = (dist < combined_sigma).all(axis=1)
                overlap_pct = overlap_per_t.mean() * 100
                overlap_matrix[i, j] = overlap_matrix[j, i] = overlap_pct
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Overlap matrix
        ax1 = axes[0]
        im = ax1.imshow(overlap_matrix, cmap='viridis', vmin=0, vmax=100)
        ax1.set_xlabel('Tube i')
        ax1.set_ylabel('Tube j')
        ax1.set_title('Tube Overlap Matrix (% timesteps)')
        plt.colorbar(im, ax=ax1, label='Overlap %')
        
        # Right: Multi-start tubes plot
        ax2 = axes[1]
        plot_multi_start_tubes(
            [(t['mu'], t['sigma']) for t in tubes],
            output_path.parent / "multi_start_tubes.png",
            dims_to_plot=[0],
        )
        # Load and display in subplot (simplified - just show one dim)
        t = np.arange(len(tubes[0]['mu']))
        for i, tube in enumerate(tubes):
            ax2.plot(t, tube['mu'][:, 0], linewidth=1.5, alpha=0.7, label=f'Tube {i}')
            ax2.fill_between(t, tube['mu'][:, 0] - tube['sigma'][:, 0],
                           tube['mu'][:, 0] + tube['sigma'][:, 0], alpha=0.1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('x[0]')
        ax2.set_title('Multi-Start Tubes')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Tube Overlap Analysis (n={n_starts} starts)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        avg_overlap = overlap_matrix[np.triu_indices(n_tubes, k=1)].mean()
        print(f"  Average Tube Overlap: {avg_overlap:.1f}%")
        print(f"  Saved tube overlap matrix to {output_path}")
        
        return {
            'overlap_matrix': overlap_matrix.tolist(),
            'avg_overlap': float(avg_overlap),
        }
    
    def run(
        self,
        actor,
        env,
        training_history: Optional[Dict] = None,
        eval_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run full universal suite.
        
        Args:
            actor: Trained actor
            env: Environment
            training_history: Optional training history dict
            eval_results: Optional evaluation results dict
        
        Returns:
            Dict with suite results
        """
        print("\n" + "="*60)
        print("UNIVERSAL EVALUATION SUITE")
        print("="*60)
        
        results = {}
        
        # 1. Learning curves
        if training_history:
            print("\n1. Learning Curves...")
            self.run_learning_curves(training_history)
            results['learning_curves'] = True
        
        # 2. Episode rollouts
        print("\n2. Episode Rollouts...")
        self.run_episode_rollouts(actor, env, n_episodes=8)
        results['episode_rollouts'] = True
        
        # 3. Tube overlap matrix
        print("\n3. Tube Overlap Matrix...")
        overlap_results = self.run_tube_overlap_matrix(actor, env, n_starts=5)
        results['tube_overlap'] = overlap_results
        
        # 4. Energy landscape
        print("\n4. Energy Landscape...")
        device = next(actor.parameters()).device
        obs = env.reset()
        s0 = torch.tensor(obs, dtype=torch.float32, device=device)
        energy_results = plot_energy_landscape(
            actor, s0, actor.z_dim, actor.pred_dim,
            self.output_dir / "energy_landscape.png",
        )
        results['energy_landscape'] = energy_results
        
        # 5. Latent health
        print("\n5. Latent Health...")
        if eval_results and 'z_samples' in eval_results:
            z_array = eval_results['z_samples']
            labels = eval_results.get('modes')
            
            plot_latent_scatter(
                z_array,
                self.output_dir / "latent_scatter.png",
                labels=labels if labels else None,
            )
            plot_pairwise_cosine_histogram(
                z_array,
                self.output_dir / "pairwise_cosine.png",
            )
            
            latent_metrics = compute_latent_metrics(z_array, labels)
            results['latent_metrics'] = latent_metrics
        
        print("\n" + "="*60)
        print("UNIVERSAL SUITE COMPLETE")
        print("="*60)
        
        return results
