"""
Training statistics recorder interface and JSONL implementation.

Provides a contract for recording system-level training statistics (binding loss,
agency cost, knot components) to JSONL files at configurable intervals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime

# Import torch for type checking in record_move
try:
    import torch
except ImportError:
    torch = None  # Type checking only

# Import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None


class StatsRecorder(ABC):
    """
    Abstract interface for recording training statistics.
    
    Tracks system-level metrics: binding loss, agency cost, and knot components.
    Does not track environment-specific data (energy, goals, etc.).
    """
    
    @abstractmethod
    def record_move(
        self,
        move: int,  # For internal tracking, but not written to file (line number = move)
        binding_loss: float,
        agency_cost: float,
        knot_components: Optional[Dict[str, Any]] = None,
        mu_t: Optional[Any] = None,  # Selected tube trajectory (T, state_dim)
        sigma_t: Optional[Any] = None,  # Selected sigma (T, state_dim)
        goals_reached: Optional[int] = None  # Number of goals reached in this move (0 or 1 typically)
    ):
        """
        Record statistics for a single move.
        
        Args:
            move: Move number (for internal tracking, not written - line number = move)
            binding_loss: Binding loss value
            agency_cost: Agency cost value
            knot_components: Optional dict with knot-related metrics:
                - selected_port: Index of selected port
                - knot_mask: Active knot mask (if available)
                - num_active_knots: Number of active knots
                - basis_weights: Basis function weights (if available)
            mu_t: Selected tube trajectory (T, state_dim) tensor or array
            sigma_t: Selected sigma (T, state_dim) tensor or array
            goals_reached: Optional number of goals reached in this move (for cumulative tracking)
        """
        pass
    
    @abstractmethod
    def flush(self):
        """Flush any buffered data to disk."""
        pass
    
    @abstractmethod
    def finalize(self):
        """Finalize recording (close files, write metadata, etc.)."""
        pass


class JSONLStatsRecorder(StatsRecorder):
    """
    JSONL implementation of StatsRecorder.
    
    Writes training statistics to JSONL files at configurable intervals.
    Also writes environment and system configuration metadata.
    """
    
    def __init__(
        self,
        output_dir: str,
        flush_interval: int = 10,
        system_config: Optional[Dict[str, Any]] = None,
        environment_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize JSONL stats recorder.
        
        Args:
            output_dir: Directory to write output files
            flush_interval: Number of moves between flushes to disk
            system_config: Optional system configuration dict
            environment_config: Optional environment configuration dict
        """
        self.output_dir = output_dir
        self.flush_interval = flush_interval
        self.system_config = system_config or {}
        self.environment_config = environment_config or {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        self.stats_file = os.path.join(output_dir, f"training_stats_{timestamp}.jsonl")
        self.config_file = os.path.join(output_dir, f"training_config_{timestamp}.json")
        self.plot_file = os.path.join(output_dir, f"training_plots_{timestamp}.png")
        
        # Buffer for stats entries
        self.stats_buffer = []
        
        # Write initial config file
        self._write_config()
    
    def _write_config(self):
        """Write configuration metadata to JSON file."""
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "system_config": self.system_config,
            "environment_config": self.environment_config
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def record_move(
        self,
        move: int,  # For internal tracking, but not written to file (line number = move)
        binding_loss: float,
        agency_cost: float,
        knot_components: Optional[Dict[str, Any]] = None,
        mu_t: Optional[Any] = None,  # Selected tube trajectory (T, state_dim)
        sigma_t: Optional[Any] = None,  # Selected sigma (T, state_dim)
        goals_reached: Optional[int] = None  # Number of goals reached in this move
    ):
        """
        Record statistics for a single move.
        
        Args:
            move: Move number (for internal tracking, not written - line number = move)
            binding_loss: Binding loss value
            agency_cost: Agency cost value
            knot_components: Optional dict with knot-related metrics
            mu_t: Selected tube trajectory (T, state_dim) tensor or array
            sigma_t: Selected sigma (T, state_dim) tensor or array
            goals_reached: Optional number of goals reached in this move (for cumulative tracking)
        """
        # Convert tensor values to float if needed
        if hasattr(binding_loss, 'item'):
            binding_loss = binding_loss.item()
        if hasattr(agency_cost, 'item'):
            agency_cost = agency_cost.item()
        
        # Prepare knot components (convert tensors to lists/numbers)
        knot_data = {}
        if knot_components:
            for key, value in knot_components.items():
                if torch is not None and isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        knot_data[key] = value.item()
                    else:
                        # Convert to list, handling multi-dimensional tensors
                        knot_data[key] = value.detach().cpu().tolist()
                else:
                    knot_data[key] = value
        
        # Convert mu_t and sigma_t to lists if provided
        mu_t_list = None
        if mu_t is not None:
            if torch is not None and isinstance(mu_t, torch.Tensor):
                mu_t_list = mu_t.detach().cpu().tolist()
            elif hasattr(mu_t, 'tolist'):
                mu_t_list = mu_t.tolist()
            else:
                mu_t_list = list(mu_t) if isinstance(mu_t, (list, tuple)) else None
        
        sigma_t_list = None
        if sigma_t is not None:
            if torch is not None and isinstance(sigma_t, torch.Tensor):
                sigma_t_list = sigma_t.detach().cpu().tolist()
            elif hasattr(sigma_t, 'tolist'):
                sigma_t_list = sigma_t.tolist()
            else:
                sigma_t_list = list(sigma_t) if isinstance(sigma_t, (list, tuple)) else None
        
        # Create stats entry
        # Note: "move" is not included - line number = move number
        entry = {
            "binding_loss": float(binding_loss),
            "agency_cost": float(agency_cost),
        }
        
        if knot_data:
            entry["knot_components"] = knot_data
        
        if mu_t_list is not None:
            entry["mu_t"] = mu_t_list
        
        if sigma_t_list is not None:
            entry["sigma_t"] = sigma_t_list
        
        if goals_reached is not None:
            entry["goals_reached"] = int(goals_reached)
        
        # Add to buffer
        self.stats_buffer.append(entry)
        
        # Flush if interval reached
        if len(self.stats_buffer) >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Flush buffered stats to JSONL file."""
        if not self.stats_buffer:
            return
        
        with open(self.stats_file, 'a') as f:
            for entry in self.stats_buffer:
                json.dump(entry, f)
                f.write('\n')
        
        self.stats_buffer.clear()
    
    def _load_stats_data(self) -> List[Dict[str, Any]]:
        """
        Load all stats data from the JSONL file.
        
        Returns:
            List of stats entries, where each entry is a dict with move number inferred from line number
        """
        stats_data = []
        if not os.path.exists(self.stats_file):
            return stats_data
        
        with open(self.stats_file, 'r') as f:
            for move, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry['move'] = move  # Add move number based on line number
                    stats_data.append(entry)
                except json.JSONDecodeError:
                    continue
        
        return stats_data
    
    def generate_plots(self):
        """
        Generate training plots from recorded statistics.
        
        Creates plots for:
        - Binding loss over moves
        - Agency cost over moves
        - Knot components (if available)
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        # Load stats data
        stats_data = self._load_stats_data()
        if not stats_data:
            print("Warning: No stats data found, skipping plot generation")
            return
        
        # Extract data arrays
        moves = [entry['move'] for entry in stats_data]
        binding_losses = [entry['binding_loss'] for entry in stats_data]
        agency_costs = [entry['agency_cost'] for entry in stats_data]
        
        # Check if knot components are available
        has_knot_components = any('knot_components' in entry for entry in stats_data)
        
        # Check if goals data is available
        has_goals_data = any('goals_reached' in entry for entry in stats_data)
        
        # Calculate cumulative goals if available
        cumulative_goals = None
        if has_goals_data:
            cumulative_goals = []
            total_goals = 0
            for entry in stats_data:
                goals_reached = entry.get('goals_reached', 0)
                total_goals += goals_reached
                cumulative_goals.append(total_goals)
        
        # Check if segment length data is available
        has_segment_length_data = False
        if has_knot_components:
            # Check if any entry has segment length statistics
            for entry in stats_data:
                if 'knot_components' in entry:
                    if 'segment_length_mean' in entry['knot_components']:
                        has_segment_length_data = True
                        break
        
        # Check if obstacle data is available
        has_obstacle_data = False
        if has_knot_components:
            for entry in stats_data:
                if 'knot_components' in entry:
                    if 'obstacle_proximity' in entry['knot_components']:
                        has_obstacle_data = True
                        break
        
        # Determine number of plots needed
        num_plots = 2  # binding_loss and agency_cost are always shown
        if has_knot_components:
            num_plots += 2  # selected_port and num_active_knots
        if has_goals_data:
            num_plots += 1  # cumulative_goals
        if has_segment_length_data:
            num_plots += 1  # segment length statistics
        if has_obstacle_data:
            num_plots += 2  # obstacle proximity vs binding loss, conditional collision rate
        
        # Create figure with subplots
        if num_plots <= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
            axes = axes.reshape(1, -1)  # Reshape for consistent indexing
        elif num_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
        elif num_plots <= 6:
            # 5 or 6 plots: use 2x3 layout
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
        else:
            # 7+ plots: use 3x3 layout (will hide unused)
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Helper function to get next axis
        def get_next_axis():
            nonlocal plot_idx
            if axes.ndim == 1:
                ax = axes[plot_idx]
            else:
                row = plot_idx // axes.shape[1]
                col = plot_idx % axes.shape[1]
                ax = axes[row, col]
            plot_idx += 1
            return ax
        
        # Calculate moving average window size (20 moves or 10% of data, whichever is smaller)
        window_size = min(20, max(5, len(moves) // 10))
        
        # Plot 1: Binding Loss (with moving average)
        ax1 = get_next_axis()
        # Plot raw values with low opacity
        ax1.plot(moves, binding_losses, 'b-', linewidth=0.5, alpha=0.3, label='Raw')
        # Add moving average to show trend
        if len(binding_losses) >= window_size:
            moving_avg = np.convolve(binding_losses, np.ones(window_size)/window_size, mode='valid')
            moving_avg_moves = moves[window_size-1:]
            ax1.plot(moving_avg_moves, moving_avg, 'b-', linewidth=2, alpha=0.9, label=f'Moving Avg ({window_size})')
        ax1.set_xlabel('Move')
        ax1.set_ylabel('Binding Loss')
        ax1.set_title('Binding Loss Over Moves')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot 2: Agency Cost (with moving average)
        ax2 = get_next_axis()
        # Plot raw values with low opacity
        ax2.plot(moves, agency_costs, 'r-', linewidth=0.5, alpha=0.3, label='Raw')
        # Add moving average to show trend
        if len(agency_costs) >= window_size:
            moving_avg = np.convolve(agency_costs, np.ones(window_size)/window_size, mode='valid')
            moving_avg_moves = moves[window_size-1:]
            ax2.plot(moving_avg_moves, moving_avg, 'r-', linewidth=2, alpha=0.9, label=f'Moving Avg ({window_size})')
        ax2.set_xlabel('Move')
        ax2.set_ylabel('Agency Cost')
        ax2.set_title('Agency Cost Over Moves')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Plot 3: Cumulative Goals (if available)
        if has_goals_data and cumulative_goals is not None:
            ax_goals = get_next_axis()
            ax_goals.plot(moves, cumulative_goals, 'g-', linewidth=1.5, alpha=0.7, label='Cumulative Goals')
            ax_goals.set_xlabel('Move')
            ax_goals.set_ylabel('Cumulative Goals Reached')
            ax_goals.set_title('Cumulative Goals Over Moves')
            ax_goals.grid(True, alpha=0.3)
            ax_goals.legend()
        
        # Plot Knot Components (if available)
        if has_knot_components:
            # Extract knot component data
            knot_data = {}
            for entry in stats_data:
                if 'knot_components' in entry:
                    for key, value in entry['knot_components'].items():
                        if key not in knot_data:
                            knot_data[key] = []
                        # Handle scalar and array values
                        if isinstance(value, (int, float)):
                            knot_data[key].append(value)
                        elif isinstance(value, list):
                            # For arrays, take mean or first element
                            if len(value) > 0 and isinstance(value[0], (int, float)):
                                knot_data[key].append(np.mean(value) if len(value) > 1 else value[0])
                            else:
                                knot_data[key].append(0)
                        else:
                            knot_data[key].append(0)
                else:
                    # Fill with None/NaN for missing entries
                    for key in knot_data:
                        knot_data[key].append(np.nan)
            
            # Plot selected_port (if available) - use scatter plot for discrete values
            if 'selected_port' in knot_data and len(knot_data['selected_port']) > 0:
                ax3 = get_next_axis()
                # Filter out NaN values for plotting
                valid_indices = [i for i, v in enumerate(knot_data['selected_port']) if not np.isnan(v)]
                if valid_indices:
                    valid_moves = [moves[i] for i in valid_indices]
                    valid_ports = [knot_data['selected_port'][i] for i in valid_indices]
                    ax3.scatter(valid_moves, valid_ports, c='c', alpha=0.6, s=15, edgecolors='none')
                    ax3.set_xlabel('Move')
                    ax3.set_ylabel('Selected Port Index')
                    ax3.set_title('Selected Port Over Moves')
                    unique_ports = sorted(set(valid_ports))
                    if unique_ports:
                        ax3.set_yticks(unique_ports)  # Set y-ticks to actual port values
                    ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot num_active_knots (if available) - use scatter plot for discrete values
            if 'num_active_knots' in knot_data and len(knot_data['num_active_knots']) > 0:
                ax4 = get_next_axis()
                # Filter out NaN values for plotting
                valid_indices = [i for i, v in enumerate(knot_data['num_active_knots']) if not np.isnan(v)]
                if valid_indices:
                    valid_moves = [moves[i] for i in valid_indices]
                    valid_knots = [knot_data['num_active_knots'][i] for i in valid_indices]
                    ax4.scatter(valid_moves, valid_knots, c='m', alpha=0.6, s=15, edgecolors='none')
                    ax4.set_xlabel('Move')
                    ax4.set_ylabel('Number of Active Knots')
                    ax4.set_title('Active Knots Over Moves')
                    unique_knots = sorted(set(valid_knots))
                    if unique_knots:
                        ax4.set_yticks(unique_knots)  # Set y-ticks to actual knot values
                    ax4.grid(True, alpha=0.3, axis='y')
            
            # Plot segment length statistics (if available)
            if has_segment_length_data:
                # Extract segment length statistics
                seg_length_min = []
                seg_length_max = []
                seg_length_mean = []
                seg_length_std = []
                
                for entry in stats_data:
                    if 'knot_components' in entry and 'segment_length_mean' in entry['knot_components']:
                        seg_length_min.append(entry['knot_components'].get('segment_length_min', np.nan))
                        seg_length_max.append(entry['knot_components'].get('segment_length_max', np.nan))
                        seg_length_mean.append(entry['knot_components'].get('segment_length_mean', np.nan))
                        seg_length_std.append(entry['knot_components'].get('segment_length_std', np.nan))
                    else:
                        seg_length_min.append(np.nan)
                        seg_length_max.append(np.nan)
                        seg_length_mean.append(np.nan)
                        seg_length_std.append(np.nan)
                
                # Plot segment length statistics
                ax_seg = get_next_axis()
                # Filter out NaN values for plotting
                valid_indices = [i for i, v in enumerate(seg_length_mean) if not np.isnan(v)]
                if valid_indices:
                    valid_moves = [moves[i] for i in valid_indices]
                    valid_min = [seg_length_min[i] for i in valid_indices]
                    valid_max = [seg_length_max[i] for i in valid_indices]
                    valid_mean = [seg_length_mean[i] for i in valid_indices]
                    valid_std = [seg_length_std[i] for i in valid_indices]
                    
                    # Plot mean with std as shaded region
                    ax_seg.plot(valid_moves, valid_mean, 'b-', linewidth=1.5, alpha=0.7, label='Mean')
                    ax_seg.fill_between(valid_moves, 
                                        [m - s for m, s in zip(valid_mean, valid_std)],
                                        [m + s for m, s in zip(valid_mean, valid_std)],
                                        alpha=0.2, color='blue', label='±1 Std')
                    
                    # Plot min and max as lighter lines
                    ax_seg.plot(valid_moves, valid_min, 'g--', linewidth=1, alpha=0.5, label='Min')
                    ax_seg.plot(valid_moves, valid_max, 'r--', linewidth=1, alpha=0.5, label='Max')
                    
                    ax_seg.set_xlabel('Move')
                    ax_seg.set_ylabel('Segment Length')
                    ax_seg.set_title('Segment Length Statistics Over Moves')
                    ax_seg.grid(True, alpha=0.3)
                    ax_seg.legend(fontsize=8)
        
        # Plot Obstacle Learning Metrics (if available)
        if has_obstacle_data:
            # Extract obstacle data
            obstacle_proximities = []
            obstacle_collisions = []
            proximity_bins = []
            
            for entry in stats_data:
                if 'knot_components' in entry:
                    kc = entry['knot_components']
                    obstacle_proximities.append(kc.get('obstacle_proximity', np.nan))
                    obstacle_collisions.append(kc.get('obstacle_collision', 0))
                    proximity_bins.append(kc.get('proximity_bin', 'unknown'))
                else:
                    obstacle_proximities.append(np.nan)
                    obstacle_collisions.append(0)
                    proximity_bins.append('unknown')
            
            # Plot 1: Binding Loss vs Obstacle Proximity (Binned)
            ax_obs1 = get_next_axis()
            valid_indices = [i for i, p in enumerate(obstacle_proximities) if not np.isnan(p) and p < 100.0]
            if valid_indices:
                valid_moves = [moves[i] for i in valid_indices]
                valid_proximities = [obstacle_proximities[i] for i in valid_indices]
                valid_losses = [binding_losses[i] for i in valid_indices]
                
                # Color by proximity bin
                colors = {'0-0.5': 'red', '0.5-1.0': 'orange', '1.0-2.0': 'yellow', '2.0+': 'green'}
                valid_bins = [proximity_bins[i] for i in valid_indices]
                for bin_name in ['0-0.5', '0.5-1.0', '1.0-2.0', '2.0+']:
                    bin_indices = [j for j, b in enumerate(valid_bins) if b == bin_name]
                    if bin_indices:
                        bin_moves = [valid_moves[j] for j in bin_indices]
                        bin_proximities = [valid_proximities[j] for j in bin_indices]
                        bin_losses = [valid_losses[j] for j in bin_indices]
                        ax_obs1.scatter(bin_proximities, bin_losses, c=colors.get(bin_name, 'gray'), 
                                       alpha=0.5, s=20, label=bin_name, edgecolors='none')
                
                # Add moving average line
                if len(valid_proximities) >= window_size:
                    # Sort by proximity for moving average
                    sorted_data = sorted(zip(valid_proximities, valid_losses))
                    sorted_prox, sorted_loss = zip(*sorted_data)
                    moving_avg_loss = np.convolve(sorted_loss, np.ones(window_size)/window_size, mode='valid')
                    moving_avg_prox = sorted_prox[window_size-1:]
                    ax_obs1.plot(moving_avg_prox, moving_avg_loss, 'k-', linewidth=2, alpha=0.7, label='Moving Avg')
                
                ax_obs1.set_xlabel('Obstacle Proximity')
                ax_obs1.set_ylabel('Binding Loss')
                ax_obs1.set_title('Binding Loss vs Obstacle Proximity (Binned)')
                ax_obs1.grid(True, alpha=0.3)
                ax_obs1.legend(fontsize=7, title='Proximity Bin')
            
            # Plot 2: Conditional Collision Rate by Proximity Bin
            ax_obs2 = get_next_axis()
            bin_names = ['0-0.5', '0.5-1.0', '1.0-2.0', '2.0+']
            collision_rates_by_bin = {bin_name: [] for bin_name in bin_names}
            
            # Calculate collision rate per bin over time (using moving windows)
            for bin_name in bin_names:
                bin_indices = [i for i, b in enumerate(proximity_bins) if b == bin_name]
                if len(bin_indices) >= window_size:
                    bin_collisions = [obstacle_collisions[i] for i in bin_indices]
                    bin_moves = [moves[i] for i in bin_indices]
                    # Calculate moving average collision rate
                    moving_collision_rate = np.convolve(bin_collisions, np.ones(window_size)/window_size, mode='valid')
                    moving_moves = bin_moves[window_size-1:]
                    collision_rates_by_bin[bin_name] = (moving_moves, moving_collision_rate)
            
            colors_bins = {'0-0.5': 'red', '0.5-1.0': 'orange', '1.0-2.0': 'yellow', '2.0+': 'green'}
            for bin_name in bin_names:
                if collision_rates_by_bin[bin_name]:
                    moves_bin, rates = collision_rates_by_bin[bin_name]
                    ax_obs2.plot(moves_bin, rates, color=colors_bins.get(bin_name, 'gray'), 
                               linewidth=1.5, alpha=0.7, label=f'{bin_name}')
            
            ax_obs2.set_xlabel('Move')
            ax_obs2.set_ylabel('Collision Rate')
            ax_obs2.set_title('Conditional Collision Rate by Proximity Bin')
            ax_obs2.grid(True, alpha=0.3)
            ax_obs2.legend(fontsize=7, title='Proximity Bin')
            ax_obs2.set_ylim([0, 1.1])
        
        # Hide any remaining unused subplots
        while plot_idx < axes.size:
            if axes.ndim == 1:
                axes[plot_idx].axis('off')
            else:
                row = plot_idx // axes.shape[1]
                col = plot_idx % axes.shape[1]
                axes[row, col].axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {self.plot_file}")
    
    def finalize(self):
        """Finalize recording (flush remaining data and generate plots)."""
        self.flush()
        print(f"Training statistics saved to: {self.stats_file}")
        print(f"Training configuration saved to: {self.config_file}")
        
        # Generate plots
        try:
            self.generate_plots()
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")
            import traceback
            traceback.print_exc()