import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import time

class LivePlotter:
    # Consistent aesthetic color palette
    COLORS = {
        'trajectory': '#6B7280',  # Cool grey for actual paths
        'tube_high_agency': '#2563EB',  # Blue (high agency/confident)
        'tube_low_agency': '#CBD5E1',  # Light blue-grey (low agency/uncertain, darker than background)
        'obstacle': '#EF4444',  # Warm red for obstacles (contrasting)
        'obstacle_alpha': 0.25,  # Obstacle transparency
        'goal': '#F59E0B',  # Amber/gold for goals (contrasting, warm)
        'goal_reached': '#FBBF24',  # Lighter gold for reached goals
        'start': '#1F2937',  # Dark grey for start point
        'current_pos': '#3B82F6',  # Blue for current position marker
        'boundary': '#9CA3AF',  # Medium grey for boundary wireframe
    }
    
    # Line widths (thinner for cleaner look)
    LINE_WIDTHS = {
        'tube': 1.5,  # Thinner tube centerlines
        'trajectory': 2.0,  # Slightly thicker for actual paths (more visible)
        'boundary': 0.8,  # Thin boundary lines
    }
    
    def __init__(self, obstacles, target_pos, bounds=None, max_history=5, record_mode=False):
        """
        Args:
            obstacles: List of (position, radius) tuples
            target_pos: Initial target position
            bounds: Optional dict with 'min' and 'max' keys for bounding box
            max_history: Maximum number of moves to display in visualization (default: 5)
            record_mode: If True, collect data for replay instead of rendering live
        """
        self.obstacles = obstacles
        self.target_pos = target_pos
        self.max_history = max_history
        self.record_mode = record_mode
        
        # Set default bounds if not provided (needed for both record and replay modes)
        if bounds is None:
            self.bounds = {
                'min': [-2.0, -2.0, -2.0],
                'max': [12.0, 12.0, 2.0]
            }
        else:
            self.bounds = bounds
        
        # In record mode, just collect data - no plotting setup
        if record_mode:
            self.recorded_data = []  # List of (mu_t, sigma_t, actual_path, current_pos, step, goal_positions)
            self.goal_positions = []  # Track goal positions over time
            return
        
        # Live mode: set up plotting
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Performance optimization: Set aspect ratio once (avoids recalculation)
        self.ax.set_box_aspect([1, 1, 1])
        
        # Bounds already set above (before record_mode check)
        
        # Plot bounding box as wireframe
        self._plot_bounding_box()
        
        # Plot obstacles with consistent color
        # OPTIMIZATION: Reduce resolution for better performance with many obstacles
        # Use lower resolution (10x10 instead of 20x20) - 4x fewer points
        obstacle_resolution = max(8, min(12, 20 // max(1, len(obstacles) // 4)))  # Adaptive resolution
        u = np.linspace(0, 2 * np.pi, obstacle_resolution)
        v = np.linspace(0, np.pi, obstacle_resolution)
        
        for obs_p, obs_r in obstacles:
            # Reuse u, v arrays for all obstacles (more efficient)
            x = obs_p[0] + obs_r * np.outer(np.cos(u), np.sin(v))
            y = obs_p[1] + obs_r * np.outer(np.sin(u), np.sin(v))
            z = obs_p[2] + obs_r * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color=self.COLORS['obstacle'], 
                               alpha=self.COLORS['obstacle_alpha'], edgecolor='none',
                               antialiased=False)  # Disable antialiasing for speed
        
        # Plot target (store marker for updates)
        self.target_marker = self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                       color=self.COLORS['goal'], s=300, marker='*', 
                       edgecolors='#D97706', linewidths=1.5, label='Target', zorder=10)
        
        # Plot start
        self.ax.scatter(0, 0, 0, color=self.COLORS['start'], s=100, 
                        marker='o', edgecolors='white', linewidths=1, label='Start')
        
        # Storage for plot elements (only current episode)
        self.tube_lines = []  # Store line segments for gradient-colored tubes
        self.path_lines = []
        self.tube_spheres = []  # Store sphere surfaces for cleanup (deprecated)
        self.tube_markers = []  # Store tube start/end markers
        self.path_markers = []  # Store path endpoint markers
        self.current_pos_marker = None
        self.current_episode = -1  # Track current episode to detect transitions
        self.reached_goal_markers = []  # Store markers for goals reached in current episode
        
        # Custom colormap for agency visualization: Blue (high agency) -> Light blue-grey (low agency)
        # Create a custom colormap from blue to light blue-grey
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [self.COLORS['tube_high_agency'], self.COLORS['tube_low_agency']]
        self.agency_cmap = LinearSegmentedColormap.from_list('agency', colors_list, N=256)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Live Training: Tube Warping Visualization\nTube color = Agency (σ): Blue=High Agency, Light Grey=Low Agency')
        self.ax.legend()
        self.ax.set_xlim(self.bounds['min'][0], self.bounds['max'][0])
        self.ax.set_ylim(self.bounds['min'][1], self.bounds['max'][1])
        self.ax.set_zlim(self.bounds['min'][2], self.bounds['max'][2])
        
        # Add colorbar for agency visualization (only once, will be updated)
        # Create a scalar mappable for the colormap
        sm = plt.cm.ScalarMappable(cmap=self.agency_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        self.cbar = self.fig.colorbar(sm, ax=self.ax, pad=0.1, shrink=0.6)
        self.cbar.set_label('Normalized σ (Agency)', rotation=270, labelpad=15)
    
    def _plot_bounding_box(self):
        """Plot the bounding box as a wireframe cube."""
        min_b = self.bounds['min']
        max_b = self.bounds['max']
        
        # Define the 8 vertices of the bounding box
        vertices = np.array([
            [min_b[0], min_b[1], min_b[2]],  # 0: min corner
            [max_b[0], min_b[1], min_b[2]],  # 1
            [max_b[0], max_b[1], min_b[2]],  # 2
            [min_b[0], max_b[1], min_b[2]],  # 3
            [min_b[0], min_b[1], max_b[2]],  # 4
            [max_b[0], min_b[1], max_b[2]],  # 5
            [max_b[0], max_b[1], max_b[2]],  # 6
            [min_b[0], max_b[1], max_b[2]],  # 7: max corner
        ])
        
        # Define the 12 edges of the cube
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]
        
        # Plot edges with consistent color
        for edge in edges:
            points = vertices[edge]
            self.ax.plot3D(*points.T, color=self.COLORS['boundary'], linestyle='--', 
                          linewidth=self.LINE_WIDTHS['boundary'], alpha=0.4, 
                          label='Boundary' if edge == edges[0] else '')
        
    def update(self, mu_t, sigma_t, actual_path, current_pos, episode, step, goal_pos=None):
        """
        Update the plot with new tube and path information.
        
        In record_mode: Collects data for later replay (no rendering)
        In live_mode: Updates visualization immediately
        
        Args:
            mu_t: (T, state_dim) planned tube trajectory (relative)
            sigma_t: (T, state_dim) tube radii (per-dimension)
            actual_path: (T, state_dim) actual path taken
            current_pos: (1, state_dim) current position
            episode: current episode number
            step: current step number
            goal_pos: Optional current goal position (for recording)
        """
        # Record mode: just collect data, no rendering
        if self.record_mode:
            # Convert tensors to numpy for storage
            mu_t_np = mu_t.detach().cpu().numpy() if isinstance(mu_t, torch.Tensor) else np.array(mu_t)
            sigma_t_np = sigma_t.detach().cpu().numpy() if isinstance(sigma_t, torch.Tensor) else np.array(sigma_t)
            actual_path_np = actual_path.detach().cpu().numpy() if isinstance(actual_path, torch.Tensor) else np.array(actual_path)
            current_pos_np = current_pos.detach().cpu().numpy() if isinstance(current_pos, torch.Tensor) else np.array(current_pos)
            goal_pos_np = goal_pos.detach().cpu().numpy() if isinstance(goal_pos, torch.Tensor) else (np.array(goal_pos) if goal_pos is not None else None)
            
            self.recorded_data.append({
                'mu_t': mu_t_np.copy(),
                'sigma_t': sigma_t_np.copy(),
                'actual_path': actual_path_np.copy(),
                'current_pos': current_pos_np.copy(),
                'step': step,
                'goal_pos': goal_pos_np.copy() if goal_pos_np is not None else None
            })
            return
        
        # Live mode: render immediately
        self._render_frame(mu_t, sigma_t, actual_path, current_pos, episode, step)
    
    def _render_frame(self, mu_t, sigma_t, actual_path, current_pos, episode, step):
        """Internal method to render a single frame (used by both update and replay)."""
        # Clear previous episode's visualization when starting a new episode
        if episode != self.current_episode:
            self._clear_episode()
            self.current_episode = episode
        
        # Keep only last few steps for clarity within current episode
        # Optimized: only clean up when we exceed limit (avoid unnecessary work)
        # Note: tube_lines can contain either Line3DCollection objects or regular line objects
        # Both support .remove() method, so cleanup works for both
        if len(self.tube_lines) > self.max_history:
            # Remove old items efficiently
            to_remove = len(self.tube_lines) - self.max_history
            for line in self.tube_lines[:to_remove]:
                line.remove()
            self.tube_lines = self.tube_lines[to_remove:]
        
        if len(self.path_lines) > self.max_history:
            to_remove = len(self.path_lines) - self.max_history
            for line in self.path_lines[:to_remove]:
                line.remove()
            self.path_lines = self.path_lines[to_remove:]
        
        # Clean up old markers (optimized: batch removal)
        max_markers = self.max_history * 2
        if len(self.tube_markers) > max_markers:
            to_remove = len(self.tube_markers) - max_markers
            for marker in self.tube_markers[:to_remove]:
                marker.remove()
            self.tube_markers = self.tube_markers[to_remove:]
        
        if len(self.path_markers) > max_markers:
            to_remove = len(self.path_markers) - max_markers
            for marker in self.path_markers[:to_remove]:
                marker.remove()
            self.path_markers = self.path_markers[to_remove:]
        
        # Spheres are disabled, but keep cleanup code for compatibility
        max_spheres = self.max_history * 5
        if len(self.tube_spheres) > max_spheres:
            to_remove = len(self.tube_spheres) - max_spheres
            for sphere in self.tube_spheres[:to_remove]:
                sphere.remove()
            self.tube_spheres = self.tube_spheres[to_remove:]
        
        # Transform tube to global coordinates
        mu_global = mu_t + current_pos
        
        # Plot planned tube with radius visualization
        mu_np = mu_global.detach().cpu().numpy()
        sigma_np = sigma_t.detach().cpu().numpy()
        
        # Ensure mu_np is 2D (T, state_dim)
        if mu_np.ndim > 2:
            mu_np = mu_np.reshape(-1, mu_np.shape[-1])
        elif mu_np.ndim == 1:
            mu_np = mu_np.reshape(1, -1)
        
        # Handle per-dimension sigmas: sigma_np is now (T, state_dim) instead of (T, 1)
        # For visualization, we'll use the mean sigma across dimensions
        if sigma_np.ndim > 1:
            # If it's (T, state_dim), take mean across dimensions
            if sigma_np.shape[-1] > 1:
                sigma_np = sigma_np.mean(axis=-1)  # (T,)
            else:
                sigma_np = sigma_np.squeeze(-1)  # (T,)
        if sigma_np.ndim == 0:
            sigma_np = np.array([sigma_np.item()])
        
        # Ensure sigma_np matches mu_np length (simple interpolation without scipy)
        if len(sigma_np) != len(mu_np):
            if len(sigma_np) == 1:
                # Single value: replicate
                sigma_np = np.full(len(mu_np), sigma_np[0])
            elif len(sigma_np) < len(mu_np):
                # Upsample: linear interpolation using numpy
                old_indices = np.linspace(0, len(sigma_np) - 1, len(sigma_np))
                new_indices = np.linspace(0, len(sigma_np) - 1, len(mu_np))
                sigma_np = np.interp(new_indices, old_indices, sigma_np)
            else:
                # Downsample: take evenly spaced samples
                indices = np.linspace(0, len(sigma_np) - 1, len(mu_np)).astype(int)
                sigma_np = sigma_np[indices]
        
        # Normalize sigma values for colormap (0 = low uncertainty, 1 = high uncertainty)
        # Use a fixed range for consistent visualization across moves
        # Typical sigma range: 0.1 to 2.0, but we'll use dynamic normalization per tube
        # Ensure sigma_np is 1D
        if sigma_np.ndim > 1:
            sigma_np = sigma_np.flatten()
        sigma_min, sigma_max = float(sigma_np.min()), float(sigma_np.max())
        if sigma_max > sigma_min:
            sigma_normalized = (sigma_np - sigma_min) / (sigma_max - sigma_min)
        else:
            # All values are the same - use middle of colormap
            sigma_normalized = np.full_like(sigma_np, 0.5)
        
        # Ensure sigma_normalized is 1D numpy array
        sigma_normalized = np.atleast_1d(sigma_normalized).flatten()
        
        # Plot tube centerline with gradient coloring based on agency (sigma)
        # OPTIMIZED: Use Line3DCollection to batch all segments of a tube into one object
        # This dramatically reduces matplotlib overhead while maintaining sequential visibility
        if len(mu_np) > 1:
            # Prepare segments for this tube as a single collection
            # Shape: (N_segments, 2, 3) - each segment has start and end points
            segments = np.array([
                [mu_np[i], mu_np[i+1]] 
                for i in range(len(mu_np) - 1)
            ])
            
            # Compute colors for each segment (average of endpoints)
            segment_colors = []
            for i in range(len(mu_np) - 1):
                seg_sigma = float((sigma_normalized[i] + sigma_normalized[i + 1]) / 2.0)
                segment_colors.append(self.agency_cmap(seg_sigma))
            
            # Create a single Line3DCollection for this entire tube
            # This is 10-40x faster than plotting segments individually
            tube_collection = Line3DCollection(
                segments,
                colors=segment_colors,
                linewidths=self.LINE_WIDTHS['tube'],
                linestyles='--',
                alpha=0.85,
                antialiased=False
            )
            
            # Add to axes and store reference
            self.ax.add_collection3d(tube_collection)
            self.tube_lines.append(tube_collection)  # Store collection, not individual lines
            
            # Add label only for first tube
            if len(self.tube_lines) == 1:
                tube_collection.set_label('Planned Tube (σ gradient)')
        else:
            # Single point: just plot as marker
            tube_line, = self.ax.plot(mu_np[:, 0], mu_np[:, 1], mu_np[:, 2], 
                                      color=self.COLORS['tube_high_agency'], alpha=0.7, 
                                      linestyle='--', linewidth=self.LINE_WIDTHS['tube'], 
                                      marker='o', markersize=6,
                                      label='Planned Tube' if len(self.tube_lines) == 0 else '')
            self.tube_lines.append(tube_line)
        
        # Plot starting point of tube (colored by sigma at start)
        start_point = mu_np[0]
        start_sigma_val = float(sigma_normalized[0]) if len(sigma_normalized) > 0 else 0.5
        start_sigma_color = self.agency_cmap(start_sigma_val)
        start_marker = self.ax.scatter(start_point[0], start_point[1], start_point[2], 
                                      color=start_sigma_color, s=20, alpha=0.85, marker='o',
                                      edgecolors='white', linewidths=0.5)
        self.tube_markers.append(start_marker)
        
        # Plot end point of tube (colored by sigma at end)
        end_point = mu_np[-1]
        end_sigma_val = float(sigma_normalized[-1]) if len(sigma_normalized) > 0 else 0.5
        end_sigma_color = self.agency_cmap(end_sigma_val)
        end_marker = self.ax.scatter(end_point[0], end_point[1], end_point[2], 
                                     color=end_sigma_color, s=35, alpha=0.9, marker='x', 
                                     linewidths=1.5, edgecolors='white')
        self.tube_markers.append(end_marker)
        
        # DISABLED: Sphere visualization is very expensive (creates many 3D surfaces)
        # The tube centerline and markers provide sufficient visual information
        # If you want to re-enable, reduce to max 1-2 spheres and lower resolution
        # try:
        #     # Only show 1 sphere at midpoint for performance
        #     if len(mu_np) > 2:
        #         mid_idx = len(mu_np) // 2
        #         cx, cy, cz = mu_np[mid_idx]
        #         radius = float(sigma_np[mid_idx] if sigma_np.size > mid_idx else sigma_np[0])
        #         # Use lower resolution (4x4 instead of 8x8)
        #         u = np.linspace(0, 2 * np.pi, 4)
        #         v = np.linspace(0, np.pi, 4)
        #         u_grid, v_grid = np.meshgrid(u, v)
        #         x = cx + radius * np.sin(v_grid) * np.cos(u_grid)
        #         y = cy + radius * np.sin(v_grid) * np.sin(u_grid)
        #         z = cz + radius * np.cos(v_grid)
        #         sphere_surface = self.ax.plot_surface(x, y, z, color='cyan', alpha=0.15, edgecolor='none')
        #         self.tube_spheres.append(sphere_surface)
        # except:
        #     pass
        
        # Plot actual path taken (optimized: only convert if we have a path)
        if actual_path is not None and len(actual_path) > 0:
            # Convert once and reuse
            if isinstance(actual_path, torch.Tensor):
                path_np = actual_path.detach().cpu().numpy()
            else:
                path_np = np.asarray(actual_path)
            
            # Ensure path_np is 2D
            if path_np.ndim == 1:
                path_np = path_np.reshape(1, 3)
            
            # Plot this step's actual path with cool grey color
            if len(path_np) > 1:
                # Ensure we have 3D coordinates
                z_coords = path_np[:, 2] if path_np.shape[1] > 2 else np.zeros(len(path_np))
                path_line, = self.ax.plot(path_np[:, 0], path_np[:, 1], z_coords,
                                         color=self.COLORS['trajectory'], 
                                         linewidth=self.LINE_WIDTHS['trajectory'], 
                                         alpha=0.9, linestyle='-',
                                         label='Actual Path' if len(self.path_lines) == 0 else '',
                                         antialiased=False)  # Disable antialiasing for better performance
                self.path_lines.append(path_line)
                
                # Plot path endpoints (reuse path_np array slicing)
                start_pt = path_np[0]
                end_pt = path_np[-1]
                start_marker = self.ax.scatter(start_pt[0], start_pt[1], start_pt[2],
                                              color=self.COLORS['trajectory'], s=25, 
                                              alpha=0.8, marker='s', edgecolors='white', 
                                              linewidths=0.5)
                self.path_markers.append(start_marker)
                
                end_marker = self.ax.scatter(end_pt[0], end_pt[1], end_pt[2],
                                            color=self.COLORS['trajectory'], s=35, 
                                            alpha=0.9, marker='D', edgecolors='white', 
                                            linewidths=0.5)
                self.path_markers.append(end_marker)
            else:
                # Single point path - just plot as a marker
                single_marker = self.ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2],
                                               color=self.COLORS['trajectory'], s=50, 
                                               alpha=0.8, marker='o', edgecolors='white', 
                                               linewidths=0.5)
                self.path_markers.append(single_marker)
        
        # Update current position marker (large, visible)
        if self.current_pos_marker is not None:
            self.current_pos_marker.remove()
        
        # Optimize: convert once and reuse
        if isinstance(current_pos, torch.Tensor):
            pos_np = current_pos.squeeze().detach().cpu().numpy()
        else:
            pos_np = np.asarray(current_pos).squeeze()
        self.current_pos_marker = self.ax.scatter(pos_np[0], pos_np[1], pos_np[2],
                                                  color=self.COLORS['current_pos'], s=150, 
                                                  marker='o', edgecolors='white', 
                                                  linewidths=2, alpha=0.9,
                                                  label='Current Position', zorder=10)
        
        # Update title with episode/step info and key metrics
        avg_sigma = float(sigma_np.mean()) if len(sigma_np) > 0 else 0.0
        goals_reached_count = len(self.reached_goal_markers)
        self.ax.set_title(f'Move {step} | Avg σ: {avg_sigma:.3f} | Goals: {goals_reached_count}')
    
    def _clear_episode(self):
        """Clear all visualization elements from the previous episode."""
        # Remove all tube lines
        for line in self.tube_lines:
            line.remove()
        self.tube_lines = []
        
        # Remove all path lines
        for line in self.path_lines:
            line.remove()
        self.path_lines = []
        
        # Remove all markers
        for marker in self.tube_markers:
            marker.remove()
        self.tube_markers = []
        
        for marker in self.path_markers:
            marker.remove()
        self.path_markers = []
        
        # Remove all sphere surfaces
        for sphere in self.tube_spheres:
            sphere.remove()
        self.tube_spheres = []
        
        # Remove current position marker
        if self.current_pos_marker is not None:
            self.current_pos_marker.remove()
            self.current_pos_marker = None
        
        # Remove reached goal markers
        for marker in self.reached_goal_markers:
            marker.remove()
        self.reached_goal_markers = []
    
    def mark_goal_reached(self, goal_pos):
        """
        Mark a goal as reached and add it to the visualization.
        
        Args:
            goal_pos: (3,) numpy array or tensor with goal position
        """
        # Convert to numpy if needed
        if isinstance(goal_pos, torch.Tensor):
            goal_np = goal_pos.detach().cpu().numpy() if goal_pos.requires_grad else goal_pos.cpu().numpy()
        else:
            goal_np = np.array(goal_pos)
        
        # Ensure it's 1D
        if goal_np.ndim > 1:
            goal_np = goal_np.squeeze()
        
        # Add a marker for the reached goal (hollow outline only for contrast with active goal)
        reached_marker = self.ax.scatter(
            goal_np[0], goal_np[1], goal_np[2],
            facecolors='none', s=200, marker='*', 
            edgecolors='#D97706', linewidths=0.8, alpha=0.7, 
            label='Reached Goal' if len(self.reached_goal_markers) == 0 else '',
            zorder=9
        )
        self.reached_goal_markers.append(reached_marker)
        self.ax.legend()
    
    def replay(self, frame_delay=0.1, start_frame=0, end_frame=None):
        """
        Replay recorded training session after training completes.
        
        Args:
            frame_delay: Delay between frames in seconds (default: 0.1 = 10 FPS)
            start_frame: Starting frame index (default: 0)
            end_frame: Ending frame index (default: None = all frames)
        """
        if not self.record_mode or len(self.recorded_data) == 0:
            print("No recorded data to replay")
            return
        
        print(f"\nReplaying {len(self.recorded_data)} frames...")
        
        # Initialize plotting (was skipped in record_mode)
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self._plot_bounding_box()
        
        # Plot obstacles
        obstacle_resolution = max(8, min(12, 20 // max(1, len(self.obstacles) // 4)))
        u = np.linspace(0, 2 * np.pi, obstacle_resolution)
        v = np.linspace(0, np.pi, obstacle_resolution)
        for obs_p, obs_r in self.obstacles:
            x = obs_p[0] + obs_r * np.outer(np.cos(u), np.sin(v))
            y = obs_p[1] + obs_r * np.outer(np.sin(u), np.sin(v))
            z = obs_p[2] + obs_r * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color=self.COLORS['obstacle'], 
                               alpha=self.COLORS['obstacle_alpha'], edgecolor='none',
                               antialiased=False)
        
        # Plot start
        self.ax.scatter(0, 0, 0, color=self.COLORS['start'], s=100, 
                        marker='o', edgecolors='white', linewidths=1, label='Start')
        
        # Initialize storage
        self.tube_lines = []
        self.path_lines = []
        self.tube_spheres = []  # Store sphere surfaces for cleanup (deprecated, but needed for _clear_episode)
        self.tube_markers = []
        self.path_markers = []
        self.current_pos_marker = None
        self.reached_goal_markers = []
        self.target_marker = None
        self.current_episode = -1
        
        # Custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [self.COLORS['tube_high_agency'], self.COLORS['tube_low_agency']]
        self.agency_cmap = LinearSegmentedColormap.from_list('agency', colors_list, N=256)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Training Replay: Tube Warping Visualization\nTube color = Agency (σ): Blue=High Agency, Light Grey=Low Agency')
        self.ax.legend()
        self.ax.set_xlim(self.bounds['min'][0], self.bounds['max'][0])
        self.ax.set_ylim(self.bounds['min'][1], self.bounds['max'][1])
        self.ax.set_zlim(self.bounds['min'][2], self.bounds['max'][2])
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=self.agency_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        self.cbar = self.fig.colorbar(sm, ax=self.ax, pad=0.1, shrink=0.6)
        self.cbar.set_label('Normalized σ (Agency)', rotation=270, labelpad=15)
        
        # Replay frames
        end_frame = end_frame if end_frame is not None else len(self.recorded_data)
        frames_to_replay = self.recorded_data[start_frame:end_frame]
        
        current_goal_pos = None
        for frame_idx, frame_data in enumerate(frames_to_replay):
            # Update goal position if it changed
            if frame_data['goal_pos'] is not None:
                new_goal_pos = frame_data['goal_pos']
                if current_goal_pos is None or not np.allclose(new_goal_pos, current_goal_pos):
                    # Update target marker
                    if self.target_marker is not None:
                        self.target_marker.remove()
                    self.target_marker = self.ax.scatter(
                        new_goal_pos[0], new_goal_pos[1], new_goal_pos[2],
                        color=self.COLORS['goal'], s=300, marker='*',
                        edgecolors='#D97706', linewidths=1.5, label='Target', zorder=10
                    )
                    current_goal_pos = new_goal_pos
            
            # Convert numpy back to format expected by _render_frame
            mu_t = torch.from_numpy(frame_data['mu_t'])
            sigma_t = torch.from_numpy(frame_data['sigma_t'])
            actual_path = torch.from_numpy(frame_data['actual_path'])
            current_pos = torch.from_numpy(frame_data['current_pos']).unsqueeze(0)
            
            # Render this frame
            self._render_frame(mu_t, sigma_t, actual_path, current_pos, 
                             episode=0, step=frame_data['step'])
            
            # Render and pause for smooth playback
            self.fig.canvas.draw()
            plt.pause(frame_delay)
            
            # Keep only last N moves visible
            if len(self.tube_lines) > self.max_history:
                to_remove = len(self.tube_lines) - self.max_history
                for line in self.tube_lines[:to_remove]:
                    line.remove()
                self.tube_lines = self.tube_lines[to_remove:]
            
            if len(self.path_lines) > self.max_history:
                to_remove = len(self.path_lines) - self.max_history
                for line in self.path_lines[:to_remove]:
                    line.remove()
                self.path_lines = self.path_lines[to_remove:]
        
        print("Replay complete. Close window to exit.")
        plt.ioff()
        plt.show()
    
    def close(self):
        """Close plotter."""
        if hasattr(self, 'fig'):
            plt.ioff()
            plt.close(self.fig)
