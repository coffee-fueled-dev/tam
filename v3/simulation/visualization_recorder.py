import torch
import numpy as np
import json
import gzip
import os
from datetime import datetime


class VisualizationRecorder:
    """
    Records training data to JSONL format for web-based visualization.
    
    Generates:
    - visualization_data_{timestamp}.jsonl: Frame-by-frame training data
    - visualization_metadata_{timestamp}.json: Static scene metadata
    
    Provides the same interface as WebPlotter/LivePlotter for drop-in replacement.
    """
    
    def __init__(self, obstacles, active_goals=None, bounds=None, max_history=5, 
                 artifacts_dir=None, config=None, world_seed=None):
        """
        Args:
            obstacles: List of (position, radius) tuples
            active_goals: List of goal positions (each is a tensor or array)
            bounds: Optional dict with 'min' and 'max' keys for bounding box
            max_history: Maximum number of moves to display (for compatibility, not used)
            artifacts_dir: Directory to save output files
            config: Full configuration dictionary (for metadata)
            world_seed: Random seed used for obstacle generation
        """
        self.obstacles = obstacles
        self.active_goals = active_goals if active_goals is not None else []
        self.max_history = max_history
        
        # Set default bounds if not provided
        if bounds is None:
            self.bounds = {
                'min': [-2.0, -2.0, -2.0],
                'max': [12.0, 12.0, 12.0]
            }
        else:
            self.bounds = bounds
        
        # Initialize goal counter and energy tracking
        self.goal_counter = 0
        self.current_energy = None
        self.max_energy = None
        
        # Recorded data storage
        self.recorded_data = []  # List of frame dictionaries
        self.goal_positions = []  # Track goal positions over time
        
        # Configuration and metadata
        self.config = config if config is not None else {}
        self.world_seed = world_seed
        
        # File paths (will be set when finalize() is called)
        self.artifacts_dir = artifacts_dir
        self.jsonl_path = None
        self.metadata_path = None
        
        # Compression settings
        self.compression = "none"  # "none", "gzip", "delta"
        
        # Track state for delta encoding
        self.previous_frame = None
    
    def update(self, mu_t, sigma_t, actual_path, current_pos, episode, step, 
               active_goals=None, energy=None, max_energy=None, loss=None):
        """
        Record a frame of training data.
        
        Args:
            mu_t: (T, state_dim) planned tube trajectory (relative)
            sigma_t: (T, state_dim) tube radii (per-dimension)
            actual_path: (T, state_dim) actual path taken
            current_pos: (1, state_dim) or (state_dim,) current position
            episode: current episode number
            step: current step number
            active_goals: Optional list of active goal positions (each is a tensor or array)
            energy: Optional current energy value
            max_energy: Optional maximum energy value
            loss: Optional loss value for this move
        """
        # Convert tensors to numpy for storage
        mu_t_np = mu_t.detach().cpu().numpy() if isinstance(mu_t, torch.Tensor) else np.array(mu_t)
        sigma_t_np = sigma_t.detach().cpu().numpy() if isinstance(sigma_t, torch.Tensor) else np.array(sigma_t)
        actual_path_np = actual_path.detach().cpu().numpy() if isinstance(actual_path, torch.Tensor) else np.array(actual_path)
        
        # Handle current_pos shape
        if isinstance(current_pos, torch.Tensor):
            current_pos_np = current_pos.squeeze().detach().cpu().numpy()
        else:
            current_pos_np = np.array(current_pos).squeeze()
        
        if current_pos_np.ndim == 0:
            current_pos_np = np.array([current_pos_np])
        elif current_pos_np.ndim > 1:
            current_pos_np = current_pos_np.flatten()
        
        # Convert active_goals to list of numpy arrays
        active_goals_np = []
        if active_goals is not None:
            for goal in active_goals:
                if isinstance(goal, torch.Tensor):
                    goal_np = goal.detach().cpu().numpy() if goal.requires_grad else goal.cpu().numpy()
                else:
                    goal_np = np.array(goal)
                if goal_np.ndim > 1:
                    goal_np = goal_np.squeeze()
                active_goals_np.append(goal_np.tolist())
        
        # Calculate agency statistics from sigma_t
        # sigma_t is (T, state_dim) - flatten to get all sigma values
        agency_stats = None
        if sigma_t_np.size > 0:
            sigma_flat = sigma_t_np.flatten()
            agency_stats = {
                'min': float(np.min(sigma_flat)),
                'max': float(np.max(sigma_flat)),
                'mean': float(np.mean(sigma_flat)),
                'std': float(np.std(sigma_flat))
            }
        
        # Store frame data
        frame_data = {
            'step': int(step),
            'episode': int(episode),
            'mu_t': mu_t_np.tolist(),  # Convert to list for JSON serialization
            'sigma_t': sigma_t_np.tolist(),
            'actual_path': actual_path_np.tolist(),
            'current_pos': current_pos_np.tolist(),
            'active_goals': active_goals_np,  # List of goal positions
            'energy': float(energy) if energy is not None else None,
            'max_energy': float(max_energy) if max_energy is not None else None,
            'goal_reached': False,  # Will be set by mark_goal_reached()
            'agency': agency_stats,
            'loss': float(loss) if loss is not None else None
        }
        
        self.recorded_data.append(frame_data)
        self.current_energy = energy
        self.max_energy = max_energy
    
    def mark_goal_reached(self, goal_pos):
        """
        Mark a goal as reached.
        
        Args:
            goal_pos: (state_dim,) numpy array or tensor with goal position
        """
        self.goal_counter += 1
        
        # Convert to numpy if needed
        if isinstance(goal_pos, torch.Tensor):
            goal_np = goal_pos.detach().cpu().numpy() if goal_pos.requires_grad else goal_pos.cpu().numpy()
        else:
            goal_np = np.array(goal_pos)
        
        if goal_np.ndim > 1:
            goal_np = goal_np.squeeze()
        
        self.goal_positions.append(goal_np.tolist())
        
        # Mark the last frame as having reached a goal
        if len(self.recorded_data) > 0:
            self.recorded_data[-1]['goal_reached'] = True
    
    def _write_frame(self, frame_data, file_handle, use_delta=False):
        """
        Write a single frame to JSONL file.
        
        Args:
            frame_data: Dictionary with frame data
            file_handle: Open file handle (or gzip file handle)
            use_delta: If True, store only changes from previous frame
        """
        if use_delta and self.previous_frame is not None:
            # Delta encoding: store only changes
            delta_frame = {}
            for key, value in frame_data.items():
                if key == 'step' or key == 'episode':
                    # Always include step/episode
                    delta_frame[key] = value
                elif key == 'goal_reached':
                    # Always include goal_reached flag
                    delta_frame[key] = value
                elif self.previous_frame.get(key) != value:
                    # Only include if changed
                    delta_frame[key] = value
            json_line = json.dumps(delta_frame, separators=(',', ':')) + '\n'
        else:
            json_line = json.dumps(frame_data, separators=(',', ':')) + '\n'
        
        file_handle.write(json_line.encode() if isinstance(file_handle, gzip.GzipFile) else json_line)
        
        # Update previous frame for delta encoding
        if use_delta:
            self.previous_frame = frame_data.copy()
    
    def _write_metadata(self):
        """
        Write metadata JSON file with static scene data, config, and seed.
        """
        metadata = {
            'obstacles': [[float(p[0]), float(p[1]), float(p[2]), float(r)] 
                          for p, r in self.obstacles],
            'bounds': {
                'min': [float(x) for x in self.bounds['min']],
                'max': [float(x) for x in self.bounds['max']]
            },
            'state_dim': len(self.bounds['min']),
            'total_steps': len(self.recorded_data),
            'compression': self.compression,
            'created_at': datetime.now().isoformat(),
            'world_seed': self.world_seed,
            'config': self.config
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def finalize(self, output_dir=None, compression="none"):
        """
        Finalize recording: write JSONL file and metadata.
        
        Args:
            output_dir: Directory to save output files (uses artifacts_dir if None)
            compression: Compression type ("none", "gzip", "delta")
        """
        self.compression = compression
        
        # Determine output directory
        if output_dir is None:
            output_dir = self.artifacts_dir
        
        if output_dir is None:
            raise ValueError("output_dir or artifacts_dir must be provided")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set file paths
        jsonl_filename = f"visualization_data_{timestamp}.jsonl"
        if compression == "gzip":
            jsonl_filename += ".gz"
        
        self.jsonl_path = os.path.join(output_dir, jsonl_filename)
        self.metadata_path = os.path.join(output_dir, f"visualization_metadata_{timestamp}.json")
        
        # Write JSONL file
        use_delta = (compression == "delta")
        if compression == "gzip":
            with gzip.open(self.jsonl_path, 'wt', encoding='utf-8') as f:
                for frame_data in self.recorded_data:
                    self._write_frame(frame_data, f, use_delta=use_delta)
        else:
            with open(self.jsonl_path, 'w') as f:
                for frame_data in self.recorded_data:
                    self._write_frame(frame_data, f, use_delta=use_delta)
        
        # Write metadata
        self._write_metadata()
        
        print(f"\nVisualization data files generated:")
        print(f"  JSONL: {self.jsonl_path}")
        print(f"  Metadata: {self.metadata_path}")
    
    def close(self):
        """Close recorder (no-op, files are written in finalize())."""
        pass
