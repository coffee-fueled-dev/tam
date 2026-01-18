import torch
import numpy as np

class SimulationWrapper:
    """
    Dimension-agnostic simulation wrapper for environment physics.
    
    Works with any N-dimensional state space. Obstacles and bounds are
    automatically handled for the specified state dimension.
    """
    def __init__(self, obstacles, state_dim=3, bounds=None, energy_config=None):
        """
        Args:
            obstacles: List of (position, radius) tuples where position is (state_dim,) array/list
            state_dim: Dimension of state space (e.g., 3 for 3D, 6 for 6D)
            bounds: Optional dict with 'min' and 'max' keys, each a list of state_dim floats
                   If None, defaults to [-2, -2, ...] to [12, 12, ...] for each dimension
            energy_config: Optional dict with energy parameters:
                          - max_energy: Maximum energy (default: 100.0)
                          - initial_energy: Starting energy (default: max_energy)
                          - energy_per_unit_distance: Energy cost per unit distance (default: 1.0)
        """
        self.state_dim = state_dim
        self.obstacles = obstacles
        
        # Set default bounds if not provided
        if bounds is None:
            self.bounds = {
                'min': [-2.0] * state_dim,
                'max': [12.0] * state_dim
            }
        else:
            # Ensure bounds match state_dim
            min_bounds = list(bounds['min'])
            max_bounds = list(bounds['max'])
            # Pad or truncate to match state_dim
            if len(min_bounds) < state_dim:
                min_bounds.extend([-2.0] * (state_dim - len(min_bounds)))
            elif len(min_bounds) > state_dim:
                min_bounds = min_bounds[:state_dim]
            if len(max_bounds) < state_dim:
                max_bounds.extend([12.0] * (state_dim - len(max_bounds)))
            elif len(max_bounds) > state_dim:
                max_bounds = max_bounds[:state_dim]
            self.bounds = {'min': min_bounds, 'max': max_bounds}
        
        # Convert to tensors for easier computation
        self.bounds_min = torch.tensor(self.bounds['min'], dtype=torch.float32)
        self.bounds_max = torch.tensor(self.bounds['max'], dtype=torch.float32)
        
        # Initialize energy system
        if energy_config is None:
            energy_config = {}
        self.max_energy = energy_config.get('max_energy', 100.0)
        self.initial_energy = energy_config.get('initial_energy', self.max_energy)
        self.energy_per_unit_distance = energy_config.get('energy_per_unit_distance', 1.0)
        self.current_energy = self.initial_energy
        self.is_dead = False  # Track death state
    
    def get_raw_observation(self, current_pos, target_pos, max_obstacles=10):
        """
        Get RAW context observation (for InferenceEngine).
        
        This is the raw, messy context from the world that the InferenceEngine
        will compress into a latent situation. The Actor never sees this directly.
        
        Args:
            current_pos: (1, state_dim) or (state_dim,) current position tensor
            target_pos: (state_dim,) or (1, state_dim) target position tensor
            max_obstacles: Maximum number of obstacles to include (default: 10)
                          Observation is always this size, padded with zeros if fewer obstacles
        
        Returns:
            raw_ctx: torch.Tensor of shape (raw_ctx_dim,)
            raw_ctx_dim = state_dim (rel_goal) 
                        + max_obstacles * (state_dim + 1) (rel_obs_pos + obs_radius)
                        + 2 * state_dim (boundary distances: dist_to_min, dist_to_max per dimension)
        """
        # Ensure current_pos is (state_dim,)
        if current_pos.dim() > 1:
            current_pos_flat = current_pos.squeeze(0)
        else:
            current_pos_flat = current_pos
        
        # Ensure target_pos is (state_dim,)
        if target_pos.dim() > 1:
            target_pos_flat = target_pos.squeeze(0)
        else:
            target_pos_flat = target_pos
        
        # Ensure dimensions match state_dim
        if current_pos_flat.shape[0] != self.state_dim:
            # Pad or truncate
            if current_pos_flat.shape[0] < self.state_dim:
                padding = torch.zeros(self.state_dim - current_pos_flat.shape[0], 
                                    dtype=current_pos_flat.dtype, device=current_pos_flat.device)
                current_pos_flat = torch.cat([current_pos_flat, padding])
            else:
                current_pos_flat = current_pos_flat[:self.state_dim]
        
        if target_pos_flat.shape[0] != self.state_dim:
            if target_pos_flat.shape[0] < self.state_dim:
                padding = torch.zeros(self.state_dim - target_pos_flat.shape[0],
                                    dtype=target_pos_flat.dtype, device=target_pos_flat.device)
                target_pos_flat = torch.cat([target_pos_flat, padding])
            else:
                target_pos_flat = target_pos_flat[:self.state_dim]
        
        # Relative goal vector
        rel_goal = target_pos_flat - current_pos_flat  # (state_dim,)
        
        # Get obstacle context: relative positions and radii
        # Sort obstacles by distance to current position (nearest first)
        device = current_pos_flat.device if hasattr(current_pos_flat, 'device') else None
        obstacle_info = []
        for obs_p, obs_r in self.obstacles:
            obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=device)
            # Ensure obstacle position matches state_dim
            if obs_pos.shape[0] != self.state_dim:
                if obs_pos.shape[0] < self.state_dim:
                    padding = torch.zeros(self.state_dim - obs_pos.shape[0], dtype=obs_pos.dtype, device=device)
                    obs_pos = torch.cat([obs_pos, padding])
                else:
                    obs_pos = obs_pos[:self.state_dim]
            
            rel_obs_pos = obs_pos - current_pos_flat  # (state_dim,)
            obs_radius = torch.tensor([obs_r], dtype=torch.float32, device=device)  # (1,)
            distance = torch.norm(rel_obs_pos)
            obstacle_info.append((distance.item(), rel_obs_pos, obs_radius))
        
        # Sort by distance and take nearest max_obstacles
        obstacle_info.sort(key=lambda x: x[0])
        obstacle_info = obstacle_info[:max_obstacles]
        
        # Build obstacle features: [rel_pos (state_dim), radius] for each obstacle
        obs_parts = [rel_goal]
        for _, rel_obs_pos, obs_radius in obstacle_info:
            obs_parts.append(rel_obs_pos)  # (state_dim,)
            obs_parts.append(obs_radius)   # (1,)
        
        # Pad with zeros if we have fewer than max_obstacles
        # The actor can learn to ignore zero-padded obstacles (radius=0 means no obstacle)
        # This decouples observation from exact obstacle count - add/remove obstacles freely!
        num_obstacles_present = len(obstacle_info)
        if num_obstacles_present < max_obstacles:
            # Pad with zero vectors: [0, ..., 0, 0] for each missing obstacle
            zeros_state = torch.zeros(self.state_dim, dtype=torch.float32, device=device)
            zeros_1d = torch.zeros(1, dtype=torch.float32, device=device)
            for _ in range(max_obstacles - num_obstacles_present):
                obs_parts.append(zeros_state)  # Zero relative position
                obs_parts.append(zeros_1d)  # Zero radius (indicates no obstacle - actor learns to ignore)
        
        # Add boundary information: distance to each boundary per dimension
        # This enables proactive boundary avoidance (same principle as obstacles)
        # For each dimension, compute distance to min and max boundaries
        boundary_distances = []
        for dim in range(self.state_dim):
            dist_to_min = current_pos_flat[dim] - self.bounds_min[dim]  # Positive = inside bounds
            dist_to_max = self.bounds_max[dim] - current_pos_flat[dim]  # Positive = inside bounds
            boundary_distances.append(dist_to_min)
            boundary_distances.append(dist_to_max)
        
        boundary_tensor = torch.stack(boundary_distances)  # (2 * state_dim,)
        
        # Add energy information: raw value and normalized (0-1)
        energy_value = torch.tensor([self.current_energy], dtype=torch.float32, device=device)
        energy_normalized = torch.tensor([self.current_energy / self.max_energy], dtype=torch.float32, device=device)
        
        # Concatenate all parts: [rel_goal, obstacles..., boundaries, energy_value, energy_normalized]
        raw_ctx = torch.cat([torch.cat(obs_parts, dim=0), boundary_tensor, energy_value, energy_normalized], dim=0)
        # Shape: (state_dim + max_obstacles*(state_dim+1) + 2*state_dim + 2,)
        
        return raw_ctx
    
    def execute(self, mu_t, sigma_t, current_pos):
        """
        Execute a tube in the world, checking for obstacles and binding violations.
        
        PHILOSOPHY: All physics (obstacles, boundaries, etc.) creates deviation from
        the planned tube. This deviation surfaces through binding failure:
        - Exception: binding failed (deviation > sigma)
        - Magnitude: deviation distance
        - Position: where the violation occurred (actual_p after physics)
        
        Args:
            mu_t: (T, 3) relative tube trajectory
            sigma_t: (T, 1) tube radii
            current_pos: (1, 3) current position
        
        Returns:
            actual_p: (T, 3) actual path taken
        """
        # Ensure mu_t and sigma_t are properly shaped - should be (T, state_dim) and (T, 1) or (T,)
        if mu_t.dim() > 2:
            mu_t = mu_t.view(-1, self.state_dim)
        elif mu_t.dim() == 1:
            mu_t = mu_t.unsqueeze(0)
        
        if sigma_t.dim() > 2:
            sigma_t = sigma_t.view(-1)
        elif sigma_t.dim() == 2:
            sigma_t = sigma_t.squeeze(-1) if sigma_t.shape[1] == 1 else sigma_t[:, 0]
        elif sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
        
        # Transform relative tube to global coordinates
        mu_global = mu_t + current_pos
        
        actual_path = []
        current_pos_squeezed = current_pos.squeeze()
        # Ensure current_pos matches state_dim
        if current_pos_squeezed.shape[0] != self.state_dim:
            if current_pos_squeezed.shape[0] < self.state_dim:
                padding = torch.zeros(self.state_dim - current_pos_squeezed.shape[0],
                                    dtype=current_pos_squeezed.dtype, device=current_pos_squeezed.device)
                current_pos_squeezed = torch.cat([current_pos_squeezed, padding])
            else:
                current_pos_squeezed = current_pos_squeezed[:self.state_dim]
        actual_path.append(current_pos_squeezed.clone())
        
        # If tube is too short (only 1 point), return current position
        if len(mu_global) <= 1:
            return torch.stack(actual_path)
        
        # Calculate maximum distance that can be traveled with current energy
        # This limits movement capability as energy depletes
        max_travel_distance = self.current_energy / self.energy_per_unit_distance if self.current_energy > 0 else 0.0
        
        # Execute step by step
        for t in range(1, len(mu_global)):
            # Check if dead (energy depleted)
            if self.is_dead or self.current_energy <= 0:
                # Can't move - return partial path (creates binding failure)
                self.is_dead = True
                break
            
            expected_p = mu_global[t]
            
            # Handle sigma_t indexing - ensure we get a scalar
            t_idx = min(t, len(sigma_t) - 1)
            affordance_r = sigma_t[t_idx].item() if hasattr(sigma_t[t_idx], 'item') else float(sigma_t[t_idx])
            
            # Start with expected position
            actual_p = expected_p.clone()
            
            # Calculate distance traveled from previous position
            prev_pos = actual_path[-1] if len(actual_path) > 0 else current_pos_squeezed
            intended_dist = torch.norm(actual_p - prev_pos).item()
            
            # Limit movement based on available energy
            # If intended distance exceeds what we can afford, scale it down
            if intended_dist > max_travel_distance and max_travel_distance > 0:
                # Scale down the movement to stay within energy budget
                direction = (actual_p - prev_pos) / (intended_dist + 1e-6)
                actual_p = prev_pos + direction * max_travel_distance
                dist_traveled = max_travel_distance
            else:
                dist_traveled = intended_dist
            
            # Consume energy based on distance traveled
            energy_consumed = dist_traveled * self.energy_per_unit_distance
            self.current_energy -= energy_consumed
            
            # Update max travel distance for next segment
            max_travel_distance = self.current_energy / self.energy_per_unit_distance if self.current_energy > 0 else 0.0
            
            # Check if energy depleted during this segment
            if self.current_energy <= 0:
                # Energy depleted - stop execution early
                # Partial path creates binding failure (planned path longer than actual)
                self.is_dead = True
                # Add current position before death
                actual_path.append(actual_p.clone())
                break
            
            # Check for obstacles (physics that creates deviation)
            device = actual_p.device
            for obs_p, obs_r in self.obstacles:
                obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=device)
                # Ensure obstacle position matches state_dim
                if obs_pos.shape[0] != self.state_dim:
                    if obs_pos.shape[0] < self.state_dim:
                        padding = torch.zeros(self.state_dim - obs_pos.shape[0], dtype=obs_pos.dtype, device=device)
                        obs_pos = torch.cat([obs_pos, padding])
                    else:
                        obs_pos = obs_pos[:self.state_dim]
                
                d = torch.norm(actual_p - obs_pos)
                if d < obs_r:
                    # Collision: push away from obstacle (creates deviation from expected)
                    vec = actual_p - obs_pos
                    if d > 1e-6:
                        actual_p = obs_pos + (vec / d) * obs_r
                    else:
                        # If exactly on obstacle, push in first dimension direction
                        push_dir = torch.zeros(self.state_dim, dtype=torch.float32, device=actual_p.device)
                        push_dir[0] = obs_r
                        actual_p = obs_pos + push_dir
            
            # Check boundaries (physics that creates deviation - same as obstacles)
            # Boundaries should push back, creating deviation that can trigger binding failure
            bounds_min = self.bounds_min.to(device)
            bounds_max = self.bounds_max.to(device)
            
            # Check each dimension for boundary violations
            for dim in range(self.state_dim):
                if actual_p[dim] < bounds_min[dim]:
                    # Violated lower bound: push to boundary (creates deviation)
                    actual_p[dim] = bounds_min[dim]
                elif actual_p[dim] > bounds_max[dim]:
                    # Violated upper bound: push to boundary (creates deviation)
                    actual_p[dim] = bounds_max[dim]
            
            # Binding check: if deviation exceeds sigma, stop early
            # This captures ALL physics deviations: obstacles, boundaries, etc.
            # Exception: binding failed
            # Magnitude: deviation distance
            # Position: where the violation occurred (actual_p)
            deviation = torch.norm(actual_p - expected_p)
            if deviation > affordance_r:
                # Binding broken - physics (obstacle/boundary) created too much deviation
                # Record the exception: position where binding failed
                actual_path.append(actual_p.clone())
                break
            
            actual_path.append(actual_p.clone())
        
        return torch.stack(actual_path)
    
    def replenish_energy(self, amount):
        """
        Replenish energy by a fixed amount (called when goal is reached).
        
        Args:
            amount: Energy amount to add
        """
        self.current_energy = min(self.max_energy, self.current_energy + amount)
    
    def reset_energy(self):
        """Reset energy to initial value (for episode resets)."""
        self.current_energy = self.initial_energy
        self.is_dead = False
      