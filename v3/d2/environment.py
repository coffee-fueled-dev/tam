"""
Simple 2D environment implementation using the abstract Environment interface.

This demonstrates how to create a fresh environment that implements the TAM
Environment contract, enabling it to work with the generic train_tam_system().
"""

import torch
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from v3.environment import Environment

if TYPE_CHECKING:
    from v3.environment_recorder import EnvironmentRecorder


class Simple2DEnvironment(Environment):
    """
    Simple 2D point-mass environment with obstacles and goals.
    
    Demonstrates a clean implementation of the Environment interface.
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        bounds: Optional[dict] = None,
        obstacles: Optional[List] = None,
        goals: Optional[List] = None,
        max_observed_obstacles: int = 5,
        max_observed_goals: int = 3,
        goal_completion_threshold: float = 0.5
    ):
        """
        Initialize simple 2D environment.
        
        Args:
            state_dim: Dimension of state space (default: 2 for 2D)
            bounds: Dict with 'min' and 'max' keys, each a list of state_dim floats
            obstacles: List of (position, radius) tuples
            goals: List of goal positions (state_dim,) tensors or arrays
            max_observed_obstacles: Maximum obstacles in observation
            max_observed_goals: Maximum goals in observation
            goal_completion_threshold: Distance threshold for goal completion (default: 0.5)
        """
        self._state_dim = state_dim
        
        # Set default bounds
        if bounds is None:
            self.bounds = {
                'min': [-5.0] * state_dim,
                'max': [5.0] * state_dim
            }
        else:
            self.bounds = bounds
        
        # Convert bounds to tensors
        self.bounds_min = torch.tensor(self.bounds['min'], dtype=torch.float32)
        self.bounds_max = torch.tensor(self.bounds['max'], dtype=torch.float32)
        
        # Initialize obstacles
        self.obstacles = obstacles if obstacles is not None else []
        
        # Initialize goals (validate they don't overlap with obstacles)
        self.active_goals = []
        if goals is not None:
            goal_radius = 0.5  # Same as used in get_context()
            min_distance_from_obstacle = goal_radius + 0.1  # Small buffer
            
            for goal in goals:
                if isinstance(goal, torch.Tensor):
                    goal_tensor = goal
                else:
                    goal_tensor = torch.tensor(goal, dtype=torch.float32)
                
                # Ensure goal has correct shape
                if goal_tensor.dim() > 1:
                    goal_tensor = goal_tensor.squeeze(0)
                if goal_tensor.shape[0] != self.state_dim:
                    if goal_tensor.shape[0] < self.state_dim:
                        padding = torch.zeros(self.state_dim - goal_tensor.shape[0],
                                            dtype=torch.float32)
                        goal_tensor = torch.cat([goal_tensor, padding])
                    else:
                        goal_tensor = goal_tensor[:self.state_dim]
                
                # Check if goal overlaps with obstacles
                valid = True
                for obs_pos, obs_r in self.obstacles:
                    if isinstance(obs_pos, torch.Tensor):
                        obs_pos_tensor = obs_pos
                    else:
                        obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32)
                    
                    # Ensure obstacle position matches state_dim
                    if obs_pos_tensor.dim() > 1:
                        obs_pos_tensor = obs_pos_tensor.squeeze(0)
                    if obs_pos_tensor.shape[0] != self.state_dim:
                        if obs_pos_tensor.shape[0] < self.state_dim:
                            padding = torch.zeros(self.state_dim - obs_pos_tensor.shape[0],
                                                dtype=torch.float32)
                            obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                        else:
                            obs_pos_tensor = obs_pos_tensor[:self.state_dim]
                    
                    distance = torch.norm(goal_tensor - obs_pos_tensor).item()
                    if distance < obs_r + min_distance_from_obstacle:
                        valid = False
                        break
                
                if valid:
                    self.active_goals.append(goal_tensor)
                else:
                    # If goal overlaps with obstacle, generate a new valid one
                    print(f"Warning: Initial goal at {goal_tensor.tolist()} overlaps with obstacle. Generating replacement.")
                    new_goal = self._generate_random_goal()
                    self.active_goals.append(new_goal)
        
        # Observation parameters
        self.max_observed_obstacles = max_observed_obstacles
        self.max_observed_goals = max_observed_goals
        
        # Goal completion threshold
        self.goal_completion_threshold = goal_completion_threshold
        
        # Energy system (optional, for compatibility)
        self.max_energy = 100.0
        self.current_energy = 100.0
        self.initial_energy = 100.0
        self.energy_per_unit_distance = 0
        self.is_dead = False
        
        # Store current position for apply() method
        self._current_position = None
    
    @property
    def state_dim(self) -> int:
        """Dimension of state space."""
        return self._state_dim
    
    @property
    def context_dim(self) -> int:
        """
        Dimension of raw context observation.
        
        Structure:
        - max_observed_obstacles * (state_dim + 2) [rel_pos, radius, color]
        - max_observed_goals * (state_dim + 2) [rel_pos, radius, color]
        - 2 * state_dim [boundary distances]
        - 2 [energy_value, energy_normalized]
        """
        return (self.max_observed_obstacles * (self.state_dim + 2) +
                self.max_observed_goals * (self.state_dim + 2) +
                2 * self.state_dim + 2)
    
    def get_context(self, current_state: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        Get prior context before binding (for port affordance evaluation).
        
        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            intent: Intent/target (state_dim,) or (1, state_dim)
            
        Returns:
            raw_context: Raw context observation c_n ∈ C (raw_ctx_dim,)
        """
        # Ensure current_state is (state_dim,)
        if current_state.dim() > 1:
            current_state = current_state.squeeze(0)
        
        device = current_state.device if hasattr(current_state, 'device') else None
        
        # Get obstacle context
        obstacle_info = []
        for obs_pos, obs_r in self.obstacles:
            obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32, device=device)
            if obs_pos_tensor.shape[0] != self.state_dim:
                if obs_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(self.state_dim - obs_pos_tensor.shape[0], 
                                        dtype=torch.float32, device=device)
                    obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                else:
                    obs_pos_tensor = obs_pos_tensor[:self.state_dim]
            
            rel_obs_pos = obs_pos_tensor - current_state
            distance = torch.norm(rel_obs_pos).item()
            obstacle_info.append((distance, rel_obs_pos, obs_r))
        
        # Sort by distance and take nearest
        obstacle_info.sort(key=lambda x: x[0])
        obstacle_info = obstacle_info[:self.max_observed_obstacles]
        
        # Build obstacle features
        obs_parts = []
        for _, rel_pos, radius in obstacle_info:
            obs_parts.append(rel_pos)
            obs_parts.append(torch.tensor([radius], dtype=torch.float32, device=device))
            obs_parts.append(torch.tensor([1.0], dtype=torch.float32, device=device))  # color = 1.0 for obstacles
        
        # Pad obstacles (each obstacle has 3 parts: position, radius, color)
        num_obstacles_added = len(obstacle_info)
        while num_obstacles_added < self.max_observed_obstacles:
            obs_parts.append(torch.zeros(self.state_dim, dtype=torch.float32, device=device))
            obs_parts.append(torch.zeros(1, dtype=torch.float32, device=device))
            obs_parts.append(torch.zeros(1, dtype=torch.float32, device=device))
            num_obstacles_added += 1
        
        # Get goal context
        goal_info = []
        for goal_pos in self.active_goals:
            if isinstance(goal_pos, torch.Tensor):
                goal_pos_tensor = goal_pos.to(device) if device else goal_pos
            else:
                goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.float32, device=device)
            
            if goal_pos_tensor.dim() > 1:
                goal_pos_tensor = goal_pos_tensor.squeeze(0)
            if goal_pos_tensor.shape[0] != self.state_dim:
                if goal_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(self.state_dim - goal_pos_tensor.shape[0],
                                        dtype=torch.float32, device=device)
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[:self.state_dim]
            
            rel_goal_pos = goal_pos_tensor - current_state
            distance = torch.norm(rel_goal_pos).item()
            goal_info.append((distance, rel_goal_pos))
        
        # Sort by distance and take nearest
        goal_info.sort(key=lambda x: x[0])
        goal_info = goal_info[:self.max_observed_goals]
        
        # Build goal features
        goal_parts = []
        for _, rel_pos in goal_info:
            goal_parts.append(rel_pos)
            goal_parts.append(torch.tensor([0.5], dtype=torch.float32, device=device))  # radius
            goal_parts.append(torch.tensor([-1.0], dtype=torch.float32, device=device))  # color = -1.0 for goals
        
        # Pad goals (each goal has 3 parts: position, radius, color)
        num_goals_added = len(goal_info)
        while num_goals_added < self.max_observed_goals:
            goal_parts.append(torch.zeros(self.state_dim, dtype=torch.float32, device=device))
            goal_parts.append(torch.zeros(1, dtype=torch.float32, device=device))
            goal_parts.append(torch.zeros(1, dtype=torch.float32, device=device))
            num_goals_added += 1
        
        # Boundary distances
        boundary_distances = []
        for dim in range(self.state_dim):
            dist_to_min = current_state[dim] - self.bounds_min[dim]
            dist_to_max = self.bounds_max[dim] - current_state[dim]
            boundary_distances.append(dist_to_min)
            boundary_distances.append(dist_to_max)
        
        boundary_tensor = torch.stack(boundary_distances)
        
        # Energy information
        energy_value = torch.tensor([self.current_energy], dtype=torch.float32, device=device)
        energy_normalized = torch.tensor([self.current_energy / self.max_energy], 
                                        dtype=torch.float32, device=device)
        
        # Concatenate all parts
        raw_ctx = torch.cat([
            torch.cat(obs_parts, dim=0) if obs_parts else torch.tensor([], dtype=torch.float32, device=device),
            torch.cat(goal_parts, dim=0) if goal_parts else torch.tensor([], dtype=torch.float32, device=device),
            boundary_tensor,
            energy_value,
            energy_normalized
        ], dim=0)
        
        return raw_ctx
    
    def apply(self, next_state: torch.Tensor, env_recorder: Optional['EnvironmentRecorder'] = None) -> int:
        """
        Apply next state and update environment internal state.
        
        This is called by the training loop for each state transition.
        Updates environment state (position, energy) and optionally records to env_recorder.
        
        Args:
            next_state: Next state to apply (state_dim,)
            env_recorder: Optional environment recorder to call
            
        Returns:
            goals_reached: Number of goals reached in this step (0 or 1 typically)
        """
        # Ensure next_state is (state_dim,)
        if next_state.dim() > 1:
            next_state = next_state.squeeze(0)
        
        # Consume energy based on distance traveled
        if self._current_position is not None:
            distance = torch.norm(next_state - self._current_position).item()
            energy_cost = distance * self.energy_per_unit_distance
            if energy_cost > self.current_energy:
                # Energy depleted
                self.current_energy = 0.0
                self.is_dead = True
            else:
                self.current_energy -= energy_cost
                if self.current_energy <= 0:
                    self.is_dead = True
        
        # Update current position
        self._current_position = next_state.clone()
        
        # Check for goal completion and spawn new goal
        goals_reached = self._check_and_update_goals(next_state)
        
        # Record to environment recorder if provided
        if env_recorder is not None:
            env_recorder.record(
                agent_position=next_state,
                active_goals=self.active_goals,
                obstacles=self.obstacles,
                energy=self.current_energy,
                max_energy=self.max_energy,
                move_number=None  # Will be set by training loop
            )
        
        return goals_reached
    
    def bind_port(self, tube: torch.Tensor, sigma: torch.Tensor, 
                  current_state: torch.Tensor) -> torch.Tensor:
        """
        Bind a port (execute tube) and receive context episode.
        
        Computes physics but does NOT update environment state.
        The training loop should call apply() for each state in the returned path.
        
        Args:
            tube: Proposed trajectory τ (affordance tube) (T, state_dim) - relative to current_state
            sigma: Precision vector (T, state_dim) or (T, 1)
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            
        Returns:
            context_episode: Actual path taken e_{n→n+1} (T', state_dim)
        """
        # Ensure current_state is (state_dim,)
        if current_state.dim() > 1:
            current_state = current_state.squeeze(0)
        
        # Ensure tube is (T, state_dim)
        if tube.dim() == 1:
            tube = tube.unsqueeze(0)
        
        # Ensure sigma is (T, state_dim) or (T,)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            # Expand to (T, state_dim) if needed
            if sigma.shape[0] == tube.shape[0]:
                sigma = sigma.unsqueeze(-1).expand(-1, self.state_dim)
            else:
                sigma = sigma.unsqueeze(0).expand(tube.shape[0], self.state_dim)
        elif sigma.dim() == 2 and sigma.shape[1] == 1:
            sigma = sigma.expand(-1, self.state_dim)
        
        # Ensure sigma length matches tube length
        if sigma.shape[0] != tube.shape[0]:
            # Pad or truncate sigma to match tube length
            if sigma.shape[0] < tube.shape[0]:
                # Pad with last sigma value
                last_sigma = sigma[-1:].expand(tube.shape[0] - sigma.shape[0], -1)
                sigma = torch.cat([sigma, last_sigma], dim=0)
            else:
                # Truncate
                sigma = sigma[:tube.shape[0]]
        
        # Transform relative tube to global coordinates
        tube_global = tube + current_state
        
        # ENFORCE CAUSALITY: Actual trajectory must start at current_state
        # The tube is interpolated from knots [k_0, k_1, k_2, ...] where k_0 should be at origin (relative)
        # We ignore k_0 entirely and construct trajectory from current_state directly to k_1
        # This teaches the actor through binding failure that k_0 must be at origin (current_state)
        
        # Use current energy/dead state for physics computation (read-only)
        # Don't modify environment state here - that's done by apply()
        # We need to track energy consumption to know when to stop
        current_energy = self.current_energy
        is_dead = self.is_dead
        
        # Execute step by step
        actual_path = [current_state.clone()]
        
        # Debug: check if tube is empty or has issues
        if len(tube_global) <= 1:
            # Empty or single-point tube - return current state
            return torch.stack(actual_path)
        
        # Construct trajectory from current_state to k_1 (first non-zero point in tube)
        # The tube[0] should be k_0 (relative to current_state, so should be near origin)
        # We skip tube[0] and start from tube[1] which should be near k_1
        # This enforces that actual trajectory starts at current_state, not at k_0
        
        # Find the first meaningful target (skip k_0 which should be at origin)
        # If tube only has 1 point, it's just k_0, so return current_state
        if len(tube_global) < 2:
            return torch.stack(actual_path)
        
        # Start trajectory from current_state to tube_global[1] (k_1 in global coords)
        # Then continue with rest of tube
        # Execute step by step
        for t in range(1, len(tube_global)):
            # Check energy
            if is_dead or current_energy <= 0:
                is_dead = True
                break
            
            # Target is tube_global[t] (k_1, k_2, ... in global coordinates)
            # We're moving from actual_path[-1] (which starts at current_state) to this target
            expected_p = tube_global[t]
            actual_p = expected_p.clone()
            
            # Calculate distance traveled
            prev_pos = actual_path[-1]
            intended_dist = torch.norm(actual_p - prev_pos).item()
            
            # Check energy constraints (but don't consume - apply() will do that)
            energy_cost = intended_dist * self.energy_per_unit_distance
            if energy_cost > current_energy:
                # Scale down movement
                direction = (actual_p - prev_pos) / (intended_dist + 1e-6)
                max_dist = current_energy / self.energy_per_unit_distance
                actual_p = prev_pos + direction * max_dist
                is_dead = True
                # Don't update current_energy here - apply() will handle it
            # Note: We don't consume energy here - apply() will consume it per step
            
            # Check obstacles
            for obs_pos, obs_r in self.obstacles:
                obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32, device=actual_p.device)
                if obs_pos_tensor.shape[0] != self.state_dim:
                    if obs_pos_tensor.shape[0] < self.state_dim:
                        padding = torch.zeros(self.state_dim - obs_pos_tensor.shape[0],
                                            dtype=torch.float32, device=actual_p.device)
                        obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                    else:
                        obs_pos_tensor = obs_pos_tensor[:self.state_dim]
                
                d = torch.norm(actual_p - obs_pos_tensor)
                if d < obs_r:
                    # Collision: push away
                    vec = actual_p - obs_pos_tensor
                    if d > 1e-6:
                        actual_p = obs_pos_tensor + (vec / d) * obs_r
                    else:
                        push_dir = torch.zeros(self.state_dim, dtype=torch.float32, device=actual_p.device)
                        push_dir[0] = obs_r
                        actual_p = obs_pos_tensor + push_dir
            
            # Check boundaries
            for dim in range(self.state_dim):
                if actual_p[dim] < self.bounds_min[dim]:
                    actual_p[dim] = self.bounds_min[dim]
                elif actual_p[dim] > self.bounds_max[dim]:
                    actual_p[dim] = self.bounds_max[dim]
            
            # Binding check: if deviation exceeds sigma, stop early
            deviation = torch.norm(actual_p - expected_p)
            # Ensure we don't index out of bounds
            # Since we skip k_0 and start at k_1, use t directly (not t-1) for sigma indexing
            t_idx = min(t, sigma.shape[0] - 1)  # t starts at 1 (k_1), so use t directly
            if t_idx < 0:
                t_idx = 0
            sigma_val = sigma[t_idx].mean().item() if sigma.dim() > 1 else sigma[t_idx].item()
            
            if deviation > sigma_val:
                actual_path.append(actual_p.clone())
                break
            
            actual_path.append(actual_p.clone())
            
            if is_dead:
                break
        
        return torch.stack(actual_path)
    
    def get_intent(self, current_state: torch.Tensor, active_goals: Optional[List] = None) -> torch.Tensor:
        """
        Get intent/target for current state.
        
        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            active_goals: Optional list of goals (if None, uses self.active_goals)
            
        Returns:
            intent: Intent/target direction (state_dim,)
        """
        # Use provided active_goals or stored ones
        goals = active_goals if active_goals is not None else self.active_goals
        
        # Ensure current_state is (state_dim,)
        if current_state.dim() > 1:
            current_state = current_state.squeeze(0)
        
        # Find nearest goal
        if len(goals) == 0:
            return torch.zeros(self.state_dim, dtype=torch.float32, 
                             device=current_state.device if hasattr(current_state, 'device') else None)
        
        min_distance = float('inf')
        nearest_goal = None
        
        for goal_pos in goals:
            if isinstance(goal_pos, torch.Tensor):
                goal_pos_tensor = goal_pos
            else:
                goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.float32)
            
            if goal_pos_tensor.dim() > 1:
                goal_pos_tensor = goal_pos_tensor.squeeze(0)
            if goal_pos_tensor.shape[0] != self.state_dim:
                if goal_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(self.state_dim - goal_pos_tensor.shape[0],
                                        dtype=torch.float32)
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[:self.state_dim]
            
            distance = torch.norm(goal_pos_tensor - current_state).item()
            if distance < min_distance:
                min_distance = distance
                nearest_goal = goal_pos_tensor
        
        if nearest_goal is None:
            return torch.zeros(self.state_dim, dtype=torch.float32)
        
        # Return relative goal vector
        rel_goal = nearest_goal - current_state
        return rel_goal
    
    def _generate_random_goal(self, max_attempts: int = 100) -> torch.Tensor:
        """
        Generate a random goal position within bounds, avoiding obstacles.
        
        Args:
            max_attempts: Maximum number of attempts to find a valid position
            
        Returns:
            goal_position: Random goal position (state_dim,) tensor that doesn't overlap with obstacles
        """
        device = self.bounds_min.device if hasattr(self.bounds_min, 'device') else None
        
        # Goal radius (used for collision checking)
        goal_radius = 0.5  # Same as used in get_context()
        min_distance_from_obstacle = goal_radius + 0.1  # Small buffer
        
        for attempt in range(max_attempts):
            # Generate random position within bounds
            goal_pos = []
            for dim in range(self.state_dim):
                min_val = self.bounds_min[dim].item()
                max_val = self.bounds_max[dim].item()
                random_val = np.random.uniform(min_val, max_val)
                goal_pos.append(random_val)
            
            goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.float32, device=device)
            
            # Check if goal overlaps with any obstacle
            valid = True
            for obs_pos, obs_r in self.obstacles:
                # Convert obstacle position to tensor
                if isinstance(obs_pos, torch.Tensor):
                    obs_pos_tensor = obs_pos.to(device) if device else obs_pos
                else:
                    obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32, device=device)
                
                # Ensure obstacle position matches state_dim
                if obs_pos_tensor.dim() > 1:
                    obs_pos_tensor = obs_pos_tensor.squeeze(0)
                if obs_pos_tensor.shape[0] != self.state_dim:
                    if obs_pos_tensor.shape[0] < self.state_dim:
                        padding = torch.zeros(self.state_dim - obs_pos_tensor.shape[0],
                                            dtype=torch.float32, device=device)
                        obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                    else:
                        obs_pos_tensor = obs_pos_tensor[:self.state_dim]
                
                # Check distance to obstacle
                distance = torch.norm(goal_pos_tensor - obs_pos_tensor).item()
                if distance < obs_r + min_distance_from_obstacle:
                    valid = False
                    break
            
            if valid:
                return goal_pos_tensor
        
        # If all attempts failed, try center position as fallback
        # This should rarely happen unless obstacles fill the entire space
        center = []
        for dim in range(self.state_dim):
            center_val = (self.bounds_min[dim].item() + self.bounds_max[dim].item()) / 2.0
            center.append(center_val)
        center_tensor = torch.tensor(center, dtype=torch.float32, device=device)
        
        # Check if center is valid (doesn't overlap with obstacles)
        center_valid = True
        for obs_pos, obs_r in self.obstacles:
            if isinstance(obs_pos, torch.Tensor):
                obs_pos_tensor = obs_pos.to(device) if device else obs_pos
            else:
                obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32, device=device)
            
            if obs_pos_tensor.dim() > 1:
                obs_pos_tensor = obs_pos_tensor.squeeze(0)
            if obs_pos_tensor.shape[0] != self.state_dim:
                if obs_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(self.state_dim - obs_pos_tensor.shape[0],
                                        dtype=torch.float32, device=device)
                    obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                else:
                    obs_pos_tensor = obs_pos_tensor[:self.state_dim]
            
            distance = torch.norm(center_tensor - obs_pos_tensor).item()
            if distance < obs_r + min_distance_from_obstacle:
                center_valid = False
                break
        
        if center_valid:
            return center_tensor
        
        # Last resort: return a position near the minimum bounds (should be less likely to have obstacles)
        fallback = []
        for dim in range(self.state_dim):
            fallback_val = self.bounds_min[dim].item() + 0.5
            fallback.append(fallback_val)
        return torch.tensor(fallback, dtype=torch.float32, device=device)
    
    def _check_and_update_goals(self, current_position: torch.Tensor) -> int:
        """
        Check if any goal has been reached and update goals accordingly.
        
        If a goal is reached (within threshold distance), remove it and add a new
        random goal.
        
        Args:
            current_position: Current agent position (state_dim,)
            
        Returns:
            goals_reached: Number of goals reached (and removed) in this step
        """
        # Ensure current_position is (state_dim,)
        if current_position.dim() > 1:
            current_position = current_position.squeeze(0)
        
        device = current_position.device if hasattr(current_position, 'device') else None
        
        # Check each goal for completion
        goals_to_remove = []
        for i, goal_pos in enumerate(self.active_goals):
            # Normalize goal position
            if isinstance(goal_pos, torch.Tensor):
                goal_pos_tensor = goal_pos.to(device) if device else goal_pos
            else:
                goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.float32, device=device)
            
            if goal_pos_tensor.dim() > 1:
                goal_pos_tensor = goal_pos_tensor.squeeze(0)
            if goal_pos_tensor.shape[0] != self.state_dim:
                if goal_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(self.state_dim - goal_pos_tensor.shape[0],
                                        dtype=torch.float32, device=device)
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[:self.state_dim]
            
            # Check distance to goal
            distance = torch.norm(goal_pos_tensor - current_position).item()
            if distance <= self.goal_completion_threshold:
                goals_to_remove.append(i)
        
        # Remove completed goals (in reverse order to maintain indices)
        for i in reversed(goals_to_remove):
            self.active_goals.pop(i)
        
        # Add new random goal for each completed goal
        for _ in goals_to_remove:
            new_goal = self._generate_random_goal()
            self.active_goals.append(new_goal)
        
        return len(goals_to_remove)
    
    def replenish_energy(self, amount: float):
        """Replenish energy by a fixed amount."""
        self.current_energy = min(self.max_energy, self.current_energy + amount)
    
    def reset_energy(self):
        """Reset energy to initial value."""
        self.current_energy = self.initial_energy
        self.is_dead = False
