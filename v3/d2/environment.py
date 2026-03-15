"""
Simple 2D environment implementation using the abstract Environment interface.

This demonstrates how to create a fresh environment that implements the TAM
Environment contract, enabling it to work with the generic train_tam_system().
"""

import torch
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from v3.environment import Environment
from v3.goal_motif import (
    GoalMotif,
    extract_motif_from_tkn_output,
    create_motif_from_position,
)

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
        max_observed_obstacles: Optional[int] = None,  # None = no limit, observe all
        max_observed_goals: Optional[int] = None,  # None = no limit, observe all
        goal_completion_threshold: float = 0.5,
    ):
        """
        Initialize simple 2D environment.

        Args:
            state_dim: Dimension of state space (default: 2 for 2D)
            bounds: Dict with 'min' and 'max' keys, each a list of state_dim floats
            obstacles: List of (position, radius) tuples
            goals: List of goal positions (state_dim,) tensors or arrays
            max_observed_obstacles: Maximum obstacles in observation (None = no limit, observe all)
            max_observed_goals: Maximum goals in observation (None = no limit, observe all)
            goal_completion_threshold: Distance threshold for goal completion (default: 0.5)
        """
        self._state_dim = state_dim

        # Set default bounds
        if bounds is None:
            self.bounds = {"min": [-5.0] * state_dim, "max": [5.0] * state_dim}
        else:
            self.bounds = bounds

        # Convert bounds to tensors
        self.bounds_min = torch.tensor(self.bounds["min"], dtype=torch.float32)
        self.bounds_max = torch.tensor(self.bounds["max"], dtype=torch.float32)

        # Initialize obstacles
        self.obstacles = obstacles if obstacles is not None else []

        # Initialize goals (validate they don't overlap with obstacles)
        # Support both position-based (for backward compatibility) and motif-based goals
        self.active_goals = []  # Keep for backward compatibility / learning
        self.active_goal_motifs = []  # Primary: motif-based goals
        self._use_motif_goals = False  # Flag to switch between modes
        self.reached_goals = []  # Track goals that have been reached (for visualization)
        self.reached_goals = []  # Track goals that have been reached (for visualization)

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
                        padding = torch.zeros(
                            self.state_dim - goal_tensor.shape[0], dtype=torch.float32
                        )
                        goal_tensor = torch.cat([goal_tensor, padding])
                    else:
                        goal_tensor = goal_tensor[: self.state_dim]

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
                            padding = torch.zeros(
                                self.state_dim - obs_pos_tensor.shape[0],
                                dtype=torch.float32,
                            )
                            obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                        else:
                            obs_pos_tensor = obs_pos_tensor[: self.state_dim]

                    distance = torch.norm(goal_tensor - obs_pos_tensor).item()
                    if distance < obs_r + min_distance_from_obstacle:
                        valid = False
                        break

                if valid:
                    self.active_goals.append(goal_tensor)
                else:
                    # If goal overlaps with obstacle, generate a new valid one
                    print(
                        f"Warning: Initial goal at {goal_tensor.tolist()} overlaps with obstacle. Generating replacement."
                    )
                    new_goal = self._generate_random_goal()
                    self.active_goals.append(new_goal)

        # Observation parameters (None = no limit, observe all)
        self.max_observed_obstacles = (
            max_observed_obstacles  # None means observe all obstacles
        )
        self.max_observed_goals = max_observed_goals  # None means observe all goals

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
        
        # Store TKN processor for extracting patterns from new goals
        self._tkn_processor = None

    @property
    def state_dim(self) -> int:
        """Dimension of state space."""
        return self._state_dim

    @property
    def context_dim(self) -> int:
        """
        Dimension of raw context observation.

        Structure:
        - num_obstacles * (state_dim + 2) [rel_pos, radius, color] (all obstacles if max_observed_obstacles is None)
        - num_goals * (state_dim + 2) [rel_pos, radius, color] (all goals if max_observed_goals is None)
        - 2 * state_dim [boundary distances]
        - 2 [energy_value, energy_normalized]

        Note: If max_observed_obstacles/goals is None, uses actual count of obstacles/goals.
        """
        num_obstacles = (
            self.max_observed_obstacles
            if self.max_observed_obstacles is not None
            else len(self.obstacles)
        )
        num_goals = (
            self.max_observed_goals
            if self.max_observed_goals is not None
            else len(self.active_goals)
        )
        return (
            num_obstacles * (self.state_dim + 2)
            + num_goals * (self.state_dim + 2)
            + 2 * self.state_dim
            + 2
        )

    def get_context(
        self, current_state: torch.Tensor, intent: torch.Tensor
    ) -> torch.Tensor:
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

        device = current_state.device if hasattr(current_state, "device") else None

        # Get obstacle context
        obstacle_info = []
        for obs_pos, obs_r in self.obstacles:
            obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32, device=device)
            if obs_pos_tensor.shape[0] != self.state_dim:
                if obs_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(
                        self.state_dim - obs_pos_tensor.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                else:
                    obs_pos_tensor = obs_pos_tensor[: self.state_dim]

            rel_obs_pos = obs_pos_tensor - current_state
            distance = torch.norm(rel_obs_pos).item()
            obstacle_info.append((distance, rel_obs_pos, obs_r))

        # Sort by distance
        obstacle_info.sort(key=lambda x: x[0])

        # Limit to max_observed_obstacles if specified, otherwise use all
        if self.max_observed_obstacles is not None:
            obstacle_info = obstacle_info[: self.max_observed_obstacles]

        # Build obstacle features
        obs_parts = []
        for _, rel_pos, radius in obstacle_info:
            obs_parts.append(rel_pos)
            obs_parts.append(torch.tensor([radius], dtype=torch.float32, device=device))
            obs_parts.append(
                torch.tensor([1.0], dtype=torch.float32, device=device)
            )  # color = 1.0 for obstacles

        # Pad obstacles only if max_observed_obstacles is set (for fixed-size context)
        if self.max_observed_obstacles is not None:
            num_obstacles_added = len(obstacle_info)
            while num_obstacles_added < self.max_observed_obstacles:
                obs_parts.append(
                    torch.zeros(self.state_dim, dtype=torch.float32, device=device)
                )
                obs_parts.append(torch.zeros(1, dtype=torch.float32, device=device))
                obs_parts.append(torch.zeros(1, dtype=torch.float32, device=device))
                num_obstacles_added += 1

        # Get goal context
        goal_info = []
        for goal_pos in self.active_goals:
            if isinstance(goal_pos, torch.Tensor):
                goal_pos_tensor = goal_pos.to(device) if device else goal_pos
            else:
                goal_pos_tensor = torch.tensor(
                    goal_pos, dtype=torch.float32, device=device
                )

            if goal_pos_tensor.dim() > 1:
                goal_pos_tensor = goal_pos_tensor.squeeze(0)
            if goal_pos_tensor.shape[0] != self.state_dim:
                if goal_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(
                        self.state_dim - goal_pos_tensor.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[: self.state_dim]

            rel_goal_pos = goal_pos_tensor - current_state
            distance = torch.norm(rel_goal_pos).item()
            goal_info.append((distance, rel_goal_pos, True))  # True = active goal

        # Add reached goals (color = -0.5 to distinguish from active goals)
        for goal_pos in self.reached_goals:
            if isinstance(goal_pos, torch.Tensor):
                goal_pos_tensor = goal_pos.to(device) if device else goal_pos
            else:
                goal_pos_tensor = torch.tensor(
                    goal_pos, dtype=torch.float32, device=device
                )

            if goal_pos_tensor.dim() > 1:
                goal_pos_tensor = goal_pos_tensor.squeeze(0)
            if goal_pos_tensor.shape[0] != self.state_dim:
                if goal_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(
                        self.state_dim - goal_pos_tensor.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[: self.state_dim]

            rel_goal_pos = goal_pos_tensor - current_state
            distance = torch.norm(rel_goal_pos).item()
            goal_info.append((distance, rel_goal_pos, False))  # False = reached goal

        # Sort by distance
        goal_info.sort(key=lambda x: x[0])

        # Limit to max_observed_goals if specified, otherwise use all
        if self.max_observed_goals is not None:
            goal_info = goal_info[: self.max_observed_goals]

        # Build goal features
        goal_parts = []
        for _, rel_pos, is_active in goal_info:
            goal_parts.append(rel_pos)
            goal_parts.append(
                torch.tensor([0.5], dtype=torch.float32, device=device)
            )  # radius
            # Use different colors: -1.0 for active goals, -0.5 for reached goals
            goal_color = -1.0 if is_active else -0.5
            goal_parts.append(
                torch.tensor([goal_color], dtype=torch.float32, device=device)
            )

        # Pad goals only if max_observed_goals is set (for fixed-size context)
        if self.max_observed_goals is not None:
            num_goals_added = len(goal_info)
            while num_goals_added < self.max_observed_goals:
                goal_parts.append(
                    torch.zeros(self.state_dim, dtype=torch.float32, device=device)
                )
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
        energy_value = torch.tensor(
            [self.current_energy], dtype=torch.float32, device=device
        )
        energy_normalized = torch.tensor(
            [self.current_energy / self.max_energy], dtype=torch.float32, device=device
        )

        # Concatenate all parts
        raw_ctx = torch.cat(
            [
                torch.cat(obs_parts, dim=0)
                if obs_parts
                else torch.tensor([], dtype=torch.float32, device=device),
                torch.cat(goal_parts, dim=0)
                if goal_parts
                else torch.tensor([], dtype=torch.float32, device=device),
                boundary_tensor,
                energy_value,
                energy_normalized,
            ],
            dim=0,
        )

        return raw_ctx

    def apply(
        self,
        next_state: torch.Tensor,
        env_recorder: Optional["EnvironmentRecorder"] = None,
    ) -> int:
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
                move_number=None,  # Will be set by training loop
            )

        return goals_reached

    def bind_port(
        self, tube: torch.Tensor, sigma: torch.Tensor, current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Bind a port (execute single next step) and receive context episode.

        Computes physics but does NOT update environment state.
        The training loop should call apply() for each state in the returned path.

        Args:
            tube: Proposed next step (1, state_dim) - relative to current_state
            sigma: Precision vector (1, state_dim) - per-dimension precision for the step
            current_state: Current state x_n (state_dim,) or (1, state_dim)

        Returns:
            context_episode: Actual path taken e_{n→n+1} (2, state_dim) - [current_state, next_state]
        """
        # Ensure current_state is (state_dim,)
        if current_state.dim() > 1:
            current_state = current_state.squeeze(0)

        # Ensure tube is (1, state_dim) - single step
        if tube.dim() == 1:
            tube = tube.unsqueeze(0)
        elif tube.dim() > 2:
            tube = tube.squeeze()
        if tube.dim() == 1:
            tube = tube.unsqueeze(0)

        # Ensure sigma is (1, state_dim)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).unsqueeze(0)
        elif sigma.dim() == 1:
            # (state_dim,) -> (1, state_dim)
            sigma = sigma.unsqueeze(0)
        elif sigma.dim() == 2:
            # Already (1, state_dim) or (T, state_dim) - take first if needed
            if sigma.shape[0] > 1:
                sigma = sigma[0:1]
            if sigma.shape[1] != self.state_dim:
                # Expand to (1, state_dim) if needed
                sigma = sigma.expand(-1, self.state_dim)

        # Transform relative tube to global coordinates
        # tube is (1, state_dim), current_state is (state_dim,)
        expected_p = tube[0] + current_state  # (state_dim,)

        # Use current energy/dead state for physics computation (read-only)
        current_energy = self.current_energy

        # Start path with current state
        actual_path = [current_state.clone()]

        # Compute actual next position with physics checks
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

        # Interpolate path and check collisions at multiple points
        # Number of interpolation steps based on distance (at least 10 steps, more for longer moves)
        num_steps = max(
            10, int(intended_dist * 10)
        )  # 10 steps per unit distance, minimum 10
        start_pos = prev_pos.clone()
        end_pos = actual_p.clone()

        # Interpolate path and check for collisions/boundaries at each step
        for step_idx in range(1, num_steps + 1):
            # Interpolation parameter: 0 = start, 1 = end
            t = step_idx / num_steps
            interpolated_pos = start_pos + (end_pos - start_pos) * t

            # Check obstacles at this interpolated position
            collision_occurred = False
            for obs_pos, obs_r in self.obstacles:
                obs_pos_tensor = torch.tensor(
                    obs_pos, dtype=torch.float32, device=interpolated_pos.device
                )
                if obs_pos_tensor.shape[0] != self.state_dim:
                    if obs_pos_tensor.shape[0] < self.state_dim:
                        padding = torch.zeros(
                            self.state_dim - obs_pos_tensor.shape[0],
                            dtype=torch.float32,
                            device=interpolated_pos.device,
                        )
                        obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                    else:
                        obs_pos_tensor = obs_pos_tensor[: self.state_dim]

                d = torch.norm(interpolated_pos - obs_pos_tensor)
                if d < obs_r:
                    # Collision detected: stop path at previous valid position
                    # Push away from obstacle at the collision point
                    vec = interpolated_pos - obs_pos_tensor
                    if d > 1e-6:
                        # Push to surface of obstacle
                        collision_pos = obs_pos_tensor + (vec / d) * obs_r
                    else:
                        # If exactly at center, push in a default direction
                        push_dir = torch.zeros(
                            self.state_dim,
                            dtype=torch.float32,
                            device=interpolated_pos.device,
                        )
                        push_dir[0] = obs_r
                        collision_pos = obs_pos_tensor + push_dir

                    # Add collision position to path and stop
                    actual_path.append(collision_pos.clone())
                    collision_occurred = True
                    break

            if collision_occurred:
                break

            # Check boundaries at this interpolated position
            boundary_violation = False
            clamped_pos = interpolated_pos.clone()
            for dim in range(self.state_dim):
                if clamped_pos[dim] < self.bounds_min[dim]:
                    clamped_pos[dim] = self.bounds_min[dim]
                    boundary_violation = True
                elif clamped_pos[dim] > self.bounds_max[dim]:
                    clamped_pos[dim] = self.bounds_max[dim]
                    boundary_violation = True

            # If boundary violated, stop at boundary
            if boundary_violation:
                actual_path.append(clamped_pos.clone())
                break

            # If this is the last step, add the final position
            if step_idx == num_steps:
                actual_path.append(interpolated_pos.clone())

        # If no collisions occurred, ensure we have at least the end position
        if len(actual_path) == 1:
            # No collisions, add final position (with boundary clamping)
            final_pos = end_pos.clone()
            for dim in range(self.state_dim):
                if final_pos[dim] < self.bounds_min[dim]:
                    final_pos[dim] = self.bounds_min[dim]
                elif final_pos[dim] > self.bounds_max[dim]:
                    final_pos[dim] = self.bounds_max[dim]
            actual_path.append(final_pos.clone())

        return torch.stack(actual_path)

    def get_intent(
        self,
        current_state: torch.Tensor,
        goal_motif: Optional[GoalMotif] = None,
        active_goals: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Get intent/target for current state.

        Uses motif-based navigation if goal_motif provided, else falls back to position-based.

        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            goal_motif: Optional GoalMotif to navigate toward (if provided, uses motif-based navigation)
            active_goals: Optional list of goal positions (for backward compatibility)

        Returns:
            intent: Intent/target direction (state_dim,)
        """
        # Ensure current_state is (state_dim,)
        if current_state.dim() > 1:
            current_state = current_state.squeeze(0)

        # If goal_motif provided, return zero intent (will be generated by Actor's intent generator)
        # The training loop will use Actor.generate_intent_from_motif() instead
        if goal_motif is not None:
            # Return zero intent - Actor will generate intent from motif
            return torch.zeros(
                self.state_dim,
                dtype=torch.float32,
                device=current_state.device
                if hasattr(current_state, "device")
                else None,
            )

        # Fallback to position-based intent (backward compatibility)
        # Use provided active_goals or stored ones
        goals = active_goals if active_goals is not None else self.active_goals

        # Find nearest goal
        if len(goals) == 0:
            return torch.zeros(
                self.state_dim,
                dtype=torch.float32,
                device=current_state.device
                if hasattr(current_state, "device")
                else None,
            )

        min_distance = float("inf")
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
                    padding = torch.zeros(
                        self.state_dim - goal_pos_tensor.shape[0], dtype=torch.float32
                    )
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[: self.state_dim]

            distance = torch.norm(goal_pos_tensor - current_state).item()
            if distance < min_distance:
                min_distance = distance
                nearest_goal = goal_pos_tensor

        if nearest_goal is None:
            return torch.zeros(self.state_dim, dtype=torch.float32)

        # Return relative goal vector
        rel_goal = nearest_goal - current_state
        return rel_goal

    def get_goal_motif(self) -> Optional[List[GoalMotif]]:
        """
        Get current active goal motifs.

        Returns:
            List of GoalMotif objects, or None if not using motif-based goals
        """
        if self._use_motif_goals and len(self.active_goal_motifs) > 0:
            return self.active_goal_motifs
        return None

    def set_goal_motifs(self, goal_motifs: List[GoalMotif]):
        """
        Set active goal motifs (enables motif-based navigation).

        Args:
            goal_motifs: List of GoalMotif objects
        """
        self.active_goal_motifs = goal_motifs
        self._use_motif_goals = True

    def set_tkn_processor(self, tkn_processor):
        """Set TKN processor for extracting patterns from new goals."""
        self._tkn_processor = tkn_processor

    def check_goal_completion(
        self, current_state: torch.Tensor, goal_motif: Optional[GoalMotif] = None
    ) -> bool:
        """
        Check if a goal motif has been reached.

        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            goal_motif: Optional GoalMotif to check against

        Returns:
            True if goal motif is matched, False otherwise
        """
        # This will be called by the training loop with the Actor's match_goal_motif
        # For now, return False (matching happens in training loop)
        return False

    def _generate_random_goal(self, max_attempts: int = 100) -> torch.Tensor:
        """
        Generate a random goal position within bounds, avoiding obstacles.

        Args:
            max_attempts: Maximum number of attempts to find a valid position

        Returns:
            goal_position: Random goal position (state_dim,) tensor that doesn't overlap with obstacles
        """
        device = self.bounds_min.device if hasattr(self.bounds_min, "device") else None

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
                    obs_pos_tensor = torch.tensor(
                        obs_pos, dtype=torch.float32, device=device
                    )

                # Ensure obstacle position matches state_dim
                if obs_pos_tensor.dim() > 1:
                    obs_pos_tensor = obs_pos_tensor.squeeze(0)
                if obs_pos_tensor.shape[0] != self.state_dim:
                    if obs_pos_tensor.shape[0] < self.state_dim:
                        padding = torch.zeros(
                            self.state_dim - obs_pos_tensor.shape[0],
                            dtype=torch.float32,
                            device=device,
                        )
                        obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                    else:
                        obs_pos_tensor = obs_pos_tensor[: self.state_dim]

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
            center_val = (
                self.bounds_min[dim].item() + self.bounds_max[dim].item()
            ) / 2.0
            center.append(center_val)
        center_tensor = torch.tensor(center, dtype=torch.float32, device=device)

        # Check if center is valid (doesn't overlap with obstacles)
        center_valid = True
        for obs_pos, obs_r in self.obstacles:
            if isinstance(obs_pos, torch.Tensor):
                obs_pos_tensor = obs_pos.to(device) if device else obs_pos
            else:
                obs_pos_tensor = torch.tensor(
                    obs_pos, dtype=torch.float32, device=device
                )

            if obs_pos_tensor.dim() > 1:
                obs_pos_tensor = obs_pos_tensor.squeeze(0)
            if obs_pos_tensor.shape[0] != self.state_dim:
                if obs_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(
                        self.state_dim - obs_pos_tensor.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    obs_pos_tensor = torch.cat([obs_pos_tensor, padding])
                else:
                    obs_pos_tensor = obs_pos_tensor[: self.state_dim]

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

    def _check_and_update_goals(
        self,
        current_position: torch.Tensor,
        goal_match_scores: Optional[List[float]] = None,
    ) -> int:
        """
        Check if any goal has been reached and update goals accordingly.

        If using motif-based goals, uses match_scores. Otherwise uses distance threshold.

        Args:
            current_position: Current agent position (state_dim,)
            goal_match_scores: Optional list of match scores for motif-based goals (from Actor.match_goal_motif)

        Returns:
            goals_reached: Number of goals reached (and removed) in this step
        """
        # Ensure current_position is (state_dim,)
        if current_position.dim() > 1:
            current_position = current_position.squeeze(0)

        device = (
            current_position.device if hasattr(current_position, "device") else None
        )

        goals_reached = 0

        # Check motif-based goals if available
        if (
            self._use_motif_goals
            and goal_match_scores is not None
            and len(goal_match_scores) > 0
        ):
            goals_to_remove = []
            match_threshold = 0.7  # Threshold for goal completion (can be configurable)

            for i, match_score in enumerate(goal_match_scores):
                if i < len(self.active_goal_motifs) and match_score >= match_threshold:
                    goals_to_remove.append(i)
                    goals_reached += 1

            # Remove completed goals (in reverse order to maintain indices)
            for i in reversed(goals_to_remove):
                if i < len(self.active_goal_motifs):
                    self.active_goal_motifs.pop(i)  # Remove reached motif
                # Also remove from active_goals and add to reached_goals
                if i < len(self.active_goals):
                    reached_goal = self.active_goals.pop(i)
                    # Add to reached_goals for visualization (keep last 50 to avoid memory growth)
                    self.reached_goals.append(
                        reached_goal.clone()
                        if isinstance(reached_goal, torch.Tensor)
                        else reached_goal
                    )
                    if len(self.reached_goals) > 50:
                        self.reached_goals.pop(0)  # Keep only recent reached goals

            # Add new random goal for each completed one
            for _ in range(goals_reached):
                new_goal = self._generate_random_goal()
                self.active_goals.append(new_goal)
                # Extract TKN pattern from new goal position
                if self._tkn_processor is not None:
                    zero_intent = torch.zeros(self.state_dim)
                    context = self.get_context(new_goal, zero_intent)
                    new_motif = create_motif_from_position(
                        new_goal, self._tkn_processor, context, self.obstacles
                    )
                else:
                    new_motif = GoalMotif(position_hint=new_goal)
                self.active_goal_motifs.append(new_motif)

            return goals_reached

        # Fallback: position-based goal checking (backward compatibility)
        goals_to_remove = []
        for i, goal_pos in enumerate(self.active_goals):
            # Normalize goal position
            if isinstance(goal_pos, torch.Tensor):
                goal_pos_tensor = goal_pos.to(device) if device else goal_pos
            else:
                goal_pos_tensor = torch.tensor(
                    goal_pos, dtype=torch.float32, device=device
                )

            if goal_pos_tensor.dim() > 1:
                goal_pos_tensor = goal_pos_tensor.squeeze(0)
            if goal_pos_tensor.shape[0] != self.state_dim:
                if goal_pos_tensor.shape[0] < self.state_dim:
                    padding = torch.zeros(
                        self.state_dim - goal_pos_tensor.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    goal_pos_tensor = torch.cat([goal_pos_tensor, padding])
                else:
                    goal_pos_tensor = goal_pos_tensor[: self.state_dim]

            # Check distance to goal
            distance = torch.norm(goal_pos_tensor - current_position).item()
            if distance <= self.goal_completion_threshold:
                goals_to_remove.append(i)

        # Remove completed goals (in reverse order to maintain indices)
        for i in reversed(goals_to_remove):
            reached_goal = self.active_goals.pop(i)
            # Add to reached_goals for visualization (keep last 50 to avoid memory growth)
            self.reached_goals.append(
                reached_goal.clone()
                if isinstance(reached_goal, torch.Tensor)
                else reached_goal
            )
            if len(self.reached_goals) > 50:
                self.reached_goals.pop(0)  # Keep only recent reached goals

        # Add new random goal for each completed goal
        for _ in goals_to_remove:
            new_goal = self._generate_random_goal()
            self.active_goals.append(new_goal)
            # Extract TKN pattern from new goal position
            if self._use_motif_goals:
                if self._tkn_processor is not None:
                    zero_intent = torch.zeros(self.state_dim)
                    context = self.get_context(new_goal, zero_intent)
                    new_motif = create_motif_from_position(
                        new_goal, self._tkn_processor, context, self.obstacles
                    )
                else:
                    new_motif = GoalMotif(position_hint=new_goal)
                self.active_goal_motifs.append(new_motif)

        return len(goals_to_remove)

    def replenish_energy(self, amount: float):
        """Replenish energy by a fixed amount."""
        self.current_energy = min(self.max_energy, self.current_energy + amount)

    def reset_energy(self):
        """Reset energy to initial value."""
        self.current_energy = self.initial_energy
        self.is_dead = False
