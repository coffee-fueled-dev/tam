"""
3D Obstacle Navigation Environment.

The agent must navigate from start to goal while avoiding unknown obstacles.
Obstacles cause binding failures - reality diverges from the predicted tube.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Obstacle:
    """Spherical obstacle in 3D space."""
    center: np.ndarray  # (3,)
    radius: float
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle."""
        return np.linalg.norm(point - self.center) < self.radius
    
    def project_out(self, point: np.ndarray) -> np.ndarray:
        """Project point to surface if inside obstacle."""
        diff = point - self.center
        dist = np.linalg.norm(diff)
        if dist < self.radius:
            # Push to surface
            if dist < 1e-6:
                # At center, push in random direction
                diff = np.random.randn(3)
                dist = np.linalg.norm(diff)
            return self.center + diff / dist * (self.radius + 0.01)
        return point


@dataclass
class ObstacleNavConfig:
    """Configuration for obstacle navigation environment."""
    # World bounds
    world_min: float = -2.0
    world_max: float = 2.0
    
    # Goal settings
    goal_radius: float = 0.15  # Success threshold
    
    # Obstacle settings
    n_obstacles: int = 5
    obstacle_radius_min: float = 0.2
    obstacle_radius_max: float = 0.5
    
    # Dynamics
    max_speed: float = 0.3  # Max distance per step
    dt: float = 1.0
    
    # Episode settings
    T: int = 20
    max_steps: int = 100
    
    # Randomization
    randomize_start: bool = True
    randomize_goal: bool = True
    randomize_obstacles: bool = True
    
    # Fixed positions (used if not randomizing)
    fixed_start: Optional[np.ndarray] = None
    fixed_goal: Optional[np.ndarray] = None
    fixed_obstacles: List[Obstacle] = field(default_factory=list)


class ObstacleNavEnv:
    """
    3D navigation environment with unknown obstacles.
    
    Observation includes ALL dynamics-relevant state:
      - current_pos (3): Where am I?
      - goal_pos (3): Where do I want to go?
      - goal_direction (3): Unit vector toward goal
      - goal_distance (1): How far is the goal?
      - max_speed (1): How fast can I move per step?
      - remaining_steps (1): How many steps left?
      - last_velocity (3): What did I actually move last step?
      - was_blocked (1): Did my last action get blocked?
    
    Total: 16D observation
    
    The actor learns through binding failure which of these matter.
    If speed limit causes binding failure, the actor learns to use max_speed.
    If obstacles block, the actor learns from was_blocked patterns.
    """
    
    def __init__(self, config: Optional[ObstacleNavConfig] = None):
        self.config = config or ObstacleNavConfig()
        self.obs_dim = 16  # Rich state for dynamics learning
        self.action_dim = 3
        self.pred_dim = 3  # Trajectory prediction dimension
        
        # State
        self.pos = np.zeros(3)
        self.goal = np.zeros(3)
        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        self.start_pos = np.zeros(3)
        self.last_velocity = np.zeros(3)
        self.was_blocked = False
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return initial observation."""
        if seed is not None:
            np.random.seed(seed)
        
        c = self.config
        
        # Generate obstacles first (so we can ensure start/goal are clear)
        if c.randomize_obstacles:
            self.obstacles = self._generate_obstacles()
        else:
            self.obstacles = list(c.fixed_obstacles)
        
        # Set start position
        if c.randomize_start:
            self.pos = self._sample_clear_position()
        else:
            self.pos = c.fixed_start.copy() if c.fixed_start is not None else np.zeros(3)
        
        self.start_pos = self.pos.copy()
        
        # Set goal position
        if c.randomize_goal:
            self.goal = self._sample_clear_position(min_dist_from=self.pos, min_dist=1.0)
        else:
            self.goal = c.fixed_goal.copy() if c.fixed_goal is not None else np.array([1.0, 1.0, 1.0])
        
        self.step_count = 0
        self.last_velocity = np.zeros(3)
        self.was_blocked = False
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action and return (obs, reward, done, info).
        
        Key for binding failure learning:
          - 'x' in info is the actual position (what really happened)
          - If obstacle blocks movement, x differs from intended position
          - Rich observation includes dynamics state for learning
        """
        action = np.array(action, dtype=np.float32)
        old_pos = self.pos.copy()
        
        # Clip action to max speed
        speed = np.linalg.norm(action)
        if speed > self.config.max_speed:
            action = action / speed * self.config.max_speed
        
        # Intended new position
        intended_pos = self.pos + action * self.config.dt
        
        # Check obstacle collisions and project out
        actual_pos = self._apply_obstacles(intended_pos)
        
        # Clamp to world bounds
        actual_pos = np.clip(actual_pos, self.config.world_min, self.config.world_max)
        
        # Track dynamics state for observation
        self.last_velocity = actual_pos - old_pos
        self.was_blocked = not np.allclose(intended_pos, actual_pos)
        
        self.pos = actual_pos
        self.step_count += 1
        
        # Compute reward and done
        dist_to_goal = np.linalg.norm(self.pos - self.goal)
        reached_goal = dist_to_goal < self.config.goal_radius
        timeout = self.step_count >= self.config.max_steps
        
        # Simple reward: negative distance + bonus for reaching goal
        reward = -dist_to_goal * 0.1
        if reached_goal:
            reward += 10.0
        
        done = reached_goal or timeout
        
        info = {
            'x': self.pos.copy(),  # Actual position (for binding failure)
            'intended': intended_pos.copy(),
            'dist_to_goal': dist_to_goal,
            'reached_goal': reached_goal,
            'collision': self.was_blocked,
            'actual_speed': np.linalg.norm(self.last_velocity),
        }
        
        return self._get_obs(), reward, done, info
    
    def _get_obs(self) -> np.ndarray:
        """
        Get rich observation with all dynamics-relevant state.
        
        The actor learns through binding failure which features matter:
        - If speed limit causes failures, it learns to use max_speed
        - If obstacles block, it learns from was_blocked and last_velocity
        - If time runs out, it learns from remaining_steps
        
        Returns: (16,) array with:
          [0:3]   current_pos
          [3:6]   goal_pos
          [6:9]   goal_direction (unit vector)
          [9]     goal_distance (scalar)
          [10]    max_speed (scalar, normalized)
          [11]    remaining_steps (scalar, normalized)
          [12:15] last_velocity
          [15]    was_blocked (0 or 1)
        """
        c = self.config
        
        # Goal-relative features
        goal_vec = self.goal - self.pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-6)
        
        # Normalized dynamics features
        remaining_steps_norm = (c.max_steps - self.step_count) / c.max_steps
        max_speed_norm = c.max_speed  # Could normalize by world size
        
        obs = np.concatenate([
            self.pos,                           # [0:3] Where am I?
            self.goal,                          # [3:6] Where do I want to go?
            goal_dir,                           # [6:9] Which way to goal?
            [goal_dist],                        # [9] How far?
            [max_speed_norm],                   # [10] How fast can I move?
            [remaining_steps_norm],             # [11] How much time left?
            self.last_velocity,                 # [12:15] What did I actually do?
            [float(self.was_blocked)],          # [15] Was I blocked?
        ]).astype(np.float32)
        
        return obs
    
    def _generate_obstacles(self) -> List[Obstacle]:
        """Generate random obstacles."""
        c = self.config
        obstacles = []
        
        for _ in range(c.n_obstacles):
            # Random center
            center = np.random.uniform(c.world_min + 0.5, c.world_max - 0.5, size=3)
            
            # Random radius
            radius = np.random.uniform(c.obstacle_radius_min, c.obstacle_radius_max)
            
            obstacles.append(Obstacle(center=center, radius=radius))
        
        return obstacles
    
    def _sample_clear_position(
        self, 
        min_dist_from: Optional[np.ndarray] = None,
        min_dist: float = 0.0,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """Sample a position that's not inside any obstacle."""
        c = self.config
        
        for _ in range(max_attempts):
            pos = np.random.uniform(c.world_min + 0.3, c.world_max - 0.3, size=3)
            
            # Check obstacles
            in_obstacle = any(obs.contains(pos) for obs in self.obstacles)
            if in_obstacle:
                continue
            
            # Check distance constraint
            if min_dist_from is not None:
                if np.linalg.norm(pos - min_dist_from) < min_dist:
                    continue
            
            return pos
        
        # Fallback: return random position
        return np.random.uniform(c.world_min, c.world_max, size=3)
    
    def _apply_obstacles(self, pos: np.ndarray) -> np.ndarray:
        """Apply obstacle constraints to position."""
        result = pos.copy()
        
        # Project out of each obstacle
        for obstacle in self.obstacles:
            result = obstacle.project_out(result)
        
        return result
    
    def get_obstacle_info(self) -> List[Dict[str, Any]]:
        """Get obstacle information (for visualization, NOT for actor)."""
        return [
            {'center': obs.center.tolist(), 'radius': obs.radius}
            for obs in self.obstacles
        ]
    
    def render_ascii(self) -> str:
        """Simple ASCII rendering (top-down view, z projected)."""
        c = self.config
        grid_size = 20
        grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        
        def to_grid(pos):
            x = int((pos[0] - c.world_min) / (c.world_max - c.world_min) * (grid_size - 1))
            y = int((pos[1] - c.world_min) / (c.world_max - c.world_min) * (grid_size - 1))
            return max(0, min(grid_size - 1, x)), max(0, min(grid_size - 1, y))
        
        # Draw obstacles
        for obs in self.obstacles:
            gx, gy = to_grid(obs.center)
            grid[gy][gx] = 'O'
        
        # Draw goal
        gx, gy = to_grid(self.goal)
        grid[gy][gx] = 'G'
        
        # Draw agent
        gx, gy = to_grid(self.pos)
        grid[gy][gx] = 'A'
        
        return '\n'.join([''.join(row) for row in grid])
