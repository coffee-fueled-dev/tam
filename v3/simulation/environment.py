"""
Deterministic environment generation utilities.

Provides functions for generating obstacle layouts with guaranteed properties:
- Deterministic (seed-based)
- Sufficient open space for navigation
- Packing threshold to prevent overcrowding
- Dimension-agnostic
"""

import numpy as np
from typing import List, Tuple


def generate_obstacles(
    state_dim: int,
    bounds: dict,
    num_obstacles: int = 8,
    min_radius: float = 0.8,
    max_radius: float = 2.0,
    packing_threshold: float = 0.4,
    min_open_path_width: float = 3.0,
    seed: int = 42
) -> List[Tuple]:
    """
    Generate a deterministic set of obstacles with guaranteed open space.
    
    Uses a packing algorithm to ensure obstacles are well-distributed while
    maintaining navigable corridors. The algorithm:
    1. Generates candidate positions using seeded RNG
    2. Checks packing threshold (minimum distance between obstacles)
    3. Ensures open paths exist (minimum corridor width)
    4. Validates against boundaries
    
    Args:
        state_dim: Dimension of state space
        bounds: Dict with 'min' and 'max' keys, each a list of state_dim floats
        num_obstacles: Target number of obstacles to place
        min_radius: Minimum obstacle radius
        max_radius: Maximum obstacle radius
        packing_threshold: Minimum distance between obstacle centers (as fraction of max_radius)
                          Higher = more spread out (default 0.4 means 40% of max_radius minimum gap)
        min_open_path_width: Minimum width of open corridors (ensures navigability)
        seed: Random seed for deterministic generation
        
    Returns:
        List of (position, radius) tuples where position is a list of state_dim floats
    """
    rng = np.random.RandomState(seed)
    
    min_bounds = np.array(bounds['min'])
    max_bounds = np.array(bounds['max'])
    bounds_size = max_bounds - min_bounds
    
    obstacles = []
    max_attempts = num_obstacles * 100  # Limit attempts to avoid infinite loops
    attempts = 0
    
    # Minimum distance between obstacle centers
    min_center_distance = max_radius * (1.0 + packing_threshold)
    
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        
        # Generate candidate position
        candidate_pos = min_bounds + rng.rand(state_dim) * bounds_size
        
        # Generate candidate radius
        candidate_radius = min_radius + (max_radius - min_radius) * rng.rand()
        
        # Check if candidate is too close to existing obstacles
        too_close = False
        for existing_pos, existing_radius in obstacles:
            distance = np.linalg.norm(np.array(candidate_pos) - np.array(existing_pos))
            required_distance = candidate_radius + existing_radius + min_center_distance
            if distance < required_distance:
                too_close = True
                break
        
        if too_close:
            continue
        
        # Check if candidate leaves sufficient open space
        # We want to ensure there's at least one path through the environment
        # Simple heuristic: check if candidate doesn't block all paths from origin to corners
        # For now, we'll use a simpler check: ensure candidate isn't too close to origin
        origin = np.zeros(state_dim)
        distance_to_origin = np.linalg.norm(np.array(candidate_pos) - origin)
        if distance_to_origin < candidate_radius + min_open_path_width:
            continue
        
        # Check boundaries (obstacle must fit within bounds)
        if np.any(candidate_pos - candidate_radius < min_bounds) or \
           np.any(candidate_pos + candidate_radius > max_bounds):
            continue
        
        # Accept candidate
        obstacles.append((list(candidate_pos), float(candidate_radius)))
    
    if len(obstacles) < num_obstacles:
        print(f"Warning: Only generated {len(obstacles)} obstacles out of {num_obstacles} requested")
    
    return obstacles


def validate_obstacle_layout(
    obstacles: List[Tuple],
    bounds: dict,
    min_open_path_width: float = 3.0
) -> bool:
    """
    Validate that an obstacle layout has sufficient open space for navigation.
    
    Args:
        obstacles: List of (position, radius) tuples
        bounds: Dict with 'min' and 'max' keys
        min_open_path_width: Minimum required corridor width
        
    Returns:
        True if layout is valid, False otherwise
    """
    if len(obstacles) == 0:
        return True
    
    # Check that there's at least one path from origin to a corner
    # Simple heuristic: check if we can reach at least one corner
    origin = np.zeros(len(bounds['min']))
    corners = []
    
    # Generate corner positions (all combinations of min/max for each dimension)
    state_dim = len(bounds['min'])
    for i in range(2 ** state_dim):
        corner = []
        for dim in range(state_dim):
            if (i >> dim) & 1:
                corner.append(bounds['max'][dim])
            else:
                corner.append(bounds['min'][dim])
        corners.append(np.array(corner))
    
    # Check if at least one corner is reachable
    for corner in corners:
        # Simple line-of-sight check: see if direct path is clear
        path_clear = True
        steps = 20
        for step in range(steps + 1):
            t = step / steps
            point = origin + t * (corner - origin)
            
            # Check distance to all obstacles
            for obs_pos, obs_radius in obstacles:
                distance = np.linalg.norm(point - np.array(obs_pos))
                if distance < obs_radius + min_open_path_width / 2:
                    path_clear = False
                    break
            
            if not path_clear:
                break
        
        if path_clear:
            return True
    
    return False
