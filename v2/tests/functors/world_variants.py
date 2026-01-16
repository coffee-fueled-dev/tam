"""
CMG world variants for functor tests.

Creates environments with the same topological structure (same K)
but different dynamics, observation mappings, or noise levels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class WorldVariantConfig:
    """Configuration for a CMG world variant."""
    
    # Core topology (keep same across variants)
    d: int = 3                    # State dimension
    K: int = 4                    # Number of modes (fork count)
    T: int = 20                   # Trajectory length
    t_gate: int = 5               # Gating timestep
    
    # Variant knobs (change these between worlds)
    noise_x: float = 0.01         # Process noise
    noise_obs: float = 0.01       # Observation noise
    action_scale: float = 1.0     # Action scaling
    damping: float = 0.1          # Velocity damping
    drift: Optional[np.ndarray] = None  # Constant drift vector
    
    # Observation projection (scramble/rotate/partial)
    obs_rotation: Optional[np.ndarray] = None  # Rotation matrix
    obs_mask: Optional[np.ndarray] = None      # Feature mask
    obs_noise_type: str = "gaussian"           # "gaussian" or "uniform"
    
    # Goal distribution
    goal_spread: float = 0.8      # How spread out goals are
    
    # Irreversibility strength
    early_divergence: bool = True
    divergence_strength: float = 1.0


def create_world_A(seed: int = 42, K: int = 4, d: int = 3) -> WorldVariantConfig:
    """
    Create World A: Baseline CMG with moderate noise.
    """
    np.random.seed(seed)
    return WorldVariantConfig(
        d=d,
        K=K,
        T=20,
        t_gate=5,
        noise_x=0.01,
        noise_obs=0.01,
        action_scale=1.0,
        damping=0.1,
        early_divergence=True,
        divergence_strength=1.0,
    )


def create_world_B(seed: int = 43, K: int = 4, d: int = 3) -> WorldVariantConfig:
    """
    Create World B: Higher noise, different action scale.
    
    Same topology (K modes, same fork structure) but different dynamics.
    """
    np.random.seed(seed)
    return WorldVariantConfig(
        d=d,
        K=K,
        T=20,
        t_gate=5,
        noise_x=0.03,           # Higher process noise
        noise_obs=0.02,         # Higher observation noise
        action_scale=0.8,       # Weaker actions
        damping=0.15,           # More damping
        early_divergence=True,
        divergence_strength=1.2,  # Stronger divergence
    )


def create_world_C(seed: int = 44, K: int = 4, d: int = 3) -> WorldVariantConfig:
    """
    Create World C: Rotated observations, different drift.
    
    Same topology but observation space is transformed.
    """
    np.random.seed(seed)
    
    # Random rotation matrix (rotate first two dimensions)
    theta = np.pi / 6  # 30 degree rotation
    R = np.eye(d, dtype=np.float32)
    if d >= 2:
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)
    
    # Drift vector
    drift = np.zeros(d, dtype=np.float32)
    if d >= 1:
        drift[0] = 0.02
    if d >= 2:
        drift[1] = -0.01
    
    return WorldVariantConfig(
        d=d,
        K=K,
        T=20,
        t_gate=5,
        noise_x=0.015,
        noise_obs=0.01,
        action_scale=1.0,
        damping=0.1,
        drift=drift,
        obs_rotation=R,
        early_divergence=True,
        divergence_strength=1.0,
    )


def create_world_pair(
    variant: str = "noise",
    seed: int = 42,
) -> Tuple[WorldVariantConfig, WorldVariantConfig]:
    """
    Create a pair of worlds with the same topology but different dynamics.
    
    Args:
        variant: Type of variation
            - "noise": Different noise levels
            - "action": Different action scales
            - "observation": Different observation transforms
            - "divergence": Different gating strength
        seed: Random seed
    
    Returns:
        (config_A, config_B)
    """
    np.random.seed(seed)
    
    base = WorldVariantConfig(d=3, K=4, T=20, t_gate=5)
    
    if variant == "noise":
        config_A = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            noise_x=0.01, noise_obs=0.01,
        )
        config_B = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            noise_x=0.03, noise_obs=0.02,
        )
    
    elif variant == "action":
        config_A = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            action_scale=1.0, damping=0.1,
        )
        config_B = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            action_scale=0.7, damping=0.2,
        )
    
    elif variant == "observation":
        theta = np.pi / 4  # 45 degree rotation
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        
        config_A = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            obs_rotation=None,
        )
        config_B = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            obs_rotation=R,
        )
    
    elif variant == "divergence":
        config_A = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            early_divergence=True,
            divergence_strength=0.5,
        )
        config_B = WorldVariantConfig(
            d=3, K=4, T=20, t_gate=5,
            early_divergence=True,
            divergence_strength=1.5,
        )
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return config_A, config_B


class CMGWorldVariant:
    """
    CMG environment with configurable dynamics/observations.
    
    Wraps the standard CMGEnv but applies transformations.
    """
    
    def __init__(self, config: WorldVariantConfig):
        from v2.environments.cmg import CMGEnv, CMGConfig
        
        self.config = config
        
        # Create base CMG environment
        base_config = CMGConfig(
            d=config.d,
            K=config.K,
            T=config.T,
            t_gate=config.t_gate,
            noise_x=config.noise_x,
        )
        self.base_env = CMGEnv(base_config)
        
        # Observation dimension from actual environment
        self.obs_dim = self.base_env.obs_dim
        self.state_dim = config.d  # State dimension (position only)
        self.T = config.T
    
    def reset(self) -> np.ndarray:
        """Reset and return transformed observation."""
        obs = self.base_env.reset()
        return self._transform_obs(obs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step with transformed action and observation."""
        # Scale action
        scaled_action = action * self.config.action_scale
        
        # Add drift
        if self.config.drift is not None:
            scaled_action = scaled_action + self.config.drift
        
        obs, reward, done, info = self.base_env.step(scaled_action)
        
        # Transform observation
        obs = self._transform_obs(obs)
        
        # Add observation noise
        if self.config.noise_obs > 0:
            if self.config.obs_noise_type == "gaussian":
                obs = obs + np.random.randn(*obs.shape).astype(np.float32) * self.config.noise_obs
            else:
                obs = obs + (np.random.rand(*obs.shape).astype(np.float32) - 0.5) * 2 * self.config.noise_obs
        
        return obs, reward, done, info
    
    def _transform_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation transformation."""
        if self.config.obs_rotation is not None:
            # obs has shape (obs_dim,) where obs_dim = d + d (position + goal)
            # Rotation matrix is (d, d), so rotate position and goal separately
            d = self.config.d
            if obs.shape[0] >= 2 * d:
                # Rotate position (first d dims)
                obs[:d] = obs[:d] @ self.config.obs_rotation.T
                # Rotate goal (next d dims)
                obs[d:2*d] = obs[d:2*d] @ self.config.obs_rotation.T
            else:
                # Fallback: if obs doesn't match expected structure, just rotate first d dims
                obs[:d] = obs[:d] @ self.config.obs_rotation.T
        
        if self.config.obs_mask is not None:
            obs = obs * self.config.obs_mask
        
        return obs.astype(np.float32)
    
    @property
    def x(self):
        """Access underlying state."""
        return self.base_env.x


def create_paired_envs(
    variant: str = "noise",
    seed: int = 42,
) -> Tuple['CMGWorldVariant', 'CMGWorldVariant']:
    """
    Create a pair of environments with same topology but different dynamics.
    
    Args:
        variant: Type of variation
        seed: Random seed
    
    Returns:
        (env_A, env_B)
    """
    config_A, config_B = create_world_pair(variant, seed)
    return CMGWorldVariant(config_A), CMGWorldVariant(config_B)


def create_topological_collision_config(
    seed: int = 42,
    d: int = 3,
) -> Tuple[WorldVariantConfig, WorldVariantConfig, WorldVariantConfig]:
    """
    Create a "Topological Collision" scenario.
    
    This tests the fundamental limit of functorial transfer:
    - Actor A sees K=2 basins as ADJACENT (no intervening structure)
    - Actor B sees K=3 basins where the MIDDLE basin sits between A's two
    - Actor C sees K=2 basins (same as A, for composition testing)
    
    The question: Can the functor F_AB handle mapping an "edge" in A's 
    topology to a "path through a vertex" in B's topology?
    
    This is where we expect:
    - Jacobian determinant spikes (topological tearing)
    - Composition violations (F_AC ≠ F_BC ∘ F_AB)
    - High condition numbers in the transition region
    
    Returns:
        (config_A, config_B, config_C) with collision topology
    """
    np.random.seed(seed)
    
    # World A: 2 basins, diametrically opposed
    # Goals at "north" and "south" poles
    config_A = WorldVariantConfig(
        d=d,
        K=2,
        T=20,
        t_gate=5,
        noise_x=0.01,
        noise_obs=0.01,
        action_scale=1.0,
        goal_spread=1.0,  # Maximum separation
        early_divergence=True,
        divergence_strength=1.0,
    )
    
    # World B: 3 basins, the MIDDLE one sits between A's two
    # Goals at "north", "equator", and "south"
    # The equator goal creates the collision
    config_B = WorldVariantConfig(
        d=d,
        K=3,
        T=20,
        t_gate=5,
        noise_x=0.01,
        noise_obs=0.01,
        action_scale=1.0,
        goal_spread=0.7,  # Tighter packing to ensure middle basin matters
        early_divergence=True,
        divergence_strength=1.0,
    )
    
    # World C: Same as A (2 basins) for composition test
    rot_C = random_rotation_matrix(d, seed + 200)
    config_C = WorldVariantConfig(
        d=d,
        K=2,
        T=20,
        t_gate=5,
        noise_x=0.02,
        noise_obs=0.02,
        action_scale=1.0,
        goal_spread=1.0,
        obs_rotation=rot_C,
        early_divergence=True,
        divergence_strength=1.0,
    )
    
    return config_A, config_B, config_C


def create_topological_collision_envs(
    seed: int = 42,
    d: int = 3,
) -> Tuple['CMGWorldVariant', 'CMGWorldVariant', 'CMGWorldVariant']:
    """
    Create environments for the topological collision test.
    
    Returns:
        (env_A, env_B, env_C) where A and C have K=2, B has K=3
    """
    config_A, config_B, config_C = create_topological_collision_config(seed, d)
    return (
        CMGWorldVariant(config_A),
        CMGWorldVariant(config_B),
        CMGWorldVariant(config_C),
    )
