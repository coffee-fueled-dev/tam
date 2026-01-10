"""
Topologically Equivalent Environments for Functor Testing.

These environments share the same uncertainty topology (similar cone structures)
but have different dynamics. This allows testing whether intent geometry
can be transported via a learned functor.

Environments:
1. LatentRuleGridworld - Original 4-rule gridworld
2. RotatedGridworld - Same rules but grid is rotated 45°
3. ScaledGridworld - Same rules but velocities are scaled
4. MirroredGridworld - Same rules but state space is mirrored
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .latent_rule_gridworld import LatentRuleGridworld, GridWorldState


class RotatedGridworld(LatentRuleGridworld):
    """
    Gridworld with rotated coordinate system.
    
    Same uncertainty structure (4 rules, same slip/hint mechanics)
    but states are in rotated coordinates.
    
    This tests whether the functor can learn the rotation transformation.
    """
    
    def __init__(
        self,
        rotation_deg: float = 45.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rotation_deg = rotation_deg
        self.rotation_rad = math.radians(rotation_deg)
        self.cos_r = math.cos(self.rotation_rad)
        self.sin_r = math.sin(self.rotation_rad)
    
    def _rotate(self, x: float, y: float) -> Tuple[float, float]:
        """Rotate coordinates."""
        x_r = x * self.cos_r - y * self.sin_r
        y_r = x * self.sin_r + y * self.cos_r
        return x_r, y_r
    
    def _normalize_pos(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize and rotate position."""
        # First normalize to [0, 1]
        nx = x / (self.W - 1)
        ny = y / (self.H - 1)
        # Then rotate
        return self._rotate(nx, ny)
    
    def reset_paired(self, source_state: Dict) -> np.ndarray:
        """Reset to paired state (same state, rotation applied in observation)."""
        self.state = GridWorldState(
            x=float(source_state['x']),
            y=float(source_state['y']),
            goal_x=float(source_state['goal_x']),
            goal_y=float(source_state['goal_y']),
            rule=source_state['rule'],
            steps=0,
            hit_wall_count=0,
            slip_count=0,
            key_visits=[],
            steps_to_first_key=None,
        )
        return self.observe()
    
    def observe(self) -> np.ndarray:
        """Return observation in rotated coordinates."""
        if self.state is None:
            raise RuntimeError("Environment not reset")
        
        s = self.state
        
        # Rotated positions
        nx, ny = self._normalize_pos(s.x, s.y)
        ngx, ngy = self._normalize_pos(s.goal_x, s.goal_y)
        
        # Get hint
        hint = self._get_hint(s.x, s.y, s.rule)
        
        obs = np.array([nx, ny, ngx, ngy, 0.0], dtype=np.float32)
        obs = np.concatenate([obs, hint])
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """Step with rotated state output."""
        obs_next, _, reward, done, info = super().step(action)
        
        # State in rotated coordinates
        state_next = np.array(self._normalize_pos(self.state.x, self.state.y), dtype=np.float32)
        
        return obs_next, state_next, reward, done, info
    
    def rollout(
        self, policy_fn, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Rollout with rotated states."""
        obs0 = self.observe()
        obs_seq = [obs0]
        state_seq = [np.array(self._normalize_pos(self.state.x, self.state.y), dtype=np.float32)]
        actions = []
        
        for _ in range(horizon):
            if self.state is None:
                break
            
            a = int(policy_fn(obs_seq[-1]))
            a = np.clip(a, 0, 3)
            actions.append([float(a)])
            
            obs_next, state_next, _, done, info = self.step(a)
            obs_seq.append(obs_next)
            state_seq.append(state_next)
            
            if done:
                break
        
        return np.asarray(obs_seq), np.asarray(state_seq), np.asarray(actions), info


class ScaledGridworld(LatentRuleGridworld):
    """
    Gridworld with scaled dynamics.
    
    Same uncertainty structure but states evolve at different rates.
    Tests whether functor can learn velocity scaling.
    """
    
    def __init__(
        self,
        velocity_scale: float = 1.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.velocity_scale = velocity_scale
    
    def reset_paired(self, source_state: Dict) -> np.ndarray:
        """Reset to paired state (same state, different dynamics)."""
        self.state = GridWorldState(
            x=float(source_state['x']),
            y=float(source_state['y']),
            goal_x=float(source_state['goal_x']),
            goal_y=float(source_state['goal_y']),
            rule=source_state['rule'],
            steps=0,
            hit_wall_count=0,
            slip_count=0,
            key_visits=[],
            steps_to_first_key=None,
        )
        return self.observe()
    
    def _apply_action(self, x: float, y: float, a_eff: int) -> Tuple[float, float, bool]:
        """Apply action with scaled velocity."""
        x_new, y_new = x, y
        step_size = self.velocity_scale
        
        if a_eff == 0:  # up
            y_new = max(0, y - step_size)
        elif a_eff == 1:  # down
            y_new = min(self.H - 1, y + step_size)
        elif a_eff == 2:  # left
            x_new = max(0, x - step_size)
        elif a_eff == 3:  # right
            x_new = min(self.W - 1, x + step_size)
        
        # Clamp to grid bounds
        x_new = np.clip(x_new, 0, self.W - 1)
        y_new = np.clip(y_new, 0, self.H - 1)
        
        hit_wall = (x_new == x and y_new == y) and (
            (a_eff == 0 and y == 0) or
            (a_eff == 1 and y == self.H - 1) or
            (a_eff == 2 and x == 0) or
            (a_eff == 3 and x == self.W - 1)
        )
        
        return x_new, y_new, hit_wall


class MirroredGridworld(LatentRuleGridworld):
    """
    Gridworld with mirrored state space.
    
    Same uncertainty structure but state coordinates are mirrored.
    Tests whether functor can learn reflection transformation.
    """
    
    def __init__(
        self,
        mirror_x: bool = True,
        mirror_y: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
    
    def _normalize_pos(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize and mirror position."""
        nx = x / (self.W - 1)
        ny = y / (self.H - 1)
        
        if self.mirror_x:
            nx = 1.0 - nx
        if self.mirror_y:
            ny = 1.0 - ny
        
        return nx, ny
    
    def reset_paired(self, source_state: Dict) -> np.ndarray:
        """
        Reset to a state paired with a source environment state.
        
        Applies mirror transform to create equivalent situation.
        
        Args:
            source_state: dict with keys 'x', 'y', 'goal_x', 'goal_y', 'rule'
        
        Returns:
            observation
        """
        # Apply mirror transform to positions
        if self.mirror_x:
            x = (self.W - 1) - source_state['x']
            goal_x = (self.W - 1) - source_state['goal_x']
        else:
            x = source_state['x']
            goal_x = source_state['goal_x']
        
        if self.mirror_y:
            y = (self.H - 1) - source_state['y']
            goal_y = (self.H - 1) - source_state['goal_y']
        else:
            y = source_state['y']
            goal_y = source_state['goal_y']
        
        # Set state
        self.state = GridWorldState(
            x=float(x),
            y=float(y),
            goal_x=float(goal_x),
            goal_y=float(goal_y),
            rule=source_state['rule'],  # Same rule
            steps=0,
            hit_wall_count=0,
            slip_count=0,
            key_visits=[],
            steps_to_first_key=None,
        )
        
        return self.observe()


class ShiftedRulesGridworld(LatentRuleGridworld):
    """
    Gridworld with shifted rule mappings.
    
    Same structure but rules are permuted: rule 0 in env A = rule 1 in env B, etc.
    Tests whether functor can learn rule correspondence.
    """
    
    def __init__(
        self,
        rule_shift: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rule_shift = rule_shift % 4
    
    def reset(self) -> np.ndarray:
        """Reset with shifted rules."""
        result = super().reset()
        # Shift the rule
        self.state.rule = (self.state.rule + self.rule_shift) % 4
        return self.observe()
    
    def reset_paired(self, source_state: Dict) -> np.ndarray:
        """Reset to paired state with shifted rule."""
        shifted_rule = (source_state['rule'] + self.rule_shift) % 4
        self.state = GridWorldState(
            x=float(source_state['x']),
            y=float(source_state['y']),
            goal_x=float(source_state['goal_x']),
            goal_y=float(source_state['goal_y']),
            rule=shifted_rule,  # Shifted!
            steps=0,
            hit_wall_count=0,
            slip_count=0,
            key_visits=[],
            steps_to_first_key=None,
        )
        return self.observe()


class ContinuousGridworld(LatentRuleGridworld):
    """
    Gridworld with continuous position updates.
    
    Instead of discrete grid steps, uses continuous velocity dynamics.
    Same uncertainty structure but fundamentally different dynamics.
    """
    
    def __init__(
        self,
        dt: float = 0.2,
        friction: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.friction = friction
        self.vx = 0.0
        self.vy = 0.0
    
    def reset(self) -> np.ndarray:
        """Reset with zero velocity."""
        result = super().reset()
        self.vx = 0.0
        self.vy = 0.0
        return result
    
    def _apply_action(self, x: float, y: float, a_eff: int) -> Tuple[float, float, bool]:
        """Apply action with continuous dynamics."""
        # Apply acceleration based on action
        ax, ay = 0.0, 0.0
        accel = 3.0
        
        if a_eff == 0:  # up
            ay = -accel
        elif a_eff == 1:  # down
            ay = accel
        elif a_eff == 2:  # left
            ax = -accel
        elif a_eff == 3:  # right
            ax = accel
        
        # Update velocity
        self.vx = self.vx * self.friction + ax * self.dt
        self.vy = self.vy * self.friction + ay * self.dt
        
        # Update position
        x_new = x + self.vx * self.dt
        y_new = y + self.vy * self.dt
        
        # Clamp and check wall collision
        hit_wall = False
        if x_new < 0:
            x_new = 0
            self.vx = 0
            hit_wall = True
        elif x_new > self.W - 1:
            x_new = self.W - 1
            self.vx = 0
            hit_wall = True
        
        if y_new < 0:
            y_new = 0
            self.vy = 0
            hit_wall = True
        elif y_new > self.H - 1:
            y_new = self.H - 1
            self.vy = 0
            hit_wall = True
        
        return x_new, y_new, hit_wall
    
    @property
    def obs_dim(self) -> int:
        """Add velocity to observation."""
        return super().obs_dim + 2
    
    def observe(self) -> np.ndarray:
        """Include velocity in observation."""
        base_obs = super().observe()
        vel = np.array([self.vx / 3.0, self.vy / 3.0], dtype=np.float32)  # Normalized velocity
        return np.concatenate([base_obs, vel])


# =============================================================================
# Factory Functions
# =============================================================================

def make_standard_gridworld(kwargs: Dict) -> LatentRuleGridworld:
    """Factory for standard gridworld."""
    return LatentRuleGridworld(**kwargs)


def make_rotated_gridworld(kwargs: Dict) -> RotatedGridworld:
    """Factory for rotated gridworld."""
    return RotatedGridworld(**kwargs)


def make_scaled_gridworld(kwargs: Dict) -> ScaledGridworld:
    """Factory for scaled gridworld."""
    return ScaledGridworld(**kwargs)


def make_mirrored_gridworld(kwargs: Dict) -> MirroredGridworld:
    """Factory for mirrored gridworld."""
    return MirroredGridworld(**kwargs)


def make_shifted_rules_gridworld(kwargs: Dict) -> ShiftedRulesGridworld:
    """Factory for shifted rules gridworld."""
    return ShiftedRulesGridworld(**kwargs)


def make_continuous_gridworld(kwargs: Dict) -> ContinuousGridworld:
    """Factory for continuous gridworld."""
    return ContinuousGridworld(**kwargs)


# =============================================================================
# Environment Pairs for Functor Testing
# =============================================================================

EQUIVALENT_ENV_PAIRS = [
    # (name, env_A_factory, env_A_kwargs, env_B_factory, env_B_kwargs, description)
    (
        "standard_to_rotated",
        make_standard_gridworld,
        {"seed": 0},
        make_rotated_gridworld,
        {"seed": 0, "rotation_deg": 45.0},
        "Tests functor learning of coordinate rotation",
    ),
    (
        "standard_to_mirrored",
        make_standard_gridworld,
        {"seed": 0},
        make_mirrored_gridworld,
        {"seed": 0, "mirror_x": True},
        "Tests functor learning of reflection transformation",
    ),
    (
        "standard_to_scaled",
        make_standard_gridworld,
        {"seed": 0},
        make_scaled_gridworld,
        {"seed": 0, "velocity_scale": 1.5},
        "Tests functor learning of velocity scaling",
    ),
    (
        "standard_to_shifted",
        make_standard_gridworld,
        {"seed": 0},
        make_shifted_rules_gridworld,
        {"seed": 0, "rule_shift": 1},
        "Tests functor learning of rule permutation",
    ),
]
