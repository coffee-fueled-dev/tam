"""
Latent Rule Gridworld Environment for Concept Splitting Experiments.

This environment forces the TAM actor to learn distinct commitments (z) for different
latent rules, testing whether the agent can split concepts in z-space.

Key features:
- 4 latent rules that remap actions differently
- Info keys that emit noisy rule hints
- Goal navigation task
- Concept-splitting diagnostics
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GridWorldState:
    """Internal state of the gridworld."""
    x: float
    y: float
    goal_x: float
    goal_y: float
    rule: int
    steps: int
    hit_wall_count: int
    slip_count: int
    key_visits: List[int]  # which keys visited
    steps_to_first_key: Optional[int]


class LatentRuleGridworld:
    """
    Gridworld with latent rules that remap actions.
    
    Rules:
    - 0: identity (up=up, down=down, left=left, right=right)
    - 1: rotate 90° clockwise (up->right, right->down, down->left, left->up)
    - 2: flip horizontal (left<->right, up/down unchanged)
    - 3: swap up/down + 20% slip probability
    """
    
    def __init__(
        self,
        W: int = 9,
        H: int = 9,
        max_steps: int = 64,
        step_cost: float = -0.01,
        wall_penalty: float = -0.2,
        goal_reward: float = 1.0,
        slip_prob: float = 0.2,
        hint_noise_std: float = 0.3,
        key_positions: Optional[List[Tuple[int, int]]] = None,
        seed: int = 0,
    ):
        self.W = W
        self.H = H
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward
        self.slip_prob = slip_prob
        self.hint_noise_std = hint_noise_std
        
        # Key positions (default: symmetric corners)
        if key_positions is None:
            self.key_positions = [(1, 1), (W-2, 1), (1, H-2), (W-2, H-2)]
        else:
            self.key_positions = key_positions
        
        self.K = len(self.key_positions)  # number of keys (should match num rules)
        assert self.K >= 4, "Need at least 4 keys for 4 rules"
        
        self.rng = np.random.RandomState(seed)
        self.state: Optional[GridWorldState] = None
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.action_names = ["up", "down", "left", "right"]
        
    @property
    def obs_dim(self) -> int:
        """Observation dimension: normalized pos(2) + goal(2) + hit_wall(1) + hint(K)"""
        return 4 + 1 + self.K
    
    @property
    def state_dim(self) -> int:
        """State dimension for tube prediction: normalized (x, y)"""
        return 2
    
    def _normalize_pos(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize position to [0, 1]"""
        return x / (self.W - 1), y / (self.H - 1)
    
    def _map_action(self, a: int, rule: int) -> int:
        """
        Map action according to latent rule.
        
        Args:
            a: original action (0=up, 1=down, 2=left, 3=right)
            rule: latent rule (0-3)
        
        Returns:
            effective action
        """
        if rule == 0:
            # Identity
            return a
        elif rule == 1:
            # Rotate 90° clockwise: up->right, right->down, down->left, left->up
            mapping = {0: 3, 1: 2, 2: 0, 3: 1}  # up->right, down->left, left->up, right->down
            return mapping[a]
        elif rule == 2:
            # Flip horizontal: left<->right
            if a == 2:  # left
                return 3  # right
            elif a == 3:  # right
                return 2  # left
            else:
                return a  # up/down unchanged
        elif rule == 3:
            # Swap up/down (no remap, but will add slip)
            if a == 0:  # up
                return 1  # down
            elif a == 1:  # down
                return 0  # up
            else:
                return a  # left/right unchanged
        else:
            return a
    
    def _apply_action(self, x: float, y: float, a_eff: int) -> Tuple[float, float, bool]:
        """
        Apply action and return new position + hit_wall flag.
        
        Args:
            x, y: current position
            a_eff: effective action (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            (x_new, y_new, hit_wall)
        """
        x_new, y_new = x, y
        
        if a_eff == 0:  # up
            y_new = max(0, y - 1)
        elif a_eff == 1:  # down
            y_new = min(self.H - 1, y + 1)
        elif a_eff == 2:  # left
            x_new = max(0, x - 1)
        elif a_eff == 3:  # right
            x_new = min(self.W - 1, x + 1)
        
        hit_wall = (x_new == x and y_new == y) and (
            (a_eff == 0 and y == 0) or
            (a_eff == 1 and y == self.H - 1) or
            (a_eff == 2 and x == 0) or
            (a_eff == 3 and x == self.W - 1)
        )
        
        return x_new, y_new, hit_wall
    
    def _get_hint(self, x: float, y: float, rule: int) -> np.ndarray:
        """
        Get rule hint from key position.
        
        Args:
            x, y: current position
            rule: true latent rule
        
        Returns:
            hint_logits [K] - noisy one-hot encoding of rule
        """
        hint = np.zeros(self.K, dtype=np.float32)
        
        # Check if agent is on a key
        for i, (kx, ky) in enumerate(self.key_positions):
            if int(x) == kx and int(y) == ky:
                # Emit hint: correct rule gets +signal, others 0
                if i < 4:  # only first 4 keys correspond to rules
                    hint[i] = 1.0 if i == rule else 0.0
                break
        
        # Add Gaussian noise
        hint += self.rng.normal(0, self.hint_noise_std, size=hint.shape)
        
        return hint
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        # Start at center
        start_x = self.W // 2
        start_y = self.H // 2
        
        # Sample goal from corners
        corners = [(0, 0), (self.W-1, 0), (0, self.H-1), (self.W-1, self.H-1)]
        goal_x, goal_y = corners[self.rng.randint(0, len(corners))]
        
        # Sample latent rule
        rule = self.rng.randint(0, 4)
        
        self.state = GridWorldState(
            x=float(start_x),
            y=float(start_y),
            goal_x=float(goal_x),
            goal_y=float(goal_y),
            rule=rule,
            steps=0,
            hit_wall_count=0,
            slip_count=0,
            key_visits=[],
            steps_to_first_key=None,
        )
        
        return self.observe()
    
    def observe(self) -> np.ndarray:
        """Return current observation."""
        if self.state is None:
            raise RuntimeError("Environment not reset")
        
        s = self.state
        
        # Normalize positions
        nx, ny = self._normalize_pos(s.x, s.y)
        ngx, ngy = self._normalize_pos(s.goal_x, s.goal_y)
        
        # Get hint
        hint = self._get_hint(s.x, s.y, s.rule)
        
        # Build observation: [pos(2), goal(2), hit_wall(1), hint(K)]
        obs = np.array([
            nx, ny,
            ngx, ngy,
            0.0,  # hit_wall flag (updated in step)
        ], dtype=np.float32)
        obs = np.concatenate([obs, hint])
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """
        Execute one step.
        
        Args:
            action: discrete action (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            (obs_next, state_next, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Environment not reset")
        
        s = self.state
        s.steps += 1
        
        # Map action according to rule
        a_eff = self._map_action(action, s.rule)
        
        # Apply slip for rule 3
        if s.rule == 3 and self.rng.rand() < self.slip_prob:
            # Slip: random action
            a_eff = self.rng.randint(0, 4)
            s.slip_count += 1
        
        # Apply action
        x_new, y_new, hit_wall = self._apply_action(s.x, s.y, a_eff)
        
        # Update state
        s.x = x_new
        s.y = y_new
        
        if hit_wall:
            s.hit_wall_count += 1
        
        # Check if on a key
        for i, (kx, ky) in enumerate(self.key_positions):
            if int(s.x) == kx and int(s.y) == ky:
                if i not in s.key_visits:
                    s.key_visits.append(i)
                    if s.steps_to_first_key is None:
                        s.steps_to_first_key = s.steps
        
        # Compute reward
        reward = self.step_cost
        if hit_wall:
            reward += self.wall_penalty
        
        # Check goal
        done = False
        if int(s.x) == int(s.goal_x) and int(s.y) == int(s.goal_y):
            reward += self.goal_reward
            done = True
        
        # Check max steps
        if s.steps >= self.max_steps:
            done = True
        
        # Build info
        info = {
            "rule": s.rule,
            "hit_wall": 1 if hit_wall else 0,
            "hit_wall_count": s.hit_wall_count,
            "slip_count": s.slip_count,
            "key_visits": s.key_visits.copy(),
            "steps_to_first_key": s.steps_to_first_key,
            "goal_reached": 1 if done and int(s.x) == int(s.goal_x) and int(s.y) == int(s.goal_y) else 0,
        }
        
        # Volatility: slip_frac + hit_wall_frac
        volatility = (s.slip_count / max(1, s.steps)) + (s.hit_wall_count / max(1, s.steps))
        info["volatility"] = float(volatility)
        
        # Get next observation
        obs_next = self.observe()
        
        # State for tube prediction: normalized (x, y)
        state_next = np.array(self._normalize_pos(s.x, s.y), dtype=np.float32)
        
        return obs_next, state_next, reward, done, info
    
    def rollout(
        self, policy_fn, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Rollout episode.
        
        Args:
            policy_fn: function that takes obs and returns action (int)
            horizon: maximum steps
        
        Returns:
            (obs_seq, state_seq, actions, info)
        """
        obs0 = self.observe()
        obs_seq = [obs0]
        state_seq = [np.array(self._normalize_pos(self.state.x, self.state.y), dtype=np.float32)]
        actions = []
        
        for _ in range(horizon):
            if self.state is None:
                break
            
            # Get action from policy
            a = int(policy_fn(obs_seq[-1]))
            a = np.clip(a, 0, 3)  # ensure valid action
            actions.append([float(a)])
            
            # Step
            obs_next, state_next, _, done, info = self.step(a)
            obs_seq.append(obs_next)
            state_seq.append(state_next)
            
            if done:
                break
        
        obs_arr = np.asarray(obs_seq)
        state_arr = np.asarray(state_seq)
        act_arr = np.asarray(actions)
        
        # Aggregate info
        info_out = {
            "rule": self.state.rule if self.state else 0,
            "hit_wall_count": self.state.hit_wall_count if self.state else 0,
            "slip_count": self.state.slip_count if self.state else 0,
            "key_visits": self.state.key_visits.copy() if self.state else [],
            "steps_to_first_key": self.state.steps_to_first_key if self.state else None,
            "goal_reached": info.get("goal_reached", 0),
            "volatility": info.get("volatility", 0.0),
        }
        
        return obs_arr, state_arr, act_arr, info_out
