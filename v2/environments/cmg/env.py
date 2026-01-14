"""
Commitment-Mode-Gating (CMG) Environment

A controlled environment where early actions choose a discrete mode k ∈ {0,...,K-1},
and mode-conditioned dynamics drive state toward a mode-conditioned goal g_k.

This forces necessary commitment because the agent's choice causally excludes futures.

Key features:
- Vectorized (batch) support for efficient training/evaluation
- Configurable gating window, noise, dynamics
- Clean separation between mode selection and trajectory execution
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
from enum import Enum

import numpy as np
import torch


class ObsMode(Enum):
    """Observation format."""
    X_GOAL = "x_goal"           # [x_t, goal_k]
    X_GOAL_TIME = "x_goal_time" # [x_t, goal_k, t/T]
    X_GOAL_MODEID = "x_goal_modeid"  # [x_t, goal_k, onehot(k)] (diagnostic)


class DynamicsType(Enum):
    """Type of dynamics matrix A."""
    IDENTITY = "identity"
    STABLE_RANDOM = "stable_random"


@dataclass
class CMGConfig:
    """Configuration for the CMG environment."""
    # Dimensions
    d: int = 4                  # State dimension
    m: int = None               # Action dimension (default = d)
    K: int = 2                  # Number of modes
    T: int = 16                 # Horizon
    t_gate: int = 2             # Gating steps (0..t_gate-1)
    
    # Time
    dt: float = 1.0             # Step size
    
    # Noise
    noise_x: float = 0.01       # Process noise std
    noise_gate: float = 0.05    # Gating noise std
    
    # Mode dynamics
    freeze_mode_after_gate: bool = True
    sticky_mode: bool = False
    sticky_prob: float = 0.95
    
    # Dynamics
    A_type: str = "identity"
    A_scale: float = 0.9        # Spectral radius for stable_random
    B_scale: float = 0.8        # Action gain
    drift_scale: float = 0.10   # Magnitude of mode drift vectors
    goal_scale: float = 1.0     # Goal distance from origin
    goal_noise: float = 0.0
    
    # Phase 1: Early divergence (makes commitment necessary)
    early_divergence: bool = False      # Enable early mode divergence
    early_divergence_scale: float = 0.5 # Strength of early divergence
    early_divergence_end: float = 0.33  # Fraction of T for early phase
    divergence_dims: int = None         # Dims with mode-specific dynamics (None = d/2)
    
    # Phase 1: Bottleneck/fork constraint
    bottleneck: bool = False            # Enable bottleneck constraint
    bottleneck_strength: float = 0.3    # Penalty for wrong-mode region
    bottleneck_time: float = 0.25       # When bottleneck is checked (fraction of T)
    
    # Mode balance
    uniform_mode: bool = False          # Force uniform mode distribution (bypass gating)
    deterministic_mode: bool = False    # Phase 2: Mode = f(x0) - deterministic from initial state
    # For K=2: mode = sign(dot(w, x0))
    # For K>2: mode = argmax(W @ x0) where W is (K, d)
    
    # Cost
    action_cost: float = 0.01
    
    # Observation
    obs_mode: str = "x_goal_time"
    
    # Random seed
    seed: int = 0
    
    def __post_init__(self):
        if self.m is None:
            self.m = self.d
        if self.divergence_dims is None:
            self.divergence_dims = max(1, self.d // 2)


@dataclass
class CMGParams:
    """Sampled parameters for the CMG environment."""
    W: np.ndarray       # (K, m) gating weights
    A: np.ndarray       # (K, d, d) mode-conditioned dynamics
    B: np.ndarray       # (d, m) shared action matrix
    b: np.ndarray       # (K, d) mode-conditioned drift
    g: np.ndarray       # (K, d) mode-conditioned goals
    # Phase 1 additions
    b_early: np.ndarray = None   # (K, d) early divergence drift (stronger, opposite)
    bottleneck_centers: np.ndarray = None  # (K, d) bottleneck target positions
    # Phase 2: Deterministic mode selection weights
    W_mode: np.ndarray = None    # (K, d) mode selection from x0


class CMGEnv:
    """
    Commitment-Mode-Gating Environment.
    
    Supports both single and batch operation.
    """
    
    def __init__(self, config: CMGConfig = None, params: CMGParams = None):
        self.config = config or CMGConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Sample or use provided params
        if params is None:
            self.params = self.sample_params()
        else:
            self.params = params
        
        # State (will be set on reset)
        self.x = None       # (B, d) or (d,)
        self.k = None       # (B,) or int
        self.t = None       # int
        self.batch_size = None
        
        # Compute observation dimension
        self._obs_dim = self._compute_obs_dim()
    
    def _compute_obs_dim(self) -> int:
        """Compute observation dimension based on config."""
        c = self.config
        base = c.d + c.d  # x + goal
        
        if c.obs_mode == "x_goal":
            return base
        elif c.obs_mode == "x_goal_time":
            return base + 1
        elif c.obs_mode == "x_goal_modeid":
            return base + c.K
        else:
            raise ValueError(f"Unknown obs_mode: {c.obs_mode}")
    
    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self.config.m
    
    @property
    def state_dim(self) -> int:
        return self.config.d
    
    @property
    def num_modes(self) -> int:
        return self.config.K
    
    def sample_params(self, seed: int = None) -> CMGParams:
        """Sample environment parameters."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        
        c = self.config
        
        # Gating weights W: (K, m) - normalized rows
        W = rng.standard_normal((c.K, c.m)).astype(np.float32)
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        
        # Action matrix B: (d, m)
        if c.m == c.d:
            B = c.B_scale * np.eye(c.d, dtype=np.float32)
        else:
            # Random orthogonal-ish matrix
            B_raw = rng.standard_normal((c.d, c.m)).astype(np.float32)
            B = c.B_scale * B_raw / (np.linalg.norm(B_raw) + 1e-8) * np.sqrt(c.d)
        
        # Dynamics matrices A: (K, d, d)
        if c.A_type == "identity":
            A = np.stack([np.eye(c.d, dtype=np.float32) for _ in range(c.K)])
        elif c.A_type == "stable_random":
            A = []
            for _ in range(c.K):
                A_k = rng.standard_normal((c.d, c.d)).astype(np.float32)
                # Project to target spectral radius
                eigvals = np.linalg.eigvals(A_k)
                max_eig = np.max(np.abs(eigvals))
                if max_eig > 0:
                    A_k = A_k * (c.A_scale / max_eig)
                A.append(A_k)
            A = np.stack(A)
        else:
            raise ValueError(f"Unknown A_type: {c.A_type}")
        
        # Mode drift vectors b: (K, d) - arranged as simplex-like directions
        b = rng.standard_normal((c.K, c.d)).astype(np.float32)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        b = b * c.drift_scale
        
        # Mode goals g: (K, d) - aligned with drifts
        g = c.goal_scale * b / (c.drift_scale + 1e-8)
        if c.goal_noise > 0:
            g = g + rng.standard_normal((c.K, c.d)).astype(np.float32) * c.goal_noise
        
        # Phase 1: Early divergence drift vectors
        # These are STRONGER and act in OPPOSITE directions to force early separation
        b_early = None
        if c.early_divergence:
            b_early = np.zeros((c.K, c.d), dtype=np.float32)
            # Only apply to divergence_dims dimensions
            dd = c.divergence_dims
            
            # Create opposing directions in the divergent dimensions
            # For K modes, arrange as simplex vertices
            for k in range(c.K):
                angle = 2 * np.pi * k / c.K
                if dd >= 2:
                    b_early[k, 0] = np.cos(angle) * c.early_divergence_scale
                    b_early[k, 1] = np.sin(angle) * c.early_divergence_scale
                else:
                    # 1D case: alternate directions
                    b_early[k, 0] = (1 if k % 2 == 0 else -1) * c.early_divergence_scale
        
        # Phase 1: Bottleneck centers (intermediate targets modes must pass through)
        bottleneck_centers = None
        if c.bottleneck:
            # Place bottleneck centers at fraction of the way to goals
            # but offset perpendicular to create disjoint regions
            bottleneck_centers = np.zeros((c.K, c.d), dtype=np.float32)
            for k in range(c.K):
                # Start with a fraction of the goal direction
                bottleneck_centers[k] = g[k] * 0.4
                # Add perpendicular offset in first 2 dims
                angle = 2 * np.pi * k / c.K + np.pi/2
                if c.d >= 2:
                    bottleneck_centers[k, 0] += np.cos(angle) * c.bottleneck_strength
                    bottleneck_centers[k, 1] += np.sin(angle) * c.bottleneck_strength
        
        # Phase 2: Deterministic mode selection weights
        # W_mode: (K, d) - each row is a direction, mode = argmax(W_mode @ x0)
        W_mode = None
        if c.deterministic_mode:
            # Create K well-separated directions for mode selection
            W_mode = np.zeros((c.K, c.d), dtype=np.float32)
            for k in range(c.K):
                angle = 2 * np.pi * k / c.K
                if c.d >= 2:
                    W_mode[k, 0] = np.cos(angle)
                    W_mode[k, 1] = np.sin(angle)
                else:
                    W_mode[k, 0] = 1.0 if k % 2 == 0 else -1.0
        
        return CMGParams(W=W, A=A, B=B, b=b, g=g, b_early=b_early,
                        bottleneck_centers=bottleneck_centers, W_mode=W_mode)
    
    def reset(self, batch_size: int = None) -> np.ndarray:
        """
        Reset environment.
        
        Args:
            batch_size: If None, single env. If int, batch of envs.
        
        Returns:
            obs: Initial observation (obs_dim,) or (B, obs_dim)
        """
        c = self.config
        p = self.params
        self.batch_size = batch_size
        self.t = 0
        
        if batch_size is None:
            # Single environment
            self.x = self.rng.uniform(-1, 1, c.d).astype(np.float32)
            
            # Mode selection
            if c.uniform_mode:
                # Random mode (for baseline)
                self.k = self.rng.integers(0, c.K)
            elif c.deterministic_mode and p.W_mode is not None:
                # Phase 2: mode = f(x0) = argmax(W_mode @ x0)
                logits = p.W_mode @ self.x  # (K,)
                self.k = int(np.argmax(logits))
            else:
                # Default: random initial mode (will be updated by gating)
                self.k = self.rng.integers(0, c.K)
        else:
            # Batch environment
            self.x = self.rng.uniform(-1, 1, (batch_size, c.d)).astype(np.float32)
            
            # Mode selection
            if c.uniform_mode:
                self.k = self.rng.integers(0, c.K, batch_size)
            elif c.deterministic_mode and p.W_mode is not None:
                # Phase 2: mode = f(x0) = argmax(W_mode @ x0)
                logits = self.x @ p.W_mode.T  # (B, K)
                self.k = np.argmax(logits, axis=1)
            else:
                self.k = self.rng.integers(0, c.K, batch_size)
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        c = self.config
        
        if self.batch_size is None:
            # Single env
            goal = self.params.g[self.k]
            
            if c.obs_mode == "x_goal":
                return np.concatenate([self.x, goal]).astype(np.float32)
            elif c.obs_mode == "x_goal_time":
                return np.concatenate([self.x, goal, [self.t / c.T]]).astype(np.float32)
            elif c.obs_mode == "x_goal_modeid":
                onehot = np.zeros(c.K, dtype=np.float32)
                onehot[self.k] = 1.0
                return np.concatenate([self.x, goal, onehot]).astype(np.float32)
        else:
            # Batch env
            goal = self.params.g[self.k]  # (B, d)
            
            if c.obs_mode == "x_goal":
                return np.concatenate([self.x, goal], axis=1).astype(np.float32)
            elif c.obs_mode == "x_goal_time":
                time_frac = np.full((self.batch_size, 1), self.t / c.T, dtype=np.float32)
                return np.concatenate([self.x, goal, time_frac], axis=1).astype(np.float32)
            elif c.obs_mode == "x_goal_modeid":
                onehot = np.zeros((self.batch_size, c.K), dtype=np.float32)
                onehot[np.arange(self.batch_size), self.k] = 1.0
                return np.concatenate([self.x, goal, onehot], axis=1).astype(np.float32)
    
    def _update_mode(self, action: np.ndarray) -> np.ndarray:
        """
        Update mode based on action (during gating window).
        
        Returns:
            switched: bool or (B,) bool indicating if mode changed
        """
        c = self.config
        p = self.params
        
        # If uniform_mode is enabled, skip gating and keep initial mode
        if c.uniform_mode:
            return False if self.batch_size is None else np.zeros(self.batch_size, dtype=bool)
        
        # Compute gating logits: l_k = w_k^T tanh(a) + noise
        phi_a = np.tanh(action)  # (m,) or (B, m)
        
        if self.batch_size is None:
            logits = p.W @ phi_a  # (K,)
            if c.noise_gate > 0:
                logits = logits + self.rng.normal(0, c.noise_gate, c.K)
            
            old_k = self.k
            
            if c.sticky_mode and self.t > 0:
                if self.rng.random() > c.sticky_prob:
                    self.k = np.argmax(logits)
            else:
                self.k = np.argmax(logits)
            
            return self.k != old_k
        else:
            logits = (p.W @ phi_a.T).T  # (B, K)
            if c.noise_gate > 0:
                logits = logits + self.rng.normal(0, c.noise_gate, (self.batch_size, c.K))
            
            old_k = self.k.copy()
            
            if c.sticky_mode and self.t > 0:
                switch_mask = self.rng.random(self.batch_size) > c.sticky_prob
                new_k = np.argmax(logits, axis=1)
                self.k = np.where(switch_mask, new_k, self.k)
            else:
                self.k = np.argmax(logits, axis=1)
            
            return self.k != old_k
    
    def _dynamics_step(self, action: np.ndarray):
        """Apply dynamics update."""
        c = self.config
        p = self.params
        
        # Phase 1: Use early divergence drift in early phase
        early_phase = c.early_divergence and self.t < int(c.early_divergence_end * c.T)
        
        if self.batch_size is None:
            # x_{t+1} = A_k x_t + B a_t + b_k + noise
            A_k = p.A[self.k]
            
            # Select drift based on phase
            if early_phase and p.b_early is not None:
                b_k = p.b_early[self.k]  # Strong divergent drift
            else:
                b_k = p.b[self.k]  # Normal drift toward goal
            
            self.x = A_k @ self.x + p.B @ action + b_k
            if c.noise_x > 0:
                self.x = self.x + self.rng.normal(0, c.noise_x, c.d).astype(np.float32)
        else:
            # Batch dynamics
            A_k = p.A[self.k]  # (B, d, d)
            
            # Select drift based on phase
            if early_phase and p.b_early is not None:
                b_k = p.b_early[self.k]  # (B, d)
            else:
                b_k = p.b[self.k]  # (B, d)
            
            # x_{t+1} = A_k @ x_t + B @ a_t + b_k
            Ax = np.einsum('bij,bj->bi', A_k, self.x)
            Ba = action @ p.B.T  # (B, m) @ (m, d) = (B, d)
            
            self.x = Ax + Ba + b_k
            if c.noise_x > 0:
                self.x = self.x + self.rng.normal(0, c.noise_x, (self.batch_size, c.d)).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: (m,) or (B, m) action
        
        Returns:
            obs: Next observation
            reward: Step reward (or array for batch)
            done: Whether episode ended
            info: Additional info
        """
        c = self.config
        
        action = np.asarray(action, dtype=np.float32)
        
        # Mode update (if in gating window)
        switched = False
        if self.t < c.t_gate:
            switched = self._update_mode(action)
        elif not c.freeze_mode_after_gate:
            if c.sticky_mode:
                switched = self._update_mode(action)
        
        # Dynamics update
        self._dynamics_step(action)
        
        # Increment time
        self.t += 1
        done = self.t >= c.T
        
        # Compute reward
        if self.batch_size is None:
            action_cost = c.action_cost * np.sum(action ** 2)
            if done:
                goal = self.params.g[self.k]
                terminal_cost = np.sum((self.x - goal) ** 2)
                reward = -terminal_cost - action_cost
            else:
                reward = -action_cost
        else:
            action_cost = c.action_cost * np.sum(action ** 2, axis=1)
            if done:
                goal = self.params.g[self.k]  # (B, d)
                terminal_cost = np.sum((self.x - goal) ** 2, axis=1)
                reward = -terminal_cost - action_cost
            else:
                reward = -action_cost
        
        obs = self._get_obs()
        
        info = {
            "k": self.k,
            "x": self.x.copy(),
            "goal": self.params.g[self.k] if self.batch_size is None else self.params.g[self.k].copy(),
            "switched": switched,
            "t": self.t,
        }
        
        return obs, reward, done, info
    
    def get_mode_goals(self) -> np.ndarray:
        """Get all mode goals. Shape: (K, d)"""
        return self.params.g.copy()
    
    def get_current_goal(self) -> np.ndarray:
        """Get goal for current mode(s). Shape: (d,) or (B, d)"""
        return self.params.g[self.k]


def rollout_with_forced_mode(
    env: CMGEnv,
    k_forced: int,
    policy,
    device: torch.device = None,
) -> Dict[str, np.ndarray]:
    """
    Rollout with a forced mode (for causal intervention testing).
    
    Args:
        env: CMG environment
        k_forced: Mode to force
        policy: Policy function (obs -> action) or "random" or "goal_seeking"
    
    Returns:
        record: Dict with trajectory data
    """
    c = env.config
    
    # Reset and force mode
    obs = env.reset()
    env.k = k_forced
    
    record = {
        "obs": [obs],
        "x": [env.x.copy()],
        "k": [env.k],
        "actions": [],
        "rewards": [],
    }
    
    for t in range(c.T):
        # Get action
        if policy == "random":
            action = np.random.randn(c.m).astype(np.float32)
        elif policy == "goal_seeking":
            # Simple proportional controller
            goal = env.get_current_goal()
            x = env.x
            action = np.clip(goal - x, -1.0, 1.0).astype(np.float32)
        elif callable(policy):
            if device is not None:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                action = policy(obs_t).cpu().numpy()
            else:
                action = policy(obs)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        
        # Force mode to stay at k_forced
        env.k = k_forced
        
        obs, reward, done, info = env.step(action)
        
        # Force mode again after step (in case dynamics changed it)
        env.k = k_forced
        
        record["obs"].append(obs)
        record["x"].append(env.x.copy())
        record["k"].append(env.k)
        record["actions"].append(action)
        record["rewards"].append(reward)
    
    # Convert to arrays
    record["obs"] = np.array(record["obs"])
    record["x"] = np.array(record["x"])
    record["k"] = np.array(record["k"])
    record["actions"] = np.array(record["actions"])
    record["rewards"] = np.array(record["rewards"])
    record["return"] = np.sum(record["rewards"])
    record["k_forced"] = k_forced
    record["final_goal_dist"] = np.linalg.norm(record["x"][-1] - env.params.g[k_forced])
    
    return record


def generate_episode(
    env: CMGEnv,
    actor=None,
    policy_mode: str = "random",
    record: bool = True,
    device: torch.device = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a single episode.
    
    Args:
        env: CMG environment
        actor: Actor object with bind() and act() methods (if policy_mode="actor")
        policy_mode: "random", "goal_seeking", or "actor"
        record: Whether to record full trajectory
        device: Torch device for actor
    
    Returns:
        record: Dict with trajectory data
    """
    c = env.config
    
    obs = env.reset()
    
    data = {
        "obs": [obs],
        "x": [env.x.copy()],
        "k": [env.k if isinstance(env.k, int) else env.k.copy()],
        "goal": [env.get_current_goal()],
        "actions": [],
        "rewards": [],
    }
    
    # Binding phase (if actor mode)
    z_star = None
    if policy_mode == "actor" and actor is not None:
        if device is not None:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32)
        
        if hasattr(actor, 'bind'):
            z_star = actor.bind(obs_t)
        elif hasattr(actor, 'select_z_cem'):
            z_star, _ = actor.select_z_cem(obs_t)
        
        data["z_star"] = z_star.cpu().numpy() if hasattr(z_star, 'cpu') else z_star
    
    for t in range(c.T):
        # Get action
        if policy_mode == "random":
            action = np.random.randn(c.m).astype(np.float32)
        elif policy_mode == "goal_seeking":
            goal = env.get_current_goal()
            x = env.x
            action = np.clip(goal - x, -1.0, 1.0).astype(np.float32)
        elif policy_mode == "actor" and actor is not None:
            if device is not None:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32)
            
            if hasattr(actor, 'act'):
                action = actor.act(obs_t, z_star)
            else:
                # Default: use actor's mu prediction as action
                mu, _ = actor.predict_tube(obs_t, z_star)
                action = mu[0, t, :].cpu().numpy() if hasattr(mu, 'cpu') else mu[0, t, :]
            
            action = np.asarray(action, dtype=np.float32)
        else:
            raise ValueError(f"Unknown policy_mode: {policy_mode}")
        
        obs, reward, done, info = env.step(action)
        
        if record:
            data["obs"].append(obs)
            data["x"].append(env.x.copy())
            data["k"].append(info["k"])
            data["goal"].append(env.get_current_goal())
            data["actions"].append(action)
            data["rewards"].append(reward)
    
    # Convert to arrays
    data["obs"] = np.array(data["obs"])
    data["x"] = np.array(data["x"])
    data["k"] = np.array(data["k"])
    data["goal"] = np.array(data["goal"])
    data["actions"] = np.array(data["actions"])
    data["rewards"] = np.array(data["rewards"])
    data["return"] = float(np.sum(data["rewards"]))
    
    return data


# =============================================================================
# Testing
# =============================================================================

def test_cmg_env():
    """Test the CMG environment."""
    print("Testing CMG Environment...")
    
    # Test single env
    config = CMGConfig(d=4, K=2, T=16, t_gate=2)
    env = CMGEnv(config)
    
    print(f"  Config: d={config.d}, K={config.K}, T={config.T}, t_gate={config.t_gate}")
    print(f"  Obs dim: {env.obs_dim}, Action dim: {env.action_dim}")
    
    obs = env.reset()
    print(f"  Reset: obs shape={obs.shape}, k={env.k}")
    
    # Run episode with random actions
    total_reward = 0
    for t in range(config.T):
        action = np.random.randn(config.m).astype(np.float32)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if t < 3 or done:
            print(f"  Step {t}: k={info['k']}, switched={info['switched']}, reward={reward:.3f}")
    
    print(f"  Total reward: {total_reward:.3f}")
    
    # Test batch env
    print("\n  Testing batch mode...")
    batch_size = 32
    obs = env.reset(batch_size=batch_size)
    print(f"  Batch reset: obs shape={obs.shape}, k shape={env.k.shape}")
    
    for t in range(config.T):
        action = np.random.randn(batch_size, config.m).astype(np.float32)
        obs, reward, done, info = env.step(action)
    
    print(f"  Batch final rewards: mean={np.mean(reward):.3f}, std={np.std(reward):.3f}")
    
    # Test goal-seeking policy
    print("\n  Testing goal-seeking policy...")
    episode = generate_episode(env, policy_mode="goal_seeking")
    print(f"  Goal-seeking return: {episode['return']:.3f}")
    print(f"  Final x: {episode['x'][-1][:2]}...")
    print(f"  Goal: {episode['goal'][-1][:2]}...")
    
    # Test forced mode rollout
    print("\n  Testing forced mode rollout...")
    for k in range(config.K):
        record = rollout_with_forced_mode(env, k, "goal_seeking")
        print(f"  Mode {k}: return={record['return']:.3f}, final_dist={record['final_goal_dist']:.3f}")
    
    print("\n✓ CMG Environment works!")


if __name__ == "__main__":
    test_cmg_env()
