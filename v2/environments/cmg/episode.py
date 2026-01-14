"""
CMG episode generation utilities.
"""

from typing import Dict, Optional
import numpy as np
import torch
from .env import CMGEnv


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
    
    if not record:
        # Fast path: just run episode, return summary
        total_reward = 0
        for t in range(c.T):
            if policy_mode == "random":
                action = np.random.randn(c.m).astype(np.float32)
            elif policy_mode == "goal_seeking":
                goal = env.get_current_goal()
                action = np.clip(goal - env.x, -1.0, 1.0).astype(np.float32)
            elif policy_mode == "actor" and actor is not None:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                action = actor.act(obs_t).cpu().numpy()
            else:
                raise ValueError(f"Unknown policy_mode: {policy_mode}")
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        return {
            "return": total_reward,
            "k": info.get("k", env.k),
        }
    
    # Full recording
    record = {
        "obs": [obs],
        "x": [env.x.copy()],
        "k": [env.k],
        "actions": [],
        "rewards": [],
    }
    
    for t in range(c.T):
        if policy_mode == "random":
            action = np.random.randn(c.m).astype(np.float32)
        elif policy_mode == "goal_seeking":
            goal = env.get_current_goal()
            action = np.clip(goal - env.x, -1.0, 1.0).astype(np.float32)
        elif policy_mode == "actor" and actor is not None:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action = actor.act(obs_t).cpu().numpy()
        else:
            raise ValueError(f"Unknown policy_mode: {policy_mode}")
        
        obs, reward, done, info = env.step(action)
        
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
    
    return record
