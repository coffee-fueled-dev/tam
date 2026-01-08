"""
Generic evaluation functions for TAM experiments.

Provides environment-agnostic evaluation that works with any environment
that implements the Environment protocol.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

try:
    from experiments import (
        EvalSnapshot,
        episode_control_cost,
        episode_metrics,
        nominal_gaussian_coverage,
        tube_predictions_for_episode,
    )
except ImportError:
    try:
        from ..experiments import (
            EvalSnapshot,
            episode_control_cost,
            episode_metrics,
            nominal_gaussian_coverage,
            tube_predictions_for_episode,
        )
    except ImportError:
        # Fallback: define minimal versions
        EvalSnapshot = None  # type: ignore
        episode_control_cost = None  # type: ignore
        episode_metrics = None  # type: ignore
        nominal_gaussian_coverage = None  # type: ignore
        tube_predictions_for_episode = None  # type: ignore


def evaluate_agent_generic(
    agent: Any,
    env: Any,
    step: int,
    ks: List[float],
    n_episodes: int = 200,
    k_star: float = 2.0,
    Hr_eval: Optional[int] = None,
    pred_dim: Optional[int] = None,
    outcome_fn: Optional[Callable[[np.ndarray, np.ndarray, Dict[str, Any]], float]] = None,
) -> Any:
    """
    Generic evaluation function that works with any environment.
    
    Args:
        agent: Actor instance
        env: Environment instance
        step: Current training step
        ks: List of k-sigma values for coverage evaluation
        n_episodes: Number of evaluation episodes
        k_star: k-sigma value for pareto error calculation
        Hr_eval: Number of reasoning steps (None = max)
        pred_dim: Prediction dimension (defaults to agent.pred_dim)
        outcome_fn: Optional function to compute episode outcome from (states, actions, info)
                    If None, uses control cost or distance-based metric
    
    Returns:
        EvalSnapshot object
    """
    if EvalSnapshot is None or episode_metrics is None:
        raise ImportError("Evaluation functions not available. Import from experiments.py")
    
    # Determine prediction dimension
    D = pred_dim if pred_dim is not None else getattr(agent, 'pred_dim', 2)
    
    cov_sum = {k: 0.0 for k in ks}
    sharp_sum = 0.0
    Et_sum = 0.0
    vol_sum = 0.0
    J_sum = 0.0
    sigma_w_sum = np.zeros(D, dtype=np.float64)
    err_w_sum = np.zeros(D, dtype=np.float64)
    z_scores_list = []
    max_r = 20000  # cap z-scores size
    
    pareto_rows = []
    J_points_list = []
    
    for _ in range(n_episodes):
        env.reset()
        obs0 = env.observe()
        s0_t = torch.tensor(obs0, dtype=torch.float32, device=agent.device).unsqueeze(0)
        z, _zmu, _zls = agent.sample_z(s0_t)
        
        # Sample reasoning steps and refined horizon
        Hr_eval_eff = agent.max_refine_steps if Hr_eval is None else int(Hr_eval)
        T, _E_T_im, _ = agent.sample_horizon_refined(s0_t, z, Hr_eval_eff)
        
        # Create deterministic evaluation policy
        def policy_fn(obs_np):
            ot = torch.tensor(
                obs_np, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            action = agent._policy_action(z, ot).detach().cpu().numpy().squeeze()
            
            # Handle discrete vs continuous actions
            if agent.action_dim == 1:
                # Continuous: clip to env limits
                if hasattr(env, 'amax'):
                    action = float(np.clip(float(action), -env.amax, env.amax))
                return float(action)
            else:
                # Discrete: take best action (no exploration)
                return int(np.argmax(action))
        
        obs_seq, state_seq, actions, info = env.rollout(policy_fn=policy_fn, horizon=T)
        
        # Compute episode outcome (control cost or custom metric)
        if outcome_fn is not None:
            J_mean = outcome_fn(state_seq, actions, info)
        elif hasattr(env, 'dt') and episode_control_cost is not None:
            # Use control cost if available (pendulum-style)
            cost_result = episode_control_cost(state_seq, actions, dt=env.dt)
            J_mean = cost_result["J_mean"]
        else:
            # Fallback: use distance-based metric if available
            # For gridworld: distance to goal
            if hasattr(env, 'state') and env.state is not None:
                if hasattr(env.state, 'goal_x') and hasattr(env.state, 'goal_y'):
                    final_pos = state_seq[-1]
                    goal_norm = env._normalize_pos(env.state.goal_x, env.state.goal_y)
                    goal_pos = np.array(goal_norm)
                    dist_cost = np.linalg.norm(final_pos - goal_pos)
                    J_mean = dist_cost + (1.0 - info.get("goal_reached", 0)) * 10.0
                else:
                    # Generic: use mean squared state magnitude
                    J_mean = float(np.mean(np.sum(state_seq**2, axis=-1)))
            else:
                # Last resort: mean squared state
                J_mean = float(np.mean(np.sum(state_seq**2, axis=-1)))
        
        J_sum += J_mean
        J_points_list.append(J_mean)
        
        # Compute episode metrics
        m = episode_metrics(agent, obs0, state_seq, z, T, ks, Hr_eval=Hr_eval)
        for k in ks:
            cov_sum[k] += m["coverage"][k]
        sharp_sum += m["sharp_log_vol"]
        Et_sum += m["E_T"]
        vol_sum += info.get("volatility", 0.0)  # Default to 0 if not available
        
        # Dashboard aggregates
        sigma_w_sum += m["sigma_w"]
        err_w_sum += m["err_w"]
        
        # Collect z-scores (with size cap)
        z_ep = m["z_scores"].flatten()
        z_scores_list.append(z_ep)
        
        # Compute coverage error for pareto
        cov_star = m["coverage"][k_star]
        nom_star = nominal_gaussian_coverage(k_star, D)
        cov_err = abs(cov_star - nom_star)
        
        pareto_rows.append([m["sharp_log_vol"], cov_err, m["E_T"], info.get("volatility", 0.0)])
    
    # Aggregate z-scores and subsample if needed
    z_all = np.concatenate(z_scores_list, axis=0)
    if len(z_all) > max_r:
        idx = np.random.choice(len(z_all), size=max_r, replace=False)
        z_all = z_all[idx]
    
    empirical = {k: cov_sum[k] / n_episodes for k in ks}
    nominal = {k: nominal_gaussian_coverage(k, D) for k in ks}
    
    pareto = np.asarray(pareto_rows, dtype=np.float64)
    return EvalSnapshot(
        step=step,
        ks=ks,
        empirical_coverage=empirical,
        nominal_coverage=nominal,
        mean_sharp_log_vol=sharp_sum / n_episodes,
        mean_E_T=Et_sum / n_episodes,
        mean_volatility=vol_sum / n_episodes,
        pareto_points=pareto,
        mean_J=J_sum / n_episodes,
        J_points=np.asarray(J_points_list, dtype=np.float64),
        mean_sigma_w=sigma_w_sum / n_episodes,
        mean_err_w=err_w_sum / n_episodes,
        z_scores=z_all,
    )
