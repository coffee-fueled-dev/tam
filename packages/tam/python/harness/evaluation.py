"""
Generic evaluation functions for TAM experiments.

Provides environment-agnostic evaluation that works with any environment
that implements the Environment protocol.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from utils import truncated_geometric_weights
except ImportError:
    try:
        from ..utils import truncated_geometric_weights  # type: ignore
    except ImportError:
        # Fallback implementation
        def truncated_geometric_weights(p_stop: torch.Tensor, T: int):
            t_idx = torch.arange(1, T + 1, device=p_stop.device, dtype=p_stop.dtype)
            one_minus = (1.0 - p_stop).clamp(1e-6, 1.0 - 1e-6)
            s = one_minus ** (t_idx - 1.0)
            w = s * p_stop
            tail = one_minus**T
            w = w.clone()
            w[-1] = w[-1] + tail
            w = w / (w.sum() + 1e-8)
            E_T = (w * t_idx).sum()
            return w, E_T


@dataclass
class EvalSnapshot:
    """Evaluation snapshot containing metrics for a single evaluation checkpoint."""
    step: int
    ks: List[float]
    empirical_coverage: Dict[float, float]
    nominal_coverage: Dict[float, float]
    mean_sharp_log_vol: float
    mean_E_T: float
    mean_volatility: float
    pareto_points: np.ndarray  # rows: [sharp_log_vol, cov_error_at_k*, E_T, volatility]
    mean_J: float = 0.0  # mean control cost
    J_points: Optional[np.ndarray] = None  # per-episode J values (optional)
    mean_sigma_w: Optional[np.ndarray] = None  # weighted mean sigma per dim
    mean_err_w: Optional[np.ndarray] = None  # weighted mean |err| per dim
    z_scores: Optional[np.ndarray] = None  # standardized residuals (1D, bounded size)


def nominal_gaussian_coverage(k: float, D: int) -> float:
    """
    Nominal coverage that all D independent dims lie within +/- k sigma.
    For 1D: erf(k/sqrt(2))
    For D dims with "all dims inside": (erf(k/sqrt(2)))^D
    """
    p1 = math.erf(k / math.sqrt(2.0))
    return float(p1 ** D)


def episode_control_cost(
    states: np.ndarray,
    actions: np.ndarray,
    dt: float = 0.05,
    w_theta: float = 1.0,
    w_omega: float = 0.1,
    w_act: float = 0.01,
) -> Dict[str, float]:
    """
    Compute episode control cost J (independent of tube self-prediction).
    
    Args:
        states: [T+1, 2] array of (theta, omega) states
        actions: [T, 1] array of actions
        dt: Time step
        w_theta: Weight for theta^2 cost
        w_omega: Weight for omega^2 cost
        w_act: Weight for action^2 cost
    
    Returns:
        Dict with 'J_sum', 'J_mean', and 'cost_t' (per-step costs)
    """
    theta = states[1:, 0]  # [T]
    omega = states[1:, 1]  # [T]
    a = actions[:, 0] if actions.ndim > 1 else actions  # [T]
    
    cost = w_theta * theta**2 + w_omega * omega**2 + w_act * a**2
    return {
        "J_sum": float(cost.sum()),
        "J_mean": float(cost.mean()),
        "cost_t": cost,
    }


@torch.no_grad()
def tube_predictions_for_episode(
    agent: Any,
    obs0: np.ndarray,
    z: torch.Tensor,
    T: int,
    Hr_eval: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get tube predictions for an episode.
    
    Returns:
      mu [T, pred_dim], log_var [T, pred_dim], w [T], p_stop (scalar tensor)
    Uses the *refined* tube after Hr_eval refinement steps.
    """
    device = agent.device
    s0_t = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Respect reasoning_mode: if "off", force Hr=0
    if agent.reasoning_mode == "off":
        Hr = 0
        muK, sigK, stop_logit = agent._tube_init(z, s0_t)
    elif agent.use_dynamic_pondering and Hr_eval is None:
        # Use dynamic pondering: agent decides when to stop refining
        muK, sigK, stop_logit, actual_Hr, _ = agent.infer_tube_dynamic(
            s0_t, z, max_steps=agent.max_refine_steps
        )
    else:
        # Use fixed Hr
        Hr = agent.max_refine_steps if Hr_eval is None else int(Hr_eval)
        muK, sigK, stop_logit = agent.infer_tube(s0_t, z, Hr)
    
    p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
    
    mu, log_var = agent._tube_traj(muK, sigK, T)
    w, _E = truncated_geometric_weights(p_stop, T)
    return mu, log_var, w, p_stop


@torch.no_grad()
def episode_metrics(
    agent: Any,
    obs0: np.ndarray,
    states: np.ndarray,  # [T_actual+1, pred_dim]
    z: torch.Tensor,
    T: int,
    k_list: List[float],
    Hr_eval: Optional[int] = None,
) -> Dict:
    """
    Computes:
      - empirical coverage for each k in k_list (fraction of steps inside k-sigma, all dims)
      - sharpness summary: weighted log cone volume
      - E[T] proxy: from halting weights over horizon T
      - sigma and error aggregates for dashboard
    
    Args:
        states: [T_actual+1, pred_dim] - actual trajectory (may be shorter than T)
        T: sampled horizon from tube (may be longer than actual trajectory)
        Hr_eval: Number of reasoning steps to use for evaluation (None = max)
    """
    # Get actual trajectory length
    T_actual = len(states) - 1
    # Use the minimum of sampled T and actual trajectory length
    T_use = min(T, T_actual)
    
    mu, log_var, w, _p_stop = tube_predictions_for_episode(agent, obs0, z, T_use, Hr_eval=Hr_eval)
    std = torch.exp(0.5 * log_var)  # [T_use, pred_dim]
    y = torch.tensor(states[1:T_use+1], dtype=torch.float32, device=agent.device)  # [T_use, pred_dim]
    
    # Assert dimension match
    assert mu.shape[0] == y.shape[0], \
        f"Time dimension mismatch: mu has T={mu.shape[0]} but y has T={y.shape[0]}"
    assert mu.shape[-1] == y.shape[-1], \
        f"pred_dim mismatch: mu has D={mu.shape[-1]} but y has D={y.shape[-1]}"
    
    # sharpness: weighted log volume
    cv_t = torch.prod(std, dim=-1)  # [T_use] - product across all dimensions
    log_vol = torch.log(cv_t + 1e-8)
    sharp = float((w * log_vol).sum().item())
    
    # E[T] proxy (over 1..T_use)
    t_idx = torch.arange(1, T_use + 1, device=agent.device, dtype=w.dtype)
    E_T = float((w * t_idx).sum().item())
    
    coverages = {}
    for k in k_list:
        inside = (torch.abs(y - mu) <= (float(k) * std + 1e-8)).all(dim=-1)  # [T_use]
        cov = float(inside.float().mean().item())
        coverages[k] = cov
    
    # Dashboard aggregates: weighted sigma and error
    sigma = std  # [T_use, pred_dim]
    err = torch.abs(y - mu)  # [T_use, pred_dim]
    sigma_w = (w[:, None] * sigma).sum(0)  # [pred_dim]
    err_w = (w[:, None] * err).sum(0)  # [pred_dim]
    
    # Z-scores for histogram
    r = (y - mu) / (sigma + 1e-8)  # [T_use, pred_dim]
    
    return {
        "coverage": coverages,
        "sharp_log_vol": sharp,
        "E_T": E_T,
        "sigma_w": sigma_w.cpu().numpy(),  # [pred_dim]
        "err_w": err_w.cpu().numpy(),  # [pred_dim]
        "z_scores": r.cpu().numpy(),  # [T_use, pred_dim]
    }


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
        n_episodes: Number of evaluation episodes (0 = skip evaluation)
        k_star: k-sigma value for pareto error calculation
        Hr_eval: Number of reasoning steps (None = max)
        pred_dim: Prediction dimension (defaults to agent.pred_dim)
        outcome_fn: Optional function to compute episode outcome from (states, actions, info)
                    If None, uses control cost or distance-based metric
    
    Returns:
        EvalSnapshot object
    """
    # Handle zero episodes case
    if n_episodes == 0:
        D = pred_dim if pred_dim is not None else getattr(agent, 'pred_dim', 2)
        return EvalSnapshot(
            step=step,
            ks=ks,
            empirical_coverage={k: 0.0 for k in ks},
            nominal_coverage={k: nominal_gaussian_coverage(k, D) for k in ks},
            mean_sharp_log_vol=0.0,
            mean_E_T=0.0,
            mean_volatility=0.0,
            pareto_points=np.array([], dtype=np.float64).reshape(0, 4),
            mean_J=0.0,
            J_points=np.array([], dtype=np.float64),
            mean_sigma_w=np.zeros(D, dtype=np.float64),
            mean_err_w=np.zeros(D, dtype=np.float64),
            z_scores=np.array([], dtype=np.float64),
        )
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
    if len(z_scores_list) > 0:
        z_all = np.concatenate(z_scores_list, axis=0)
        if len(z_all) > max_r:
            idx = np.random.choice(len(z_all), size=max_r, replace=False)
            z_all = z_all[idx]
    else:
        # No episodes evaluated - create empty array
        z_all = np.array([], dtype=np.float64)
    
    # Handle case where no episodes were evaluated
    if n_episodes == 0:
        # Return empty snapshot
        empirical = {k: 0.0 for k in ks}
        nominal = {k: nominal_gaussian_coverage(k, D) for k in ks}
        pareto = np.array([], dtype=np.float64).reshape(0, 4)
    else:
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
