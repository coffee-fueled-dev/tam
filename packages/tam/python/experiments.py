"""
TAM experiment + eval suite:
- Hidden-regime + bursty switching + actuator faults
- Partial observability (obs = noisy theta only, omega hidden)
- Evaluation metrics:
  * calibration curve over k (empirical coverage vs nominal Gaussian coverage)
  * sharpness–coverage Pareto (log cone volume vs coverage error), colored by E[T]
  * dual “phase portrait” (soft_bind-target vs E[T]-target_ET) from training history
  * horizon vs cone volume scatter colored by volatility index

Assumes you have:
- actor.py with class Actor (your current interface: sample_z, sample_horizon, _policy_action,
  _tube_params, _tube_traj, train_on_episode, history dict, target_bind, target_ET)
- utils.py providing truncated_geometric_weights and interp1d_linear (or we fall back)

Run:
    python experiments.py
"""

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- imports from your codebase ---
try:
    from actor import Actor
except ImportError:
    from .actor import Actor  # type: ignore

# Prefer your utils if present; fallback to minimal versions if not.
try:
    from utils import truncated_geometric_weights
except ImportError:
    try:
        from .utils import truncated_geometric_weights  # type: ignore
    except ImportError:

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


# -----------------------------
# Run directory + saving utilities (Patch 0)
# -----------------------------
def make_run_dir(base: str = "runs", tag: str = "tam") -> Path:
    """
    Create a timestamped run directory with subfolders for figures and data.
    
    Returns:
        Path to the created run directory
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fig").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    return run_dir


def save_fig(fig, path: Path, dpi: int = 150):
    """
    Save a matplotlib figure and close it to prevent memory leaks.
    
    Args:
        fig: matplotlib figure object
        path: Path to save the figure
        dpi: Resolution for saving
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Environment: hidden regime + faults + POMDP observation
# -----------------------------
@dataclass
class FaultEvent:
    kind: str  # "invert" | "gain_drop" | "deadzone"
    duration: int
    strength: float  # meaning depends on kind


class HiddenRegimeFaultEnv:
    """
    Like your ControlledPendulumEnv but:
      - internal hidden regime (0/1), can switch stochastically and in bursts
      - actuator faults (invert, gain drop, deadzone) applied for random durations
      - partial observability: observe noisy theta only (optionally theta-only)
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        obs_noise_std: float = 0.02,
        dt: float = 0.05,
        amax: float = 2.0,
        # base regimes: physics
        g0: float = 9.8,
        f0: float = 0.1,
        g1: float = 25.0,
        f1: float = 1.5,
        # base action semantics
        base_invert_regime1: bool = True,
        base_gain_regime0: float = 1.0,
        base_gain_regime1: float = 1.25,
        base_deadzone_regime1: float = 0.0,
        # regime switching
        switch_mid_episode: bool = True,
        switch_prob_per_step: float = 0.15,
        # bursty switching windows
        burst_prob_per_step: float = 0.01,
        burst_len_range: Tuple[int, int] = (8, 20),
        burst_switch_prob: float = 0.6,
        # faults
        fault_prob_per_step: float = 0.01,
        fault_duration_range: Tuple[int, int] = (10, 35),
        # obs
        obs_mode: str = "theta_only",  # "theta_only" or "theta_omega"
        seed: int = 0,
    ):
        self.rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.obs_noise_std = obs_noise_std
        self.dt = dt
        self.amax = amax

        self.g0, self.f0 = g0, f0
        self.g1, self.f1 = g1, f1

        self.base_invert_regime1 = base_invert_regime1
        self.base_gain_regime0 = base_gain_regime0
        self.base_gain_regime1 = base_gain_regime1
        self.base_deadzone_regime1 = base_deadzone_regime1

        self.switch_mid_episode = switch_mid_episode
        self.switch_prob_per_step = switch_prob_per_step

        self.burst_prob_per_step = burst_prob_per_step
        self.burst_len_range = burst_len_range
        self.burst_switch_prob = burst_switch_prob
        self._burst_steps_left = 0

        self.fault_prob_per_step = fault_prob_per_step
        self.fault_duration_range = fault_duration_range
        self._fault: Optional[FaultEvent] = None
        self._fault_steps_left = 0

        self.obs_mode = obs_mode

        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)
        self.regime = 0

        # episode stats for volatility
        self._ep_switches = 0
        self._ep_fault_steps = 0
        self._ep_steps = 0

    @property
    def obs_dim(self) -> int:
        return 1 if self.obs_mode == "theta_only" else 2

    @property
    def state_dim(self) -> int:
        return 2

    def reset(self):
        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)
        self.regime = int(self.rng.rand() < 0.5)
        self._burst_steps_left = 0
        self._fault = None
        self._fault_steps_left = 0

        self._ep_switches = 0
        self._ep_fault_steps = 0
        self._ep_steps = 0

        return self.observe()

    def observe(self) -> np.ndarray:
        theta, omega = float(self.state[0]), float(self.state[1])
        if self.obs_mode == "theta_only":
            obs = np.array([theta], dtype=np.float64)
        else:
            obs = np.array([theta, omega], dtype=np.float64)

        obs += self.rng.normal(0, self.obs_noise_std, size=obs.shape)
        return obs

    def _maybe_start_burst(self):
        if self._burst_steps_left <= 0 and (self.rng.rand() < self.burst_prob_per_step):
            self._burst_steps_left = self.rng.randint(
                self.burst_len_range[0], self.burst_len_range[1] + 1
            )

    def _maybe_switch_regime(self):
        if not self.switch_mid_episode:
            return

        self._maybe_start_burst()
        p = (
            self.burst_switch_prob
            if self._burst_steps_left > 0
            else self.switch_prob_per_step
        )
        if self.rng.rand() < p:
            self.regime = 1 - self.regime
            self._ep_switches += 1

        if self._burst_steps_left > 0:
            self._burst_steps_left -= 1

    def _maybe_start_fault(self):
        if self._fault_steps_left > 0:
            return
        if self.rng.rand() < self.fault_prob_per_step:
            dur = self.rng.randint(
                self.fault_duration_range[0], self.fault_duration_range[1] + 1
            )
            kind = self.rng.choice(["invert", "gain_drop", "deadzone"])
            if kind == "invert":
                strength = 1.0
            elif kind == "gain_drop":
                strength = float(self.rng.uniform(0.2, 0.7))  # multiply gain by this
            else:
                strength = float(
                    self.rng.uniform(0.2, 0.8)
                )  # deadzone threshold in |a|
            self._fault = FaultEvent(kind=kind, duration=dur, strength=strength)
            self._fault_steps_left = dur

    def _effective_action(self, action: float) -> float:
        a = float(np.clip(action, -self.amax, self.amax))

        # base semantics by regime
        if self.regime == 0:
            gain = self.base_gain_regime0
            invert = False
            deadzone = 0.0
        else:
            gain = self.base_gain_regime1
            invert = self.base_invert_regime1
            deadzone = self.base_deadzone_regime1

        # apply fault modifiers if active
        if self._fault_steps_left > 0 and self._fault is not None:
            self._ep_fault_steps += 1
            if self._fault.kind == "invert":
                invert = not invert
            elif self._fault.kind == "gain_drop":
                gain *= self._fault.strength
            elif self._fault.kind == "deadzone":
                deadzone = max(deadzone, self._fault.strength)

        if abs(a) < deadzone:
            a = 0.0
        if invert:
            a = -a
        return gain * a

    def step(self, action: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Returns:
          obs_next, state_next, info
        """
        self._ep_steps += 1

        # update hidden processes
        self._maybe_switch_regime()
        self._maybe_start_fault()

        # countdown fault if active
        if self._fault_steps_left > 0:
            self._fault_steps_left -= 1
            if self._fault_steps_left == 0:
                self._fault = None

        theta, omega = float(self.state[0]), float(self.state[1])
        g, f = (self.g0, self.f0) if self.regime == 0 else (self.g1, self.f1)

        a_eff = self._effective_action(action)
        acc = -g * math.sin(theta) - f * omega + a_eff

        omega = omega + acc * self.dt
        theta = theta + omega * self.dt

        # process noise on latent state
        theta += self.rng.normal(0, self.noise_std)
        omega += self.rng.normal(0, self.noise_std)

        self.state = np.array([theta, omega], dtype=np.float64)
        obs = self.observe()

        info = {
            "regime": int(self.regime),
            "fault_active": int(self._fault is not None),
        }
        return obs, self.state.copy(), info

    def rollout(
        self, policy_fn, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Rollout:
          obs:   [T+1, obs_dim]
          states:[T+1, 2]
          actions:[T, 1]
          info: dict with volatility stats
        """
        obs0 = self.observe()
        obs_seq = [obs0]
        state_seq = [self.state.copy()]
        acts = []
        regimes = [int(self.regime)]

        for _ in range(horizon):
            a = float(policy_fn(obs_seq[-1]))
            acts.append([a])
            obs_next, s_next, info = self.step(a)
            obs_seq.append(obs_next)
            state_seq.append(s_next)
            regimes.append(int(info["regime"]))

        obs_arr = np.asarray(obs_seq)
        state_arr = np.asarray(state_seq)
        act_arr = np.asarray(acts)
        regimes = np.asarray(regimes, dtype=np.int32)

        # volatility index: switches per step + fault fraction
        switches = int(self._ep_switches)
        fault_frac = float(self._ep_fault_steps / max(1, self._ep_steps))
        vol = float(switches / max(1, self._ep_steps) + fault_frac)

        info_out = {
            "switches": switches,
            "fault_frac": fault_frac,
            "volatility": vol,
            "regime_path": regimes,
        }
        return obs_arr, state_arr, act_arr, info_out


# -----------------------------
# Evaluation helpers
# -----------------------------
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
    a = actions[:, 0]  # [T]
    
    cost = w_theta * theta**2 + w_omega * omega**2 + w_act * a**2
    return {
        "J_sum": float(cost.sum()),
        "J_mean": float(cost.mean()),
        "cost_t": cost,
    }


def nominal_gaussian_coverage(k: float, D: int) -> float:
    """
    Nominal coverage that all D independent dims lie within +/- k sigma.
    For 1D: erf(k/sqrt(2))
    For D dims with "all dims inside": (erf(k/sqrt(2)))^D
    """
    p1 = math.erf(k / math.sqrt(2.0))
    return float(p1**D)


@torch.no_grad()
def tube_predictions_for_episode(
    agent: Actor,
    obs0: np.ndarray,
    z: torch.Tensor,
    T: int,
    Hr_eval: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      mu [T, state_dim], log_var [T, state_dim], w [T], p_stop (scalar tensor)
    Uses the *refined* tube after Hr_eval refinement steps.

    Args:
        agent: Actor instance
        obs0: Initial observation
        z: Latent commitment
        T: Horizon
        Hr_eval:
          - None => use agent.max_refine_steps (or dynamic pondering if enabled)
          - int  => fixed refinement steps

    NOTE: This assumes your tube is trained on latent state targets (theta, omega).
    If you later switch to tube-on-obs, adjust targets and D accordingly.
    """
    device = agent.device
    s0_t = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)

    if agent.use_dynamic_pondering and Hr_eval is None:
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
    agent: Actor,
    obs0: np.ndarray,
    states: np.ndarray,  # [T+1, 2]
    z: torch.Tensor,
    T: int,
    k_list: List[float],
    Hr_eval: int = None,
) -> Dict:
    """
    Computes:
      - empirical coverage for each k in k_list (fraction of steps inside k-sigma, all dims)
      - sharpness summary: weighted log cone volume
      - E[T] proxy: from halting weights over horizon T
      - sigma and error aggregates for dashboard

    Args:
        Hr_eval: Number of reasoning steps to use for evaluation (None = max)
    """
    mu, log_var, w, _p_stop = tube_predictions_for_episode(agent, obs0, z, T, Hr_eval=Hr_eval)
    std = torch.exp(0.5 * log_var)  # [T,2]
    y = torch.tensor(states[1:], dtype=torch.float32, device=agent.device)  # [T,2]

    # sharpness: weighted log volume
    cv_t = torch.prod(std, dim=-1)  # [T] - product across all dimensions
    log_vol = torch.log(cv_t + 1e-8)
    sharp = float((w * log_vol).sum().item())

    # E[T] proxy (over 1..T)
    t_idx = torch.arange(1, T + 1, device=agent.device, dtype=w.dtype)
    E_T = float((w * t_idx).sum().item())

    coverages = {}
    for k in k_list:
        inside = (torch.abs(y - mu) <= (float(k) * std + 1e-8)).all(dim=-1)  # [T]
        cov = float(inside.float().mean().item())
        coverages[k] = cov

    # Dashboard aggregates: weighted sigma and error
    sigma = std  # [T,2]
    err = torch.abs(y - mu)  # [T,2]
    sigma_w = (w[:, None] * sigma).sum(0)  # [2]
    err_w = (w[:, None] * err).sum(0)  # [2]

    # Z-scores for histogram
    r = (y - mu) / (sigma + 1e-8)  # [T,2]

    return {
        "coverage": coverages,
        "sharp_log_vol": sharp,
        "E_T": E_T,
        "sigma_w": sigma_w.cpu().numpy(),  # [2]
        "err_w": err_w.cpu().numpy(),  # [2]
        "z_scores": r.cpu().numpy(),  # [T,2]
    }


# -----------------------------
# Train + periodic evaluation
# -----------------------------
@dataclass
class EvalSnapshot:
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
    mean_sigma_w: Optional[np.ndarray] = None  # weighted mean sigma per dim [2]
    mean_err_w: Optional[np.ndarray] = None  # weighted mean |err| per dim [2]
    z_scores: Optional[np.ndarray] = None  # standardized residuals (1D, bounded size)


def train_with_eval(
    seed: int = 0,
    train_steps: int = 6000,
    eval_every: int = 1000,
    eval_episodes: int = 200,
    maxH: int = 64,
    reasoning_mode: str = "fixed",
) -> Tuple[Actor, HiddenRegimeFaultEnv, List[EvalSnapshot], List[EvalSnapshot]]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # POMDP env: theta-only obs
    env = HiddenRegimeFaultEnv(
        noise_std=0.005,
        obs_noise_std=0.02,
        dt=0.05,
        amax=2.0,
        obs_mode="theta_only",
        seed=seed,
        switch_prob_per_step=0.12,
        burst_prob_per_step=0.01,
        fault_prob_per_step=0.01,
    )

    # IMPORTANT: your ActorNet input dim must match obs_dim now.
    # If your Actor/Tube nets were built with state_dim=2, set state_dim=env.obs_dim here.
    agent = Actor(
        state_dim=env.obs_dim,  # <<< key: obs-conditioned commitments
        maxH=maxH,
        minT=2,
        M=16,
        z_dim=8,
        k_sigma=2.0,
        bind_success_frac=0.85,
        lambda_h=0.002,
        beta_kl=3e-4,
        halt_bias=-1.0,
        use_dynamic_pondering=False,
        reasoning_mode=reasoning_mode,
    )

    ks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    snapshots_lo: List[EvalSnapshot] = []  # Hr=1 (minimal reasoning)
    snapshots_hi: List[EvalSnapshot] = []  # Hr=max (full reasoning)

    env.reset()
    for step in range(train_steps):
        obs0 = env.observe()
        s0_t = torch.tensor(obs0, dtype=torch.float32, device=agent.device).unsqueeze(0)
        z, z_mu, z_logstd = agent.sample_z(s0_t)

        # Sample reasoning steps first
        Hr, E_Hr_imagine, p_ref_stop = agent.sample_reasoning_steps(z, obs0)

        # Sample horizon from refined tube after Hr steps
        T, E_T_imagine, p_stop_val = agent.sample_horizon_refined(s0_t, z, Hr)

        # policy uses observation (POMDP)
        def policy_fn(obs_np):
            ot = torch.tensor(
                obs_np, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            a = agent._policy_action(z, ot).detach().cpu().numpy().squeeze()
            a = float(a + np.random.normal(0, 0.05))  # exploration
            return float(np.clip(a, -env.amax, env.amax))

        obs_seq, state_seq, actions, info = env.rollout(policy_fn=policy_fn, horizon=T)
        
        # Note: volatility is in info but train_on_episode doesn't take it yet
        # Gate score will use NLL0 by default (gate_kind="nll0")

        # Train: NOTE targets are latent states (theta, omega) just like your original
        agent.train_on_episode(
            step=step,
            regime=0,  # regime label not used meaningfully anymore; keep for logging compatibility
            s0=obs0,  # situation = observation at start
            states=state_seq,  # latent targets
            actions=actions,
            z=z,
            z_mu=z_mu,
            z_logstd=z_logstd,
            E_T_imagine=E_T_imagine,
        )

        # periodic evaluation
        if (step + 1) % eval_every == 0 or step == 0:
            # Evaluate with minimal reasoning
            snap_lo = evaluate_agent(
                agent, env, step=step + 1, ks=ks, n_episodes=eval_episodes, k_star=2.0, Hr_eval=1
            )
            snapshots_lo.append(snap_lo)

            # Evaluate with full reasoning
            snap_hi = evaluate_agent(
                agent, env, step=step + 1, ks=ks, n_episodes=eval_episodes, k_star=2.0, Hr_eval=agent.max_refine_steps
            )
            snapshots_hi.append(snap_hi)

            print(
                f"[eval @ {step + 1:5d}] "
                f"Hr=1: cov(k=2)={snap_lo.empirical_coverage[2.0]:.3f}, logvol={snap_lo.mean_sharp_log_vol:.3f} | "
                f"Hr=max: cov(k=2)={snap_hi.empirical_coverage[2.0]:.3f}, logvol={snap_hi.mean_sharp_log_vol:.3f}"
            )

    return agent, env, snapshots_lo, snapshots_hi


@torch.no_grad()
def evaluate_agent(
    agent: Actor,
    env: HiddenRegimeFaultEnv,
    step: int,
    ks: List[float],
    n_episodes: int = 200,
    k_star: float = 2.0,
    Hr_eval: int = None,
) -> EvalSnapshot:
    """
    Evaluate calibration + pareto statistics by sampling fresh episodes.

    Args:
        Hr_eval: Number of reasoning steps to use for evaluation (None = max)

    Returns:
      - empirical coverage curve over ks
      - nominal coverage curve over ks (independent Gaussian, all-dims-inside)
      - pareto points per-episode: [sharp_log_vol, |cov(k*)-nom(k*)|, E_T, volatility]
      - control cost J
      - dashboard aggregates (sigma_w, err_w, z_scores)
    """
    D = 2  # latent state dims (theta, omega)

    cov_sum = {k: 0.0 for k in ks}
    sharp_sum = 0.0
    Et_sum = 0.0
    vol_sum = 0.0
    J_sum = 0.0
    sigma_w_sum = np.zeros(2, dtype=np.float64)
    err_w_sum = np.zeros(2, dtype=np.float64)
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

        # deterministic-ish eval policy (no exploration noise)
        def policy_fn(obs_np):
            ot = torch.tensor(
                obs_np, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            a = agent._policy_action(z, ot).detach().cpu().numpy().squeeze()
            return float(np.clip(float(a), -env.amax, env.amax))

        obs_seq, state_seq, actions, info = env.rollout(policy_fn=policy_fn, horizon=T)

        # Compute control cost J
        cost_result = episode_control_cost(state_seq, actions, dt=env.dt)
        J_mean = cost_result["J_mean"]
        J_sum += J_mean
        J_points_list.append(J_mean)

        m = episode_metrics(agent, obs0, state_seq, z, T, ks, Hr_eval=Hr_eval)
        for k in ks:
            cov_sum[k] += m["coverage"][k]
        sharp_sum += m["sharp_log_vol"]
        Et_sum += m["E_T"]
        vol_sum += info["volatility"]

        # Dashboard aggregates
        sigma_w_sum += m["sigma_w"]
        err_w_sum += m["err_w"]
        
        # Collect z-scores (with size cap)
        z_ep = m["z_scores"].flatten()  # [T*2]
        z_scores_list.append(z_ep)

        cov_star = m["coverage"][k_star]
        nom_star = nominal_gaussian_coverage(k_star, D)
        cov_err = abs(cov_star - nom_star)

        pareto_rows.append([m["sharp_log_vol"], cov_err, m["E_T"], info["volatility"]])

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


# -----------------------------
# Plots
# -----------------------------
def plot_eval_snapshots(
    snapshots_lo: List[EvalSnapshot],
    snapshots_hi: List[EvalSnapshot],
    k_star: float = 2.0,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plots evaluation metrics comparing Hr=1 (minimal reasoning) vs Hr=max (full reasoning).
    Generates 3 separate figures.
    """
    # 1) calibration curves at each snapshot (empirical vs nominal as function of k)
    fig1 = plt.figure(figsize=(12, 6))
    for snap in snapshots_lo:
        xs = snap.ks
        ys = [snap.empirical_coverage[k] for k in xs]
        plt.plot(xs, ys, alpha=0.5, linestyle='-', label=f"Hr=1 step {snap.step}")
    for snap in snapshots_hi:
        xs = snap.ks
        ys = [snap.empirical_coverage[k] for k in xs]
        plt.plot(xs, ys, alpha=0.9, linestyle='--', label=f"Hr=max step {snap.step}")
    # nominal line (same for all)
    if len(snapshots_hi) > 0:
        xs = snapshots_hi[-1].ks
        nom = [snapshots_hi[-1].nominal_coverage[k] for k in xs]
        plt.plot(xs, nom, linestyle=":", linewidth=2, alpha=0.8, label="nominal Gaussian (all-dims)")
    plt.xlabel("k (tube radius in σ)")
    plt.ylabel("Empirical coverage (fraction inside)")
    plt.ylim(-0.05, 1.05)
    plt.title("Calibration curve: Hr=1 (solid) vs Hr=max (dashed)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        save_fig(fig1, save_path)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # 2) pareto front: sharpness vs coverage error at k*
    fig2 = plt.figure(figsize=(12, 6))
    for snap in snapshots_lo:
        P = snap.pareto_points
        x = P[:, 0]  # sharp_log_vol
        y = P[:, 1]  # cov error at k*
        plt.scatter(x, y, alpha=0.3, s=15, marker='o', label=f"Hr=1 step {snap.step}")
    for snap in snapshots_hi:
        P = snap.pareto_points
        x = P[:, 0]  # sharp_log_vol
        y = P[:, 1]  # cov error at k*
        plt.scatter(x, y, alpha=0.6, s=15, marker='s', label=f"Hr=max step {snap.step}")
    plt.xlabel("Sharpness = weighted log cone volume (lower is tighter)")
    plt.ylabel(f"Coverage error at k={k_star} (|emp - nominal|)")
    plt.title("Sharpness–Calibration Pareto: circles=Hr=1, squares=Hr=max")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        # Save with _pareto suffix
        pareto_path = save_path.parent / f"{save_path.stem}_pareto{save_path.suffix}"
        save_fig(fig2, pareto_path)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # 3) horizon vs cone volume scatter for both Hr levels (use last snapshot)
    if len(snapshots_lo) > 0 and len(snapshots_hi) > 0:
        snap_lo = snapshots_lo[-1]
        snap_hi = snapshots_hi[-1]
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Hr=1
        P_lo = snap_lo.pareto_points
        sc1 = ax1.scatter(P_lo[:, 0], P_lo[:, 2], c=P_lo[:, 3], alpha=0.5, s=22)
        ax1.set_xlabel("Sharpness = weighted log cone volume")
        ax1.set_ylabel("E[T]")
        ax1.set_title(f"Hr=1 (step {snap_lo.step})")
        plt.colorbar(sc1, ax=ax1, label="Volatility")

        # Hr=max
        P_hi = snap_hi.pareto_points
        sc2 = ax2.scatter(P_hi[:, 0], P_hi[:, 2], c=P_hi[:, 3], alpha=0.5, s=22)
        ax2.set_xlabel("Sharpness = weighted log cone volume")
        ax2.set_ylabel("E[T]")
        ax2.set_title(f"Hr=max (step {snap_hi.step})")
        plt.colorbar(sc2, ax=ax2, label="Volatility")

        plt.tight_layout()
        if save_path:
            # Save with _horizon suffix
            horizon_path = save_path.parent / f"{save_path.stem}_horizon{save_path.suffix}"
            save_fig(fig3, horizon_path)
        elif show:
            plt.show()
        else:
            plt.close(fig3)


def plot_dual_phase_portrait(
    agent: Actor,
    smooth: int = 200,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plots the "bargaining dynamics" in constraint space:
      x = soft_bind - target_bind
      y = E_T_train - target_ET
    If your history doesn't contain these keys yet, it will raise KeyError.
    """
    h = agent.history
    x = np.asarray(h["soft_bind"], dtype=np.float64) - float(agent.target_bind)
    y = np.asarray(h["E_T_train"], dtype=np.float64) - float(agent.target_ET)

    if smooth > 1:
        k = np.ones(smooth) / smooth
        x_s = np.convolve(x, k, mode="same")
        y_s = np.convolve(y, k, mode="same")
    else:
        x_s, y_s = x, y

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x_s, y_s, alpha=0.9)
    plt.scatter([0.0], [0.0], s=80, marker="x")
    plt.axvline(0.0, linestyle="--", alpha=0.4)
    plt.axhline(0.0, linestyle="--", alpha=0.4)
    plt.xlabel("soft_bind - target_bind")
    plt.ylabel("E[T]_train - target_ET")
    plt.title("Dual constraint phase portrait (oscillatory homeostasis)")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_overview(
    agent: Actor,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Quick recap of training metrics including reasoning compute:
      - agency proxy
      - cone volume
      - soft bind vs target
      - E[T] vs target (commitment horizon)
      - E[Hr] vs target (reasoning steps)
      - delta_nll (improvement from reasoning)
      - lambda trajectories (all three dual variables)
    """
    h = agent.history
    steps = np.asarray(h["step"], dtype=np.int32)

    exp_nll = np.asarray(h["exp_nll"], dtype=np.float64)
    agency = np.exp(-exp_nll)

    cone_vol = np.asarray(h["cone_volume"], dtype=np.float64)
    soft_bind = np.asarray(h["soft_bind"], dtype=np.float64)
    E_T_train = np.asarray(h["E_T_train"], dtype=np.float64)
    E_Hr_train = np.asarray(h["E_Hr_train"], dtype=np.float64)
    delta_nll = np.asarray(h["delta_nll"], dtype=np.float64)
    lam_bind = np.asarray(h["lambda_bind"], dtype=np.float64)
    lam_T = np.asarray(h["lambda_T"], dtype=np.float64)
    lam_r = np.asarray(h["lambda_r"], dtype=np.float64)

    fig, axes = plt.subplots(7, 1, figsize=(12, 20), sharex=True)

    axes[0].plot(steps, agency, alpha=0.85)
    axes[0].set_ylabel("exp(-E[NLL])")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Training overview with reasoning compute")

    axes[1].plot(steps, cone_vol, alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("cone vol")

    axes[2].plot(steps, soft_bind, alpha=0.85)
    axes[2].axhline(agent.target_bind, linestyle="--", alpha=0.7, label="target")
    axes[2].set_ylabel("soft bind")
    axes[2].legend()

    axes[3].plot(steps, E_T_train, alpha=0.85)
    axes[3].axhline(agent.target_ET, linestyle="--", alpha=0.7, label="target")
    axes[3].set_ylabel("E[T]_train (commitment)")
    axes[3].legend()

    axes[4].plot(steps, E_Hr_train, alpha=0.85)
    axes[4].axhline(agent.target_Hr, linestyle="--", alpha=0.7, label="target")
    axes[4].set_ylabel("E[Hr]_train (reasoning)")
    axes[4].legend()

    # Smooth delta_nll with moving average for clarity
    window = min(50, len(delta_nll) // 10 + 1)
    if len(delta_nll) > window:
        delta_nll_smooth = np.convolve(delta_nll, np.ones(window)/window, mode='valid')
        steps_smooth = steps[window-1:]
        axes[5].plot(steps_smooth, delta_nll_smooth, alpha=0.85)
    else:
        axes[5].plot(steps, delta_nll, alpha=0.85)
    axes[5].axhline(0, linestyle=":", alpha=0.5, color='gray')
    axes[5].set_ylabel("Δ NLL (reasoning improvement)")

    axes[6].plot(steps, lam_bind, alpha=0.85, label="λ_bind")
    axes[6].plot(steps, lam_T, alpha=0.85, label="λ_T")
    axes[6].plot(steps, lam_r, alpha=0.85, label="λ_r")
    axes[6].set_ylabel("dual prices")
    axes[6].set_xlabel("training step")
    axes[6].legend()

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def extract_memory(agent: Actor):
    """
    Extract memory buffer data as numpy arrays.

    Returns:
        zs: [N, z_dim] commitment vectors
        soft: [N] soft bind rates
        cone: [N] cone volumes
        lam: [N] lambda_bind values
    """
    if len(agent.mem) == 0:
        return None, None, None, None

    zs = np.stack([m["z"].numpy() for m in agent.mem], axis=0)  # [N,z_dim]
    soft = np.array([m["soft_bind"] for m in agent.mem])
    cone = np.array([m["cone_vol"] for m in agent.mem])
    lam = np.array([m["lambda_bind"] for m in agent.mem])
    return zs, soft, cone, lam


def pca2(X):
    """
    Project data to 2D using PCA.

    Args:
        X: [N, D] data matrix

    Returns:
        [N, 2] projected data
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T  # [N,2]


def plot_performance_dashboard(
    snapshots_lo: List[EvalSnapshot],
    snapshots_hi: List[EvalSnapshot],
    run_dir: Path,
    prefix: str = "perf_dashboard",
):
    """
    Generate three dashboard figures comparing Hr=1 vs Hr=max:
    A) J vs training step (performance curve)
    B) weighted σ vs weighted |err| per dimension (last snapshot)
    C) z-score histogram at 3 checkpoints
    """
    # Plot A: J vs training step
    steps_lo = [snap.step for snap in snapshots_lo]
    steps_hi = [snap.step for snap in snapshots_hi]
    J_lo = [snap.mean_J for snap in snapshots_lo]
    J_hi = [snap.mean_J for snap in snapshots_hi]

    fig_a = plt.figure(figsize=(10, 6))
    plt.plot(steps_lo, J_lo, marker='o', label="Hr=1", alpha=0.7)
    plt.plot(steps_hi, J_hi, marker='s', label="Hr=max", alpha=0.7)
    plt.xlabel("Training step")
    plt.ylabel("Mean control cost J_mean")
    plt.title("Control performance (lower is better): J_mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig_a, run_dir / "fig" / f"{prefix}_J_curve.png")

    # Plot B: weighted σ vs weighted |err| per dimension (last snapshot)
    if len(snapshots_lo) > 0 and len(snapshots_hi) > 0:
        snap_lo = snapshots_lo[-1]
        snap_hi = snapshots_hi[-1]
        
        if snap_lo.mean_sigma_w is not None and snap_lo.mean_err_w is not None:
            fig_b, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Theta dimension
            ax1.bar([0, 1], [snap_lo.mean_sigma_w[0], snap_hi.mean_sigma_w[0]], 
                   width=0.6, label="σ", alpha=0.7, color='blue')
            ax1.bar([0.3, 1.3], [snap_lo.mean_err_w[0], snap_hi.mean_err_w[0]], 
                   width=0.6, label="|err|", alpha=0.7, color='red')
            ax1.set_xticks([0.15, 1.15])
            ax1.set_xticklabels(["Hr=1", "Hr=max"])
            ax1.set_ylabel("Weighted mean")
            ax1.set_title("Theta dimension")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Omega dimension
            ax2.bar([0, 1], [snap_lo.mean_sigma_w[1], snap_hi.mean_sigma_w[1]], 
                   width=0.6, label="σ", alpha=0.7, color='blue')
            ax2.bar([0.3, 1.3], [snap_lo.mean_err_w[1], snap_hi.mean_err_w[1]], 
                   width=0.6, label="|err|", alpha=0.7, color='red')
            ax2.set_xticks([0.15, 1.15])
            ax2.set_xticklabels(["Hr=1", "Hr=max"])
            ax2.set_ylabel("Weighted mean")
            ax2.set_title("Omega dimension")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_fig(fig_b, run_dir / "fig" / f"{prefix}_sigma_vs_error_last.png")

    # Plot C: z-score histogram at 3 checkpoints
    n_snapshots = min(len(snapshots_lo), len(snapshots_hi))
    if n_snapshots > 0:
        # Pick checkpoints: first, middle, last
        idx_early = 0
        idx_mid = n_snapshots // 2
        idx_late = n_snapshots - 1
        
        fig_c, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        checkpoints = [
            (idx_early, "Early"),
            (idx_mid, "Mid"),
            (idx_late, "Late"),
        ]
        
        for col, (idx, label) in enumerate(checkpoints):
            # Hr=1
            if idx < len(snapshots_lo) and snapshots_lo[idx].z_scores is not None:
                z_lo = snapshots_lo[idx].z_scores
                axes[0, col].hist(z_lo, bins=50, alpha=0.7, density=True, color='blue')
                axes[0, col].set_xlim(-5, 5)
                axes[0, col].set_title(f"Hr=1: {label} (step {snapshots_lo[idx].step})")
                axes[0, col].set_xlabel("Z-score")
                axes[0, col].set_ylabel("Density")
                axes[0, col].grid(True, alpha=0.3)
            
            # Hr=max
            if idx < len(snapshots_hi) and snapshots_hi[idx].z_scores is not None:
                z_hi = snapshots_hi[idx].z_scores
                axes[1, col].hist(z_hi, bins=50, alpha=0.7, density=True, color='red')
                axes[1, col].set_xlim(-5, 5)
                axes[1, col].set_title(f"Hr=max: {label} (step {snapshots_hi[idx].step})")
                axes[1, col].set_xlabel("Z-score")
                axes[1, col].set_ylabel("Density")
                axes[1, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig_c, run_dir / "fig" / f"{prefix}_zscore_hists.png")


def plot_reasoning_analysis(
    agent: Actor,
    run_dir: Path,
    prefix: str = "reasoning",
):
    """
    Plot two additional analysis plots:
    1. improve vs E_Hr_train scatter (proves compute is spent where it buys fit)
    2. gate_score over training (and fraction gated-on)
    """
    h = agent.history
    steps = np.asarray(h["step"], dtype=np.int32)
    
    # Plot 1: improve vs E_Hr_train scatter
    if "improve" in h and "E_Hr_train" in h:
        improve = np.asarray(h["improve"], dtype=np.float64)
        E_Hr_train = np.asarray(h["E_Hr_train"], dtype=np.float64)
        
        fig1 = plt.figure(figsize=(10, 6))
        plt.scatter(E_Hr_train, improve, alpha=0.3, s=10)
        plt.xlabel("E[Hr]_train (expected reasoning steps)")
        plt.ylabel("Improve (NLL0 - NLLr)")
        plt.title("Reasoning effectiveness: compute spent where it buys fit")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig1, run_dir / "fig" / f"{prefix}_improve_vs_Hr.png")
    
    # Plot 2: gate_score over training and fraction gated-on
    if "gate_score" in h and "gate_on" in h:
        gate_score = np.asarray(h["gate_score"], dtype=np.float64)
        gate_on = np.asarray(h["gate_on"], dtype=np.float64)
        
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Gate score over time
        ax1.plot(steps, gate_score, alpha=0.7, label="gate_score")
        if "gate_thresh" in agent.__dict__:
            ax1.axhline(agent.gate_thresh, linestyle="--", alpha=0.5, color='red', label="threshold")
        ax1.set_ylabel("Gate score")
        ax1.set_title("Gate score over training")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fraction gated-on (moving average)
        window = min(100, len(gate_on) // 10 + 1)
        if len(gate_on) > window:
            gate_on_smooth = np.convolve(gate_on, np.ones(window)/window, mode='valid')
            steps_smooth = steps[window-1:]
            ax2.plot(steps_smooth, gate_on_smooth, alpha=0.7, label=f"fraction gated-on (MA{window})")
        else:
            ax2.plot(steps, gate_on, alpha=0.7, label="fraction gated-on")
        ax2.set_ylabel("Fraction gated-on")
        ax2.set_xlabel("Training step")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig2, run_dir / "fig" / f"{prefix}_gate_analysis.png")


def plot_z_memory_map(
    agent: Actor,
    max_points=5000,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Visualize z-space memory with 2D PCA projection.
    Shows regions colored by:
      - soft_bind (reliability)
      - log(cone_vol) (sharpness)
      - lambda_bind (reliability cost)
      - risk (regularizer target)
    """
    zs, soft, cone, lam = extract_memory(agent)

    if zs is None or zs.shape[0] == 0:
        print("No memory to plot.")
        return

    # subsample for speed
    N = zs.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        zs, soft, cone, lam = zs[idx], soft[idx], cone[idx], lam[idx]

    Z2 = pca2(zs)

    # Compute risk using same formula as actor
    risk = (1.0 - soft) + 0.2 * np.log(cone + 1e-8) + 0.05 * lam

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sc0 = axes[0, 0].scatter(Z2[:, 0], Z2[:, 1], c=soft, s=10, alpha=0.5, cmap='viridis')
    axes[0, 0].set_title("z-memory: soft_bind (high=reliable)")
    plt.colorbar(sc0, ax=axes[0, 0])

    sc1 = axes[0, 1].scatter(Z2[:, 0], Z2[:, 1], c=np.log(cone + 1e-8), s=10, alpha=0.5, cmap='viridis')
    axes[0, 1].set_title("z-memory: log cone_vol (low=tight)")
    plt.colorbar(sc1, ax=axes[0, 1])

    sc2 = axes[1, 0].scatter(Z2[:, 0], Z2[:, 1], c=lam, s=10, alpha=0.5, cmap='viridis')
    axes[1, 0].set_title("z-memory: lambda_bind (low=cheap reliability)")
    plt.colorbar(sc2, ax=axes[1, 0])

    sc3 = axes[1, 1].scatter(Z2[:, 0], Z2[:, 1], c=risk, s=10, alpha=0.5, cmap='hot')
    axes[1, 1].set_title("z-memory: risk (regularizer target)")
    plt.colorbar(sc3, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    elif show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    # Create run directory
    run_dir = make_run_dir(base="runs", tag="tam_hidden_fault")
    print(f"Run directory: {run_dir}")

    # Training parameters
    seed = 0
    train_steps = 6000
    eval_every = 1000
    eval_episodes = 200
    maxH = 64
    reasoning_mode = "fixed"  # "off" | "fixed" | "gated" | "dynamic"

    # Train
    agent, env, snapshots_lo, snapshots_hi = train_with_eval(
        seed=seed,
        train_steps=train_steps,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        maxH=maxH,
        reasoning_mode=reasoning_mode,
    )

    # Save config.json
    config = {
        "seed": seed,
        "train_steps": train_steps,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
        "maxH": maxH,
        "env": {
            "noise_std": env.noise_std,
            "obs_noise_std": env.obs_noise_std,
            "dt": env.dt,
            "amax": env.amax,
            "obs_mode": env.obs_mode,
            "switch_prob_per_step": env.switch_prob_per_step,
            "burst_prob_per_step": env.burst_prob_per_step,
            "fault_prob_per_step": env.fault_prob_per_step,
        },
        "actor": {
            "state_dim": agent.state_dim,
            "z_dim": agent.z_dim,
            "maxH": agent.maxH,
            "minT": agent.minT,
            "M": agent.M,
            "k_sigma": agent.k_sigma,
            "bind_success_frac": agent.bind_success_frac,
            "lambda_h": agent.lambda_h,
            "beta_kl": agent.beta_kl,
            "max_refine_steps": agent.max_refine_steps,
            "target_ET": agent.target_ET,
            "target_Hr": agent.target_Hr,
            "target_bind": agent.target_bind,
            "use_dynamic_pondering": agent.use_dynamic_pondering,
            "reasoning_mode": agent.reasoning_mode,
            "freeze_sigma_refine": agent.freeze_sigma_refine,
            "c_improve": agent.c_improve,
            "gate_kind": agent.gate_kind,
            "gate_thresh": agent.gate_thresh,
        },
    }
    with open(run_dir / "data" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate and save plots
    plot_training_overview(agent, save_path=run_dir / "fig" / "training_overview.png", show=False)
    plot_dual_phase_portrait(agent, smooth=200, save_path=run_dir / "fig" / "dual_phase_portrait.png", show=False)
    plot_eval_snapshots(
        snapshots_lo,
        snapshots_hi,
        k_star=2.0,
        save_path=run_dir / "fig" / "eval_calibration_pareto_horizon.png",
        show=False,
    )
    plot_z_memory_map(agent, max_points=5000, save_path=run_dir / "fig" / "z_memory_map.png", show=False)
    plot_performance_dashboard(snapshots_lo, snapshots_hi, run_dir, prefix="perf_dashboard")
    plot_reasoning_analysis(agent, run_dir, prefix="reasoning")

    # Save evaluation snapshots
    def snapshot_to_dict(snap: EvalSnapshot) -> Dict:
        return {
            "step": snap.step,
            "ks": snap.ks,
            "empirical_coverage": snap.empirical_coverage,
            "nominal_coverage": snap.nominal_coverage,
            "mean_sharp_log_vol": snap.mean_sharp_log_vol,
            "mean_E_T": snap.mean_E_T,
            "mean_volatility": snap.mean_volatility,
            "mean_J": snap.mean_J,
            "mean_sigma_w": snap.mean_sigma_w.tolist() if snap.mean_sigma_w is not None else None,
            "mean_err_w": snap.mean_err_w.tolist() if snap.mean_err_w is not None else None,
            "pareto_points": snap.pareto_points.tolist(),
            "J_points": snap.J_points.tolist() if snap.J_points is not None else None,
            "z_scores": snap.z_scores.tolist() if snap.z_scores is not None else None,
        }

    # Save snapshots as npz (more efficient for arrays)
    def save_snapshots_npz(snapshots: List[EvalSnapshot], path: Path):
        if len(snapshots) == 0:
            return
        data = {
            "step": np.array([s.step for s in snapshots]),
            "mean_sharp_log_vol": np.array([s.mean_sharp_log_vol for s in snapshots]),
            "mean_E_T": np.array([s.mean_E_T for s in snapshots]),
            "mean_volatility": np.array([s.mean_volatility for s in snapshots]),
            "mean_J": np.array([s.mean_J for s in snapshots]),
        }
        if snapshots[0].mean_sigma_w is not None:
            data["mean_sigma_w"] = np.stack([s.mean_sigma_w for s in snapshots])
            data["mean_err_w"] = np.stack([s.mean_err_w for s in snapshots])
        np.savez(path, **data)

    save_snapshots_npz(snapshots_lo, run_dir / "data" / "eval_hr1.npz")
    save_snapshots_npz(snapshots_hi, run_dir / "data" / "eval_hrmax.npz")

    # Save history
    history_dict = {k: np.asarray(v) for k, v in agent.history.items()}
    np.savez(run_dir / "data" / "history.npz", **history_dict)

    # Save memory if present
    zs, soft, cone, lam = extract_memory(agent)
    if zs is not None:
        risk = (1.0 - soft) + 0.2 * np.log(cone + 1e-8) + 0.05 * lam
        np.savez(
            run_dir / "data" / "memory.npz",
            zs=zs,
            soft=soft,
            cone=cone,
            lam=lam,
            risk=risk,
        )

    print(f"\nAll outputs saved to: {run_dir}")
    print(f"  Figures: {run_dir / 'fig'}")
    print(f"  Data: {run_dir / 'data'}")


if __name__ == "__main__":
    main()
