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

import math
from dataclasses import dataclass
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
          - None => use agent.max_refine_steps (upper bound eval)
          - int  => fixed refinement steps

    NOTE: This assumes your tube is trained on latent state targets (theta, omega).
    If you later switch to tube-on-obs, adjust targets and D accordingly.
    """
    device = agent.device
    s0_t = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)

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

    return {
        "coverage": coverages,
        "sharp_log_vol": sharp,
        "E_T": E_T,
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


def train_with_eval(
    seed: int = 0,
    train_steps: int = 6000,
    eval_every: int = 1000,
    eval_episodes: int = 200,
    maxH: int = 64,
) -> Tuple[Actor, List[EvalSnapshot]]:
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

    return agent, snapshots_lo, snapshots_hi


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
    """
    D = 2  # latent state dims (theta, omega)

    cov_sum = {k: 0.0 for k in ks}
    sharp_sum = 0.0
    Et_sum = 0.0
    vol_sum = 0.0

    pareto_rows = []

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

        m = episode_metrics(agent, obs0, state_seq, z, T, ks, Hr_eval=Hr_eval)
        for k in ks:
            cov_sum[k] += m["coverage"][k]
        sharp_sum += m["sharp_log_vol"]
        Et_sum += m["E_T"]
        vol_sum += info["volatility"]

        cov_star = m["coverage"][k_star]
        nom_star = nominal_gaussian_coverage(k_star, D)
        cov_err = abs(cov_star - nom_star)

        pareto_rows.append([m["sharp_log_vol"], cov_err, m["E_T"], info["volatility"]])

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
    )


# -----------------------------
# Plots
# -----------------------------
def plot_eval_snapshots(
    snapshots_lo: List[EvalSnapshot],
    snapshots_hi: List[EvalSnapshot],
    k_star: float = 2.0
):
    """
    Plots evaluation metrics comparing Hr=1 (minimal reasoning) vs Hr=max (full reasoning).
    """
    # 1) calibration curves at each snapshot (empirical vs nominal as function of k)
    plt.figure(figsize=(12, 6))
    for snap in snapshots_lo:
        xs = snap.ks
        ys = [snap.empirical_coverage[k] for k in xs]
        plt.plot(xs, ys, alpha=0.5, linestyle='-', label=f"Hr=1 step {snap.step}")
    for snap in snapshots_hi:
        xs = snap.ks
        ys = [snap.empirical_coverage[k] for k in xs]
        plt.plot(xs, ys, alpha=0.9, linestyle='--', label=f"Hr=max step {snap.step}")
    # nominal line (same for all)
    xs = snapshots_hi[-1].ks
    nom = [snapshots_hi[-1].nominal_coverage[k] for k in xs]
    plt.plot(xs, nom, linestyle=":", linewidth=2, alpha=0.8, label="nominal Gaussian (all-dims)")
    plt.xlabel("k (tube radius in σ)")
    plt.ylabel("Empirical coverage (fraction inside)")
    plt.ylim(-0.05, 1.05)
    plt.title("Calibration curve: Hr=1 (solid) vs Hr=max (dashed)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2) pareto front: sharpness vs coverage error at k*
    plt.figure(figsize=(12, 6))
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
    plt.show()

    # 3) horizon vs cone volume scatter for both Hr levels (use last snapshot)
    snap_lo = snapshots_lo[-1]
    snap_hi = snapshots_hi[-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
    plt.show()


def plot_dual_phase_portrait(agent: Actor, smooth: int = 200):
    """
    Plots the “bargaining dynamics” in constraint space:
      x = soft_bind - target_bind
      y = E_T_train - target_ET
    If your history doesn’t contain these keys yet, it will raise KeyError.
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

    plt.figure(figsize=(7, 7))
    plt.plot(x_s, y_s, alpha=0.9)
    plt.scatter([0.0], [0.0], s=80, marker="x")
    plt.axvline(0.0, linestyle="--", alpha=0.4)
    plt.axhline(0.0, linestyle="--", alpha=0.4)
    plt.xlabel("soft_bind - target_bind")
    plt.ylabel("E[T]_train - target_ET")
    plt.title("Dual constraint phase portrait (oscillatory homeostasis)")
    plt.tight_layout()
    plt.show()


def plot_training_overview(agent: Actor):
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
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    agent, snapshots_lo, snapshots_hi = train_with_eval(
        seed=0,
        train_steps=6000,
        eval_every=1000,
        eval_episodes=200,
        maxH=64,
    )

    plot_training_overview(agent)
    plot_dual_phase_portrait(agent, smooth=200)
    plot_eval_snapshots(snapshots_lo, snapshots_hi, k_star=2.0)


if __name__ == "__main__":
    main()
