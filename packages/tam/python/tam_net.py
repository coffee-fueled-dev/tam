"""
Action-Binding TAM with GEOMETRIC TUBE (no 64-step NN rollouts per port)

Key idea (#2): Each port predicts a *tube in trajectory space* via knot values
and linear interpolation, rather than enumerating predicted states by unrolling the NN.

- Port policy: torque action a = pi_p(x) (shared policy backbone + per-port LoRA)
- Port tube: predicts mu_t, sigma_t for t=1..T via knot values + linear interpolation
            (shared tube backbone + per-port LoRA)
- Learned horizon: tube also predicts p_stop (geometric halting distribution), truncated at maxH
- Bind length T is sampled from that distribution (so horizon is learned, cheaply)
- Binding predicate: realized trajectory stays within k-sigma tube for >= frac steps
- Training: expected per-step NLL weighted by geometric halting weights (Option 3), plus
           cone regularizer + ponder cost lambda_h * E[T]

Python 3.8+
"""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utilities
# -----------------------------
@torch.no_grad()
def gaussian_nll(
    mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Elementwise Gaussian NLL up to constant. Shapes [*, D]."""
    var = torch.exp(log_var)
    sq = (target - mu) ** 2
    return 0.5 * (log_var + sq / (var + 1e-8))


def poly_basis(t: torch.Tensor, K: int) -> torch.Tensor:
    """
    DEPRECATED: Polynomial basis function (replaced by knot-based interpolation).
    Kept for reference only.

    Polynomial basis phi=[1,t,t^2,...] for normalized time t in [0,1]
    t: [T] float
    returns: [T, K]
    """
    # K=4 => [1,t,t^2,t^3]
    out = [torch.ones_like(t)]
    for k in range(1, K):
        out.append(out[-1] * t)
    return torch.stack(out, dim=-1)  # [T,K]


def truncated_geometric_weights(
    p_stop: torch.Tensor, T: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Option-3 style weights for an observed horizon T with a geometric stop prob p_stop.
    p_stop: scalar tensor in (0,1)
    Returns:
      w: [T] weights summing to 1 (with tail mass forced into step T)
      E_T: scalar tensor expected step index under w over 1..T
    """
    # survival s_t = (1-p)^(t-1)
    t_idx = torch.arange(1, T + 1, device=p_stop.device, dtype=p_stop.dtype)
    one_minus = (1.0 - p_stop).clamp(1e-6, 1.0 - 1e-6)

    s = one_minus ** (t_idx - 1.0)  # [T]
    w = s * p_stop  # [T], unnormalized "stop at t"

    # tail mass beyond T gets added to last step
    tail = one_minus**T  # prob not stopped by step T
    w = w.clone()
    w[-1] = w[-1] + tail

    w = w / (w.sum() + 1e-8)
    E_T = (w * t_idx).sum()
    return w, E_T


def sample_truncated_geometric(p_stop: float, maxH: int, minT: int = 1) -> int:
    """
    Sample T in [1..maxH] from truncated geometric with parameter p_stop.
    Enforce minT by collapsing probability mass below minT into minT.
    """
    p = float(np.clip(p_stop, 1e-4, 1.0 - 1e-4))
    one_minus = 1.0 - p
    # P(T=t) = (1-p)^(t-1) p for t<maxH, and tail mass at maxH
    probs = np.array(
        [(one_minus ** (t - 1)) * p for t in range(1, maxH)], dtype=np.float64
    )
    tail = one_minus ** (maxH - 1)
    probs = np.concatenate(
        [probs, np.array([tail], dtype=np.float64)], axis=0
    )  # len=maxH

    probs = probs / (probs.sum() + 1e-12)

    if minT > 1:
        m = float(np.sum(probs[: minT - 1]))
        probs[minT - 1] += m
        probs[: minT - 1] = 0.0
        probs = probs / (probs.sum() + 1e-12)

    return int(np.random.choice(np.arange(1, maxH + 1), p=probs))


def interp1d_linear(knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation from knot values.
    knots: [M, D]
    t: [T] in [0,1]
    returns: [T, D]
    """
    M, D = knots.shape
    # map t to continuous knot index in [0, M-1]
    u = t * (M - 1)
    i0 = torch.floor(u).long().clamp(0, M - 2)  # [T]
    i1 = i0 + 1  # [T]
    w = (u - i0.float()).unsqueeze(-1)  # [T, 1]

    v0 = knots[i0]  # [T, D]
    v1 = knots[i1]  # [T, D]
    return (1 - w) * v0 + w * v1


# -----------------------------
# LoRA building blocks
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0
    ):
        super().__init__()
        self.scaling = alpha / max(r, 1)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.t()) @ self.B.t() * self.scaling

    @torch.no_grad()
    def copy_from(self, other: "LoRALinear", noise: float = 0.0):
        self.A.copy_(other.A + noise * torch.randn_like(other.A))
        self.B.copy_(other.B + noise * torch.randn_like(other.B))


# -----------------------------
# Shared policy backbone: action = pi(state, embedding)
# -----------------------------
class SharedPolicy(nn.Module):
    def __init__(self, state_dim: int, identity_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + identity_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


# -----------------------------
# Shared tube backbone: predicts coeffs for mu(t), log_sigma(t), and stop prob
# Input: [s0, embedding]
# Output: vec of size state_dim*K_mu + state_dim*K_sig + 1
# -----------------------------
class SharedTube(nn.Module):
    def __init__(
        self, state_dim: int, identity_dim: int, hidden_dim: int = 64, out_dim: int = 1
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + identity_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()


# -----------------------------
# Port definition: embedding + LoRA for policy and tube nets + stats
# -----------------------------
class Port(nn.Module):
    def __init__(
        self,
        port_id: int,
        identity_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        lora_r: int = 4,
    ):
        super().__init__()
        self.id = port_id
        self.embedding = nn.Parameter(torch.randn(identity_dim) * 0.1)

        # policy LoRA
        pol_in = state_dim + identity_dim
        self.pol_lora1 = LoRALinear(pol_in, hidden_dim, r=lora_r)
        self.pol_lora2 = LoRALinear(hidden_dim, hidden_dim, r=lora_r)

        # tube LoRA
        tube_in = state_dim + identity_dim
        self.tube_lora1 = LoRALinear(tube_in, hidden_dim, r=lora_r)
        self.tube_lora2 = LoRALinear(hidden_dim, hidden_dim, r=lora_r)

        # surprise stats (proliferation)
        self.surprise_history: List[float] = []
        self.mu_surp = 1.0
        self.mad_surp = 0.5

        self.usage_ema = 0.0
        self.birth_step = 0
        self.select_count = 0

        self.bind_success_ema = 0.5

        # concession tracking (for detecting hedging behavior)
        self.cone_mu = 0.0  # baseline log(cone_volume) when stable
        self.cone_mad = 1.0  # MAD of log(cone_volume)
        self.T_mu = 16.0  # baseline horizon when stable
        self.T_mad = 8.0  # MAD of horizon
        self.concession_ema = 0.0  # EMA of concession index

    @torch.no_grad()
    def update_bind_success(self, success: bool, beta: float = 0.02):
        x = 1.0 if success else 0.0
        self.bind_success_ema = (1 - beta) * self.bind_success_ema + beta * x

    @torch.no_grad()
    def update_usage(self, selected: bool, beta: float = 0.01):
        self.usage_ema = (1 - beta) * self.usage_ema + beta * (1.0 if selected else 0.0)

    @torch.no_grad()
    def update_stats(self, surp: float, is_bound: bool):
        self.surprise_history.append(float(surp))
        alpha = 0.02 if is_bound else 0.005
        old_mu = self.mu_surp
        self.mu_surp = (1 - alpha) * self.mu_surp + alpha * float(surp)
        self.mad_surp = (1 - alpha) * self.mad_surp + alpha * abs(float(surp) - old_mu)

    @torch.no_grad()
    def update_concession_baseline(
        self, log_cone_vol: float, T_val: float, is_good_bind: bool
    ):
        """
        Update baseline cone volume and horizon only on 'good' episodes.
        A good bind is one with low NLL and successful binding.
        """
        if not is_good_bind:
            return

        alpha = 0.05
        old_cone_mu = self.cone_mu
        old_T_mu = self.T_mu

        self.cone_mu = (1 - alpha) * self.cone_mu + alpha * log_cone_vol
        self.cone_mad = (1 - alpha) * self.cone_mad + alpha * abs(
            log_cone_vol - old_cone_mu
        )

        self.T_mu = (1 - alpha) * self.T_mu + alpha * T_val
        self.T_mad = (1 - alpha) * self.T_mad + alpha * abs(T_val - old_T_mu)

    @torch.no_grad()
    def compute_concession(
        self,
        log_cone_vol: float,
        T_val: float,
        avg_margin: float,
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.3,
        margin_target: float = 0.5,
    ) -> float:
        """
        Compute concession index as weighted sum of:
          c1: cone volume inflation (z-score)
          c2: horizon shrinkage (z-score)
          c3: margin slack usage
        """
        # c1: cone inflation
        z_cone = (log_cone_vol - self.cone_mu) / (self.cone_mad + 1e-6)
        c1 = max(0.0, z_cone)

        # c2: horizon shrinkage
        z_T = (self.T_mu - T_val) / (self.T_mad + 1e-6)
        c2 = max(0.0, z_T)

        # c3: slack usage
        c3 = max(0.0, margin_target - avg_margin)

        return w1 * c1 + w2 * c2 + w3 * c3

    @torch.no_grad()
    def update_concession_ema(self, concession: float, beta: float = 0.02):
        self.concession_ema = (1 - beta) * self.concession_ema + beta * concession

    def z_score(self) -> float:
        n = len(self.surprise_history)
        if n < 25:
            return 0.0
        recent = float(np.mean(self.surprise_history[-10:]))
        z = (recent - self.mu_surp) / (self.mad_surp + 1e-6)
        evidence = n / (n + 200.0)
        return float(np.clip(z * evidence, -10.0, 10.0))

    @torch.no_grad()
    def copy_from(self, parent: "Port", noise: float = 0.05):
        self.embedding.copy_(
            parent.embedding + noise * torch.randn_like(parent.embedding)
        )
        self.pol_lora1.copy_from(parent.pol_lora1, noise=0.01)
        self.pol_lora2.copy_from(parent.pol_lora2, noise=0.01)
        self.tube_lora1.copy_from(parent.tube_lora1, noise=0.01)
        self.tube_lora2.copy_from(parent.tube_lora2, noise=0.01)


# -----------------------------
# Environment (your regime-dependent actuator semantics)
# -----------------------------
class ControlledPendulumEnv:
    def __init__(
        self,
        noise_std: float = 0.005,
        dt: float = 0.05,
        amax: float = 2.0,
        action_invert_regime1: bool = True,
        action_gain_regime0: float = 1.0,
        action_gain_regime1: float = 1.25,
        action_deadzone_regime1: float = 0.0,
        switch_mid_episode: bool = True,
        switch_prob_per_step: float = 0.15,
    ):
        self.noise_std = noise_std
        self.dt = dt
        self.amax = amax

        self.action_invert_regime1 = action_invert_regime1
        self.action_gain_regime0 = action_gain_regime0
        self.action_gain_regime1 = action_gain_regime1
        self.action_deadzone_regime1 = action_deadzone_regime1

        self.switch_mid_episode = switch_mid_episode
        self.switch_prob_per_step = switch_prob_per_step

        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)

    def reset(self):
        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)

    def _effective_action(self, regime_id: int, action: float) -> float:
        a = float(np.clip(action, -self.amax, self.amax))

        if regime_id == 0:
            return self.action_gain_regime0 * a

        if abs(a) < self.action_deadzone_regime1:
            a = 0.0
        if self.action_invert_regime1:
            a = -a
        return self.action_gain_regime1 * a

    def step(self, regime_id: int, action: float) -> np.ndarray:
        g, f = (9.8, 0.1) if regime_id == 0 else (25.0, 1.5)
        theta, omega = float(self.state[0]), float(self.state[1])

        a_eff = self._effective_action(regime_id, action)
        acc = -g * math.sin(theta) - f * omega + a_eff

        omega = omega + acc * self.dt
        theta = theta + omega * self.dt

        noise = np.random.normal(0, self.noise_std, size=(2,))
        theta += noise[0]
        omega += noise[1]

        self.state = np.array([theta, omega], dtype=np.float64)
        return self.state.copy()

    def rollout(
        self, regime_id: int, policy_fn, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        states = [self.state.copy()]
        actions = []
        r = int(regime_id)

        for _ in range(horizon):
            if self.switch_mid_episode and (
                np.random.rand() < self.switch_prob_per_step
            ):
                r = 1 - r

            a = float(policy_fn(states[-1]))
            actions.append([a])
            s_next = self.step(r, a)
            states.append(s_next)

        return np.asarray(states), np.asarray(actions)


# -----------------------------
# Action-Binding TAM with knot-based tube
# -----------------------------
class ActionBindingTAM:
    def __init__(
        self,
        state_dim: int = 2,
        identity_dim: int = 8,
        hidden_dim: int = 64,
        lr: float = 7e-4,
        amax: float = 2.0,
        maxH: int = 64,
        minT: int = 2,
        K: int = 4,  # polynomial basis degree count (DEPRECATED: now using knots)
        M: int = 16,  # number of knots for interpolation
        logstd_clip: Tuple[float, float] = (-4.0, 2.0),
        temperature: float = 0.8,
        k_sigma: float = 2.0,
        bind_success_frac: float = 0.85,
        cone_reg: float = 0.001,
        goal_weight: float = 0.25,
        action_l2: float = 0.01,
        lambda_h: float = 0.002,  # ponder cost (encourage shorter horizons)
        halt_bias: float = -1.0,  # init bias for p_stop
        max_ports: int = 32,
        proliferation_z: float = 6.0,
        proliferation_cooldown: int = 300,
        min_steps_before_prolif: int = 500,
        # concession parameters
        concession_threshold: float = 2.0,  # threshold for proliferation
        concession_weight_select: float = 0.3,  # weight in selection
        concession_w1: float = 1.0,  # cone inflation weight
        concession_w2: float = 0.5,  # horizon shrinkage weight
        concession_w3: float = 0.3,  # slack usage weight
        margin_target: float = 0.5,  # target safety margin
    ):
        self.device = torch.device("cpu")
        self.state_dim = state_dim
        self.identity_dim = identity_dim
        self.hidden_dim = hidden_dim

        self.amax = amax
        self.maxH = int(maxH)
        self.minT = int(minT)
        self.K = int(K)  # kept for backward compatibility but unused
        self.M = int(M)  # number of knots
        self.logstd_min, self.logstd_max = logstd_clip

        self.temperature = temperature
        self.k_sigma = k_sigma
        self.bind_success_frac = bind_success_frac

        self.cone_reg = cone_reg
        self.goal_weight = goal_weight
        self.action_l2 = action_l2
        self.lambda_h = lambda_h

        self.max_ports = max_ports
        self.proliferation_z = proliferation_z
        self.proliferation_cooldown = proliferation_cooldown
        self.min_steps_before_prolif = min_steps_before_prolif
        self.last_prolif_step = -(10**9)

        # concession parameters
        self.concession_threshold = concession_threshold
        self.concession_weight_select = concession_weight_select
        self.concession_w1 = concession_w1
        self.concession_w2 = concession_w2
        self.concession_w3 = concession_w3
        self.margin_target = margin_target

        # Shared backbones
        self.pol = SharedPolicy(state_dim, identity_dim, hidden_dim).to(self.device)

        out_dim = (
            (state_dim * self.M) + (state_dim * self.M) + 1
        )  # mu knots + logsig knots + stop_logit
        self.tube = SharedTube(state_dim, identity_dim, hidden_dim, out_dim=out_dim).to(
            self.device
        )

        # init stop bias
        with torch.no_grad():
            self.tube.out_head.bias[-1].fill_(halt_bias)

        self.ports = nn.ModuleList(
            [Port(0, identity_dim, state_dim, hidden_dim).to(self.device)]
        )
        self.ports[0].birth_step = 0

        self.optimizer = optim.Adam(
            list(self.pol.parameters())
            + list(self.tube.parameters())
            + list(self.ports.parameters()),
            lr=lr,
        )

        self.history = {
            "step": [],
            "regime": [],
            "selected_port": [],
            "n_ports": [],
            "bind_success": [],
            "avg_abs_action": [],
            "T": [],
            "E_T": [],
            "exp_nll": [],
            "cone_volume": [],
            "z": [],
            "concession": [],
            "avg_margin": [],
        }

    # -------- policy ----------
    def _policy_action(self, p: Port, state_t: torch.Tensor) -> torch.Tensor:
        emb = p.embedding.unsqueeze(0).expand(state_t.size(0), -1)
        x = torch.cat([state_t, emb], dim=-1)

        h1 = self.pol.relu(self.pol.fc1(x) + p.pol_lora1(x))
        h2 = self.pol.relu(self.pol.fc2(h1) + p.pol_lora2(h1))
        a = torch.tanh(self.pol.act_head(h2)) * self.amax
        return a  # [B,1]

    # -------- tube predictor ----------
    def _tube_params(
        self, p: Port, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          mu_knots: [M, D]
          logsig_knots: [M, D]
          p_stop: scalar in (0,1)
        """
        emb = p.embedding.unsqueeze(0).expand(s0_t.size(0), -1)
        x = torch.cat([s0_t, emb], dim=-1)

        h1 = self.tube.relu(self.tube.fc1(x) + p.tube_lora1(x))
        h2 = self.tube.relu(self.tube.fc2(h1) + p.tube_lora2(h1))
        out = self.tube.out_head(h2).squeeze(0)  # [out_dim]

        D, M = self.state_dim, self.M
        mu_flat = out[: D * M]
        sig_flat = out[D * M : 2 * D * M]
        stop_logit = out[-1]

        mu_knots = mu_flat.view(M, D)  # [M, D]
        logsig_knots = sig_flat.view(M, D)  # [M, D]

        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        return mu_knots, logsig_knots, p_stop

    def _tube_traj(
        self, mu_knots: torch.Tensor, logsig_knots: torch.Tensor, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate tube at discrete t=1..T via linear interpolation of knots.
        mu_knots/logsig_knots: [M, D]
        Returns:
          mu: [T, D]
          log_var: [T, D]
        """
        t = torch.linspace(
            1.0 / T, 1.0, steps=T, device=mu_knots.device, dtype=mu_knots.dtype
        )  # [T]
        mu = interp1d_linear(mu_knots, t)  # [T, D]
        log_sigma = interp1d_linear(logsig_knots, t)  # [T, D]
        log_sigma = torch.clamp(log_sigma, self.logstd_min, self.logstd_max)
        log_var = 2.0 * log_sigma
        return mu, log_var

    @torch.no_grad()
    def select_port(self, step: int, s0: np.ndarray) -> int:
        """
        Cheap selection: one tube forward per port, then compute geometric cone score.
        """
        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)

        # weights for "mode of interaction" selection
        w_commit = 0.20  # prefer tighter tubes
        w_rely = 0.60  # prefer ports that keep promises
        w_z = 0.10  # discourage currently surprising ports
        w_goal = 0.25  # keep near 0 in predicted terminal mean
        w_ponder = 0.40  # prefer shorter expected horizon (in selection)
        w_concession = self.concession_weight_select  # discourage high-concession ports

        scores = []
        for p in self.ports:
            mu_knots, logsig_knots, p_stop = self._tube_params(p, s0_t)

            # use a small evaluation horizon for scoring geometry (interpolation is cheap)
            Teval = min(self.maxH, 32)
            mu, log_var = self._tube_traj(mu_knots, logsig_knots, Teval)
            std = torch.exp(0.5 * log_var)
            cone_vol = float((std[:, 0] * std[:, 1]).mean().item())

            # terminal goal
            goal_cost = float((mu[-1] ** 2).mean().item())

            # expected horizon under truncated geometric (approx via formula, Teval)
            # E[T] approx over 1..Teval with tail at Teval; good enough for selection
            w, E_T = truncated_geometric_weights(p_stop, Teval)
            E_T_val = float(E_T.item())

            commitment_bonus = -math.log(cone_vol + 1e-8)
            reliability_bonus = math.log(float(p.bind_success_ema) + 1e-8)
            z_pen = max(p.z_score(), 0.0)

            score = (
                +w_commit * commitment_bonus
                + w_rely * reliability_bonus
                - w_z * z_pen
                - w_goal * goal_cost
                - w_ponder * (E_T_val / max(Teval, 1))
                - w_concession * p.concession_ema
            )
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        probs = torch.softmax(scores_t / max(self.temperature, 1e-4), dim=0)
        idx = int(torch.multinomial(probs, 1).item())

        for i, p in enumerate(self.ports):
            p.update_usage(i == idx)
        self.ports[idx].select_count += 1
        return idx

    @torch.no_grad()
    def sample_horizon(self, p: Port, s0: np.ndarray) -> Tuple[int, float]:
        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        _cmu, _csig, p_stop = self._tube_params(p, s0_t)
        p_stop_val = float(p_stop.item())

        # approximate E[T] (truncated at maxH) numerically once with cheap weights
        w, E_T = truncated_geometric_weights(p_stop, self.maxH)
        E_T_val = float(E_T.item())

        T = sample_truncated_geometric(p_stop_val, self.maxH, minT=self.minT)
        return T, E_T_val

    def binding_predicate(
        self, mu: torch.Tensor, log_var: torch.Tensor, real: torch.Tensor
    ) -> bool:
        """
        mu/log_var: [T,D], real: [T,D]
        success if >= frac steps inside k-sigma in all dims.
        """
        std = torch.exp(0.5 * log_var)
        inside = (torch.abs(real - mu) <= (self.k_sigma * std + 1e-8)).all(dim=-1)
        frac = float(inside.float().mean().item())
        return frac >= self.bind_success_frac

    def train_on_episode(
        self,
        step: int,
        regime: int,
        s0: np.ndarray,
        states: np.ndarray,  # [T+1,D]
        actions: np.ndarray,  # [T,1]
        port_idx: int,
        E_T_imagine: float,
    ):
        p = self.ports[port_idx]
        T = int(actions.shape[0])

        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu_knots, logsig_knots, p_stop = self._tube_params(p, s0_t)

        # evaluate tube for this observed T
        mu, log_var = self._tube_traj(mu_knots, logsig_knots, T)

        # realized trajectory x1..xT
        y = torch.tensor(states[1:], dtype=torch.float32, device=self.device)  # [T,D]

        # per-step NLL
        nll_t = gaussian_nll(mu, log_var, y).mean(dim=-1)  # [T]

        # option-3 expected loss under (learned) geometric halting, forced at T
        w, E_T_train = truncated_geometric_weights(p_stop, T)
        exp_nll = (w * nll_t).sum()

        # cone regularization: keep log-stds "reasonable" (prevents escape by vagueness)
        cone_reg = self.cone_reg * torch.mean((0.5 * log_var) ** 2)

        # Optional: smoothness regularization on knots (uncomment to enable)
        # Encourages smooth tube trajectories by penalizing second differences
        # smooth_mu = ((mu_knots[2:] - 2*mu_knots[1:-1] + mu_knots[:-2])**2).mean()
        # smooth_sig = ((logsig_knots[2:] - 2*logsig_knots[1:-1] + logsig_knots[:-2])**2).mean()
        # cone_reg = cone_reg + 1e-4 * (smooth_mu + smooth_sig)

        # goal shaping: stabilize predicted terminal mean near 0
        goal_cost = torch.mean(mu[-1] ** 2)

        # action penalty (real actions taken)
        a = torch.tensor(actions, dtype=torch.float32, device=self.device)
        act_cost = torch.mean(a**2)

        # ponder cost: shorter horizons preferred
        ponder = self.lambda_h * E_T_train

        loss = (
            exp_nll
            + cone_reg
            + self.goal_weight * goal_cost
            + self.action_l2 * act_cost
            + ponder
        )

        # binding predicate based on the tube itself
        with torch.no_grad():
            is_bound = self.binding_predicate(mu, log_var, y)
            p.update_bind_success(is_bound)

            surp = float(exp_nll.detach().item())
            p.update_stats(surp, is_bound)
            z = p.z_score()

            # compute concession metrics
            std = torch.exp(0.5 * log_var)
            cone_vol = float((std[:, 0] * std[:, 1]).mean().item())
            log_cone_vol = math.log(cone_vol + 1e-8)

            # compute average margin (how much slack we have in the tube)
            margin_per_step = []
            for t in range(T):
                margins_per_dim = []
                for d in range(self.state_dim):
                    numerator = (
                        self.k_sigma * float(std[t, d].item())
                        - abs(float(y[t, d].item()) - float(mu[t, d].item()))
                    )
                    denominator = self.k_sigma * float(std[t, d].item()) + 1e-8
                    margin_d = numerator / denominator
                    margins_per_dim.append(margin_d)
                margin_per_step.append(min(margins_per_dim))  # tightest dimension
            avg_margin = float(np.mean(margin_per_step))

            # compute concession index
            concession = p.compute_concession(
                log_cone_vol=log_cone_vol,
                T_val=float(T),
                avg_margin=avg_margin,
                w1=self.concession_w1,
                w2=self.concession_w2,
                w3=self.concession_w3,
                margin_target=self.margin_target,
            )
            p.update_concession_ema(concession)

            # update baselines (only on good binds: success + low NLL)
            is_good_bind = is_bound and (surp < 2.0)
            p.update_concession_baseline(log_cone_vol, float(T), is_good_bind)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.pol.parameters())
            + list(self.tube.parameters())
            + list(self.ports.parameters()),
            1.0,
        )
        self.optimizer.step()

        # proliferation if sustained surprise OR sustained concessions
        proliferate_z = z > self.proliferation_z
        proliferate_concession = p.concession_ema > self.concession_threshold

        if (
            step > self.min_steps_before_prolif
            and (step - self.last_prolif_step) > self.proliferation_cooldown
            and len(self.ports) < self.max_ports
            and (proliferate_z or proliferate_concession)
        ):
            reason = "z-score" if proliferate_z else "concession"
            self._proliferate(parent_idx=port_idx, step=step, reason=reason)

        # logging
        self.history["step"].append(step)
        self.history["regime"].append(regime)
        self.history["selected_port"].append(port_idx)
        self.history["n_ports"].append(len(self.ports))
        self.history["bind_success"].append(1.0 if is_bound else 0.0)
        self.history["avg_abs_action"].append(float(np.mean(np.abs(actions))))
        self.history["T"].append(int(T))
        self.history["E_T"].append(float(E_T_imagine))
        self.history["exp_nll"].append(float(exp_nll.detach().item()))
        self.history["cone_volume"].append(cone_vol)
        self.history["z"].append(float(z))
        self.history["concession"].append(float(concession))
        self.history["avg_margin"].append(float(avg_margin))

    def _proliferate(self, parent_idx: int, step: int, reason: str = "z-score"):
        pid = len(self.ports)
        new_p = Port(pid, self.identity_dim, self.state_dim, self.hidden_dim).to(
            self.device
        )
        new_p.birth_step = step
        with torch.no_grad():
            new_p.copy_from(self.ports[parent_idx], noise=0.05)

        self.ports.append(new_p)
        # rebuild optimizer so new params are tracked
        self.optimizer = optim.Adam(
            list(self.pol.parameters())
            + list(self.tube.parameters())
            + list(self.ports.parameters()),
            lr=self.optimizer.param_groups[0]["lr"],
        )
        self.last_prolif_step = step
        print(
            f"Step {step}: {reason} trigger. Proliferating Port {pid} from {parent_idx}."
        )


# -----------------------------
# Experiment
# -----------------------------
def run_experiment(seed: int = 0, steps: int = 8000):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ControlledPendulumEnv(noise_std=0.005, dt=0.05, amax=2.0)
    agent = ActionBindingTAM(
        maxH=64,
        minT=2,
        M=16,  # number of knots (increase to 24 or 32 for sharper switching)
        temperature=0.8,
        k_sigma=2.0,
        bind_success_frac=0.85,
        lambda_h=0.002,  # try 0.001..0.01
        halt_bias=-1.0,  # try -2.0 (longer), 0.0 (shorter)
        proliferation_z=6.0,
    )

    env.reset()
    print("Starting Action-Binding TAM (Geometric Tube)...")

    for i in range(steps):
        regime = 0 if (i // 1000) % 2 == 0 else 1
        s0 = env.state.copy()

        idx = agent.select_port(i, s0)
        p = agent.ports[idx]

        T, E_T = agent.sample_horizon(p, s0)

        # bind: execute policy for T steps
        def policy_fn(state_np):
            st = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = agent._policy_action(p, st).cpu().numpy().squeeze()
            a = float(a + np.random.normal(0, 0.05))  # exploration
            return float(np.clip(a, -env.amax, env.amax))

        states, actions = env.rollout(regime_id=regime, policy_fn=policy_fn, horizon=T)

        agent.train_on_episode(
            step=i,
            regime=regime,
            s0=s0,
            states=states,
            actions=actions,
            port_idx=idx,
            E_T_imagine=E_T,
        )

    h = agent.history
    steps_arr = np.array(h["step"], dtype=np.int32)

    bind_success = np.array(h["bind_success"], dtype=np.float64)
    cone_volume = np.array(h["cone_volume"], dtype=np.float64)
    exp_nll = np.array(h["exp_nll"], dtype=np.float64)
    n_ports = np.array(h["n_ports"], dtype=np.int32)
    sel = np.array(h["selected_port"], dtype=np.int32)
    avg_abs_action = np.array(h["avg_abs_action"], dtype=np.float64)
    T_samp = np.array(h["T"], dtype=np.float64)
    E_T = np.array(h["E_T"], dtype=np.float64)
    concession = np.array(h["concession"], dtype=np.float64)
    avg_margin = np.array(h["avg_margin"], dtype=np.float64)

    agency = np.exp(-exp_nll)

    fig, axes = plt.subplots(9, 1, figsize=(12, 26), sharex=True)

    axes[0].plot(steps_arr, agency, alpha=0.9)
    axes[0].set_ylabel("Agency exp(-E[NLL])")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Action-Binding TAM: Knot-Based Tube (no long rollouts)")

    axes[1].plot(steps_arr, cone_volume, alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Cone Vol (σ0*σ1)")

    axes[2].plot(
        steps_arr,
        np.convolve(bind_success, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
    )
    axes[2].set_ylabel("Bind Success")
    axes[2].set_ylim(-0.05, 1.05)

    axes[3].scatter(steps_arr, sel, c=sel, s=4, alpha=0.6, cmap="tab10")
    axes[3].set_ylabel("Selected Port")

    axes[4].plot(steps_arr, n_ports, drawstyle="steps-post")
    axes[4].set_ylabel("# Ports")

    axes[5].plot(steps_arr, avg_abs_action, alpha=0.85)
    axes[5].set_ylabel("Avg |action|")

    axes[6].plot(
        steps_arr,
        np.convolve(T_samp, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        label="T sampled (smoothed)",
    )
    axes[6].plot(
        steps_arr,
        np.convolve(E_T, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        label="E[T] (smoothed)",
    )
    axes[6].set_ylabel("Horizon")
    axes[6].legend()

    axes[7].plot(
        steps_arr,
        np.convolve(concession, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        color="red",
    )
    axes[7].set_ylabel("Concession Index")
    axes[7].axhline(
        y=agent.concession_threshold, color="k", linestyle="--", alpha=0.5, label="Threshold"
    )
    axes[7].legend()

    axes[8].plot(
        steps_arr,
        np.convolve(avg_margin, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        color="green",
    )
    axes[8].set_ylabel("Avg Tube Margin")
    axes[8].axhline(
        y=agent.margin_target, color="k", linestyle="--", alpha=0.5, label="Target"
    )
    axes[8].set_xlabel("Training Step")
    axes[8].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment(seed=0, steps=8000)
