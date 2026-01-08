"""
Action-Binding TAM with Continuous Ports and Dual Controller.

Architecture:
- Continuous ports: z ~ q(z|s0) via ActorNet (VAE-style encoder)
- Policy: a = pi(x, z) - actions conditioned on commitment
- Tube: predicts mu_t, sigma_t for t=1..T via knot values + linear interpolation
- Learned horizon: tube also predicts p_stop (geometric halting distribution)
- Binding predicate: realized trajectory stays within k-sigma tube for >= frac steps

Training (dual tit-for-tat controllers):
Two-constraint optimization via dual controllers:
1) Reliability constraint (bind success rate)
2) Compute constraint (expected horizon)

Key prevention of exploits:
- Cone volume weighted by geometric halting distribution (prevents "early tight, late wide")
- Soft bind rate weighted by geometric halting distribution (consistent with exp_nll)
- Result: oscillatory homeostasis at the Pareto boundary of tightest cone + shortest horizon
"""

from typing import Tuple

import numpy as np
import torch
import torch.optim as optim

# Handle both package import and direct execution
try:
    from .networks import ActorNet, SharedPolicy, SharedTube
    from .utils import (
        gaussian_nll,
        interp1d_linear,
        kl_diag_gaussian_to_standard,
        sample_truncated_geometric,
        truncated_geometric_weights,
    )
except ImportError:
    from networks import ActorNet, SharedPolicy, SharedTube
    from utils import (
        gaussian_nll,
        interp1d_linear,
        kl_diag_gaussian_to_standard,
        sample_truncated_geometric,
        truncated_geometric_weights,
    )


class Actor:
    """
    TAM Actor with continuous ports in latent space and dual controller
    for tightest feasible cones with bounded compute.
    """

    def __init__(
        self,
        state_dim: int = 2,
        z_dim: int = 8,  # latent commitment dimension
        hidden_dim: int = 64,
        lr: float = 7e-4,
        amax: float = 2.0,
        maxH: int = 64,
        minT: int = 2,
        M: int = 16,  # number of knots for interpolation
        logstd_clip: Tuple[float, float] = (-4.0, 2.0),
        k_sigma: float = 2.0,
        bind_success_frac: float = 0.85,
        cone_reg: float = 0.001,
        goal_weight: float = 0.25,
        action_l2: float = 0.01,
        lambda_h: float = 0.002,  # ponder cost (encourage shorter horizons)
        beta_kl: float = 3e-4,  # KL regularization weight (start small)
        halt_bias: float = -1.0,  # init bias for p_stop
    ):
        self.device = torch.device("cpu")
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.amax = amax
        self.maxH = int(maxH)
        self.minT = int(minT)
        self.target_ET = 16.0  # pick something reasonable vs maxH
        self.lambda_T = 0.0
        self.lambda_T_lr = 0.02  # try 0.005..0.05
        self.lambda_T_clip = (0.0, 50.0)

        self.M = int(M)  # number of knots
        self.logstd_min, self.logstd_max = logstd_clip

        self.k_sigma = k_sigma
        self.bind_success_frac = bind_success_frac

        self.cone_reg = cone_reg
        self.goal_weight = goal_weight
        self.action_l2 = action_l2
        self.lambda_h = lambda_h
        self.beta_kl = beta_kl

        # Tit-for-tat dual controller for tightest feasible cone
        self.target_bind = 0.90  # r* target bind success rate
        self.lambda_bind = 0.0  # λ (dual variable / "price of reliability")
        self.lambda_lr = 0.05  # dual update step size (try 0.01..0.2)
        self.lambda_clip = (0.0, 50.0)  # keep stable
        self.margin_temp = 0.25  # softness for differentiable "inside" indicator
        self.w_cone = 0.05  # weight on cone volume minimization

        # Networks
        self.actor = ActorNet(state_dim, z_dim, hidden_dim).to(self.device)
        self.pol = SharedPolicy(state_dim, z_dim, hidden_dim).to(self.device)

        out_dim = (
            (state_dim * self.M) + (state_dim * self.M) + 1
        )  # mu knots + logsig knots + stop_logit
        self.tube = SharedTube(state_dim, z_dim, hidden_dim, out_dim=out_dim).to(
            self.device
        )

        # init stop bias
        with torch.no_grad():
            self.tube.out_head.bias[-1].fill_(halt_bias)

        self.optimizer = optim.Adam(
            list(self.actor.parameters())
            + list(self.pol.parameters())
            + list(self.tube.parameters()),
            lr=lr,
        )

        self.history = {
            "step": [],
            "regime": [],
            "bind_success": [],
            "avg_abs_action": [],
            "T": [],
            "E_T": [],
            "E_T_train": [],  # actual E[T] from training (not imagined)
            "exp_nll": [],
            "cone_volume": [],
            "kl": [],
            "z_norm": [],
            "soft_bind": [],
            "lambda_bind": [],
            "lambda_T": [],  # compute dual variable
            "cone_vol": [],
        }

    def sample_z(
        self, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterized z ~ q(z|s0). Returns z, z_mu, z_logstd.
        s0_t: [1, state_dim]
        """
        z_mu, z_logstd = self.actor(s0_t)  # [1,z_dim], [1,z_dim]
        eps = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logstd) * eps
        return z, z_mu, z_logstd

    def _policy_action(self, z: torch.Tensor, state_t: torch.Tensor) -> torch.Tensor:
        """
        z: [B, z_dim] or [1,z_dim]
        state_t: [B, state_dim]
        """
        if z.size(0) != state_t.size(0):
            z = z.expand(state_t.size(0), -1)
        x = torch.cat([state_t, z], dim=-1)

        h1 = self.pol.relu(self.pol.fc1(x))
        h2 = self.pol.relu(self.pol.fc2(h1))
        a = torch.tanh(self.pol.act_head(h2)) * self.amax
        return a  # [B,1]

    def _tube_params(
        self, z: torch.Tensor, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: [1,z_dim] or [B,z_dim]
        s0_t: [1,state_dim]
        Returns:
          mu_knots: [M, D]
          logsig_knots: [M, D]
          p_stop: scalar in (0,1)
        """
        if z.size(0) != s0_t.size(0):
            z = z.expand(s0_t.size(0), -1)
        x = torch.cat([s0_t, z], dim=-1)

        h1 = self.tube.relu(self.tube.fc1(x))
        h2 = self.tube.relu(self.tube.fc2(h1))
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
    def sample_horizon(
        self, z: torch.Tensor, s0: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Returns (T, E_T, p_stop_val) given z and s0.
        """
        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        _muK, _sigK, p_stop = self._tube_params(z, s0_t)
        p_stop_val = float(p_stop.item())

        w, E_T = truncated_geometric_weights(p_stop, self.maxH)
        E_T_val = float(E_T.item())

        T = sample_truncated_geometric(p_stop_val, self.maxH, minT=self.minT)
        return T, E_T_val, p_stop_val

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
        z: torch.Tensor,
        z_mu: torch.Tensor,
        z_logstd: torch.Tensor,
        E_T_imagine: float,
    ):
        """Train on a single episode with the dual controller."""
        T = int(actions.shape[0])

        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu_knots, logsig_knots, p_stop = self._tube_params(z, s0_t)

        # evaluate tube for this observed T
        mu, log_var = self._tube_traj(mu_knots, logsig_knots, T)

        # realized trajectory x1..xT
        y = torch.tensor(states[1:], dtype=torch.float32, device=self.device)  # [T,D]

        # per-step NLL
        nll_t = gaussian_nll(mu, log_var, y).mean(dim=-1)  # [T]

        # option-3 expected loss under (learned) geometric halting, forced at T
        w, E_T_train = truncated_geometric_weights(p_stop, T)
        exp_nll = (w * nll_t).sum()

        # Compute soft (differentiable) bind rate for dual controller
        std = torch.exp(0.5 * log_var)  # [T,D]
        err = torch.abs(y - mu)  # [T,D]
        margin = (self.k_sigma * std) - err  # [T,D]

        # soft "inside": sigmoid(margin/temp) in each dim
        soft_inside_dim = torch.sigmoid(margin / self.margin_temp)  # [T,D]

        # soft per-step inside: require all dims (use product)
        soft_inside_step = torch.prod(soft_inside_dim, dim=-1)  # [T]
        # Weight by geometric halting distribution (consistent with exp_nll and cone)
        soft_bind_rate = (w.detach() * soft_inside_step).sum()  # scalar in [0,1]

        # cone regularization: keep log-stds "reasonable" (prevents escape by vagueness)
        cone_reg = self.cone_reg * torch.mean((0.5 * log_var) ** 2)

        # goal shaping: stabilize predicted terminal mean near 0
        goal_cost = torch.mean(mu[-1] ** 2)

        # action penalty (real actions taken)
        a = torch.tensor(actions, dtype=torch.float32, device=self.device)
        act_cost = torch.mean(a**2)

        # ponder cost: shorter horizons preferred
        ponder = (self.lambda_h + self.lambda_T) * E_T_train

        # KL regularization: encourages meaningful but not collapsed latent space
        kl = kl_diag_gaussian_to_standard(z_mu, z_logstd).mean()

        # Dual controller terms for tightest feasible cone
        # Cone volume term (push tighter), weighted by geometric halting distribution
        # This prevents "early tight, late wide" exploits by aligning cone cost
        # with the port's own notion of where the episode ends
        cv_t = std[:, 0] * std[:, 1]  # [T]
        cone_vol_w = (
            w.detach() * cv_t
        ).sum()  # scalar (detach w to avoid weird coupling)
        loss_cone = self.w_cone * cone_vol_w

        # constraint penalty: λ * (target - achieved)
        constraint_violation = self.target_bind - soft_bind_rate
        loss_constraint = self.lambda_bind * constraint_violation

        loss = (
            exp_nll
            + cone_reg
            + self.goal_weight * goal_cost
            + self.action_l2 * act_cost
            + ponder
            + self.beta_kl * kl
            + loss_cone
            + loss_constraint
        )

        # binding predicate based on the tube itself
        with torch.no_grad():
            is_bound = self.binding_predicate(mu, log_var, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters())
            + list(self.pol.parameters())
            + list(self.tube.parameters()),
            1.0,
        )
        self.optimizer.step()

        # Tit-for-tat dual updates: adjust λ based on bind rate and horizon performance
        with torch.no_grad():
            # Reliability controller: increase λ_bind if we underperform, decrease otherwise
            self.lambda_bind += self.lambda_lr * float(
                (self.target_bind - soft_bind_rate).item()
            )
            lo, hi = self.lambda_clip
            self.lambda_bind = float(np.clip(self.lambda_bind, lo, hi))

            # Compute controller: increase λ_T if horizon exceeds target
            self.lambda_T += self.lambda_T_lr * float(
                (E_T_train - self.target_ET).item()
            )
            lo_T, hi_T = self.lambda_T_clip
            self.lambda_T = float(np.clip(self.lambda_T, lo_T, hi_T))

        # logging
        with torch.no_grad():
            z_norm_val = float(torch.norm(z).item())

        self.history["step"].append(step)
        self.history["regime"].append(regime)
        self.history["bind_success"].append(1.0 if is_bound else 0.0)
        self.history["avg_abs_action"].append(float(np.mean(np.abs(actions))))
        self.history["T"].append(int(T))
        self.history["E_T"].append(float(E_T_imagine))
        self.history["E_T_train"].append(float(E_T_train.detach().item()))
        self.history["exp_nll"].append(float(exp_nll.detach().item()))
        self.history["cone_volume"].append(float(cone_vol_w.detach().item()))
        self.history["kl"].append(float(kl.detach().item()))
        self.history["z_norm"].append(z_norm_val)
        self.history["soft_bind"].append(float(soft_bind_rate.detach().item()))
        self.history["lambda_bind"].append(float(self.lambda_bind))
        self.history["lambda_T"].append(float(self.lambda_T))
        self.history["cone_vol"].append(float(cone_vol_w.detach().item()))
