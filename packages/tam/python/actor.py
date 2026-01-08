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

from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Handle both package import and direct execution
try:
    from .networks import ActorNet, SharedPolicy, SharedTube, TubeRefiner, PonderHead, DynamicPonderHead
    from .utils import (
        gaussian_nll,
        interp1d_linear,
        kl_diag_gaussian_to_standard,
        sample_truncated_geometric,
        truncated_geometric_weights,
    )
except ImportError:
    from networks import ActorNet, SharedPolicy, SharedTube, TubeRefiner, PonderHead, DynamicPonderHead
    from utils import (
        gaussian_nll,
        interp1d_linear,
        kl_diag_gaussian_to_standard,
        sample_truncated_geometric,
        truncated_geometric_weights,
    )


class Actor(nn.Module):
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
        w_horizon_bonus: float = 0.01,  # reward for longer valid commitments
        # Reasoning parameters
        max_refine_steps: int = 8,  # max reasoning steps
        refine_step_scale: float = 0.1,  # refinement delta scale
        lambda_r: float = 0.001,  # ponder cost per reasoning step
        target_Hr: float = 4.0,  # target reasoning steps for dual controller
        lambda_r_lr: float = 0.02,  # dual controller learning rate for reasoning
        use_dynamic_pondering: bool = False,  # use dynamic pondering instead of fixed Hr
        # Patch A: Reasoning mode and gating
        reasoning_mode: str = "fixed",  # "off" | "fixed" | "gated" | "dynamic"
        freeze_sigma_refine: bool = True,  # prevent sigma gaming during refinement
        c_improve: float = 1.0,  # exchange rate: NLL improvement "pays for" Hr
        improve_detach: bool = True,  # prevent hacking improve via baseline coupling
        gate_kind: str = "nll0",  # "nll0" | "mem_risk" | "volatility" | "combo"
        gate_thresh: float = 0.5,  # scalar threshold for gating
        gate_kappa: float = 0.1,  # softness if using sigmoid gating
        Hr_default: int = 0,  # default Hr when gate is off
    ):
        super().__init__()
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
        self.w_horizon_bonus = w_horizon_bonus  # reward longer horizons when binding succeeds

        # --- cone algebra (latent operators) ---
        self.algebra_Teval = min(32, self.maxH)
        self.algebra_delta = 0.25
        self.algebra_margin_C = 0.0
        self.algebra_margin_H = 0.0
        self.w_algebra = 0.02  # start small

        self.w_algebra_ortho = 0.01  # optional disentanglement
        self.w_algebra_comm = 0.00  # optional (mostly 0 for now)

        # --- reasoning / iterative refinement ---
        self.max_refine_steps = int(max_refine_steps)
        self.refine_step_scale = refine_step_scale
        self.target_Hr = target_Hr
        self.lambda_r = lambda_r  # dual variable for reasoning compute
        self.lambda_r_lr = lambda_r_lr
        self.lambda_r_clip = (0.0, 50.0)
        self.use_dynamic_pondering = use_dynamic_pondering
        # Patch A: Reasoning mode and gating
        self.reasoning_mode = reasoning_mode
        self.freeze_sigma_refine = freeze_sigma_refine
        self.c_improve = c_improve
        self.improve_detach = improve_detach
        self.gate_kind = gate_kind
        self.gate_thresh = gate_thresh
        self.gate_kappa = gate_kappa
        self.Hr_default = Hr_default
        self.Hr_max = self.max_refine_steps

        # --- commitment memory (replay in z-space) ---
        self.mem_size = 50_000
        self.mem_min = 500          # don't apply regularizer until enough data
        self.mem_k = 64             # neighbors
        self.mem_sigma = 1.0        # kernel width in z-space
        self.w_mem = 0.02           # strength of memory regularizer (start small)
        self.mem_detach_targets = True

        self.mem = deque(maxlen=self.mem_size)  # store dicts per episode

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

        # Learnable operator directions (algebra generators)
        self.u_wide = nn.Parameter(torch.randn(self.z_dim) * 0.05)
        self.u_ext = nn.Parameter(torch.randn(self.z_dim) * 0.05)

        # Reasoning networks
        self.refiner = TubeRefiner(state_dim, z_dim, self.M, hidden_dim).to(self.device)
        self.ponder_head = PonderHead(state_dim, z_dim, hidden_dim).to(self.device)
        self.dynamic_ponder_head = DynamicPonderHead(state_dim, z_dim, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
            "algebra_loss": [],
            "dC_wide": [],
            "dH_ext": [],
            "horizon_bonus": [],
            "Hr": [],  # sampled reasoning steps
            "E_Hr": [],  # expected reasoning steps (imagined)
            "E_Hr_train": [],  # expected reasoning steps (trained)
            "p_refine_stop": [],
            "delta_nll": [],  # improvement from reasoning
            "lambda_r": [],  # reasoning compute dual
            "mem_loss": [],  # memory regularizer loss
            "mem_risk": [],  # estimated risk from memory
            "mem_n": [],  # number of items in memory
            # Volume tracking for dynamic pondering
            "vol_init": [],  # initial cone volume (before refinement)
            "vol_final": [],  # final cone volume (after refinement)
            "vol_reduction": [],  # total volume reduction during refinement
            "vol_improvement_rate": [],  # average improvement per step
            # Patch A: NLL0/NLLr/improve tracking
            "NLL0": [],  # baseline expected NLL
            "NLLr": [],  # refined expected NLL
            "improve": [],  # improvement from reasoning
            "ponder_reason_raw": [],  # raw reasoning ponder cost
            "ponder_reason_eff": [],  # effective reasoning ponder cost (after improve discount)
            "gate_score": [],  # gating score
            "gate_on": [],  # whether gate is on (0 or 1)
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

    def _tube_init(
        self, z: torch.Tensor, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get initial tube state (before refinement).

        Args:
            z: [1,z_dim] or [B,z_dim]
            s0_t: [1,state_dim]

        Returns:
            mu_knots: [M, D]
            logsig_knots: [M, D]
            stop_logit: scalar (pre-sigmoid)
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

        return mu_knots, logsig_knots, stop_logit

    def _tube_params(
        self, z: torch.Tensor, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backwards compatibility wrapper. Returns tube with p_stop.
        NOTE: This doesn't use refinement! Use infer_tube for reasoning.

        Args:
            z: [1,z_dim] or [B,z_dim]
            s0_t: [1,state_dim]

        Returns:
            mu_knots: [M, D]
            logsig_knots: [M, D]
            p_stop: scalar in (0,1)
        """
        mu_knots, logsig_knots, stop_logit = self._tube_init(z, s0_t)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        return mu_knots, logsig_knots, p_stop

    def infer_tube(
        self, s0_t: torch.Tensor, z: torch.Tensor, Hr: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Iteratively refine tube prediction with Hr reasoning steps.

        Args:
            s0_t: [1, state_dim]
            z: [1, z_dim]
            Hr: number of refinement steps

        Returns:
            mu_knots: [M, D] - refined
            logsig_knots: [M, D] - refined
            stop_logit: scalar - refined
        """
        # Get initial tube state
        mu_knots, logsig_knots, stop_logit = self._tube_init(z, s0_t)

        # Iterative refinement loop
        logsig_knots_init = logsig_knots.clone()  # save initial for freeze_sigma_refine
        for _ in range(Hr):
            # Compute deltas
            delta_mu, delta_logsig, delta_stop = self.refiner(
                s0_t, z, mu_knots, logsig_knots, stop_logit
            )

            # Apply deltas with step scale
            mu_knots = mu_knots + self.refine_step_scale * delta_mu
            if not self.freeze_sigma_refine:
                logsig_knots = logsig_knots + self.refine_step_scale * delta_logsig
            # else: keep logsig_knots from initial (already set above)
            stop_logit = stop_logit + self.refine_step_scale * delta_stop

            # Clamp logsig to reasonable range
            logsig_knots = torch.clamp(logsig_knots, self.logstd_min, self.logstd_max)

        return mu_knots, logsig_knots, stop_logit

    def infer_tube_dynamic(
        self, s0_t: torch.Tensor, z: torch.Tensor, max_steps: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list]:
        """
        Iteratively refine tube with dynamic halting based on cone volume derivative.
        The agent measures the "rate of clarification" and stops when improvement slows.

        Args:
            s0_t: [1, state_dim]
            z: [1, z_dim]
            max_steps: optional max refinement steps (defaults to self.max_refine_steps)

        Returns:
            mu_knots: [M, D] - refined
            logsig_knots: [M, D] - refined
            stop_logit: scalar - refined
            refined_steps: number of refinement steps taken
            halting_probs: list of p_stop at each step
        """
        max_steps_eff = int(self.max_refine_steps if max_steps is None else max_steps)

        # 1. Initial Guess
        mu_knots, logsig_knots, stop_logit = self._tube_init(z, s0_t)
        logsig_knots_init = logsig_knots.clone()  # save initial for freeze_sigma_refine

        # Volume proxy: sum of exp(logsig) across all knots and dimensions
        # This is a simplified proxy for cone volume
        def get_vol(ls):
            return torch.exp(ls).sum()

        prev_vol = get_vol(logsig_knots)
        current_vol = prev_vol

        refined_steps = 0
        halting_probs = []

        for k in range(max_steps_eff):
            # Calculate derivative (improvement rate)
            delta_vol = prev_vol - current_vol

            # 2. DECIDE: Should we stop?
            # Detach gradients to avoid "sabotaging" the tube to make PonderHead happy
            p_stop = self.dynamic_ponder_head(
                s0_t, z, current_vol.detach(), delta_vol.detach()
            )
            halting_probs.append(p_stop)

            # Soft halting logic
            if self.training:
                # During training, we might run fixed steps and weight losses
                # For now, continue refinement
                pass
            else:
                # During inference, sample whether to stop
                if torch.rand(1, device=s0_t.device) < p_stop:
                    break

            # 3. ACT: Refine
            delta_mu, delta_logsig, delta_stop = self.refiner(
                s0_t, z, mu_knots, logsig_knots, stop_logit
            )

            mu_knots = mu_knots + self.refine_step_scale * delta_mu
            if not self.freeze_sigma_refine:
                logsig_knots = logsig_knots + self.refine_step_scale * delta_logsig
            else:
                logsig_knots = logsig_knots_init.clone()  # keep initial
            stop_logit = stop_logit + self.refine_step_scale * delta_stop

            # Clamp logsig to reasonable range
            logsig_knots = torch.clamp(logsig_knots, self.logstd_min, self.logstd_max)

            # Update stats
            prev_vol = current_vol
            current_vol = get_vol(logsig_knots)
            refined_steps += 1

        return mu_knots, logsig_knots, stop_logit, refined_steps, halting_probs

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

    def _expected_nll(
        self,
        mu_knots: torch.Tensor,
        logsig_knots: torch.Tensor,
        stop_logit: torch.Tensor,
        y: torch.Tensor,
        T: int,
        detach_w: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute expected NLL under a given tube.
        
        Args:
            mu_knots: [M, D] knot means
            logsig_knots: [M, D] knot log-stds
            stop_logit: scalar stop logit
            y: [T, D] target states
            T: horizon
            detach_w: whether to detach weights (default True)
        
        Returns:
            exp_nll: scalar expected NLL
            w: [T] weights
        """
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        w, _ = truncated_geometric_weights(p_stop, T)
        if detach_w:
            w = w.detach()
        
        mu, log_var = self._tube_traj(mu_knots, logsig_knots, T)
        nll_t = gaussian_nll(mu, log_var, y).mean(dim=-1)  # [T]
        exp_nll = (w * nll_t).sum()
        
        return exp_nll, w

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

    @torch.no_grad()
    def sample_reasoning_steps(
        self, z: torch.Tensor, s0: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Sample number of reasoning refinement steps via geometric distribution.

        Returns:
            Hr: sampled number of refinement steps
            E_Hr: expected number of steps
            p_refine_stop: stopping probability
        """
        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_refine_stop = self.ponder_head(s0_t, z)
        p_refine_stop_val = float(p_refine_stop.item())

        w, E_Hr = truncated_geometric_weights(p_refine_stop, self.max_refine_steps)
        E_Hr_val = float(E_Hr.item())

        Hr = sample_truncated_geometric(
            p_refine_stop_val, self.max_refine_steps, minT=1
        )
        return Hr, E_Hr_val, p_refine_stop_val

    @torch.no_grad()
    def sample_horizon_refined(
        self,
        s0_t: torch.Tensor,
        z: torch.Tensor,
        Hr: int = None,
        maxH: int = None,
    ) -> Tuple[int, float, float]:
        """
        Sample T from the *refined* p_stop after Hr refinement steps.
        This ensures acting and training use the same (refined) halting distribution.

        Args:
            s0_t: [1, state_dim]
            z: [1, z_dim]
            Hr: number of refinement steps (optional if using dynamic pondering)
            maxH: optional max horizon (defaults to self.maxH)

        Returns:
            T: sampled horizon
            E_T: expected horizon
            p_stop_val: stopping probability
        """
        maxH_eff = int(self.maxH if maxH is None else maxH)

        if self.use_dynamic_pondering:
            # Use dynamic pondering: adaptively determine when to stop refining
            muK, sigK, stop_logit, actual_Hr, _ = self.infer_tube_dynamic(
                s0_t, z, max_steps=self.max_refine_steps
            )
        else:
            # Use fixed pondering with Hr steps
            if Hr is None:
                raise ValueError("Hr must be provided when not using dynamic pondering")
            muK, sigK, stop_logit = self.infer_tube(s0_t, z, Hr)

        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        p_stop_val = float(p_stop.item())

        w, E_T = truncated_geometric_weights(p_stop, maxH_eff)
        E_T_val = float(E_T.item())

        T = sample_truncated_geometric(p_stop_val, maxH_eff, minT=self.minT)
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

    def _mem_tensors(self, device):
        """
        Extract tensors from memory buffer.

        Returns:
            Z: [N, z_dim] - commitment vectors
            soft: [N] - soft bind rates
            cone: [N] - cone volumes
            lam: [N] - lambda_bind values
        """
        zs = torch.stack([m["z"] for m in self.mem], dim=0).to(device)  # [N,z_dim]
        soft = torch.tensor([m["soft_bind"] for m in self.mem], device=device)
        cone = torch.tensor([m["cone_vol"] for m in self.mem], device=device)
        lam = torch.tensor([m["lambda_bind"] for m in self.mem], device=device)
        return zs, soft, cone, lam

    def _mem_risk_targets(self, soft, cone, lam):
        """
        Compute scalar risk target per memory item.

        Risk = high when:
          - low bind rate (unreliable)
          - high cone volume (vague)
          - high lambda_bind (had to pay for reliability)

        Args:
            soft: [N] soft bind rates
            cone: [N] cone volumes
            lam: [N] lambda_bind values

        Returns:
            risk: [N] scalar risk targets
        """
        eps = 1e-8
        a, b, c = 1.0, 0.2, 0.05  # weights for each term
        return a * (1.0 - soft) + b * torch.log(cone + eps) + c * lam

    def gate_score(
        self,
        NLL0: Optional[torch.Tensor] = None,
        volatility: Optional[float] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gating score to decide when to use reasoning.
        
        Args:
            NLL0: baseline expected NLL (scalar tensor)
            volatility: episode volatility (float)
            z: commitment vector [1, z_dim] for mem_risk
        
        Returns:
            gate_score: scalar tensor
        """
        """
        Compute gating score to decide when to use reasoning.
        
        Args:
            NLL0: baseline expected NLL (scalar tensor)
            volatility: episode volatility (float)
            z: commitment vector [1, z_dim] for mem_risk
        
        Returns:
            gate_score: scalar tensor
        """
        scores = []
        
        if self.gate_kind == "nll0":
            if NLL0 is None:
                raise ValueError("NLL0 required for gate_kind='nll0'")
            score = NLL0.detach() if NLL0.requires_grad else NLL0
            scores.append(score)
        elif self.gate_kind == "mem_risk":
            if z is None:
                raise ValueError("z required for gate_kind='mem_risk'")
            score = self.memory_risk(z).detach()
            scores.append(score)
        elif self.gate_kind == "volatility":
            if volatility is None:
                raise ValueError("volatility required for gate_kind='volatility'")
            score = torch.tensor(volatility, device=self.device, dtype=torch.float32)
            scores.append(score)
        elif self.gate_kind == "combo":
            # Weighted sum of normalized versions
            if NLL0 is not None:
                nll_norm = (NLL0.detach() - 0.0) / (1.0 + 1e-6)  # rough normalization
                scores.append(nll_norm)
            if z is not None:
                mem = self.memory_risk(z).detach()
                mem_norm = (mem - 0.0) / (1.0 + 1e-6)
                scores.append(mem_norm)
            if volatility is not None:
                vol_norm = torch.tensor(volatility, device=self.device, dtype=torch.float32)
                scores.append(vol_norm)
        else:
            raise ValueError(f"Unknown gate_kind: {self.gate_kind}")
        
        if len(scores) == 0:
            return torch.tensor(0.0, device=self.device)
        elif len(scores) == 1:
            return scores[0]
        else:
            # Average of normalized scores
            return torch.stack(scores).mean()

    def memory_risk(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate risk in the neighborhood of z using kNN + Gaussian kernel.

        Args:
            z: [1, z_dim] current commitment

        Returns:
            scalar estimated risk in neighborhood of z
        """
        if len(self.mem) < self.mem_min:
            return torch.tensor(0.0, device=z.device)

        Z, soft, cone, lam = self._mem_tensors(z.device)  # [N,z_dim], [N]
        r = self._mem_risk_targets(soft, cone, lam)       # [N]

        zq = z.squeeze(0)                                  # [z_dim]
        d2 = torch.sum((Z - zq) ** 2, dim=-1)             # [N]

        # pick k nearest
        k = min(self.mem_k, d2.numel())
        vals, idx = torch.topk(d2, k=k, largest=False)

        d2_k = vals
        r_k = r[idx]

        # gaussian kernel weights
        sigma2 = float(self.mem_sigma) ** 2
        w = torch.exp(-d2_k / (2.0 * sigma2))
        w = w / (w.sum() + 1e-8)

        # local expected risk
        return torch.sum(w * r_k)

    def _cone_summaries(
        self, s0_t: torch.Tensor, z: torch.Tensor, Teval: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cone summaries for algebra regularization.
        Uses refined tube with max_refine_steps for stable semantics.

        Returns:
          C: weighted log cone volume (scalar)
          H: expected horizon proxy E[T] (scalar)
        """
        # Use fully refined tube for stable algebra evaluation
        muK, sigK, stop_logit = self.infer_tube(s0_t, z, self.max_refine_steps)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)

        _, log_var = self._tube_traj(muK, sigK, Teval)
        std = torch.exp(0.5 * log_var)  # [Teval, D]
        cv_t = torch.prod(std, dim=-1)  # [Teval] - product across all dimensions

        w, E_T = truncated_geometric_weights(p_stop, Teval)  # w:[Teval]
        C = (w * torch.log(cv_t + 1e-8)).sum()  # scalar
        H = E_T  # scalar
        return C, H

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
        """Train on a single episode with the dual controller and reasoning."""
        T = int(actions.shape[0])

        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.tensor(states[1:], dtype=torch.float32, device=self.device)  # [T,D]

        # Patch B: Compute baseline tube and NLL0
        mu_knots_0, logsig_knots_0, stop_logit_0 = self._tube_init(z, s0_t)
        NLL0, w_0 = self._expected_nll(mu_knots_0, logsig_knots_0, stop_logit_0, y, T, detach_w=True)
        if self.improve_detach:
            NLL0_detached = NLL0.detach()
        else:
            NLL0_detached = NLL0

        # Patch E: Compute gate score (need volatility from env - will add later)
        # For now, compute gate_score with available info
        gate_score_val = self.gate_score(NLL0=NLL0, z=z, volatility=None)
        gate_on_prob = torch.sigmoid((gate_score_val - self.gate_thresh) / (self.gate_kappa + 1e-8))
        gate_on = (gate_on_prob > 0.5).float()

        # Patch A + E: Choose reasoning approach based on reasoning_mode
        Hr = 0
        E_Hr_imagine = 0.0
        p_refine_stop_val = 0.0
        
        if self.reasoning_mode == "off":
            Hr = 0
            mu_knots = mu_knots_0
            logsig_knots = logsig_knots_0
            stop_logit = stop_logit_0
            # NLLr will equal NLL0 (no refinement)
        elif self.reasoning_mode == "fixed":
            # Fixed pondering: sample Hr from geometric distribution
            Hr, E_Hr_imagine, p_refine_stop_val = self.sample_reasoning_steps(z, s0)
            mu_knots, logsig_knots, stop_logit = self.infer_tube(s0_t, z, Hr)
        elif self.reasoning_mode == "gated":
            if gate_on > 0.5:
                # Gate is on: use reasoning
                Hr, E_Hr_imagine, p_refine_stop_val = self.sample_reasoning_steps(z, s0)
                # Use Hr_max when gate is on (simplest approach)
                Hr = self.Hr_max
                mu_knots, logsig_knots, stop_logit = self.infer_tube(s0_t, z, Hr)
            else:
                # Gate is off: use default
                Hr = self.Hr_default
                if Hr == 0:
                    mu_knots = mu_knots_0
                    logsig_knots = logsig_knots_0
                    stop_logit = stop_logit_0
                else:
                    mu_knots, logsig_knots, stop_logit = self.infer_tube(s0_t, z, Hr)
        elif self.reasoning_mode == "dynamic":
            # Dynamic pondering: adaptively decide when to stop thinking
            mu_knots, logsig_knots, stop_logit, Hr, halting_probs = self.infer_tube_dynamic(
                s0_t, z, max_steps=self.max_refine_steps
            )
            # For logging, compute expected Hr from the halting probs
            if len(halting_probs) > 0:
                p_stops = torch.stack(halting_probs)
                E_Hr_imagine = float(Hr)  # actual steps taken
                p_refine_stop_val = float(p_stops[-1].item()) if len(p_stops) > 0 else 0.0
            else:
                E_Hr_imagine = 0.0
                p_refine_stop_val = 0.0
        else:
            raise ValueError(f"Unknown reasoning_mode: {self.reasoning_mode}")

        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)

        # Patch B: Compute refined NLLr
        NLLr, w_r = self._expected_nll(mu_knots, logsig_knots, stop_logit, y, T, detach_w=False)
        
        # Patch B: Compute improve
        improve = torch.relu(NLL0_detached - NLLr)

        # Volume tracking: measure cone tightening during refinement
        with torch.no_grad():
            vol_init = torch.exp(logsig_knots_0).sum()
            vol_final = torch.exp(logsig_knots).sum()
            vol_reduction = vol_init - vol_final
            vol_improvement_rate = (vol_reduction / Hr) if Hr > 0 else torch.tensor(0.0)

        # evaluate refined tube for this observed T
        mu, log_var = self._tube_traj(mu_knots, logsig_knots, T)

        # per-step NLL (using refined tube)
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

        # ponder cost: shorter horizons preferred (commitment compute)
        ponder_commit = (self.lambda_h + self.lambda_T) * E_T_train

        # Patch C: Pay-for-progress ponder cost
        p_refine_stop_t = self.ponder_head(s0_t, z)
        _, E_Hr_train = truncated_geometric_weights(p_refine_stop_t, self.max_refine_steps)
        ponder_reason_raw = self.lambda_r * E_Hr_train
        # Effective ponder cost: discount by improvement
        improve_detached = improve.detach()  # always detach improve here
        ponder_reason_eff = self.lambda_r * torch.relu(E_Hr_train - self.c_improve * improve_detached)

        # Total ponder cost
        ponder = ponder_commit + ponder_reason_eff

        # KL regularization: encourages meaningful but not collapsed latent space
        kl = kl_diag_gaussian_to_standard(z_mu, z_logstd).mean()

        # Dual controller terms for tightest feasible cone
        # Cone volume term (push tighter), weighted by geometric halting distribution
        # This prevents "early tight, late wide" exploits by aligning cone cost
        # with the port's own notion of where the episode ends
        cv_t = torch.prod(std, dim=-1)  # [T] - product across all dimensions
        cone_vol_w = (
            w.detach() * cv_t
        ).sum()  # scalar (detach w to avoid weird coupling)
        loss_cone = self.w_cone * cone_vol_w

        # constraint penalty: λ * (target - achieved)
        constraint_violation = self.target_bind - soft_bind_rate
        loss_constraint = self.lambda_bind * constraint_violation

        # Efficiency-based horizon bonus: reward long commitments ONLY when cones are tight
        # This prevents gaming by widening cones to get longer horizons
        tight = torch.exp(-torch.log(cone_vol_w + 1e-8))  # ~ 1/vol
        horizon_bonus = -self.w_horizon_bonus * E_T_train * soft_bind_rate * tight.detach()

        loss = (
            exp_nll
            + cone_reg
            + self.goal_weight * goal_cost
            + self.action_l2 * act_cost
            + ponder
            + self.beta_kl * kl
            + loss_cone
            + loss_constraint
            + horizon_bonus
        )

        # Memory regularizer: penalize sampling z in high-risk neighborhoods
        mem_risk = self.memory_risk(z)
        mem_loss = self.w_mem * mem_risk
        loss = loss + mem_loss

        # Monotone algebra regularizer
        Teval = self.algebra_Teval
        delta = self.algebra_delta

        uw = self.u_wide / (self.u_wide.norm() + 1e-8)
        ue = self.u_ext / (self.u_ext.norm() + 1e-8)

        C0, H0 = self._cone_summaries(s0_t, z, Teval)

        Cw, _ = self._cone_summaries(s0_t, z + delta * uw, Teval)
        _, He = self._cone_summaries(s0_t, z + delta * ue, Teval)

        dC_wide = Cw - C0
        dH_ext = He - H0

        L_wide = torch.relu(self.algebra_margin_C - dC_wide)
        L_ext = torch.relu(self.algebra_margin_H - dH_ext)

        # optional inverse checks
        Cn, _ = self._cone_summaries(s0_t, z - delta * uw, Teval)
        _, Hc = self._cone_summaries(s0_t, z - delta * ue, Teval)
        L_narrow = torch.relu(self.algebra_margin_C - (C0 - Cn))
        L_contr = torch.relu(self.algebra_margin_H - (H0 - Hc))

        mono_loss = L_wide + L_ext + 0.5 * (L_narrow + L_contr)

        # optional orthogonality (disentangle widen vs extend)
        ortho_loss = torch.tensor(0.0, device=s0_t.device)
        if self.w_algebra_ortho > 0.0:
            cos = torch.dot(uw, ue)
            ortho_loss = cos * cos

        algebra_loss = mono_loss + self.w_algebra_ortho * ortho_loss
        loss = loss + self.w_algebra * algebra_loss

        # binding predicate based on the tube itself
        with torch.no_grad():
            is_bound = self.binding_predicate(mu, log_var, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
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

            # Reasoning compute controller: increase λ_r if reasoning steps exceed target
            self.lambda_r += self.lambda_r_lr * float(
                (E_Hr_train - self.target_Hr).item()
            )
            lo_r, hi_r = self.lambda_r_clip
            self.lambda_r = float(np.clip(self.lambda_r, lo_r, hi_r))

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
        self.history["algebra_loss"].append(float(algebra_loss.detach().item()))
        self.history["dC_wide"].append(float(dC_wide.detach().item()))
        self.history["dH_ext"].append(float(dH_ext.detach().item()))
        self.history["horizon_bonus"].append(float(horizon_bonus.detach().item()))
        self.history["Hr"].append(int(Hr))
        self.history["E_Hr"].append(float(E_Hr_imagine))
        self.history["E_Hr_train"].append(float(E_Hr_train.detach().item()))
        self.history["p_refine_stop"].append(float(p_refine_stop_val))
        # Patch B: Log NLL0, NLLr, improve
        self.history["NLL0"].append(float(NLL0.detach().item()))
        self.history["NLLr"].append(float(NLLr.detach().item()))
        self.history["improve"].append(float(improve.detach().item()))
        # Patch C: Log ponder costs
        self.history["ponder_reason_raw"].append(float(ponder_reason_raw.detach().item()))
        self.history["ponder_reason_eff"].append(float(ponder_reason_eff.detach().item()))
        # Patch E: Log gate info
        self.history["gate_score"].append(float(gate_score_val.detach().item()))
        self.history["gate_on"].append(float(gate_on.item()))
        # Keep delta_nll for backward compatibility (use improve)
        delta_nll_compat = NLL0_detached - NLLr
        self.history["delta_nll"].append(float(delta_nll_compat.detach().item()))
        self.history["lambda_r"].append(float(self.lambda_r))
        self.history["mem_loss"].append(float(mem_loss.detach().item()))
        self.history["mem_risk"].append(float(mem_risk.detach().item()))
        self.history["mem_n"].append(int(len(self.mem)))
        self.history["vol_init"].append(float(vol_init.item()))
        self.history["vol_final"].append(float(vol_final.item()))
        self.history["vol_reduction"].append(float(vol_reduction.item()))
        self.history["vol_improvement_rate"].append(float(vol_improvement_rate.item()))

        # Store commitment outcome in memory
        with torch.no_grad():
            self.mem.append({
                "z": z.detach().cpu().squeeze(0),  # [z_dim]
                "soft_bind": float(soft_bind_rate.item()),
                "cone_vol": float(cone_vol_w.item()),
                "E_T": float(E_T_train.item()),
                "lambda_bind": float(self.lambda_bind),
                "lambda_T": float(self.lambda_T),
                "Hr": int(Hr),
            })
