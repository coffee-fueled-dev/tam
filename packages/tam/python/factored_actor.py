"""
Factored TAM Actor: Separates z into z_intent (commitment geometry) and z_real (realization).

The key insight is that:
- z_intent captures WHAT we commit to: uncertainty shape, horizon, bind rate
- z_real captures HOW we achieve it: environment-specific actions and trajectories

z_intent should be transferable across environments via a learned functor.
z_real encodes environment-specific execution details.

Cone geometry (sigma, stop) depends ONLY on z_intent, making it transportable.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import base actor and networks
try:
    from .actor import Actor
    from .networks import (
        FactoredActorNet,
        FactoredSharedTube,
        SharedPolicy,
        TubeRefiner,
        PonderHead,
        DynamicPonderHead,
    )
    from .utils import (
        gaussian_nll,
        interp1d_linear,
        kl_diag_gaussian_to_standard,
        sample_truncated_geometric,
        truncated_geometric_weights,
    )
except ImportError:
    from actor import Actor
    from networks import (
        FactoredActorNet,
        FactoredSharedTube,
        SharedPolicy,
        TubeRefiner,
        PonderHead,
        DynamicPonderHead,
    )
    from utils import (
        gaussian_nll,
        interp1d_linear,
        kl_diag_gaussian_to_standard,
        sample_truncated_geometric,
        truncated_geometric_weights,
    )


@dataclass
class FactoredZSample:
    """Container for factored z samples with all components."""
    z: torch.Tensor  # Full z = cat([z_intent, z_real])
    z_intent: torch.Tensor
    z_real: torch.Tensor
    z_mu: torch.Tensor  # Full mean
    z_logstd: torch.Tensor  # Full logstd
    z_intent_mu: torch.Tensor
    z_intent_logstd: torch.Tensor
    z_real_mu: torch.Tensor
    z_real_logstd: torch.Tensor


class FactoredActor(Actor):
    """
    TAM Actor with factored latent space.
    
    z = [z_intent, z_real] where:
    - z_intent (dim: z_intent_dim): commitment geometry, transportable across environments
    - z_real (dim: z_real_dim): environment-specific realization
    
    Key differences from base Actor:
    - Uses FactoredActorNet for separate posteriors
    - Tube sigma/stop depend ONLY on z_intent (via FactoredSharedTube)
    - Policy uses both z_intent and z_real
    - Separate KL tracking for intent and real
    - Memory stores both components
    """

    def __init__(
        self,
        obs_dim: int = 2,
        pred_dim: Optional[int] = None,
        action_dim: int = 1,
        # Factored z dimensions
        z_intent_dim: int = 4,  # Start small for intent
        z_real_dim: int = 4,  # Remainder for realization
        # Standard parameters
        hidden_dim: int = 64,
        lr: float = 7e-4,
        amax: float = 2.0,
        maxH: int = 64,
        minT: int = 2,
        M: int = 16,
        logstd_clip: Tuple[float, float] = (-4.0, 2.0),
        k_sigma: float = 2.0,
        bind_success_frac: float = 0.85,
        cone_reg: float = 0.001,
        goal_weight: float = 0.25,
        action_l2: float = 0.01,
        lambda_h: float = 0.002,
        beta_kl: float = 3e-4,
        halt_bias: float = -1.0,
        w_horizon_bonus: float = 0.01,
        # Reasoning parameters
        max_refine_steps: int = 8,
        refine_step_scale: float = 0.1,
        lambda_r: float = 0.001,
        target_Hr: float = 4.0,
        lambda_r_lr: float = 0.02,
        use_dynamic_pondering: bool = False,
        reasoning_mode: str = "fixed",
        freeze_sigma_refine: bool = True,
        c_improve: float = 1.0,
        improve_detach: bool = True,
        gate_kind: str = "nll0",
        gate_thresh: float = 0.5,
        gate_kappa: float = 0.1,
        Hr_default: int = 0,
        # Factored-specific parameters
        intent_only_sigma: bool = True,  # Sigma depends only on z_intent
        intent_only_stop: bool = True,  # Stop depends only on z_intent
        beta_kl_intent: Optional[float] = None,  # Separate KL weight for intent
        beta_kl_real: Optional[float] = None,  # Separate KL weight for real
    ):
        # Don't call super().__init__() yet - we need to set up dimensions first
        nn.Module.__init__(self)
        
        self.device = torch.device("cpu")
        
        # Set dimensions
        self.obs_dim = obs_dim
        self.pred_dim = pred_dim if pred_dim is not None else obs_dim
        self.action_dim = action_dim
        
        # Factored z dimensions
        self.z_intent_dim = z_intent_dim
        self.z_real_dim = z_real_dim
        self.z_dim = z_intent_dim + z_real_dim  # Total z dimension
        
        self.hidden_dim = hidden_dim
        self.intent_only_sigma = intent_only_sigma
        self.intent_only_stop = intent_only_stop
        
        # KL weights (separate for intent and real)
        self.beta_kl = beta_kl
        self.beta_kl_intent = beta_kl_intent if beta_kl_intent is not None else beta_kl
        self.beta_kl_real = beta_kl_real if beta_kl_real is not None else beta_kl * 0.1  # Weaker on real
        
        # Standard parameters
        self.amax = amax
        self.maxH = int(maxH)
        self.minT = int(minT)
        self.target_ET = 16.0
        self.lambda_T = 0.0
        self.lambda_T_lr = 0.02
        self.lambda_T_clip = (0.0, 50.0)
        self.M = int(M)
        self.logstd_min, self.logstd_max = logstd_clip
        self.k_sigma = k_sigma
        self.bind_success_frac = bind_success_frac
        self.cone_reg = cone_reg
        self.goal_weight = goal_weight
        self.action_l2 = action_l2
        self.lambda_h = lambda_h
        
        # Dual controller
        self.target_bind = 0.90
        self.lambda_bind = 0.0
        self.lambda_lr = 0.05
        self.lambda_clip = (0.0, 50.0)
        self.margin_temp = 0.25
        self.w_cone = 0.05
        self.w_horizon_bonus = w_horizon_bonus
        
        # Cone algebra
        self.algebra_Teval = min(32, self.maxH)
        self.algebra_delta = 0.25
        self.algebra_margin_C = 0.0
        self.algebra_margin_H = 0.0
        self.w_algebra = 0.02
        self.w_algebra_ortho = 0.01
        self.w_algebra_comm = 0.00
        
        # Reasoning parameters
        self.max_refine_steps = int(max_refine_steps)
        self.refine_step_scale = refine_step_scale
        self.target_Hr = target_Hr
        self.lambda_r = lambda_r
        self.lambda_r_lr = lambda_r_lr
        self.lambda_r_clip = (0.0, 50.0)
        self.use_dynamic_pondering = use_dynamic_pondering
        self.reasoning_mode = reasoning_mode
        self.freeze_sigma_refine = freeze_sigma_refine
        self.c_improve = c_improve
        self.improve_detach = improve_detach
        self.gate_kind = gate_kind
        self.gate_thresh = gate_thresh
        self.gate_kappa = gate_kappa
        self.Hr_default = Hr_default
        self.Hr_max = self.max_refine_steps
        
        # Memory
        self.mem_size = 50_000
        self.mem_min = 500
        self.mem_k = 64
        self.mem_sigma = 1.0
        self.w_mem = 0.02
        self.mem_detach_targets = True
        self.mem = deque(maxlen=self.mem_size)
        
        # === FACTORED NETWORKS ===
        
        # Factored actor network (separate posteriors for intent and real)
        self.actor = FactoredActorNet(
            state_dim=obs_dim,
            z_intent_dim=z_intent_dim,
            z_real_dim=z_real_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # Policy uses full z = [z_intent, z_real]
        self.pol = SharedPolicy(obs_dim, self.z_dim, action_dim, hidden_dim).to(self.device)
        
        # Factored tube: sigma/stop depend only on z_intent
        self.tube = FactoredSharedTube(
            state_dim=obs_dim,
            z_intent_dim=z_intent_dim,
            z_real_dim=z_real_dim,
            hidden_dim=hidden_dim,
            M=self.M,
            pred_dim=self.pred_dim,
            intent_only_sigma=intent_only_sigma,
            intent_only_stop=intent_only_stop,
        ).to(self.device)
        
        # Learnable operator directions (operate on z_intent only for transportability)
        self.u_wide = nn.Parameter(torch.randn(self.z_intent_dim) * 0.05)
        self.u_ext = nn.Parameter(torch.randn(self.z_intent_dim) * 0.05)
        
        # Reasoning networks (use full z for now)
        self.refiner = TubeRefiner(obs_dim, self.z_dim, self.M, hidden_dim, pred_dim=self.pred_dim).to(self.device)
        self.ponder_head = PonderHead(obs_dim, self.z_dim, hidden_dim).to(self.device)
        self.dynamic_ponder_head = DynamicPonderHead(obs_dim, self.z_dim, hidden_dim).to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Extended history with factored tracking
        self.history = {
            "step": [],
            "regime": [],
            "bind_success": [],
            "avg_abs_action": [],
            "T": [],
            "E_T": [],
            "E_T_train": [],
            "exp_nll": [],
            "cone_volume": [],
            "kl": [],
            "kl_intent": [],  # New: separate KL tracking
            "kl_real": [],  # New: separate KL tracking
            "z_norm": [],
            "z_intent_norm": [],  # New
            "z_real_norm": [],  # New
            "soft_bind": [],
            "lambda_bind": [],
            "lambda_T": [],
            "cone_vol": [],
            "algebra_loss": [],
            "dC_wide": [],
            "dH_ext": [],
            "horizon_bonus": [],
            "Hr": [],
            "E_Hr": [],
            "E_Hr_train": [],
            "p_refine_stop": [],
            "delta_nll": [],
            "lambda_r": [],
            "mem_loss": [],
            "mem_risk": [],
            "mem_n": [],
            "vol_init": [],
            "vol_final": [],
            "vol_reduction": [],
            "vol_improvement_rate": [],
            "NLL0": [],
            "NLLr": [],
            "improve": [],
            "ponder_reason_raw": [],
            "ponder_reason_eff": [],
            "gate_score": [],
            "gate_on": [],
        }
    
    def sample_z(
        self, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterized z ~ q(z|s0). Returns z, z_mu, z_logstd (full).
        
        For backward compatibility, returns concatenated z.
        Use sample_z_factored() for full factored output.
        """
        sample = self.sample_z_factored(s0_t)
        return sample.z, sample.z_mu, sample.z_logstd
    
    def sample_z_factored(self, s0_t: torch.Tensor) -> FactoredZSample:
        """
        Sample factored z with separate intent and real components.
        
        Args:
            s0_t: [1, obs_dim]
        
        Returns:
            FactoredZSample with all components
        """
        (
            z_intent_mu, z_intent_logstd,
            z_real_mu, z_real_logstd,
            z_mu, z_logstd
        ) = self.actor(s0_t)
        
        # Reparameterized sampling
        eps_intent = torch.randn_like(z_intent_mu)
        z_intent = z_intent_mu + torch.exp(z_intent_logstd) * eps_intent
        
        eps_real = torch.randn_like(z_real_mu)
        z_real = z_real_mu + torch.exp(z_real_logstd) * eps_real
        
        # Full z
        z = torch.cat([z_intent, z_real], dim=-1)
        
        return FactoredZSample(
            z=z,
            z_intent=z_intent,
            z_real=z_real,
            z_mu=z_mu,
            z_logstd=z_logstd,
            z_intent_mu=z_intent_mu,
            z_intent_logstd=z_intent_logstd,
            z_real_mu=z_real_mu,
            z_real_logstd=z_real_logstd,
        )
    
    def _split_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split full z into z_intent and z_real components."""
        z_intent = z[..., :self.z_intent_dim]
        z_real = z[..., self.z_intent_dim:]
        return z_intent, z_real
    
    def _tube_init(
        self, z: torch.Tensor, s0_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get initial tube state (before refinement).
        Uses factored tube where sigma/stop depend only on z_intent.
        """
        z_intent, z_real = self._split_z(z)
        
        if z.size(0) != s0_t.size(0):
            z_intent = z_intent.expand(s0_t.size(0), -1)
            z_real = z_real.expand(s0_t.size(0), -1)
        
        mu_knots, logsig_knots, stop_logit = self.tube(s0_t, z_intent, z_real)
        
        return mu_knots, logsig_knots, stop_logit
    
    def _cone_summaries_intent(
        self, s0_t: torch.Tensor, z_intent: torch.Tensor, Teval: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cone summaries using ONLY z_intent.
        
        This is the key for transportability: cone geometry is determined
        solely by z_intent, making it possible to learn a functor.
        """
        # Create dummy z_real (zeros) since cone geometry shouldn't depend on it
        z_real = torch.zeros(z_intent.size(0), self.z_real_dim, device=z_intent.device)
        z = torch.cat([z_intent, z_real], dim=-1)
        
        muK, sigK, stop_logit = self.infer_tube(s0_t, z, self.max_refine_steps)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        
        _, log_var = self._tube_traj(muK, sigK, Teval)
        std = torch.exp(0.5 * log_var)
        cv_t = torch.prod(std, dim=-1)
        
        w, E_T = truncated_geometric_weights(p_stop, Teval)
        C = (w * torch.log(cv_t + 1e-8)).sum()
        H = E_T
        return C, H
    
    def compute_kl_factored(
        self,
        z_intent_mu: torch.Tensor,
        z_intent_logstd: torch.Tensor,
        z_real_mu: torch.Tensor,
        z_real_logstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence for factored posteriors.
        
        Returns:
            kl_intent: KL for intent posterior
            kl_real: KL for real posterior
            kl_total: kl_intent + kl_real
        """
        kl_intent = kl_diag_gaussian_to_standard(z_intent_mu, z_intent_logstd)
        kl_real = kl_diag_gaussian_to_standard(z_real_mu, z_real_logstd)
        kl_total = kl_intent + kl_real
        return kl_intent, kl_real, kl_total
    
    def train_on_episode(
        self,
        step: int,
        regime: int,
        s0: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        z: torch.Tensor,
        z_mu: torch.Tensor,
        z_logstd: torch.Tensor,
        E_T_imagine: float,
        # Optional factored inputs (for when using sample_z_factored)
        z_intent: Optional[torch.Tensor] = None,
        z_real: Optional[torch.Tensor] = None,
        z_intent_mu: Optional[torch.Tensor] = None,
        z_intent_logstd: Optional[torch.Tensor] = None,
        z_real_mu: Optional[torch.Tensor] = None,
        z_real_logstd: Optional[torch.Tensor] = None,
    ):
        """
        Train on a single episode with factored KL and logging.
        
        Extends base class to track separate KL for intent and real.
        """
        T = int(actions.shape[0])
        s0_t = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.tensor(states[1:], dtype=torch.float32, device=self.device)
        
        # Split z if factored inputs not provided
        if z_intent is None:
            z_intent, z_real = self._split_z(z)
        
        # Get factored posterior params if not provided
        if z_intent_mu is None:
            (
                z_intent_mu, z_intent_logstd,
                z_real_mu, z_real_logstd,
                _, _
            ) = self.actor(s0_t)
        
        # Compute baseline tube and NLL0
        mu_knots_0, logsig_knots_0, stop_logit_0 = self._tube_init(z, s0_t)
        NLL0, w_0 = self._expected_nll(mu_knots_0, logsig_knots_0, stop_logit_0, y, T, detach_w=True)
        if self.improve_detach:
            NLL0_detached = NLL0.detach()
        else:
            NLL0_detached = NLL0
        
        # Gate score
        gate_score_val = self.gate_score(NLL0=NLL0, z=z, volatility=None)
        gate_on_prob = torch.sigmoid((gate_score_val - self.gate_thresh) / (self.gate_kappa + 1e-8))
        gate_on = (gate_on_prob > 0.5).float()
        
        # Reasoning
        Hr = 0
        E_Hr_imagine = 0.0
        p_refine_stop_val = 0.0
        
        if self.reasoning_mode == "off":
            Hr = 0
            mu_knots = mu_knots_0
            logsig_knots = logsig_knots_0
            stop_logit = stop_logit_0
        elif self.reasoning_mode == "fixed":
            Hr, E_Hr_imagine, p_refine_stop_val = self.sample_reasoning_steps(z, s0)
            mu_knots, logsig_knots, stop_logit = self.infer_tube(s0_t, z, Hr)
        elif self.reasoning_mode == "gated":
            if gate_on > 0.5:
                Hr, E_Hr_imagine, p_refine_stop_val = self.sample_reasoning_steps(z, s0)
                Hr = self.Hr_max
                mu_knots, logsig_knots, stop_logit = self.infer_tube(s0_t, z, Hr)
            else:
                Hr = self.Hr_default
                if Hr == 0:
                    mu_knots = mu_knots_0
                    logsig_knots = logsig_knots_0
                    stop_logit = stop_logit_0
                else:
                    mu_knots, logsig_knots, stop_logit = self.infer_tube(s0_t, z, Hr)
        elif self.reasoning_mode == "dynamic":
            mu_knots, logsig_knots, stop_logit, Hr, halting_probs = self.infer_tube_dynamic(
                s0_t, z, max_steps=self.max_refine_steps
            )
            if len(halting_probs) > 0:
                p_stops = torch.stack(halting_probs)
                E_Hr_imagine = float(Hr)
                p_refine_stop_val = float(p_stops[-1].item()) if len(p_stops) > 0 else 0.0
            else:
                E_Hr_imagine = 0.0
                p_refine_stop_val = 0.0
        
        # Compute refined NLL
        NLLr, w = self._expected_nll(mu_knots, logsig_knots, stop_logit, y, T, detach_w=True)
        improve = NLL0_detached - NLLr
        
        # Factored KL
        kl_intent, kl_real, kl_total = self.compute_kl_factored(
            z_intent_mu, z_intent_logstd, z_real_mu, z_real_logstd
        )
        
        # Expected horizon
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        _, E_T_train = truncated_geometric_weights(p_stop, T)
        
        # Soft bind rate
        mu_traj, log_var = self._tube_traj(mu_knots, logsig_knots, T)
        std_traj = torch.exp(0.5 * log_var)
        diff = torch.abs(y - mu_traj)
        margin = diff - self.k_sigma * std_traj
        soft_bind_t = torch.sigmoid(-margin / self.margin_temp)
        soft_bind_all = torch.prod(soft_bind_t, dim=-1)
        soft_bind_rate = (w * soft_bind_all).sum()
        
        # Cone volume (weighted)
        cv_t = torch.prod(std_traj, dim=-1)
        cone_vol_w = (w * cv_t).sum()
        
        # Main loss: NLLr (refined)
        loss = NLLr
        
        # Ponder cost (pay-for-progress)
        ponder_reason_raw = self.lambda_r * Hr
        ponder_reason_eff = ponder_reason_raw - self.c_improve * improve.detach()
        ponder_reason_eff = torch.clamp(ponder_reason_eff, min=0.0)
        loss = loss + ponder_reason_eff
        
        # Factored KL loss (separate weights)
        kl_loss = self.beta_kl_intent * kl_intent + self.beta_kl_real * kl_real
        loss = loss + kl_loss
        
        # Dual constraint: soft bind rate
        bind_penalty = self.lambda_bind * (self.target_bind - soft_bind_rate)
        loss = loss + bind_penalty
        
        # Cone volume minimization
        loss = loss + self.w_cone * cone_vol_w
        
        # Horizon regularization
        loss = loss + self.lambda_T * E_T_train
        loss = loss + self.lambda_h * E_T_train
        
        # Intent-horizon alignment loss: encourage z_intent norm to correlate with E[T]
        # This forces z_intent to actually control the horizon
        if hasattr(self, 'w_intent_align') and self.w_intent_align > 0:
            # Perturb z_intent and check if E[T] changes appropriately
            z_intent_norm = z_intent.norm(dim=-1)
            # Target: higher z_intent norm → different E[T] (either direction)
            # We use the gradient magnitude as proxy
            # Simple version: just encourage z_intent to have reasonable norm
            intent_align_loss = -self.w_intent_align * z_intent_norm.mean()
            loss = loss + intent_align_loss
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        
        # Dual updates
        with torch.no_grad():
            bind_gap = self.target_bind - soft_bind_rate.item()
            self.lambda_bind = float(np.clip(
                self.lambda_bind + self.lambda_lr * bind_gap,
                *self.lambda_clip
            ))
            
            ET_gap = E_T_train.item() - self.target_ET
            self.lambda_T = float(np.clip(
                self.lambda_T + self.lambda_T_lr * ET_gap,
                *self.lambda_T_clip
            ))
        
        # Store in memory with factored z
        mem_entry = {
            "z": z.detach().cpu().squeeze(0),
            "z_intent": z_intent.detach().cpu().squeeze(0),
            "z_real": z_real.detach().cpu().squeeze(0),
            "soft_bind": float(soft_bind_rate.item()),
            "cone_vol": float(cone_vol_w.item()),
            "E_T": float(E_T_train.item()),
            "lambda_bind": float(self.lambda_bind),
            "lambda_T": float(self.lambda_T),
            "Hr": int(Hr),
            "rule": int(regime),
            "step": int(step),
            "kl_intent": float(kl_intent.item()),
            "kl_real": float(kl_real.item()),
        }
        self.mem.append(mem_entry)
        
        # Log history
        self.history["step"].append(step)
        self.history["regime"].append(regime)
        self.history["bind_success"].append(float(soft_bind_rate.item()))
        self.history["T"].append(T)
        self.history["E_T"].append(E_T_imagine)
        self.history["E_T_train"].append(float(E_T_train.item()))
        self.history["exp_nll"].append(float(NLLr.item()))
        self.history["cone_volume"].append(float(cone_vol_w.item()))
        self.history["kl"].append(float(kl_total.item()))
        self.history["kl_intent"].append(float(kl_intent.item()))
        self.history["kl_real"].append(float(kl_real.item()))
        self.history["z_norm"].append(float(z.norm().item()))
        self.history["z_intent_norm"].append(float(z_intent.norm().item()))
        self.history["z_real_norm"].append(float(z_real.norm().item()))
        self.history["soft_bind"].append(float(soft_bind_rate.item()))
        self.history["lambda_bind"].append(float(self.lambda_bind))
        self.history["lambda_T"].append(float(self.lambda_T))
        self.history["cone_vol"].append(float(cone_vol_w.item()))
        self.history["Hr"].append(Hr)
        self.history["E_Hr"].append(E_Hr_imagine)
        self.history["p_refine_stop"].append(p_refine_stop_val)
        self.history["NLL0"].append(float(NLL0.item()))
        self.history["NLLr"].append(float(NLLr.item()))
        self.history["improve"].append(float(improve.item()))
        self.history["ponder_reason_raw"].append(float(ponder_reason_raw))
        self.history["ponder_reason_eff"].append(float(ponder_reason_eff.item()))
        self.history["gate_score"].append(float(gate_score_val.item()))
        self.history["gate_on"].append(float(gate_on.item()))
        self.history["mem_n"].append(len(self.mem))
    
    def get_intent_embedding(self, s0_t: torch.Tensor) -> torch.Tensor:
        """Get deterministic z_intent (mean) for functor learning."""
        (
            z_intent_mu, _, _, _, _, _
        ) = self.actor(s0_t)
        return z_intent_mu
    
    def set_intent(
        self, 
        s0_t: torch.Tensor, 
        z_intent: torch.Tensor
    ) -> FactoredZSample:
        """
        Set z_intent externally (e.g., from functor) while sampling z_real natively.
        
        This is the key method for transfer: use transported intent,
        but sample realization locally.
        """
        # Get native z_real
        (
            _, _,
            z_real_mu, z_real_logstd,
            _, _
        ) = self.actor(s0_t)
        
        eps_real = torch.randn_like(z_real_mu)
        z_real = z_real_mu + torch.exp(z_real_logstd) * eps_real
        
        # Combine with provided intent
        z = torch.cat([z_intent, z_real], dim=-1)
        
        # Create sample object (intent params are external, real are from posterior)
        return FactoredZSample(
            z=z,
            z_intent=z_intent,
            z_real=z_real,
            z_mu=torch.cat([z_intent, z_real_mu], dim=-1),
            z_logstd=torch.cat([torch.zeros_like(z_intent), z_real_logstd], dim=-1),
            z_intent_mu=z_intent,
            z_intent_logstd=torch.zeros_like(z_intent),
            z_real_mu=z_real_mu,
            z_real_logstd=z_real_logstd,
        )


def create_factored_actor(**kwargs) -> FactoredActor:
    """Factory function for creating a FactoredActor with sensible defaults."""
    defaults = {
        "z_intent_dim": 4,
        "z_real_dim": 4,
        "intent_only_sigma": True,
        "intent_only_stop": True,
    }
    defaults.update(kwargs)
    return FactoredActor(**defaults)
