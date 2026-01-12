"""
Competitive Port Binding Actor: CEM-based z selection for forecasting.

Key insight: The actor can CHOOSE among multiple candidate commitments (z)
by planning in z-space to optimize a tradeoff between:
- Forecast intent (accuracy of prediction)
- Agency (tight tube / small volume)
- Reliability (low probability of cone violation)

This extends the homeostatic actor with:
1. RiskNet: Learned critic r_ψ(s0, z) → p_fail
2. CEM binding: Iterative search over z-space
3. Scoring function: S(s0, z) = -intent_proxy + α*agency - λ*risk
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BindingMode(Enum):
    """Binding strategies for selecting z."""
    RANDOM = "random"       # Sample one z ~ q(z|s0)
    BEST_OF_K = "best_of_k" # Sample K, pick best score
    CEM = "cem"             # Iterative CEM refinement
    ORACLE = "oracle"       # Use true outcome (diagnostic only)


class RunningStats:
    """Track running mean and std for z-score normalization."""
    
    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.mean: Optional[float] = None
        self.var: Optional[float] = None
        self.count = 0
    
    def update(self, value: float):
        self.count += 1
        if self.mean is None:
            self.mean = value
            self.var = 0.0
        else:
            delta = value - self.mean
            self.mean = self.momentum * self.mean + (1 - self.momentum) * value
            self.var = self.momentum * self.var + (1 - self.momentum) * (delta ** 2)
    
    @property
    def std(self) -> float:
        if self.var is None:
            return 1.0
        return max(np.sqrt(self.var), 1e-6)
    
    def z_score(self, value: float) -> float:
        if self.mean is None:
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class TubeOutput:
    """Output from tube prediction."""
    mu: torch.Tensor      # [T, pred_dim] trajectory mean
    sigma: torch.Tensor   # [T, pred_dim] tube width (std)
    z: torch.Tensor       # [z_dim] latent commitment
    z_mu: torch.Tensor    # [z_dim] posterior mean
    z_logstd: torch.Tensor  # [z_dim] posterior log-std


@dataclass
class BindingResult:
    """Result from competitive binding."""
    z_star: torch.Tensor   # Selected commitment
    mu: torch.Tensor       # [T, pred_dim] trajectory mean
    sigma: torch.Tensor    # [T, pred_dim] tube width
    score: float           # Score of selected z
    intent_proxy: float    # ||mu_T - goal||^2
    agency: float          # -mean(log sigma)
    risk_pred: float       # Predicted risk from RiskNet
    mode: BindingMode      # Which strategy was used


class TubeNet(nn.Module):
    """Tube network: sigma depends ONLY on z."""
    
    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 4,
        pred_dim: int = 2,
        hidden_dim: int = 64,
        T: int = 16,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.pred_dim = pred_dim
        self.hidden_dim = hidden_dim
        self.T = T
        
        # Encoder: s0 → q(z|s0)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logstd_head = nn.Linear(hidden_dim, z_dim)
        
        # Mu network: (s0, z) → mu[1:T]
        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        
        # Sigma network: z → sigma[1:T] (ONLY z!)
        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        
        nn.init.zeros_(self.sigma_net[-1].bias)
    
    def encode(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(s0)
        z_mu = self.z_mu_head(h)
        z_logstd = self.z_logstd_head(h).clamp(-4, 2)
        return z_mu, z_logstd
    
    def sample_z(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        std = torch.exp(z_logstd)
        eps = torch.randn_like(std)
        return z_mu + std * eps
    
    def predict_tube(self, s0: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict tube for given s0 and z. Supports batched inputs."""
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        B = z.shape[0]
        
        # Expand s0 if needed for batched z
        if s0.shape[0] == 1 and B > 1:
            s0 = s0.expand(B, -1)
        
        mu = self.mu_net(torch.cat([s0, z], dim=-1))
        mu = mu.view(B, self.T, self.pred_dim)
        
        logsig = self.sigma_net(z).clamp(-4, 3)
        sigma = torch.exp(logsig.view(B, self.T, self.pred_dim))
        
        return mu, sigma
    
    def forward(self, s0: torch.Tensor) -> TubeOutput:
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        
        z_mu, z_logstd = self.encode(s0)
        z = self.sample_z(z_mu, z_logstd)
        mu, sigma = self.predict_tube(s0, z)
        
        return TubeOutput(
            mu=mu.squeeze(0),
            sigma=sigma.squeeze(0),
            z=z.squeeze(0),
            z_mu=z_mu.squeeze(0),
            z_logstd=z_logstd.squeeze(0),
        )


class RiskNet(nn.Module):
    """
    Risk critic: r_ψ(s0, z) → p_fail ∈ [0,1]
    
    Predicts probability of tube violation given state and commitment.
    """
    
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output p_fail ∈ [0,1]
        )
    
    def forward(self, s0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict failure probability."""
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Expand s0 if needed for batched z
        if s0.shape[0] == 1 and z.shape[0] > 1:
            s0 = s0.expand(z.shape[0], -1)
        
        x = torch.cat([s0, z], dim=-1)
        return self.net(x).squeeze(-1)


class CompetitiveActor(nn.Module):
    """
    Actor with competitive port binding via CEM.
    
    Features:
    - Tube network with homeostatic training
    - Risk critic for risk-aware z selection
    - CEM-based z selection optimizing intent/agency/risk tradeoff
    - Multiple binding modes for comparison
    """
    
    def __init__(
        self,
        obs_dim: int,
        pred_dim: int = 2,
        z_dim: int = 4,
        hidden_dim: int = 64,
        T: int = 16,
        k_sigma: float = 2.0,
        lr: float = 1e-3,
        lr_risk: float = 1e-3,
        # Homeostasis parameters
        alpha_vol: float = 0.5,
        beta_kl: float = 0.001,
        eta_lambda: float = 0.05,
        lambda_init: float = 1.0,
        lambda_max: float = 50.0,
        gamma: float = 25.0,
        stats_momentum: float = 0.99,
        # CEM parameters
        cem_iters: int = 4,
        cem_samples: int = 128,
        cem_elites: int = 16,
        cem_smoothing: float = 0.25,
        cem_std_floor: float = 0.2,
        # Scoring weights
        intent_weight: float = 1.0,    # Weight on intent proxy
        agency_weight: float = 0.5,    # Weight on agency (tightness)
        risk_weight: float = 1.0,      # Weight on risk penalty
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.pred_dim = pred_dim
        self.z_dim = z_dim
        self.T = T
        self.k_sigma = k_sigma
        
        # Homeostasis config
        self.alpha_vol = alpha_vol
        self.beta_kl = beta_kl
        self.eta_lambda = eta_lambda
        self.lambda_max = lambda_max
        self.gamma = gamma
        
        # CEM config
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_elites = cem_elites
        self.cem_smoothing = cem_smoothing
        self.cem_std_floor = cem_std_floor
        
        # Scoring weights
        self.intent_weight = intent_weight
        self.agency_weight = agency_weight
        self.risk_weight = risk_weight
        
        # Networks
        self.tube = TubeNet(obs_dim, z_dim, pred_dim, hidden_dim, T)
        self.risk_net = RiskNet(obs_dim, z_dim, hidden_dim)
        
        # Optimizers
        self.optimizer = optim.Adam(self.tube.parameters(), lr=lr)
        self.risk_optimizer = optim.Adam(self.risk_net.parameters(), lr=lr_risk)
        
        # Dual variable
        self.lambda_fail = lambda_init
        
        # Running statistics
        self.stats_residual = RunningStats(stats_momentum)
        self.stats_sigma = RunningStats(stats_momentum)
        self.stats_hardness = RunningStats(stats_momentum)
        self.stats_fail = RunningStats(stats_momentum)
        
        # History
        self.history: Dict[str, List[float]] = {
            "mse": [],
            "log_vol": [],
            "bind_hard": [],
            "bind_soft": [],
            "fail_soft": [],
            "hardness": [],
            "hardness_z": [],
            "lambda": [],
            "kl": [],
            "loss": [],
            "z_norm": [],
            "sigma_mean": [],
            # CEM-specific
            "score": [],
            "intent_proxy": [],
            "agency": [],
            "risk_pred": [],
            "risk_loss": [],
            "cem_improvement": [],  # Score improvement from CEM iters
        }
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def extract_goal(self, s0: torch.Tensor) -> torch.Tensor:
        """Extract goal coordinates from observation."""
        # s0 = [x, y, goal_x, goal_y, rule_oh...]
        return s0[2:4]  # goal_x, goal_y
    
    def compute_intent_proxy(
        self, mu: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Intent proxy: ||mu_T - goal||^2
        
        This is available at bind time (goal is in observation).
        """
        # mu: [T, pred_dim] or [B, T, pred_dim]
        if mu.dim() == 2:
            mu_T = mu[-1]  # Last timestep
        else:
            mu_T = mu[:, -1, :]  # [B, pred_dim]
        
        return ((mu_T - goal) ** 2).sum(dim=-1)
    
    def compute_agency(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Agency: -mean(log sigma)
        
        Higher agency = tighter tubes = smaller sigma.
        """
        if sigma.dim() == 2:
            return -torch.log(sigma).mean()
        else:
            return -torch.log(sigma).mean(dim=(1, 2))
    
    def score_commitment(
        self,
        s0: torch.Tensor,
        z: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Score a commitment z for selection.
        
        S(s0, z) = -intent_weight * intent_proxy 
                   + agency_weight * agency 
                   - risk_weight * risk
        
        Returns (scores, details) where details has component values.
        """
        # Get tube predictions
        mu, sigma = self.tube.predict_tube(s0, z)
        
        # Compute components
        intent_proxy = self.compute_intent_proxy(mu, goal)
        agency = self.compute_agency(sigma)
        risk = self.risk_net(s0, z)
        
        # Combine into score (higher = better)
        score = (
            -self.intent_weight * intent_proxy +
            self.agency_weight * agency -
            self.risk_weight * risk
        )
        
        details = {
            "intent_proxy": intent_proxy,
            "agency": agency,
            "risk": risk,
            "mu": mu,
            "sigma": sigma,
        }
        
        return score, details
    
    def select_z_random(self, s0: torch.Tensor) -> BindingResult:
        """Random binding: sample one z ~ q(z|s0)."""
        with torch.no_grad():
            z_mu, z_logstd = self.tube.encode(s0.unsqueeze(0))
            z = self.tube.sample_z(z_mu, z_logstd).squeeze(0)
            
            goal = self.extract_goal(s0)
            score, details = self.score_commitment(s0, z, goal)
            
            return BindingResult(
                z_star=z,
                mu=details["mu"].squeeze(0),
                sigma=details["sigma"].squeeze(0),
                score=float(score.item()),
                intent_proxy=float(details["intent_proxy"].item()),
                agency=float(details["agency"].item()),
                risk_pred=float(details["risk"].item()),
                mode=BindingMode.RANDOM,
            )
    
    def select_z_best_of_k(self, s0: torch.Tensor, K: int = None) -> BindingResult:
        """Best-of-K: sample K candidates, pick best score."""
        if K is None:
            K = self.cem_samples
        
        with torch.no_grad():
            z_mu, z_logstd = self.tube.encode(s0.unsqueeze(0))
            z_mu = z_mu.squeeze(0)
            z_logstd = z_logstd.squeeze(0)
            
            # Sample K candidates
            std = torch.exp(z_logstd)
            eps = torch.randn(K, self.z_dim, device=s0.device)
            z_candidates = z_mu.unsqueeze(0) + std.unsqueeze(0) * eps  # [K, z_dim]
            
            # Score all candidates
            goal = self.extract_goal(s0)
            scores, details = self.score_commitment(s0, z_candidates, goal)
            
            # Pick best
            best_idx = scores.argmax()
            z_star = z_candidates[best_idx]
            
            return BindingResult(
                z_star=z_star,
                mu=details["mu"][best_idx],
                sigma=details["sigma"][best_idx],
                score=float(scores[best_idx].item()),
                intent_proxy=float(details["intent_proxy"][best_idx].item()),
                agency=float(details["agency"][best_idx].item()),
                risk_pred=float(details["risk"][best_idx].item()),
                mode=BindingMode.BEST_OF_K,
            )
    
    def select_z_cem(self, s0: torch.Tensor) -> BindingResult:
        """
        CEM binding: iterative refinement over z-space.
        
        1. Initialize: mean = encoder's z_mu, std = max(encoder std, floor)
        2. For J iterations:
           - Sample K candidates from N(mean, std)
           - Score each
           - Keep top E elites
           - Refit mean/std to elites with smoothing
        3. Return best z*
        """
        with torch.no_grad():
            # Initialize from encoder
            z_mu, z_logstd = self.tube.encode(s0.unsqueeze(0))
            cem_mean = z_mu.squeeze(0).clone()
            cem_std = torch.exp(z_logstd.squeeze(0)).clamp(min=self.cem_std_floor)
            
            goal = self.extract_goal(s0)
            initial_score = None
            
            for iteration in range(self.cem_iters):
                # Sample candidates
                eps = torch.randn(self.cem_samples, self.z_dim, device=s0.device)
                z_candidates = cem_mean.unsqueeze(0) + cem_std.unsqueeze(0) * eps
                
                # Score all candidates
                scores, details = self.score_commitment(s0, z_candidates, goal)
                
                if initial_score is None:
                    initial_score = float(scores.max().item())
                
                # Select elites (top E)
                elite_indices = scores.argsort(descending=True)[:self.cem_elites]
                elite_z = z_candidates[elite_indices]
                
                # Refit distribution with smoothing
                new_mean = elite_z.mean(dim=0)
                new_std = elite_z.std(dim=0).clamp(min=self.cem_std_floor)
                
                cem_mean = (1 - self.cem_smoothing) * cem_mean + self.cem_smoothing * new_mean
                cem_std = (1 - self.cem_smoothing) * cem_std + self.cem_smoothing * new_std
            
            # Final evaluation
            best_idx = elite_indices[0]
            z_star = z_candidates[best_idx]
            final_score = float(scores[best_idx].item())
            
            # Track CEM improvement
            cem_improvement = final_score - initial_score if initial_score else 0.0
            self.history["cem_improvement"].append(cem_improvement)
            
            return BindingResult(
                z_star=z_star,
                mu=details["mu"][best_idx],
                sigma=details["sigma"][best_idx],
                score=final_score,
                intent_proxy=float(details["intent_proxy"][best_idx].item()),
                agency=float(details["agency"][best_idx].item()),
                risk_pred=float(details["risk"][best_idx].item()),
                mode=BindingMode.CEM,
            )
    
    def select_z_oracle(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
        K: int = None,
    ) -> BindingResult:
        """
        Oracle binding: use true outcome to select best z.
        
        NOT deployable - only for diagnostic upper bound.
        """
        if K is None:
            K = self.cem_samples
        
        with torch.no_grad():
            z_mu, z_logstd = self.tube.encode(s0.unsqueeze(0))
            z_mu = z_mu.squeeze(0)
            z_logstd = z_logstd.squeeze(0)
            
            # Sample K candidates
            std = torch.exp(z_logstd)
            eps = torch.randn(K, self.z_dim, device=s0.device)
            z_candidates = z_mu.unsqueeze(0) + std.unsqueeze(0) * eps
            
            # Evaluate TRUE outcome for each candidate
            best_score = float('-inf')
            best_idx = 0
            best_mu = None
            best_sigma = None
            
            for i in range(K):
                z = z_candidates[i]
                mu, sigma = self.tube.predict_tube(s0, z)
                mu = mu.squeeze(0)
                sigma = sigma.squeeze(0)
                
                # True bind
                bind = self.compute_bind_hard(mu, sigma, trajectory)
                log_vol = self.compute_log_volume(sigma)
                
                # Oracle score: high bind, low volume
                score = float(bind.item()) - 0.5 * float(log_vol.item())
                
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_mu = mu
                    best_sigma = sigma
            
            z_star = z_candidates[best_idx]
            goal = self.extract_goal(s0)
            intent_proxy = self.compute_intent_proxy(best_mu, goal)
            agency = self.compute_agency(best_sigma)
            risk = self.risk_net(s0, z_star)
            
            return BindingResult(
                z_star=z_star,
                mu=best_mu,
                sigma=best_sigma,
                score=best_score,
                intent_proxy=float(intent_proxy.item()),
                agency=float(agency.item()),
                risk_pred=float(risk.item()),
                mode=BindingMode.ORACLE,
            )
    
    def select_z(
        self,
        s0: torch.Tensor,
        mode: BindingMode = BindingMode.CEM,
        trajectory: torch.Tensor = None,  # Only for oracle mode
    ) -> BindingResult:
        """Select z using specified binding mode."""
        if mode == BindingMode.RANDOM:
            return self.select_z_random(s0)
        elif mode == BindingMode.BEST_OF_K:
            return self.select_z_best_of_k(s0)
        elif mode == BindingMode.CEM:
            return self.select_z_cem(s0)
        elif mode == BindingMode.ORACLE:
            assert trajectory is not None, "Oracle mode requires trajectory"
            return self.select_z_oracle(s0, trajectory)
        else:
            raise ValueError(f"Unknown binding mode: {mode}")
    
    # --- Tube training (homeostasis) ---
    
    def compute_mse(self, mu: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        return ((trajectory - mu) ** 2).mean()
    
    def compute_log_volume(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.log(sigma).mean()
    
    def compute_bind_hard(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        inside = (torch.abs(trajectory - mu) < self.k_sigma * sigma).all(dim=-1).float()
        return inside.mean()
    
    def compute_soft_bind_fail(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        margin = self.k_sigma * sigma - torch.abs(trajectory - mu)
        p_dim = torch.sigmoid(self.gamma * margin)
        p_t = torch.prod(p_dim, dim=-1)
        bind_soft = p_t.mean()
        fail_soft = 1.0 - bind_soft
        return bind_soft, fail_soft
    
    def compute_residual(self, mu: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        return torch.abs(trajectory - mu).mean()
    
    def compute_kl(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2 * z_logstd)
        kl = 0.5 * (var + z_mu**2 - 1 - 2 * z_logstd).sum()
        return kl
    
    def train_step(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
        binding_result: BindingResult,
    ) -> Dict[str, float]:
        """
        Training step for both tube model and risk critic.
        
        Args:
            s0: Initial state
            trajectory: True trajectory [T, pred_dim]
            binding_result: Result from select_z (contains z*, mu, sigma)
        """
        self.train()
        
        z_star = binding_result.z_star
        mu = binding_result.mu
        sigma = binding_result.sigma
        
        # === Tube training (homeostasis) ===
        # Recompute with gradients
        z_mu, z_logstd = self.tube.encode(s0.unsqueeze(0))
        mu_grad, sigma_grad = self.tube.predict_tube(s0, z_star)
        mu_grad = mu_grad.squeeze(0)
        sigma_grad = sigma_grad.squeeze(0)
        
        mse = self.compute_mse(mu_grad, trajectory)
        log_vol = self.compute_log_volume(sigma_grad)
        bind_hard = self.compute_bind_hard(mu_grad, sigma_grad, trajectory)
        bind_soft, fail_soft = self.compute_soft_bind_fail(mu_grad, sigma_grad, trajectory)
        residual = self.compute_residual(mu_grad, trajectory)
        kl = self.compute_kl(z_mu.squeeze(0), z_logstd.squeeze(0))
        
        sigma_mean = float(sigma_grad.mean().item())
        residual_val = float(residual.item())
        fail_val = float(fail_soft.item())
        
        # Update running statistics
        self.stats_residual.update(residual_val)
        self.stats_sigma.update(sigma_mean)
        self.stats_fail.update(fail_val)
        
        hardness_raw = residual_val / (sigma_mean + 1e-6)
        self.stats_hardness.update(hardness_raw)
        hardness_z = self.stats_hardness.z_score(hardness_raw)
        
        # Tube loss
        loss = mse
        loss = loss + self.alpha_vol * log_vol
        loss = loss + self.lambda_fail * fail_soft
        loss = loss + self.beta_kl * kl
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.tube.parameters(), 1.0)
        self.optimizer.step()
        
        # Lambda update (homeostasis)
        surp_fail = max(0.0, -hardness_z)
        success_amt = 1.0 - fail_val
        delta = self.eta_lambda * (
            fail_val * (1.0 + surp_fail) -
            success_amt * 0.2
        )
        self.lambda_fail = float(np.clip(
            self.lambda_fail + delta,
            0.1,
            self.lambda_max
        ))
        
        # === Risk critic training ===
        risk_pred = self.risk_net(s0, z_star.detach())
        risk_target = fail_soft.detach()
        risk_loss = F.binary_cross_entropy(risk_pred, risk_target.unsqueeze(0))
        
        self.risk_optimizer.zero_grad()
        risk_loss.backward()
        nn.utils.clip_grad_norm_(self.risk_net.parameters(), 1.0)
        self.risk_optimizer.step()
        
        # Track metrics
        metrics = {
            "mse": float(mse.item()),
            "log_vol": float(log_vol.item()),
            "bind_hard": float(bind_hard.item()),
            "bind_soft": float(bind_soft.item()),
            "fail_soft": float(fail_soft.item()),
            "hardness": float(hardness_raw),
            "hardness_z": float(hardness_z),
            "lambda": float(self.lambda_fail),
            "kl": float(kl.item()),
            "loss": float(loss.item()),
            "z_norm": float(z_star.norm().item()),
            "sigma_mean": sigma_mean,
            "score": binding_result.score,
            "intent_proxy": binding_result.intent_proxy,
            "agency": binding_result.agency,
            "risk_pred": binding_result.risk_pred,
            "risk_loss": float(risk_loss.item()),
        }
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics
    
    def get_equilibrium_stats(self, window: int = 500) -> Dict[str, float]:
        if len(self.history["bind_hard"]) < window:
            window = len(self.history["bind_hard"])
        if window == 0:
            return {}
        
        return {
            "emergent_bind": float(np.mean(self.history["bind_hard"][-window:])),
            "bind_std": float(np.std(self.history["bind_hard"][-window:])),
            "mean_log_vol": float(np.mean(self.history["log_vol"][-window:])),
            "mean_lambda": float(np.mean(self.history["lambda"][-window:])),
            "lambda_stable": float(np.std(self.history["lambda"][-window:]) / 
                                  (np.mean(self.history["lambda"][-window:]) + 1e-8) < 0.2),
            "mean_fail": float(np.mean(self.history["fail_soft"][-window:])),
            "mean_sigma": float(np.mean(self.history["sigma_mean"][-window:])),
            "mean_score": float(np.mean(self.history["score"][-window:])),
            "mean_risk_loss": float(np.mean(self.history["risk_loss"][-window:])),
        }


def test_competitive_actor():
    """Quick test of competitive binding."""
    print("Testing CompetitiveActor...")
    
    actor = CompetitiveActor(
        obs_dim=8,
        pred_dim=2,
        z_dim=4,
        T=16,
    )
    
    # Test all binding modes
    s0 = torch.randn(8)
    trajectory = torch.randn(16, 2)
    
    for mode in [BindingMode.RANDOM, BindingMode.BEST_OF_K, BindingMode.CEM]:
        result = actor.select_z(s0, mode=mode)
        print(f"  {mode.value}: score={result.score:.3f}, "
              f"intent={result.intent_proxy:.3f}, agency={result.agency:.3f}, "
              f"risk={result.risk_pred:.3f}")
    
    # Test oracle
    result = actor.select_z(s0, mode=BindingMode.ORACLE, trajectory=trajectory)
    print(f"  oracle: score={result.score:.3f}")
    
    # Test training
    result = actor.select_z(s0, mode=BindingMode.CEM)
    metrics = actor.train_step(s0, trajectory, result)
    print(f"  train_step: loss={metrics['loss']:.3f}, risk_loss={metrics['risk_loss']:.3f}")
    
    print("✓ CompetitiveActor works!")
    return actor


if __name__ == "__main__":
    test_competitive_actor()
