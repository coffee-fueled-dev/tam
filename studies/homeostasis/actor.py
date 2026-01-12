"""
Self-Calibrating Homeostasis Actor: Optimal tubes via commitment market.

Key insight: Instead of a fixed bind target ρ, let the agent discover the
optimal operating point where marginal benefit of tightening equals
marginal cost of failures.

Free energy formulation:
    L = L_mu + α * V(Φ) + λ * Fail_soft
    
λ update based on SURPRISE of failure (continuous):
    δ = η * (fail_amt * (1 + surp_fail) - success_amt * 0.2)
    
Key fixes (from diagnosis):
1. Use SOFT fail for gradients (differentiable)
2. Use CONTINUOUS fail amount in λ update (not boolean)
3. Normalize hardness by sigma (creates negative feedback when σ too small)

Domain-independent via z-scores:
- All metrics normalized by running statistics
- Hardness, surprise computed in z-score space
- No domain-specific hyperparameters needed
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
            # Welford's online variance (with momentum)
            self.var = self.momentum * self.var + (1 - self.momentum) * (delta ** 2)
    
    @property
    def std(self) -> float:
        if self.var is None:
            return 1.0
        return max(np.sqrt(self.var), 1e-6)
    
    def z_score(self, value: float) -> float:
        """Return z-score: (value - mean) / std."""
        if self.mean is None:
            return 0.0
        return (value - self.mean) / self.std
    
    def quantile(self, value: float) -> float:
        """Approximate quantile using normal CDF on z-score."""
        z = self.z_score(value)
        # Approximate normal CDF
        return 0.5 * (1 + np.tanh(z * 0.7978845608))  # Approximation


@dataclass
class TubeOutput:
    """Output from tube prediction."""
    mu: torch.Tensor      # [T, pred_dim] trajectory mean
    sigma: torch.Tensor   # [T, pred_dim] tube width (std)
    z: torch.Tensor       # [z_dim] latent commitment
    z_mu: torch.Tensor    # [z_dim] posterior mean
    z_logstd: torch.Tensor  # [z_dim] posterior log-std


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
        
        # Initialize sigma to moderate values
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
        B = s0.shape[0]
        
        mu = self.mu_net(torch.cat([s0, z], dim=-1))
        mu = mu.view(B, self.T, self.pred_dim)
        
        # Wide clamp range
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


class Actor(nn.Module):
    """
    Actor with self-calibrating homeostasis.
    
    No fixed bind target. λ adjusts based on failure surprise.
    Uses SOFT fail for gradients and CONTINUOUS fail amount in λ update.
    
    Domain-independent via z-score normalization:
    - Hardness computed as z-score of residual/sigma ratio
    - Surprise computed relative to running statistics
    - No domain-specific hyperparameters needed
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
        # Homeostasis parameters (domain-independent defaults)
        alpha_vol: float = 0.5,      # Volume minimization weight
        beta_kl: float = 0.001,      # KL weight
        eta_lambda: float = 0.05,    # λ learning rate
        lambda_init: float = 1.0,    # Initial λ
        lambda_max: float = 50.0,    # Cap λ
        # Soft bind parameters
        gamma: float = 25.0,         # Sharpness of soft bind sigmoid
        # Running stats momentum
        stats_momentum: float = 0.99,
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
        self.gamma = gamma  # Sigmoid sharpness for soft bind
        
        # Networks
        self.tube = TubeNet(obs_dim, z_dim, pred_dim, hidden_dim, T)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Dual variable - price of failure
        self.lambda_fail = lambda_init
        
        # Running statistics for domain-independent normalization
        self.stats_residual = RunningStats(stats_momentum)
        self.stats_sigma = RunningStats(stats_momentum)
        self.stats_hardness = RunningStats(stats_momentum)  # residual / sigma
        self.stats_fail = RunningStats(stats_momentum)
        
        # History
        self.history: Dict[str, List[float]] = {
            "mse": [],
            "log_vol": [],
            "bind_hard": [],
            "bind_soft": [],
            "fail_soft": [],
            "hardness": [],
            "hardness_z": [],  # z-score of hardness
            "lambda": [],
            "kl": [],
            "loss": [],
            "z_norm": [],
            "sigma_mean": [],
        }
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_commitment(self, s0: torch.Tensor) -> TubeOutput:
        return self.tube(s0)
    
    def compute_mse(self, mu: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        return ((trajectory - mu) ** 2).mean()
    
    def compute_log_volume(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.log(sigma).mean()
    
    def compute_bind_hard(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Hard bind: fraction of timesteps inside kσ box (for reporting only)."""
        inside = (torch.abs(trajectory - mu) < self.k_sigma * sigma).all(dim=-1).float()
        return inside.mean()
    
    def compute_soft_bind_fail(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable soft bind/fail using sigmoid.
        
        margin = k*σ - |τ - μ|  (positive means inside)
        p_dim = sigmoid(γ * margin)  (smooth 0→1)
        p_t = prod_d(p_dim)  (all dims must be inside)
        bind_soft = mean_t(p_t)
        fail_soft = 1 - bind_soft
        """
        margin = self.k_sigma * sigma - torch.abs(trajectory - mu)  # [T, D]
        p_dim = torch.sigmoid(self.gamma * margin)  # [T, D]
        p_t = torch.prod(p_dim, dim=-1)  # [T]
        bind_soft = p_t.mean()
        fail_soft = 1.0 - bind_soft
        return bind_soft, fail_soft
    
    def compute_residual(self, mu: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        """Mean absolute residual |τ - μ|."""
        return torch.abs(trajectory - mu).mean()
    
    def compute_hardness(self, residual: float, sigma_mean: float) -> float:
        """
        Hardness normalized by sigma (creates negative feedback).
        
        If sigma is small, hardness is high → λ increases → sigma grows.
        This prevents collapse to floor.
        """
        return residual / (sigma_mean + 1e-6)
    
    def compute_kl(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2 * z_logstd)
        kl = 0.5 * (var + z_mu**2 - 1 - 2 * z_logstd).sum()
        return kl
    
    def train_step(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Training step with self-calibrating homeostasis.
        
        Domain-independent via z-score normalization:
        1. Use soft fail in loss (differentiable)
        2. Use continuous fail_amt in λ update
        3. Hardness computed as z-score (domain-independent)
        4. Surprise based on z-score deviation
        """
        self.train()
        
        # Forward pass
        out = self.tube(s0)
        
        # Compute metrics
        mse = self.compute_mse(out.mu, trajectory)
        log_vol = self.compute_log_volume(out.sigma)
        bind_hard = self.compute_bind_hard(out.mu, out.sigma, trajectory)
        bind_soft, fail_soft = self.compute_soft_bind_fail(out.mu, out.sigma, trajectory)
        residual = self.compute_residual(out.mu, trajectory)
        kl = self.compute_kl(out.z_mu, out.z_logstd)
        
        sigma_mean = float(out.sigma.mean().item())
        residual_val = float(residual.item())
        fail_val = float(fail_soft.item())
        
        # Update running statistics (domain adaptation)
        self.stats_residual.update(residual_val)
        self.stats_sigma.update(sigma_mean)
        self.stats_fail.update(fail_val)
        
        # Compute hardness in original units, then z-score it
        hardness_raw = residual_val / (sigma_mean + 1e-6)
        self.stats_hardness.update(hardness_raw)
        hardness_z = self.stats_hardness.z_score(hardness_raw)
        
        # Loss: use SOFT fail for gradients
        loss = mse
        loss = loss + self.alpha_vol * log_vol
        loss = loss + self.lambda_fail * fail_soft  # Soft fail for gradients!
        loss = loss + self.beta_kl * kl
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # λ update using Z-SCORES (domain-independent)
        # Fail z-score: how unusual is this failure rate?
        fail_z = self.stats_fail.z_score(fail_val)
        
        # Surprise = negative hardness z-score
        # Low hardness z-score (easy) + failure = high surprise
        # High hardness z-score (hard) + failure = expected
        surp_fail = max(0.0, -hardness_z)  # Easy (below mean) = high surprise
        
        # λ update: increase on surprising failures, decrease on expected successes
        success_amt = 1.0 - fail_val
        delta = self.eta_lambda * (
            fail_val * (1.0 + surp_fail) -  # Failures push λ up (more if surprising)
            success_amt * 0.2                # Successes push λ down
        )
        
        self.lambda_fail = float(np.clip(
            self.lambda_fail + delta,
            0.1,  # Minimum
            self.lambda_max
        ))
        
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
            "z_norm": float(out.z.norm().item()),
            "sigma_mean": sigma_mean,
        }
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics
    
    def get_equilibrium_stats(self, window: int = 500) -> Dict[str, float]:
        """Get stats about the emergent equilibrium."""
        if len(self.history["bind_hard"]) < window:
            window = len(self.history["bind_hard"])
        
        if window == 0:
            return {}
        
        recent_bind = np.array(self.history["bind_hard"][-window:])
        recent_vol = np.array(self.history["log_vol"][-window:])
        recent_lambda = np.array(self.history["lambda"][-window:])
        recent_fail = np.array(self.history["fail_soft"][-window:])
        recent_sigma = np.array(self.history["sigma_mean"][-window:])
        
        return {
            "emergent_bind": float(np.mean(recent_bind)),
            "bind_std": float(np.std(recent_bind)),
            "mean_log_vol": float(np.mean(recent_vol)),
            "mean_lambda": float(np.mean(recent_lambda)),
            "lambda_std": float(np.std(recent_lambda)),
            "lambda_stable": float(np.std(recent_lambda) / (np.mean(recent_lambda) + 1e-8) < 0.2),
            "mean_fail": float(np.mean(recent_fail)),
            "mean_sigma": float(np.mean(recent_sigma)),
        }


def test_self_calibrating_actor():
    """Quick test."""
    print("Testing Actor with z-score normalization...")
    
    actor = Actor(
        obs_dim=8,
        pred_dim=2,
        z_dim=4,
        T=16,
        alpha_vol=0.5,
    )
    
    # Run a few steps to build up running statistics
    for _ in range(10):
        s0 = torch.randn(8)
        trajectory = torch.randn(16, 2)
        metrics = actor.train_step(s0, trajectory)
    
    print(f"  bind_hard: {metrics['bind_hard']:.3f}, bind_soft: {metrics['bind_soft']:.3f}")
    print(f"  fail_soft: {metrics['fail_soft']:.3f}")
    print(f"  hardness: {metrics['hardness']:.3f}, hardness_z: {metrics['hardness_z']:.3f}")
    print(f"  sigma_mean: {metrics['sigma_mean']:.3f}, λ: {metrics['lambda']:.3f}")
    print(f"  Running stats - residual μ={actor.stats_residual.mean:.3f}, σ={actor.stats_residual.std:.3f}")
    
    print("✓ Actor with z-scores works!")
    return actor


if __name__ == "__main__":
    test_self_calibrating_actor()
