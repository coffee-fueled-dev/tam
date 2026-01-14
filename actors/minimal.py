"""
Minimal actor.

Stripped down to the essential TAM components:
1. TubeNet: μ(s0, z), σ(z) - σ depends only on z
2. Encoder: s0 → (z_mu, z_logstd) proposal distribution  
3. Competitive binding: random, best_of_k, CEM
4. Training: Gaussian NLL + α log_vol (clean, interpretable)
5. Scoring: mode-prototype max-fit + agency (drives mode commitment)
"""

from enum import Enum
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BindingMode(Enum):
    RANDOM = "random"
    BEST_OF_K = "best_of_k"
    CEM = "cem"


class TubeNet(nn.Module):
    """
    Predicts trajectory tube: μ(s0, z), σ(z).
    
    Key constraint: σ depends ONLY on z (not s0).
    This forces the "port" z to determine acceptance geometry.
    """
    
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        pred_dim: int = 2,
        T: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.T = T
        self.pred_dim = pred_dim
        
        # μ: from (s0, z)
        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        
        # σ: from z ONLY (critical constraint)
        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        nn.init.zeros_(self.sigma_net[-1].bias)  # Start with unit σ
    
    def forward(
        self, s0: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: [B, T, pred_dim]
            sigma: [B, T, pred_dim]  (always positive)
        """
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        B = z.shape[0]
        if s0.shape[0] == 1 and B > 1:
            s0 = s0.expand(B, -1)
        
        # μ(s0, z)
        h = torch.cat([s0, z], dim=-1)
        mu = self.mu_net(h).view(B, self.T, self.pred_dim)
        
        # σ(z) only
        logsig = self.sigma_net(z).clamp(-4, 3)
        sigma = torch.exp(logsig).view(B, self.T, self.pred_dim)
        
        return mu, sigma


class Actor(nn.Module):
    """
    Minimal actor for testing topological commitment transfer.
    
    Training objective: Gaussian NLL + α log_vol
    Selection objective: mode-prototype max-fit + β agency
    
    This is the minimal mechanism that forces z to choose a mode:
    - Small σ → high agency but risky if τ doesn't match μ
    - Large σ → safe but low agency  
    - Mode-prototype scoring rewards committing to one side
    """
    
    def __init__(
        self,
        obs_dim: int = 4,
        z_dim: int = 4,
        pred_dim: int = 2,
        T: int = 16,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        # Loss weights
        alpha_vol: float = 0.5,      # Volume penalty weight
        beta_kl: float = 0.001,      # KL regularization
        alpha_fail: float = 1.0,     # Phase 1: Max-fail penalty weight
        # Phase 2: Tube objective improvements
        alpha_smooth: float = 0.1,   # σ smoothness regularizer weight
        use_global_vol: bool = True, # Use sum instead of mean for volume
        max_fail_gamma: float = 5.0, # Softmax temperature for worst-timestep fail
        # CEM parameters
        cem_iters: int = 4,
        cem_samples: int = 128,
        cem_elites: int = 16,
        cem_smoothing: float = 0.25,
        cem_std_floor: float = 0.2,
        # Scoring weights
        agency_weight: float = 0.5,  # Weight on -log(vol) in CEM score
        k_sigma: float = 1.0,        # Phase 1: Lowered from 1.5 to make hedging harder
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.pred_dim = pred_dim
        self.T = T
        
        self.alpha_vol = alpha_vol
        self.beta_kl = beta_kl
        self.alpha_fail = alpha_fail
        self.k_sigma = k_sigma
        
        # Phase 2 parameters
        self.alpha_smooth = alpha_smooth
        self.use_global_vol = use_global_vol
        self.max_fail_gamma = max_fail_gamma
        
        # CEM
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_elites = cem_elites
        self.cem_smoothing = cem_smoothing
        self.cem_std_floor = cem_std_floor
        self.agency_weight = agency_weight
        
        # Networks
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logstd_head = nn.Linear(hidden_dim, z_dim)
        
        self.tube_net = TubeNet(obs_dim, z_dim, pred_dim, T, hidden_dim)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # History for tracking
        self.history: Dict[str, List[float]] = defaultdict(list)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    # =========================================================================
    # Encoding & Tube Prediction
    # =========================================================================
    
    def encode(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode s0 → proposal distribution q(z|s0)."""
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        h = self.encoder(s0)
        z_mu = self.z_mu_head(h)
        z_logstd = self.z_logstd_head(h).clamp(-4, 2)
        return z_mu, z_logstd
    
    def sample_z(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample from q(z|s0)."""
        std = torch.exp(z_logstd)
        eps = torch.randn_like(std)
        return z_mu + std * eps
    
    def predict_tube(
        self, s0: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict tube given state and commitment."""
        return self.tube_net(s0, z)
    
    # =========================================================================
    # Mode Measurement (for bimodal experiment)
    # =========================================================================
    
    def compute_signed_deviation(
        self, mu: torch.Tensor, s0: torch.Tensor, t_star: int = None
    ) -> torch.Tensor:
        """
        Compute signed perpendicular deviation d(z) at t*.
        
        For bimodal environment where s0 = [x, y, goal_x, goal_y] (2D):
        - d > 0 → committed to mode +1 (bend left)
        - d < 0 → committed to mode -1 (bend right)
        - d ≈ 0 → hedging (midline)
        
        For general environments, returns deviation from midpoint of
        start→goal trajectory.
        """
        if t_star is None:
            t_star = self.T // 2
        
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
        
        # Handle different observation formats
        obs_dim = s0.shape[1]
        
        if obs_dim >= 4 and self.pred_dim == 2:
            # Bimodal 2D format: s0 = [x, y, goal_x, goal_y, ...]
            start = s0[:, :2]
            goal = s0[:, 2:4]
            
            direction = goal - start
            dist = torch.norm(direction, dim=-1, keepdim=True) + 1e-6
            d_unit = direction / dist
            
            # Perpendicular direction (rotate 90°)
            d_perp = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=-1)
            
            # Expected baseline position at t*
            progress = (t_star + 1) / self.T
            expected = start + direction * progress
            
            # Actual μ at t*
            mu_t_star = mu[:, t_star, :2]  # Only first 2 dims
            
            # Signed projection onto perpendicular
            deviation = mu_t_star - expected
            d_signed = (deviation * d_perp).sum(dim=-1)
        else:
            # General format: return scalar deviation magnitude (unsigned)
            # Assumes s0 = [state, goal] with state and goal same dim
            state_dim = obs_dim // 2
            if state_dim * 2 <= obs_dim:
                start = s0[:, :state_dim]
                goal = s0[:, state_dim:state_dim*2]
                
                # Expected position at t*
                progress = (t_star + 1) / self.T
                expected = start + (goal - start) * progress
                
                # Actual μ at t*
                mu_t_star = mu[:, t_star, :state_dim]
                
                # Euclidean deviation (not signed in general case)
                deviation = mu_t_star - expected
                d_signed = torch.norm(deviation, dim=-1)
            else:
                # Fallback: return zero
                d_signed = torch.zeros(s0.shape[0], device=s0.device)
        
        return d_signed
    
    def compute_mci(self, d_signed: torch.Tensor, bend_amplitude: float) -> torch.Tensor:
        """Mode Commitment Index: MCI = |d| / A (normalized commitment)."""
        return torch.abs(d_signed) / bend_amplitude
    
    # =========================================================================
    # Scoring for CEM (mode-prototype max-fit + agency)
    # =========================================================================
    
    def score_commitment(
        self,
        s0: torch.Tensor,
        z: torch.Tensor,
        mode_prototypes: Tuple[torch.Tensor, ...] = None,
        bend_amplitude: float = 0.15,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Score a commitment z for CEM selection.
        
        Score = max_fit_to_prototype + agency
        
        This forces z to commit to ONE mode:
        - Hedging μ has poor fit to all prototypes
        - Committed μ matches one prototype well and can shrink σ
        
        Args:
            s0: Current state
            z: Candidate commitment(s) [B, z_dim]
            mode_prototypes: Tuple of K prototype trajectories (τ_0, τ_1, ..., τ_{K-1})
            bend_amplitude: For computing mode penalty if no prototypes
        """
        mu, sigma = self.predict_tube(s0, z)
        
        # Agency: higher score for smaller σ
        log_vol = torch.log(sigma).mean(dim=(1, 2)) if sigma.dim() == 3 else torch.log(sigma).mean()
        agency = -log_vol
        
        if mode_prototypes is not None:
            # Mode-prototype max-fit scoring
            # S(z) = max_k{-d(μ(z), τ_k)} + β*agency
            fits = []
            for tau in mode_prototypes:
                if tau.dim() == 2:
                    tau = tau.unsqueeze(0)
                # Negative MSE to this prototype (higher = better fit)
                fit = -((mu - tau) ** 2).mean(dim=(1, 2))
                fits.append(fit)
            
            fits = torch.stack(fits, dim=1)  # [B, K]
            
            # Max-fit: reward fitting ONE mode well
            mode_fit, best_mode = fits.max(dim=1)
            
            score = mode_fit + self.agency_weight * agency
            
            details = {
                "mu": mu, "sigma": sigma, "agency": agency,
                "fits": fits, "mode_fit": mode_fit, "best_mode": best_mode,
            }
        else:
            # Fallback: penalize staying near midline
            d_signed = self.compute_signed_deviation(mu, s0)
            mode_penalty = torch.exp(-torch.abs(d_signed) / (bend_amplitude * 0.5))
            
            score = agency - mode_penalty
            
            details = {
                "mu": mu, "sigma": sigma, "agency": agency,
                "d_signed": d_signed, "mode_penalty": mode_penalty,
            }
        
        return score, details
    
    # =========================================================================
    # Binding Modes
    # =========================================================================
    
    def select_z_random(self, s0: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Random binding: single sample from q(z|s0)."""
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            z = self.sample_z(z_mu, z_logstd).squeeze(0)
            mu, sigma = self.predict_tube(s0, z)
            return z, {"mu": mu.squeeze(0), "sigma": sigma.squeeze(0)}
    
    def select_z_best_of_k(
        self,
        s0: torch.Tensor,
        k: int = 32,
        mode_prototypes: Tuple[torch.Tensor, torch.Tensor] = None,
        bend_amplitude: float = 0.15,
    ) -> Tuple[torch.Tensor, Dict]:
        """Best-of-K binding: sample K, take highest score."""
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            z_mu = z_mu.squeeze(0)
            z_logstd = z_logstd.squeeze(0)
            
            # Sample K candidates
            eps = torch.randn(k, self.z_dim, device=s0.device)
            z_candidates = z_mu.unsqueeze(0) + torch.exp(z_logstd).unsqueeze(0) * eps
            
            # Score
            scores, details = self.score_commitment(s0, z_candidates, mode_prototypes, bend_amplitude)
            
            # Best
            best_idx = scores.argmax()
            z_star = z_candidates[best_idx]
            
            return z_star, {
                "mu": details["mu"][best_idx],
                "sigma": details["sigma"][best_idx],
                "score": float(scores[best_idx].item()),
            }
    
    def select_z_cem(
        self,
        s0: torch.Tensor,
        mode_prototypes: Tuple[torch.Tensor, torch.Tensor] = None,
        bend_amplitude: float = 0.15,
    ) -> Tuple[torch.Tensor, Dict]:
        """CEM binding: iterative refinement in z-space."""
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            cem_mean = z_mu.squeeze(0).clone()
            cem_std = torch.exp(z_logstd.squeeze(0)).clamp(min=self.cem_std_floor)
            
            for _ in range(self.cem_iters):
                # Sample
                eps = torch.randn(self.cem_samples, self.z_dim, device=s0.device)
                z_candidates = cem_mean.unsqueeze(0) + cem_std.unsqueeze(0) * eps
                
                # Score
                scores, _ = self.score_commitment(s0, z_candidates, mode_prototypes, bend_amplitude)
                
                # Elites
                elite_indices = scores.argsort(descending=True)[:self.cem_elites]
                elite_z = z_candidates[elite_indices]
                
                # Refit
                new_mean = elite_z.mean(dim=0)
                new_std = elite_z.std(dim=0).clamp(min=self.cem_std_floor)
                
                cem_mean = (1 - self.cem_smoothing) * cem_mean + self.cem_smoothing * new_mean
                cem_std = (1 - self.cem_smoothing) * cem_std + self.cem_smoothing * new_std
            
            # Final best
            eps = torch.randn(self.cem_samples, self.z_dim, device=s0.device)
            z_candidates = cem_mean.unsqueeze(0) + cem_std.unsqueeze(0) * eps
            scores, details = self.score_commitment(s0, z_candidates, mode_prototypes, bend_amplitude)
            best_idx = scores.argmax()
            z_star = z_candidates[best_idx]
            
            return z_star, {
                "mu": details["mu"][best_idx],
                "sigma": details["sigma"][best_idx],
                "score": float(scores[best_idx].item()),
            }
    
    def select_z(
        self,
        s0: torch.Tensor,
        mode: BindingMode = BindingMode.CEM,
        mode_prototypes: Tuple[torch.Tensor, torch.Tensor] = None,
        bend_amplitude: float = 0.15,
        k: int = 32,
    ) -> Tuple[torch.Tensor, Dict]:
        """Unified binding interface."""
        if mode == BindingMode.RANDOM:
            return self.select_z_random(s0)
        elif mode == BindingMode.BEST_OF_K:
            return self.select_z_best_of_k(s0, k, mode_prototypes, bend_amplitude)
        elif mode == BindingMode.CEM:
            return self.select_z_cem(s0, mode_prototypes, bend_amplitude)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # =========================================================================
    # Bind/Fail Metrics
    # =========================================================================
    
    def compute_bind_hard(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Hard bind: all points within k_sigma * σ."""
        inside = (torch.abs(trajectory - mu) < self.k_sigma * sigma).all(dim=-1).float()
        return inside.mean()
    
    def compute_bind_max(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 2.10: Max-like bind focusing on worst timestep.
        
        Uses softmax-weighted aggregation so gradients flow through worst timesteps.
        Returns the bind rate weighted by how "bad" each timestep is.
        """
        # Normalized deviation at each timestep
        deviation = torch.abs(trajectory - mu)  # (T, pred_dim)
        normalized = deviation / (self.k_sigma * sigma + 1e-8)  # (T, pred_dim)
        max_dev_per_t = normalized.max(dim=-1).values  # (T,) worst dim per timestep
        
        # Softmax weights (higher weight on worse timesteps)
        weights = torch.softmax(self.max_fail_gamma * max_dev_per_t, dim=0)
        
        # Inside check per timestep
        inside = (deviation < self.k_sigma * sigma).all(dim=-1).float()  # (T,)
        
        # Weighted bind rate (focuses on worst timesteps)
        return (weights * inside).sum()
    
    def compute_nll(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian NLL: (τ - μ)² / σ² + 2 log σ."""
        residual_sq = (trajectory - mu) ** 2
        nll = (residual_sq / (sigma ** 2) + 2 * torch.log(sigma)).mean()
        return nll
    
    def compute_kl(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        """KL(q(z|s0) || N(0,I))."""
        var = torch.exp(2 * z_logstd)
        return 0.5 * (var + z_mu**2 - 1 - 2 * z_logstd).sum()
    
    # =========================================================================
    # Training (Gaussian NLL + α log_vol)
    # =========================================================================
    
    def train_step(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
        z: torch.Tensor,
        bend_amplitude: float = 0.15,
    ) -> Dict[str, float]:
        """
        Training step with Phase 2 improvements.
        
        Loss = NLL + α log_vol + β KL + α_smooth L_smooth
        
        Phase 2 improvements:
        - σ smoothness: penalize rapid σ changes over time
        - Global volume: sum instead of mean
        - Max-like failure: focus on worst timestep
        """
        self.train()
        
        # Forward
        z_mu, z_logstd = self.encode(s0)
        mu, sigma = self.predict_tube(s0, z)
        
        if mu.dim() == 3:
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)
        
        # Losses
        nll = self.compute_nll(mu, sigma, trajectory)
        
        # Phase 1: TRUE global volume = sum over ALL (t, d)
        # This makes "inflate everywhere" scale with T×d
        if self.use_global_vol:
            log_vol = torch.log(sigma).sum()  # Sum over ALL dimensions
        else:
            log_vol = torch.log(sigma).mean()
        
        # Phase 1: Soft max-fail term in loss (not just metrics)
        # Penalize worst-timestep violation with differentiable soft-max
        deviation = torch.abs(trajectory - mu)  # (T, pred_dim)
        normalized_dev = deviation / (self.k_sigma * sigma + 1e-8)  # (T, pred_dim)
        max_dev_per_t = normalized_dev.max(dim=-1).values  # (T,) worst dim per timestep
        # Soft-max over time: focuses on worst timestep
        soft_max_fail = torch.logsumexp(self.max_fail_gamma * max_dev_per_t, dim=0) / self.max_fail_gamma
        
        # Phase 2.8: σ smoothness regularizer
        # Penalize rapid changes: L_smooth = mean_t ||logσ_{t+1} - logσ_t||²
        log_sigma = torch.log(sigma)  # (T, pred_dim)
        sigma_diff = log_sigma[1:] - log_sigma[:-1]  # (T-1, pred_dim)
        smooth_loss = (sigma_diff ** 2).mean()
        
        kl = self.compute_kl(z_mu.squeeze(0), z_logstd.squeeze(0))
        
        # Combined loss: NLL + vol + fail + KL + smooth
        loss = (nll + self.alpha_vol * log_vol + self.alpha_fail * soft_max_fail 
                + self.beta_kl * kl + self.alpha_smooth * smooth_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Phase 2.10: Metrics with max-like failure
        bind_hard = self.compute_bind_hard(mu, sigma, trajectory)
        bind_max = self.compute_bind_max(mu, sigma, trajectory)  # Worst timestep
        mse = ((trajectory - mu) ** 2).mean()
        d_signed = self.compute_signed_deviation(mu.unsqueeze(0), s0)
        mci = self.compute_mci(d_signed, bend_amplitude)
        
        metrics = {
            "loss": float(loss.item()),
            "nll": float(nll.item()),
            "log_vol": float(log_vol.item()),
            "soft_max_fail": float(soft_max_fail.item()),
            "kl": float(kl.item()),
            "smooth_loss": float(smooth_loss.item()),
            "mse": float(mse.item()),
            "bind_hard": float(bind_hard.item()),
            "bind_max": float(bind_max.item()),
            "mci": float(mci.item()),
            "d_signed": float(d_signed.item()),
        }
        
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics


# =============================================================================
# Testing
# =============================================================================

def test_minimal_actor():
    """Quick test of Actor."""
    print("Testing Actor...")
    
    actor = Actor(obs_dim=4, z_dim=4, T=16)
    
    # Test encoding
    s0 = torch.tensor([0.5, 0.5, 0.8, 0.8])
    z_mu, z_logstd = actor.encode(s0)
    print(f"  encode: z_mu shape={z_mu.shape}, z_logstd shape={z_logstd.shape}")
    
    # Test tube prediction
    z = actor.sample_z(z_mu, z_logstd)
    mu, sigma = actor.predict_tube(s0, z)
    print(f"  predict_tube: mu shape={mu.shape}, sigma shape={sigma.shape}")
    
    # Test binding modes
    z_rand, _ = actor.select_z_random(s0)
    print(f"  select_z_random: z shape={z_rand.shape}")
    
    z_bok, _ = actor.select_z_best_of_k(s0, k=16)
    print(f"  select_z_best_of_k: z shape={z_bok.shape}")
    
    z_cem, details = actor.select_z_cem(s0)
    print(f"  select_z_cem: z shape={z_cem.shape}, score={details.get('score', 'N/A')}")
    
    # Test training
    trajectory = torch.randn(16, 2) * 0.1 + 0.5
    metrics = actor.train_step(s0, trajectory, z_cem)
    print(f"  train_step: loss={metrics['loss']:.4f}, bind={metrics['bind_hard']:.3f}")
    
    # Test mode measurement
    d_signed = actor.compute_signed_deviation(details["mu"].unsqueeze(0), s0)
    print(f"  compute_signed_deviation: d={d_signed.item():.4f}")
    
    print("✓ Actor works!")


if __name__ == "__main__":
    test_minimal_actor()
