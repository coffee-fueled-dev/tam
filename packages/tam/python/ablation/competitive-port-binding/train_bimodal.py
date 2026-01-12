"""
Bimodal Commitment Experiment: Testing if CEM chooses one future vs hedging.

This experiment tests the core TAM claim:
- In a bimodal world where two incompatible futures are equally predictable,
- Competitive binding (CEM) should select a low-volume commitment aligned with ONE mode
- Rather than expanding acceptance to cover BOTH (hedging)

Key metrics:
- MCI (Mode Commitment Index): How much the tube commits to one side
- mode_match: Whether the chosen side matches the realized mode
- Volume: Tighter tubes = more agency

Expected results:
- CEM: High MCI, low volume, bimodal d(z) histogram
- Random: Low MCI (midline), higher volume
- Hedge: Lowest MCI, highest bind, worst agency
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from actor import CompetitiveActor, BindingMode, TubeNet, RiskNet, RunningStats


class BimodalEnv:
    """
    Environment with truly bimodal trajectories.
    
    Same s0 can produce two equally likely, mutually incompatible futures.
    The mode m ∈ {+1, -1} determines which future is realized.
    """
    
    def __init__(
        self,
        seed: int = 0,
        bend_amplitude: float = 0.15,  # How far modes diverge
        base_noise: float = 0.02,      # Small noise on both modes
    ):
        self.rng = np.random.default_rng(seed)
        self.bend_amplitude = bend_amplitude
        self.base_noise = base_noise
        self.mode = None  # Hidden mode: +1 or -1
        self.reset()
    
    def reset(self, force_mode: int = None) -> np.ndarray:
        """Reset and sample a new hidden mode."""
        self.x = self.rng.uniform(0.2, 0.8)
        self.y = self.rng.uniform(0.2, 0.8)
        self.goal_x = self.rng.uniform(0.2, 0.8)
        self.goal_y = self.rng.uniform(0.2, 0.8)
        
        # Sample hidden mode (50/50)
        if force_mode is not None:
            self.mode = force_mode
        else:
            self.mode = self.rng.choice([-1, 1])
        
        return self.observe()
    
    def observe(self) -> np.ndarray:
        """Observation does NOT include mode - that's the hidden stochasticity."""
        return np.array([
            self.x, self.y, self.goal_x, self.goal_y
        ], dtype=np.float32)
    
    def get_baseline_direction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get baseline direction and perpendicular."""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        
        # Unit direction
        d_unit = np.array([dx / dist, dy / dist])
        # Perpendicular (rotate 90 degrees)
        d_perp = np.array([-d_unit[1], d_unit[0]])
        
        return d_unit, d_perp
    
    def bend_function(self, t: float, T: int) -> float:
        """
        Bend function that peaks at mid-horizon and returns to zero at end.
        t ∈ [0, T], returns bend magnitude.
        """
        progress = t / T
        # Sinusoidal: peaks at t=T/2, zero at t=0 and t=T
        return np.sin(np.pi * progress)
    
    def generate_trajectory(self, T: int = 16) -> np.ndarray:
        """Generate trajectory based on hidden mode."""
        traj = np.zeros((T, 2), dtype=np.float32)
        
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        _, d_perp = self.get_baseline_direction()
        
        for t in range(T):
            progress = (t + 1) / T
            
            # Baseline straight-line
            base_x = self.x + dx * progress
            base_y = self.y + dy * progress
            
            # Mode-dependent bend (perpendicular deviation)
            bend = self.bend_function(t + 1, T) * self.bend_amplitude * self.mode
            
            # Add bend in perpendicular direction
            traj[t, 0] = base_x + bend * d_perp[0]
            traj[t, 1] = base_y + bend * d_perp[1]
            
            # Small base noise
            traj[t, 0] += self.rng.normal(0, self.base_noise)
            traj[t, 1] += self.rng.normal(0, self.base_noise)
        
        return traj
    
    def generate_both_modes(self, T: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trajectories for both modes (for visualization)."""
        original_mode = self.mode
        
        self.mode = 1
        traj_plus = self.generate_trajectory(T)
        
        self.mode = -1
        traj_minus = self.generate_trajectory(T)
        
        self.mode = original_mode
        return traj_plus, traj_minus
    
    def get_mode_separation(self, T: int = 16) -> float:
        """Get the separation between modes at mid-horizon."""
        t_star = T // 2
        _, d_perp = self.get_baseline_direction()
        
        # Separation = 2 * bend_amplitude * bend(t_star)
        bend = self.bend_function(t_star, T)
        separation = 2 * self.bend_amplitude * bend
        
        return separation


class BimodalActor(nn.Module):
    """
    Actor for bimodal environment.
    
    Key difference: μ predicts deviation from baseline, z controls deviation sign.
    This allows competitive binding to choose which mode to commit to.
    """
    
    def __init__(
        self,
        obs_dim: int = 4,
        z_dim: int = 4,
        pred_dim: int = 2,
        hidden_dim: int = 64,
        T: int = 16,
        k_sigma: float = 1.5,
        lr: float = 1e-3,
        # Homeostasis
        alpha_vol: float = 0.5,
        beta_kl: float = 0.001,
        eta_lambda: float = 0.05,
        lambda_init: float = 1.0,
        lambda_max: float = 50.0,
        gamma: float = 25.0,
        # CEM
        cem_iters: int = 4,
        cem_samples: int = 128,
        cem_elites: int = 16,
        cem_smoothing: float = 0.25,
        cem_std_floor: float = 0.2,
        # Scoring
        intent_weight: float = 1.0,
        agency_weight: float = 0.5,
        risk_weight: float = 1.0,
        mode_commit_weight: float = 0.5,  # Weight on mode commitment term
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.pred_dim = pred_dim
        self.T = T
        self.k_sigma = k_sigma
        
        # Homeostasis
        self.alpha_vol = alpha_vol
        self.beta_kl = beta_kl
        self.eta_lambda = eta_lambda
        self.lambda_max = lambda_max
        self.gamma = gamma
        
        # CEM
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_elites = cem_elites
        self.cem_smoothing = cem_smoothing
        self.cem_std_floor = cem_std_floor
        
        # Scoring
        self.intent_weight = intent_weight
        self.agency_weight = agency_weight
        self.risk_weight = risk_weight
        self.mode_commit_weight = mode_commit_weight
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logstd_head = nn.Linear(hidden_dim, z_dim)
        
        # Baseline predictor (deterministic, from s0)
        self.baseline_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        
        # Deviation predictor (from z ONLY) - z controls mode
        # By making deviation ONLY depend on z, we force z to control which mode
        self.deviation_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Bounded output to encourage diverse deviations
            nn.Linear(hidden_dim, T * pred_dim),
        )
        # Initialize with large weights to encourage deviation from midline
        nn.init.xavier_uniform_(self.deviation_net[0].weight, gain=2.0)
        nn.init.xavier_uniform_(self.deviation_net[2].weight, gain=2.0)
        
        # Sigma network (from z only)
        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        nn.init.zeros_(self.sigma_net[-1].bias)
        
        # Risk network
        self.risk_net = RiskNet(obs_dim, z_dim, hidden_dim)
        
        # Optimizers
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.z_mu_head.parameters()) +
            list(self.z_logstd_head.parameters()) +
            list(self.baseline_net.parameters()) +
            list(self.deviation_net.parameters()) +
            list(self.sigma_net.parameters()),
            lr=lr
        )
        self.risk_optimizer = optim.Adam(self.risk_net.parameters(), lr=lr)
        
        # Dual variable
        self.lambda_fail = lambda_init
        
        # Running stats
        self.stats_hardness = RunningStats()
        
        # History
        self.history: Dict[str, List[float]] = {
            "mse": [], "log_vol": [], "bind_hard": [], "bind_soft": [],
            "fail_soft": [], "lambda": [], "kl": [], "loss": [],
            "mci": [],  # Mode Commitment Index
            "d_signed": [],  # Signed deviation
            "risk_loss": [], "score": [],
        }
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        h = self.encoder(s0)
        z_mu = self.z_mu_head(h)
        z_logstd = self.z_logstd_head(h).clamp(-4, 2)
        return z_mu, z_logstd
    
    def sample_z(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        std = torch.exp(z_logstd)
        eps = torch.randn_like(std)
        return z_mu + std * eps
    
    def predict_tube(
        self, s0: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict tube: μ = baseline + deviation(z), σ = f(z)
        """
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        B = z.shape[0]
        if s0.shape[0] == 1 and B > 1:
            s0 = s0.expand(B, -1)
        
        # Baseline (deterministic from s0)
        baseline = self.baseline_net(s0).view(B, self.T, self.pred_dim)
        
        # Deviation (controlled by z ONLY - not s0)
        deviation = self.deviation_net(z)
        deviation = deviation.view(B, self.T, self.pred_dim)
        
        # μ = baseline + deviation
        mu = baseline + deviation
        
        # σ from z only
        logsig = self.sigma_net(z).clamp(-4, 3)
        sigma = torch.exp(logsig.view(B, self.T, self.pred_dim))
        
        return mu, sigma
    
    def compute_signed_deviation(
        self, mu: torch.Tensor, s0: torch.Tensor, t_star: int = None
    ) -> torch.Tensor:
        """
        Compute signed perpendicular deviation d(z) at t*.
        
        Positive d → committed to mode +1
        Negative d → committed to mode -1
        d ≈ 0 → hedging (midline)
        """
        if t_star is None:
            t_star = self.T // 2
        
        # Get baseline direction from s0
        # s0 = [x, y, goal_x, goal_y]
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        
        start = s0[:, :2]  # [B, 2]
        goal = s0[:, 2:4]  # [B, 2]
        
        direction = goal - start
        dist = torch.norm(direction, dim=-1, keepdim=True) + 1e-6
        d_unit = direction / dist
        
        # Perpendicular
        d_perp = torch.stack([-d_unit[:, 1], d_unit[:, 0]], dim=-1)  # [B, 2]
        
        # Expected position at t*
        progress = (t_star + 1) / self.T
        expected = start + direction * progress  # [B, 2]
        
        # Actual μ at t*
        if mu.dim() == 2:
            mu = mu.unsqueeze(0)
        mu_t_star = mu[:, t_star, :]  # [B, 2]
        
        # Deviation from expected
        deviation = mu_t_star - expected  # [B, 2]
        
        # Signed projection onto perpendicular
        d_signed = (deviation * d_perp).sum(dim=-1)  # [B]
        
        return d_signed
    
    def compute_mci(self, d_signed: torch.Tensor, bend_amplitude: float) -> torch.Tensor:
        """
        Mode Commitment Index: MCI = |d| / A
        
        MCI ≈ 0 → hedging (midline)
        MCI ≈ 1 → fully committed to one mode
        """
        return torch.abs(d_signed) / bend_amplitude
    
    def score_commitment(
        self, s0: torch.Tensor, z: torch.Tensor, bend_amplitude: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Score a commitment z for CEM selection.
        
        J(z) = -intent + agency - risk - mode_commit_penalty
        """
        mu, sigma = self.predict_tube(s0, z)
        
        # Intent: negative log volume (we want small volume = high agency)
        agency = -torch.log(sigma).mean(dim=(1, 2)) if sigma.dim() == 3 else -torch.log(sigma).mean()
        
        # Risk
        risk = self.risk_net(s0, z)
        
        # Mode commitment: penalize staying near midline
        d_signed = self.compute_signed_deviation(mu, s0)
        # H_mode = exp(-|d|/s) → small when |d| large, big when |d| small
        mode_penalty = torch.exp(-torch.abs(d_signed) / (bend_amplitude * 0.5))
        
        # Combined score (higher = better)
        score = (
            self.agency_weight * agency -
            self.risk_weight * risk -
            self.mode_commit_weight * mode_penalty
        )
        
        details = {
            "mu": mu,
            "sigma": sigma,
            "agency": agency,
            "risk": risk,
            "d_signed": d_signed,
            "mode_penalty": mode_penalty,
        }
        
        return score, details
    
    def select_z_random(self, s0: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Random binding: sample one z ~ q(z|s0)."""
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            z = self.sample_z(z_mu, z_logstd).squeeze(0)
            mu, sigma = self.predict_tube(s0, z)
            return z, {"mu": mu.squeeze(0), "sigma": sigma.squeeze(0)}
    
    def select_z_cem(
        self, s0: torch.Tensor, bend_amplitude: float
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
                scores, _ = self.score_commitment(s0, z_candidates, bend_amplitude)
                
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
            scores, details = self.score_commitment(s0, z_candidates, bend_amplitude)
            best_idx = scores.argmax()
            z_star = z_candidates[best_idx]
            
            return z_star, {
                "mu": details["mu"][best_idx],
                "sigma": details["sigma"][best_idx],
                "score": float(scores[best_idx].item()),
            }
    
    def select_z_hedge(self, s0: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Hedge baseline: explicitly widen tubes to cover both modes.
        
        Strategy: sample z with highest sigma (widest tube).
        """
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            
            # Sample many and pick widest
            eps = torch.randn(self.cem_samples, self.z_dim, device=s0.device)
            z_candidates = z_mu + torch.exp(z_logstd) * eps
            
            # Compute tube widths
            widths = []
            for i in range(self.cem_samples):
                mu, sigma = self.predict_tube(s0, z_candidates[i:i+1])
                widths.append(sigma.mean().item())
            
            # Pick widest
            widest_idx = np.argmax(widths)
            z_hedge = z_candidates[widest_idx]
            mu, sigma = self.predict_tube(s0, z_hedge)
            
            return z_hedge, {"mu": mu.squeeze(0), "sigma": sigma.squeeze(0)}
    
    def compute_soft_bind_fail(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        margin = self.k_sigma * sigma - torch.abs(trajectory - mu)
        p_dim = torch.sigmoid(self.gamma * margin)
        p_t = torch.prod(p_dim, dim=-1)
        bind_soft = p_t.mean()
        fail_soft = 1.0 - bind_soft
        return bind_soft, fail_soft
    
    def compute_bind_hard(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        inside = (torch.abs(trajectory - mu) < self.k_sigma * sigma).all(dim=-1).float()
        return inside.mean()
    
    def compute_kl(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2 * z_logstd)
        return 0.5 * (var + z_mu**2 - 1 - 2 * z_logstd).sum()
    
    def train_step(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        bend_amplitude: float,
    ) -> Dict[str, float]:
        """Training step with homeostasis."""
        self.train()
        
        # Recompute with gradients
        z_mu, z_logstd = self.encode(s0)
        mu_grad, sigma_grad = self.predict_tube(s0, z)
        mu_grad = mu_grad.squeeze(0)
        sigma_grad = sigma_grad.squeeze(0)
        
        mse = ((trajectory - mu_grad) ** 2).mean()
        log_vol = torch.log(sigma_grad).mean()
        bind_hard = self.compute_bind_hard(mu_grad, sigma_grad, trajectory)
        bind_soft, fail_soft = self.compute_soft_bind_fail(mu_grad, sigma_grad, trajectory)
        kl = self.compute_kl(z_mu.squeeze(0), z_logstd.squeeze(0))
        
        # Mode commitment
        d_signed = self.compute_signed_deviation(mu_grad.unsqueeze(0), s0)
        mci = self.compute_mci(d_signed, bend_amplitude)
        
        # Homeostasis
        sigma_mean = float(sigma_grad.mean().item())
        residual = float(torch.abs(trajectory - mu_grad).mean().item())
        fail_val = float(fail_soft.item())
        
        hardness_raw = residual / (sigma_mean + 1e-6)
        self.stats_hardness.update(hardness_raw)
        hardness_z = self.stats_hardness.z_score(hardness_raw)
        
        # Diversity loss: encourage |d| to be larger (away from midline)
        # This helps z learn to control mode rather than collapsing to midline
        diversity_loss = torch.exp(-torch.abs(d_signed) / bend_amplitude)
        
        # Loss
        loss = mse + self.alpha_vol * log_vol + self.lambda_fail * fail_soft + self.beta_kl * kl + 0.1 * diversity_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Lambda update
        surp_fail = max(0.0, -hardness_z)
        success_amt = 1.0 - fail_val
        delta = self.eta_lambda * (fail_val * (1.0 + surp_fail) - success_amt * 0.2)
        self.lambda_fail = float(np.clip(self.lambda_fail + delta, 0.1, self.lambda_max))
        
        # Risk critic
        risk_pred = self.risk_net(s0, z.detach())
        risk_target = fail_soft.detach()
        risk_loss = F.binary_cross_entropy(risk_pred, risk_target.unsqueeze(0))
        
        self.risk_optimizer.zero_grad()
        risk_loss.backward()
        nn.utils.clip_grad_norm_(self.risk_net.parameters(), 1.0)
        self.risk_optimizer.step()
        
        metrics = {
            "mse": float(mse.item()),
            "log_vol": float(log_vol.item()),
            "bind_hard": float(bind_hard.item()),
            "bind_soft": float(bind_soft.item()),
            "fail_soft": fail_val,
            "lambda": self.lambda_fail,
            "kl": float(kl.item()),
            "loss": float(loss.item()),
            "mci": float(mci.item()),
            "d_signed": float(d_signed.item()),
            "risk_loss": float(risk_loss.item()),
        }
        
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics


def train_bimodal(
    actor: BimodalActor,
    env: BimodalEnv,
    n_steps: int = 10000,
    use_cem: bool = True,
    log_every: int = 1000,
) -> Dict[str, List[float]]:
    """Train the bimodal actor."""
    mode_name = "CEM" if use_cem else "Random"
    print(f"Training BimodalActor for {n_steps} steps with {mode_name} binding...")
    
    for step in range(n_steps):
        s0 = env.reset()
        traj = env.generate_trajectory(T=actor.T)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Select z
        if use_cem:
            z, details = actor.select_z_cem(s0_t, env.bend_amplitude)
        else:
            z, details = actor.select_z_random(s0_t)
        
        mu = details["mu"]
        sigma = details["sigma"]
        
        # Train
        metrics = actor.train_step(s0_t, traj_t, z, mu, sigma, env.bend_amplitude)
        
        if step % log_every == 0:
            print(f"  Step {step}: mse={metrics['mse']:.4f}, bind={metrics['bind_hard']:.3f}, "
                  f"mci={metrics['mci']:.3f}, d={metrics['d_signed']:.3f}")
    
    return actor.history


def evaluate_mode_commitment(
    actor: BimodalActor,
    env: BimodalEnv,
    n_samples: int = 200,
) -> Dict[str, Dict]:
    """Evaluate mode commitment across different selection strategies."""
    actor.eval()
    
    results = {
        "random": defaultdict(list),
        "cem": defaultdict(list),
        "hedge": defaultdict(list),
    }
    
    print(f"\nEvaluating mode commitment on {n_samples} samples...")
    
    for i in range(n_samples):
        s0 = env.reset()
        traj = env.generate_trajectory(T=actor.T)
        true_mode = env.mode
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        for method, select_fn in [
            ("random", lambda: actor.select_z_random(s0_t)),
            ("cem", lambda: actor.select_z_cem(s0_t, env.bend_amplitude)),
            ("hedge", lambda: actor.select_z_hedge(s0_t)),
        ]:
            with torch.no_grad():
                z, details = select_fn()
                mu = details["mu"]
                sigma = details["sigma"]
                
                bind = actor.compute_bind_hard(mu, sigma, traj_t).item()
                log_vol = torch.log(sigma).mean().item()
                d_signed = actor.compute_signed_deviation(mu.unsqueeze(0), s0_t).item()
                mci = actor.compute_mci(torch.tensor([d_signed]), env.bend_amplitude).item()
                
                # Mode match: did we commit to the correct mode?
                mode_match = (np.sign(d_signed) == true_mode) if abs(d_signed) > 0.02 else False
            
            results[method]["bind"].append(bind)
            results[method]["log_vol"].append(log_vol)
            results[method]["d_signed"].append(d_signed)
            results[method]["mci"].append(mci)
            results[method]["mode_match"].append(mode_match)
            results[method]["true_mode"].append(true_mode)
            results[method]["z"].append(z.numpy())
    
    # Summarize
    summary = {}
    for method in ["random", "cem", "hedge"]:
        summary[method] = {
            "bind_mean": np.mean(results[method]["bind"]),
            "bind_std": np.std(results[method]["bind"]),
            "log_vol_mean": np.mean(results[method]["log_vol"]),
            "mci_mean": np.mean(results[method]["mci"]),
            "mci_std": np.std(results[method]["mci"]),
            "mode_match_rate": np.mean(results[method]["mode_match"]),
            "d_signed": np.array(results[method]["d_signed"]),
            "z": np.array(results[method]["z"]),
            "true_modes": np.array(results[method]["true_mode"]),
        }
        print(f"  {method}: bind={summary[method]['bind_mean']:.3f}, "
              f"vol={summary[method]['log_vol_mean']:.2f}, "
              f"MCI={summary[method]['mci_mean']:.3f}, "
              f"mode_match={summary[method]['mode_match_rate']:.3f}")
    
    return summary


def plot_bimodal_trajectories(env: BimodalEnv, out_path: Path, n_examples: int = 5):
    """Visualize the bimodal structure."""
    fig, axes = plt.subplots(1, n_examples, figsize=(4*n_examples, 4))
    
    for i, ax in enumerate(axes):
        env.reset()
        traj_plus, traj_minus = env.generate_both_modes(T=16)
        
        # Plot both modes
        ax.plot(traj_plus[:, 0], traj_plus[:, 1], 'b-', linewidth=2, label='Mode +1')
        ax.plot(traj_minus[:, 0], traj_minus[:, 1], 'r-', linewidth=2, label='Mode -1')
        
        # Start/goal
        ax.scatter([env.x], [env.y], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([env.goal_x], [env.goal_y], c='purple', s=100, marker='*', label='Goal', zorder=5)
        
        # Midpoint separation
        t_star = 8
        ax.scatter([traj_plus[t_star, 0]], [traj_plus[t_star, 1]], c='blue', s=50, marker='x')
        ax.scatter([traj_minus[t_star, 0]], [traj_minus[t_star, 1]], c='red', s=50, marker='x')
        
        ax.set_title(f"Example {i+1}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle("Bimodal Trajectories: Same s0, Two Incompatible Futures", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_d_histogram(summary: Dict[str, Dict], out_path: Path):
    """Plot histogram of signed deviation d for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ["random", "cem", "hedge"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    titles = ["Random (Posterior Sample)", "CEM (Competitive Binding)", "Hedge (Wide Tube)"]
    
    for ax, method, color, title in zip(axes, methods, colors, titles):
        d_values = summary[method]["d_signed"]
        
        ax.hist(d_values, bins=30, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel("Signed Deviation d")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nMCI={summary[method]['mci_mean']:.3f}")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Mode Commitment: d(z) Distribution\n(Bimodal = commitment, Unimodal at 0 = hedging)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_z_space_by_mode(summary: Dict[str, Dict], out_path: Path):
    """Visualize z-space colored by mode commitment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ["random", "cem", "hedge"]
    titles = ["Random", "CEM", "Hedge"]
    
    for ax, method, title in zip(axes, methods, titles):
        z_values = summary[method]["z"]
        d_signed = summary[method]["d_signed"]
        
        # PCA if needed
        if z_values.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_values)
        else:
            z_2d = z_values
        
        # Color by d_signed
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=d_signed, cmap='RdBu', 
                            vmin=-0.2, vmax=0.2, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='d (signed deviation)')
        
        ax.set_xlabel("z PC1")
        ax.set_ylabel("z PC2")
        ax.set_title(f"{title}: z* colored by mode commitment")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Z-Space Clustering by Mode\n(Two clusters = commitment attractors)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_commitment_comparison(summary: Dict[str, Dict], out_path: Path):
    """Compare commitment metrics across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ["random", "cem", "hedge"]
    method_names = ["Random", "CEM", "Hedge"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    
    # 1. Bind rate
    ax = axes[0]
    binds = [summary[m]["bind_mean"] for m in methods]
    bind_stds = [summary[m]["bind_std"] for m in methods]
    ax.bar(method_names, binds, yerr=bind_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_ylabel("Bind Rate")
    ax.set_title("Reliability\n(higher = more coverage)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Volume (negative = tighter)
    ax = axes[1]
    vols = [summary[m]["log_vol_mean"] for m in methods]
    ax.bar(method_names, vols, color=colors, edgecolor='black')
    ax.set_ylabel("Log Volume")
    ax.set_title("Agency\n(lower = tighter tubes)")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. MCI
    ax = axes[2]
    mcis = [summary[m]["mci_mean"] for m in methods]
    mci_stds = [summary[m]["mci_std"] for m in methods]
    ax.bar(method_names, mcis, yerr=mci_stds, color=colors, capsize=5, edgecolor='black')
    ax.set_ylabel("Mode Commitment Index")
    ax.set_title("Commitment\n(higher = chose a side)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Bimodal Commitment: Method Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training(history: Dict[str, List[float]], out_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    window = 200
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # Row 1
    ax = axes[0, 0]
    ax.plot(smooth(history["bind_hard"]), color='blue')
    ax.set_title("Bind Rate")
    ax.set_ylabel("Bind")
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(smooth(history["log_vol"]), color='green')
    ax.set_title("Log Volume")
    ax.set_ylabel("Log σ")
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(smooth(history["mci"]), color='purple')
    ax.set_title("Mode Commitment Index")
    ax.set_ylabel("MCI")
    ax.grid(True, alpha=0.3)
    
    # Row 2
    ax = axes[1, 0]
    ax.plot(smooth(history["d_signed"]), color='orange')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("Signed Deviation d")
    ax.set_ylabel("d")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(history["lambda"], color='red')
    ax.set_title("λ (failure price)")
    ax.set_ylabel("λ")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(smooth(history["mse"]), color='brown')
    ax.set_title("MSE")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Bimodal Commitment Training", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--bend-amplitude", type=float, default=0.15,
                        help="Mode separation amplitude (higher = more incompatible)")
    parser.add_argument("--base-noise", type=float, default=0.02)
    parser.add_argument("--k-sigma", type=float, default=1.5)
    parser.add_argument("--mode-commit-weight", type=float, default=0.5,
                        help="Weight on mode commitment term in CEM score")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"bimodal_{timestamp}"
    if args.name:
        run_name = f"bimodal_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Bimodal Commitment Experiment")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create environment and actor
    env = BimodalEnv(
        seed=args.seed,
        bend_amplitude=args.bend_amplitude,
        base_noise=args.base_noise,
    )
    
    actor = BimodalActor(
        obs_dim=4,
        z_dim=args.z_dim,
        k_sigma=args.k_sigma,
        mode_commit_weight=args.mode_commit_weight,
    )
    
    # Visualize bimodal structure
    print("Visualizing bimodal trajectories...")
    plot_bimodal_trajectories(env, out_dir / "bimodal_structure.png")
    
    # Check mode separation
    separation = env.get_mode_separation()
    print(f"Mode separation at t*: {separation:.3f}")
    print(f"  (Should be > 6 * k_sigma * sigma_min for incompatibility)")
    
    # Train
    history = train_bimodal(actor, env, n_steps=args.steps, use_cem=True)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING MODE COMMITMENT")
    print("="*60)
    summary = evaluate_mode_commitment(actor, env)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training(history, out_dir / "training.png")
    plot_d_histogram(summary, out_dir / "d_histogram.png")
    plot_commitment_comparison(summary, out_dir / "commitment_comparison.png")
    
    try:
        plot_z_space_by_mode(summary, out_dir / "z_space_modes.png")
    except ImportError:
        print("  (Skipping z_space plot - sklearn not available)")
    
    # Save model
    torch.save(actor.state_dict(), out_dir / "model.pt")
    
    # Final analysis
    print(f"\n{'='*60}")
    print("BIMODAL COMMITMENT RESULTS")
    print(f"{'='*60}")
    
    cem_mci = summary["cem"]["mci_mean"]
    random_mci = summary["random"]["mci_mean"]
    hedge_mci = summary["hedge"]["mci_mean"]
    
    cem_vol = summary["cem"]["log_vol_mean"]
    hedge_vol = summary["hedge"]["log_vol_mean"]
    
    cem_bind = summary["cem"]["bind_mean"]
    hedge_bind = summary["hedge"]["bind_mean"]
    
    print(f"\nMode Commitment Index (higher = chose a side):")
    print(f"  CEM:    {cem_mci:.3f}")
    print(f"  Random: {random_mci:.3f}")
    print(f"  Hedge:  {hedge_mci:.3f}")
    
    print(f"\nLog Volume (lower = tighter tubes):")
    print(f"  CEM:    {cem_vol:.2f}")
    print(f"  Hedge:  {hedge_vol:.2f}")
    
    # Success criteria
    cem_commits = cem_mci > random_mci + 0.1
    cem_tight = cem_vol < hedge_vol - 0.3
    cem_reliable = cem_bind > 0.6
    
    print(f"\nSuccess Criteria:")
    print(f"  {'✓' if cem_commits else '✗'} CEM commits to a mode (MCI {cem_mci:.3f} > {random_mci:.3f} + 0.1)")
    print(f"  {'✓' if cem_tight else '✗'} CEM uses tighter tubes than hedge (vol {cem_vol:.2f} < {hedge_vol:.2f})")
    print(f"  {'✓' if cem_reliable else '✗'} CEM maintains reliability (bind {cem_bind:.3f} > 0.6)")
    
    if sum([cem_commits, cem_tight, cem_reliable]) >= 2:
        print("\n🎉 SUCCESS: Bimodal commitment achieved!")
        print("   CEM chooses one future (high agency) rather than hedging (coverage).")
        print("   This separates belief (representing both) from commitment (excluding one).")
    else:
        print("\n⚠ Commitment not fully achieved. May need tuning.")
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
