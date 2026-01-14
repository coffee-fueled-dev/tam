from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BindingMode(Enum):
    RANDOM = "random"
    BEST_OF_K = "best_of_k"
    CEM = "cem"


def _lin_interp_knots_to_T(knots: torch.Tensor, T: int) -> torch.Tensor:
    """Differentiable linear interpolation from n_knots -> T.

    Args:
        knots: [B, n_knots, d]
        T: number of timesteps

    Returns:
        values: [B, T, d]

    Notes:
        - Assumes knots are placed at uniform times over [0, 1].
        - If n_knots == 1, repeats the single knot for all timesteps.
    """
    if knots.dim() != 3:
        raise ValueError(f"knots must be [B,n_knots,d], got shape {tuple(knots.shape)}")

    B, n_knots, d = knots.shape
    if n_knots == 1:
        return knots.expand(B, T, d)

    device = knots.device
    dtype = knots.dtype

    # Uniform time locations for knots and timesteps
    knot_pos = torch.linspace(0.0, 1.0, n_knots, device=device, dtype=dtype)  # [n_knots]
    t_pos = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)          # [T]

    # For each t, find rightmost knot index i such that knot_pos[i] <= t
    # We'll interpolate between i and i+1.
    # idx in [0, n_knots-2]
    idx = torch.searchsorted(knot_pos, t_pos, right=True) - 1
    idx = idx.clamp(0, n_knots - 2)  # [T]

    i0 = idx
    i1 = idx + 1

    p0 = knot_pos[i0]  # [T]
    p1 = knot_pos[i1]  # [T]

    # Avoid divide-by-zero (shouldn't happen with linspace, but safe)
    denom = (p1 - p0).clamp(min=1e-8)  # [T]
    w = ((t_pos - p0) / denom).clamp(0.0, 1.0)  # [T]

    # Gather knots at i0 and i1
    # Expand indices for gather: [B, T, d]
    i0e = i0.view(1, T, 1).expand(B, T, d)
    i1e = i1.view(1, T, 1).expand(B, T, d)

    k0 = torch.gather(knots, dim=1, index=i0e)  # [B, T, d]
    k1 = torch.gather(knots, dim=1, index=i1e)  # [B, T, d]

    we = w.view(1, T, 1).expand(B, T, d)
    return (1.0 - we) * k0 + we * k1


class KnotTubeNet(nn.Module):
    """Tube network with knot bottleneck.

    - mu_knots depends on (s0, z)
    - sigma_knots depends only on z
    - both are interpolated to length T

    This biases ports toward *globally coherent* trajectories and widths.
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        pred_dim: int = 2,
        T: int = 16,
        n_knots: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if n_knots < 1:
            raise ValueError("n_knots must be >= 1")

        self.T = T
        self.n_knots = n_knots
        self.pred_dim = pred_dim

        # μ knots: from (s0, z)
        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_knots * pred_dim),
        )

        # log σ knots: from z ONLY (critical constraint)
        self.logsig_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_knots * pred_dim),
        )
        nn.init.zeros_(self.logsig_net[-1].bias)  # start around log σ = 0

    def forward(self, s0: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-timestep tube and the underlying knots.

        Returns:
            mu: [B, T, d]
            sigma: [B, T, d]
            mu_knots: [B, n_knots, d]
            logsig_knots: [B, n_knots, d]
        """
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B = z.shape[0]
        if s0.shape[0] == 1 and B > 1:
            s0 = s0.expand(B, -1)

        # Predict knots
        h = torch.cat([s0, z], dim=-1)
        mu_knots = self.mu_net(h).view(B, self.n_knots, self.pred_dim)

        logsig_knots = self.logsig_net(z).clamp(-4.0, 3.0).view(B, self.n_knots, self.pred_dim)

        # Interpolate to per-timestep
        mu = _lin_interp_knots_to_T(mu_knots, self.T)
        logsig = _lin_interp_knots_to_T(logsig_knots, self.T)
        sigma = torch.exp(logsig)

        return mu, sigma, mu_knots, logsig_knots


class KnotActor(nn.Module):
    """Minimal TAM actor with knot-based ports.

    Keeps the same high-level structure as your minimal Actor:
      1) Encoder q(z|s0)
      2) Tube prediction (now knot-based)
      3) Competitive binding (random / best-of-k / CEM)
      4) Training objective: Gaussian NLL + alpha_vol * log_vol + ...
      5) Scoring: mode-prototype max-fit + agency

    Differences vs minimal Actor:
      - mu, sigma are generated from knots (n_knots x d) and interpolated to T.
      - optional knot smoothness regularizer encourages "knot coherence" directly.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        z_dim: int = 4,
        pred_dim: int = 2,
        T: int = 16,
        n_knots: int = 5,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        # Loss weights
        alpha_vol: float = 0.5,
        beta_kl: float = 0.001,
        alpha_fail: float = 1.0,
        # Smoothness
        alpha_smooth: float = 0.1,          # time-domain smoothness on interpolated log σ
        alpha_knot_smooth: float = 0.0,     # optional: smoothness directly on knots (2nd-diff)
        use_global_vol: bool = True,
        max_fail_gamma: float = 5.0,
        # CEM
        cem_iters: int = 4,
        cem_samples: int = 128,
        cem_elites: int = 16,
        cem_smoothing: float = 0.25,
        cem_std_floor: float = 0.2,
        # Scoring
        agency_weight: float = 0.5,
        k_sigma: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.pred_dim = pred_dim
        self.T = T
        self.n_knots = n_knots

        self.alpha_vol = alpha_vol
        self.beta_kl = beta_kl
        self.alpha_fail = alpha_fail
        self.k_sigma = k_sigma

        self.alpha_smooth = alpha_smooth
        self.alpha_knot_smooth = alpha_knot_smooth
        self.use_global_vol = use_global_vol
        self.max_fail_gamma = max_fail_gamma

        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_elites = cem_elites
        self.cem_smoothing = cem_smoothing
        self.cem_std_floor = cem_std_floor

        self.agency_weight = agency_weight

        # Encoder q(z|s0)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logstd_head = nn.Linear(hidden_dim, z_dim)

        # Knot tube
        self.tube_net = KnotTubeNet(
            obs_dim=obs_dim,
            z_dim=z_dim,
            pred_dim=pred_dim,
            T=T,
            n_knots=n_knots,
            hidden_dim=hidden_dim,
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.history: Dict[str, List[float]] = defaultdict(list)

    @property
    def device(self):
        return next(self.parameters()).device

    # -------------------- encode / sample --------------------

    def encode(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        h = self.encoder(s0)
        z_mu = self.z_mu_head(h)
        z_logstd = self.z_logstd_head(h).clamp(-4.0, 2.0)
        return z_mu, z_logstd

    def sample_z(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        std = torch.exp(z_logstd)
        eps = torch.randn_like(std)
        return z_mu + std * eps

    # -------------------- tube prediction --------------------

    def predict_tube(self, s0: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tube_net(s0, z)

    # -------------------- scoring --------------------

    def score_commitment(
        self,
        s0: torch.Tensor,
        z: torch.Tensor,
        mode_prototypes: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Score = max_fit_to_prototype + agency.

        Same as your minimal actor, but uses knot tube.
        """
        mu, sigma, mu_knots, logsig_knots = self.predict_tube(s0, z)

        # Agency from per-timestep sigma
        log_vol = torch.log(sigma).mean(dim=(1, 2))
        agency = -log_vol

        if mode_prototypes is None:
            # If you want a fallback later, add it here; for now keep minimal.
            score = self.agency_weight * agency
            details = {
                "mu": mu,
                "sigma": sigma,
                "mu_knots": mu_knots,
                "logsig_knots": logsig_knots,
                "agency": agency,
                "mode_fit": torch.zeros_like(agency),
                "best_mode": torch.zeros_like(agency, dtype=torch.long),
            }
            return score, details

        fits = []
        for tau in mode_prototypes:
            if tau.dim() == 2:
                tau = tau.unsqueeze(0)
            # tau expected [1, T, d] matching mu
            fit = -((mu - tau) ** 2).mean(dim=(1, 2))
            fits.append(fit)
        fits = torch.stack(fits, dim=1)  # [B, K]

        mode_fit, best_mode = fits.max(dim=1)
        score = mode_fit + self.agency_weight * agency

        details = {
            "mu": mu,
            "sigma": sigma,
            "mu_knots": mu_knots,
            "logsig_knots": logsig_knots,
            "agency": agency,
            "fits": fits,
            "mode_fit": mode_fit,
            "best_mode": best_mode,
        }
        return score, details

    # -------------------- binding --------------------

    def select_z_random(self, s0: torch.Tensor):
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            z = self.sample_z(z_mu, z_logstd).squeeze(0)
            mu, sigma, mu_knots, logsig_knots = self.predict_tube(s0, z)
            return z, {
                "mu": mu.squeeze(0),
                "sigma": sigma.squeeze(0),
                "mu_knots": mu_knots.squeeze(0),
                "logsig_knots": logsig_knots.squeeze(0),
            }

    def select_z_best_of_k(self, s0: torch.Tensor, k: int = 32, mode_prototypes=None):
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            z_mu = z_mu.squeeze(0)
            z_logstd = z_logstd.squeeze(0)
            eps = torch.randn(k, self.z_dim, device=s0.device)
            z_candidates = z_mu.unsqueeze(0) + torch.exp(z_logstd).unsqueeze(0) * eps

            scores, details = self.score_commitment(s0, z_candidates, mode_prototypes)
            best_idx = scores.argmax()
            z_star = z_candidates[best_idx]

            return z_star, {
                "mu": details["mu"][best_idx],
                "sigma": details["sigma"][best_idx],
                "mu_knots": details["mu_knots"][best_idx],
                "logsig_knots": details["logsig_knots"][best_idx],
                "score": float(scores[best_idx].item()),
                "best_mode": int(details["best_mode"][best_idx].item()) if "best_mode" in details else 0,
            }

    def select_z_cem(self, s0: torch.Tensor, mode_prototypes=None):
        with torch.no_grad():
            z_mu, z_logstd = self.encode(s0)
            cem_mean = z_mu.squeeze(0).clone()
            cem_std = torch.exp(z_logstd.squeeze(0)).clamp(min=self.cem_std_floor)

            for _ in range(self.cem_iters):
                eps = torch.randn(self.cem_samples, self.z_dim, device=s0.device)
                z_candidates = cem_mean.unsqueeze(0) + cem_std.unsqueeze(0) * eps

                scores, _ = self.score_commitment(s0, z_candidates, mode_prototypes)

                elite_idx = scores.argsort(descending=True)[: self.cem_elites]
                elite_z = z_candidates[elite_idx]

                new_mean = elite_z.mean(dim=0)
                new_std = elite_z.std(dim=0).clamp(min=self.cem_std_floor)

                cem_mean = (1.0 - self.cem_smoothing) * cem_mean + self.cem_smoothing * new_mean
                cem_std = (1.0 - self.cem_smoothing) * cem_std + self.cem_smoothing * new_std

            # final sample and pick best
            eps = torch.randn(self.cem_samples, self.z_dim, device=s0.device)
            z_candidates = cem_mean.unsqueeze(0) + cem_std.unsqueeze(0) * eps
            scores, details = self.score_commitment(s0, z_candidates, mode_prototypes)
            best_idx = scores.argmax()
            z_star = z_candidates[best_idx]

            return z_star, {
                "mu": details["mu"][best_idx],
                "sigma": details["sigma"][best_idx],
                "mu_knots": details["mu_knots"][best_idx],
                "logsig_knots": details["logsig_knots"][best_idx],
                "score": float(scores[best_idx].item()),
                "best_mode": int(details["best_mode"][best_idx].item()) if "best_mode" in details else 0,
            }

    def select_z(self, s0: torch.Tensor, mode: BindingMode = BindingMode.CEM, mode_prototypes=None, k: int = 32):
        if mode == BindingMode.RANDOM:
            return self.select_z_random(s0)
        if mode == BindingMode.BEST_OF_K:
            return self.select_z_best_of_k(s0, k=k, mode_prototypes=mode_prototypes)
        if mode == BindingMode.CEM:
            return self.select_z_cem(s0, mode_prototypes=mode_prototypes)
        raise ValueError(f"Unknown mode: {mode}")

    # -------------------- metrics / loss --------------------

    def compute_bind_hard(self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        inside = (torch.abs(trajectory - mu) < self.k_sigma * sigma).all(dim=-1).float()
        return inside.mean()

    def compute_nll(self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        residual_sq = (trajectory - mu) ** 2
        return (residual_sq / (sigma ** 2) + 2.0 * torch.log(sigma)).mean()

    def compute_kl(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * z_logstd)
        return 0.5 * (var + z_mu ** 2 - 1.0 - 2.0 * z_logstd).sum()

    def _knot_second_diff(self, knots: torch.Tensor) -> torch.Tensor:
        """Second-difference smoothness: mean ||k_{i+1} - 2k_i + k_{i-1}||^2."""
        if knots.shape[1] < 3:
            return torch.zeros((), device=knots.device, dtype=knots.dtype)
        dd = knots[:, 2:, :] - 2.0 * knots[:, 1:-1, :] + knots[:, :-2, :]
        return (dd ** 2).mean()

    def train_step(self, s0: torch.Tensor, trajectory: torch.Tensor, z: torch.Tensor) -> Dict[str, float]:
        self.train()

        z_mu, z_logstd = self.encode(s0)
        mu, sigma, mu_knots, logsig_knots = self.predict_tube(s0, z)

        if mu.dim() == 3:
            mu0 = mu.squeeze(0)
            sigma0 = sigma.squeeze(0)
        else:
            mu0, sigma0 = mu, sigma

        nll = self.compute_nll(mu0, sigma0, trajectory)

        if self.use_global_vol:
            log_vol = torch.log(sigma0).sum()
        else:
            log_vol = torch.log(sigma0).mean()

        # worst-step soft max fail (same as your phase-2 variant)
        deviation = torch.abs(trajectory - mu0)
        normalized_dev = deviation / (self.k_sigma * sigma0 + 1e-8)
        max_dev_per_t = normalized_dev.max(dim=-1).values  # [T]
        soft_max_fail = torch.logsumexp(self.max_fail_gamma * max_dev_per_t, dim=0) / self.max_fail_gamma

        # time-domain smoothness on interpolated log sigma
        log_sigma = torch.log(sigma0)
        sigma_diff = log_sigma[1:] - log_sigma[:-1]
        smooth_loss = (sigma_diff ** 2).mean()

        # optional knot smoothness directly (recommended to start at 0.0 then turn on)
        knot_smooth = self._knot_second_diff(mu_knots) + self._knot_second_diff(logsig_knots)

        kl = self.compute_kl(z_mu.squeeze(0), z_logstd.squeeze(0))

        loss = (
            nll
            + self.alpha_vol * log_vol
            + self.alpha_fail * soft_max_fail
            + self.beta_kl * kl
            + self.alpha_smooth * smooth_loss
            + self.alpha_knot_smooth * knot_smooth
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        bind_hard = self.compute_bind_hard(mu0, sigma0, trajectory)
        mse = ((trajectory - mu0) ** 2).mean()

        metrics = {
            "loss": float(loss.item()),
            "nll": float(nll.item()),
            "log_vol": float(log_vol.item()),
            "soft_max_fail": float(soft_max_fail.item()),
            "kl": float(kl.item()),
            "smooth_loss": float(smooth_loss.item()),
            "knot_smooth": float(knot_smooth.item()),
            "mse": float(mse.item()),
            "bind_hard": float(bind_hard.item()),
        }

        for k, v in metrics.items():
            self.history[k].append(v)

        return metrics


def _quick_test():
    print("Testing KnotActor...")
    actor = KnotActor(obs_dim=4, z_dim=4, pred_dim=2, T=16, n_knots=5)
    s0 = torch.tensor([0.5, 0.5, 0.8, 0.8])
    z_mu, z_logstd = actor.encode(s0)
    z = actor.sample_z(z_mu, z_logstd)
    mu, sigma, mu_k, ls_k = actor.predict_tube(s0, z)
    print("  mu", mu.shape, "sigma", sigma.shape, "mu_knots", mu_k.shape, "logsig_knots", ls_k.shape)

    z_cem, details = actor.select_z_cem(s0, mode_prototypes=None)
    print("  cem z", z_cem.shape, "score", details.get("score", None))

    traj = torch.randn(16, 2) * 0.1 + 0.5
    metrics = actor.train_step(s0, traj, z_cem)
    print("  train loss", metrics["loss"], "bind", metrics["bind_hard"])
    print("✓ ok")


if __name__ == "__main__":
    _quick_test()
