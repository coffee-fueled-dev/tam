"""
TAM Actor: Minimal implementation based on ablation insights.

Key design principles (from ablation tests):
1. Fixed horizon T - no learned stopping (simplifies everything)
2. Sigma depends ONLY on z - forces z to control geometry
3. Mu depends on (s0, z) - trajectory mean needs situational info
4. z is sampled from q(z|s0) - encoder learns to map situations to geometry

Architecture:
    s0 → encoder → z_mu, z_logstd
    z ~ N(z_mu, exp(z_logstd))
    
    (s0, z) → mu_net → mu[1:T]  (trajectory mean)
    z → sigma_net → sigma[1:T]  (tube width - ONLY z!)
    
    policy: (obs, z) → action

Loss:
    L = NLL(trajectory | tube) + β_kl * KL(q(z|s0) || p(z)) + λ * (target_bind - bind_rate)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TubeOutput:
    """Output from tube prediction."""
    mu: torch.Tensor      # [T, pred_dim] trajectory mean
    sigma: torch.Tensor   # [T, pred_dim] tube width (std)
    z: torch.Tensor       # [z_dim] latent commitment
    z_mu: torch.Tensor    # [z_dim] posterior mean
    z_logstd: torch.Tensor  # [z_dim] posterior log-std


class TubeNet(nn.Module):
    """
    Tube network with key insight: sigma depends ONLY on z.
    
    This forces z to be the sole controller of uncertainty geometry.
    """
    
    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 4,
        pred_dim: int = 2,
        hidden_dim: int = 64,
        T: int = 16,  # Fixed horizon
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
        
        # Sigma network: z → sigma[1:T]  (ONLY z!)
        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T * pred_dim),
        )
        
        # Initialize sigma to reasonable values (log(1) = 0)
        nn.init.zeros_(self.sigma_net[-1].bias)
    
    def encode(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to z posterior."""
        h = self.encoder(s0)
        z_mu = self.z_mu_head(h)
        z_logstd = self.z_logstd_head(h).clamp(-4, 2)
        return z_mu, z_logstd
    
    def sample_z(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample."""
        std = torch.exp(z_logstd)
        eps = torch.randn_like(std)
        return z_mu + std * eps
    
    def predict_tube(self, s0: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict tube given s0 and z.
        
        Returns:
            mu: [B, T, pred_dim]
            sigma: [B, T, pred_dim]
        """
        B = s0.shape[0]
        
        # Mu from (s0, z)
        mu = self.mu_net(torch.cat([s0, z], dim=-1))
        mu = mu.view(B, self.T, self.pred_dim)
        
        # Sigma from z ONLY
        logsig = self.sigma_net(z)
        sigma = torch.exp(logsig.view(B, self.T, self.pred_dim)).clamp(0.01, 10.0)
        
        return mu, sigma
    
    def forward(self, s0: torch.Tensor) -> TubeOutput:
        """Full forward pass: encode, sample, predict."""
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


class Policy(nn.Module):
    """Simple policy: (obs, z) → action."""
    
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        discrete: bool = True,
    ):
        super().__init__()
        self.discrete = discrete
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            If discrete: logits [action_dim]
            If continuous: action [action_dim]
        """
        x = torch.cat([obs, z], dim=-1)
        out = self.net(x)
        
        if not self.discrete:
            out = torch.tanh(out) * 2.0  # Scale to [-2, 2]
        
        return out


class Actor(nn.Module):
    """
    TAM Actor with minimal complexity.
    
    Core insight: z controls tube geometry (sigma depends only on z).
    """
    
    def __init__(
        self,
        obs_dim: int,
        pred_dim: int = 2,
        action_dim: int = 4,
        z_dim: int = 4,
        hidden_dim: int = 64,
        T: int = 16,
        k_sigma: float = 2.0,
        lr: float = 1e-3,
        beta_kl: float = 0.001,
        target_bind: float = 0.85,
        discrete_actions: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.pred_dim = pred_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.T = T
        self.k_sigma = k_sigma
        self.beta_kl = beta_kl
        self.target_bind = target_bind
        
        # Networks
        self.tube = TubeNet(obs_dim, z_dim, pred_dim, hidden_dim, T)
        self.policy = Policy(obs_dim, z_dim, action_dim, hidden_dim, discrete_actions)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Dual variable for bind constraint
        self.lambda_bind = 1.0
        
        # History
        self.history: Dict[str, List[float]] = {
            "nll": [], "bind": [], "kl": [], "cone_vol": [], "z_norm": [],
        }
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_commitment(self, s0: torch.Tensor) -> TubeOutput:
        """Get tube commitment for initial observation."""
        return self.tube(s0)
    
    def get_action(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Get action given current observation and commitment z."""
        return self.policy(obs, z)
    
    def compute_nll(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Compute NLL of trajectory under tube."""
        # trajectory: [T, pred_dim]
        var = sigma ** 2
        sq_err = (trajectory - mu) ** 2
        nll_t = 0.5 * (sq_err / var + torch.log(var)).sum(dim=-1)  # [T]
        return nll_t.mean()
    
    def compute_bind_rate(
        self, mu: torch.Tensor, sigma: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Compute fraction of trajectory inside tube."""
        inside = (torch.abs(trajectory - mu) < self.k_sigma * sigma).all(dim=-1).float()
        return inside.mean()
    
    def compute_kl(self, z_mu: torch.Tensor, z_logstd: torch.Tensor) -> torch.Tensor:
        """KL divergence to standard normal."""
        var = torch.exp(2 * z_logstd)
        kl = 0.5 * (var + z_mu**2 - 1 - 2 * z_logstd).sum()
        return kl
    
    def train_step(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            s0: initial observation [obs_dim]
            trajectory: realized trajectory [T, pred_dim]
            actions: actions taken [T, action_dim] (optional, for policy learning)
        """
        self.train()
        
        # Get tube prediction
        out = self.tube(s0)
        
        # Compute losses
        nll = self.compute_nll(out.mu, out.sigma, trajectory)
        bind_rate = self.compute_bind_rate(out.mu, out.sigma, trajectory)
        kl = self.compute_kl(out.z_mu, out.z_logstd)
        
        # Cone volume (for tracking)
        cone_vol = torch.prod(out.sigma.mean(dim=0))
        
        # Total loss
        loss = nll
        loss = loss + self.beta_kl * kl
        loss = loss + self.lambda_bind * (self.target_bind - bind_rate)
        
        # Policy loss (if actions provided)
        if actions is not None:
            # Simple behavior cloning for now
            z = out.z.unsqueeze(0).expand(trajectory.shape[0], -1)
            # Would need obs sequence here - skip for now
            pass
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update lambda_bind
        bind_gap = self.target_bind - bind_rate.item()
        self.lambda_bind = float(np.clip(self.lambda_bind + 0.01 * bind_gap, 0.01, 10.0))
        
        # Track
        metrics = {
            "nll": float(nll.item()),
            "bind": float(bind_rate.item()),
            "kl": float(kl.item()),
            "cone_vol": float(cone_vol.item()),
            "z_norm": float(out.z.norm().item()),
        }
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics
    
    def evaluate_z_control(self, s0: torch.Tensor, n_samples: int = 20) -> Dict:
        """Check if z controls tube geometry."""
        self.eval()
        
        with torch.no_grad():
            z_mu, z_logstd = self.tube.encode(s0.unsqueeze(0))
        
        results = []
        
        for d in range(self.z_dim):
            for scale in np.linspace(-2, 2, n_samples):
                z = z_mu.clone()
                z[0, d] = z[0, d] + scale
                
                with torch.no_grad():
                    mu, sigma = self.tube.predict_tube(s0.unsqueeze(0), z)
                
                cone_vol = torch.prod(sigma.mean(dim=1)).item()
                
                results.append({
                    "dim": d,
                    "scale": scale,
                    "cone_vol": np.log(cone_vol + 1e-8),
                })
        
        return results


def test_actor():
    """Quick test of the Actor."""
    print("Testing Actor...")
    
    # Create actor
    actor = Actor(
        obs_dim=8,
        pred_dim=2,
        action_dim=4,
        z_dim=4,
        T=16,
    )
    
    # Dummy data
    s0 = torch.randn(8)
    trajectory = torch.randn(16, 2)
    
    # Get commitment
    out = actor.get_commitment(s0)
    print(f"  z shape: {out.z.shape}")
    print(f"  mu shape: {out.mu.shape}")
    print(f"  sigma shape: {out.sigma.shape}")
    
    # Train step
    metrics = actor.train_step(s0, trajectory)
    print(f"  nll: {metrics['nll']:.3f}, bind: {metrics['bind']:.3f}, kl: {metrics['kl']:.3f}")
    
    # Check z control
    results = actor.evaluate_z_control(s0)
    print(f"  z control test: {len(results)} samples")
    
    print("✓ Actor works!")
    return actor


if __name__ == "__main__":
        test_actor()
