"""
Functor Learning for Cross-Environment Commitment Transfer.

Learns a mapping Φ: Z_A → Z_B such that commitments from environment A,
when translated via Φ, induce equivalent cone behavior in environment B.

Key principle: Match behavioral invariants (cone semantics), not actions/trajectories.

Cone signature: [C, H, r, λ_bind]
- C: weighted log cone volume
- H: expected horizon E[T]
- r: soft bind rate
- λ_bind: dual pressure (reliability price)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from utils import truncated_geometric_weights
except ImportError:
    try:
        from ..utils import truncated_geometric_weights
    except ImportError:
        def truncated_geometric_weights(p_stop: torch.Tensor, T: int):
            t_idx = torch.arange(1, T + 1, device=p_stop.device, dtype=p_stop.dtype)
            one_minus = (1.0 - p_stop).clamp(1e-6, 1.0 - 1e-6)
            s = one_minus ** (t_idx - 1.0)
            w = s * p_stop
            tail = one_minus**T
            w = w.clone()
            w[-1] = w[-1] + tail
            w = w / (w.sum() + 1e-8)
            E_T = (w * t_idx).sum()
            return w, E_T

try:
    from actor import Actor
except ImportError:
    try:
        from ..actor import Actor
    except ImportError:
        Actor = None


@dataclass
class ConeSignature:
    """Cone signature for a commitment z."""
    z: np.ndarray  # [z_dim]
    log_cone_vol: float  # C: weighted log cone volume
    E_T: float  # H: expected horizon
    bind_rate: float  # r: soft bind rate
    lambda_bind: float  # dual pressure
    
    def to_vector(self) -> np.ndarray:
        """Convert to 4D signature vector."""
        return np.array([self.log_cone_vol, self.E_T, self.bind_rate, self.lambda_bind])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "z": self.z.tolist(),
            "log_cone_vol": self.log_cone_vol,
            "E_T": self.E_T,
            "bind_rate": self.bind_rate,
            "lambda_bind": self.lambda_bind,
        }


@dataclass
class FunctorConfig:
    """Configuration for functor learning."""
    z_dim: int = 8
    hidden_dim: int = 64
    
    # Functor architecture
    functor_type: str = "linear"  # "linear", "affine", "mlp", "residual"
    
    # Training
    lr: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 100
    n_eval_episodes: int = 50  # episodes to compute cone stats
    
    # Normalization
    normalize_signatures: bool = True
    
    # Evaluation
    k_sigma: float = 2.0
    eval_horizon: int = 16
    
    # Output
    output_dir: Optional[Path] = None
    seed: int = 0


class LinearFunctor(nn.Module):
    """Linear functor: Φ(z) = Wz + b"""
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.linear = nn.Linear(z_dim, z_dim)
        # Initialize close to identity
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class AffineFunctor(nn.Module):
    """Affine functor with learned scale and shift."""
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(z_dim))
        self.shift = nn.Parameter(torch.zeros(z_dim))
        self.linear = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_scaled = z * self.scale + self.shift
        return self.linear(z_scaled)


class MLPFunctor(nn.Module):
    """MLP functor with residual connection."""
    
    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # Small init for near-identity at start
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)  # Residual connection


class ResidualFunctor(nn.Module):
    """Low-rank residual functor: Φ(z) = z + UV^T z"""
    
    def __init__(self, z_dim: int, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(z_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(z_dim, rank) * 0.01)
        self.bias = nn.Parameter(torch.zeros(z_dim))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z + U @ V^T @ z + bias
        delta = z @ self.V  # [batch, rank]
        delta = delta @ self.U.T  # [batch, z_dim]
        return z + delta + self.bias


def create_functor(config: FunctorConfig) -> nn.Module:
    """Create functor module based on config."""
    if config.functor_type == "linear":
        return LinearFunctor(config.z_dim)
    elif config.functor_type == "affine":
        return AffineFunctor(config.z_dim)
    elif config.functor_type == "mlp":
        return MLPFunctor(config.z_dim, config.hidden_dim)
    elif config.functor_type == "residual":
        return ResidualFunctor(config.z_dim)
    else:
        raise ValueError(f"Unknown functor type: {config.functor_type}")


# =============================================================================
# Cone Signature Extraction
# =============================================================================

@torch.no_grad()
def get_cone_signature(
    env: Any,
    actor: Actor,
    z: torch.Tensor,
    n_episodes: int = 20,
    k_sigma: float = 2.0,
    eval_horizon: int = 16,
) -> ConeSignature:
    """
    Extract cone signature for a given commitment z.
    
    Runs multiple episodes and computes:
    - C: weighted log cone volume
    - H: expected horizon E[T]
    - r: soft bind rate
    - λ: current dual pressure
    
    Args:
        env: Environment instance
        actor: Actor instance (frozen)
        z: Commitment tensor [1, z_dim] or [z_dim]
        n_episodes: Number of episodes to average over
        k_sigma: k-sigma for bind rate calculation
        eval_horizon: Horizon for tube evaluation
    
    Returns:
        ConeSignature
    """
    device = actor.device
    z = z.to(device)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    
    log_vol_list = []
    E_T_list = []
    bind_list = []
    
    for _ in range(n_episodes):
        env.reset()
        obs0 = env.observe()
        s0_t = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get tube predictions
        muK, sigK, stop_logit = actor._tube_init(z, s0_t)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        
        # Get trajectory predictions
        T = eval_horizon
        mu, log_var = actor._tube_traj(muK, sigK, T)
        std = torch.exp(0.5 * log_var)  # [T, pred_dim]
        
        # Compute weighted log cone volume
        w, E_T = truncated_geometric_weights(p_stop, T)
        cone_vol_t = torch.prod(std, dim=-1)  # [T]
        log_vol = torch.log(cone_vol_t + 1e-8)
        weighted_log_vol = float((w * log_vol).sum().item())
        
        log_vol_list.append(weighted_log_vol)
        E_T_list.append(float(E_T.item()))
        
        # Roll out to get bind rate
        def policy_fn(obs_np):
            ot = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor._policy_action(z, ot).detach().cpu().numpy().squeeze()
            if actor.action_dim == 1:
                return float(np.clip(float(action), -getattr(env, 'amax', 2.0), getattr(env, 'amax', 2.0)))
            else:
                return int(np.argmax(action))
        
        _, state_seq, _, _ = env.rollout(policy_fn=policy_fn, horizon=T)
        
        # Compute bind rate
        y = torch.tensor(state_seq[1:T+1], dtype=torch.float32, device=device)
        T_actual = min(len(state_seq) - 1, T)
        if T_actual > 0:
            mu_use = mu[:T_actual]
            std_use = std[:T_actual]
            y_use = y[:T_actual]
            inside = (torch.abs(y_use - mu_use) <= k_sigma * std_use + 1e-8).all(dim=-1)
            bind_rate = float(inside.float().mean().item())
        else:
            bind_rate = 0.0
        
        bind_list.append(bind_rate)
    
    return ConeSignature(
        z=z.squeeze().cpu().numpy(),
        log_cone_vol=float(np.mean(log_vol_list)),
        E_T=float(np.mean(E_T_list)),
        bind_rate=float(np.mean(bind_list)),
        lambda_bind=float(actor.lambda_bind),
    )


def collect_cone_dataset(
    env: Any,
    actor: Actor,
    n_samples: int = 200,
    n_eval_episodes: int = 20,
    k_sigma: float = 2.0,
    eval_horizon: int = 16,
) -> List[ConeSignature]:
    """
    Collect dataset of (z, cone_signature) pairs from an environment.
    
    Samples z from the actor's prior and evaluates cone stats.
    """
    device = actor.device
    dataset = []
    
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  Collecting sample {i}/{n_samples}")
        
        # Reset and sample z
        env.reset()
        obs0 = env.observe()
        s0_t = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
        z, _, _ = actor.sample_z(s0_t)
        
        # Get cone signature
        sig = get_cone_signature(
            env, actor, z,
            n_episodes=n_eval_episodes,
            k_sigma=k_sigma,
            eval_horizon=eval_horizon,
        )
        dataset.append(sig)
    
    return dataset


# =============================================================================
# Functor Training
# =============================================================================

class FunctorTrainer:
    """
    Trains a functor Φ: Z_A → Z_B by matching cone signatures.
    
    Training procedure:
    1. Sample z_A from source dataset
    2. Map: z_B = Φ(z_A)
    3. Evaluate cone stats in env B with z_B
    4. Minimize: ||normalize(sig_A) - normalize(sig_B)||²
    """
    
    def __init__(
        self,
        functor: nn.Module,
        actor_A: Actor,
        actor_B: Actor,
        env_factory_A: Callable,
        env_factory_B: Callable,
        env_kwargs_A: Dict[str, Any],
        env_kwargs_B: Dict[str, Any],
        config: FunctorConfig,
    ):
        self.functor = functor
        self.actor_A = actor_A
        self.actor_B = actor_B
        self.env_factory_A = env_factory_A
        self.env_factory_B = env_factory_B
        self.env_kwargs_A = env_kwargs_A
        self.env_kwargs_B = env_kwargs_B
        self.config = config
        
        # Freeze actors
        for param in actor_A.parameters():
            param.requires_grad = False
        for param in actor_B.parameters():
            param.requires_grad = False
        
        # Optimizer (only for functor)
        self.optimizer = optim.Adam(functor.parameters(), lr=config.lr)
        
        # Output directory
        if config.output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path("runs") / f"functor_{timestamp}"
        else:
            self.run_dir = Path(config.output_dir)
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "fig").mkdir(exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        
        # Storage
        self.source_dataset: List[ConeSignature] = []
        self.train_losses: List[float] = []
        self.eval_results: List[Dict[str, Any]] = []
        
        # Normalization stats (computed from source dataset)
        self.sig_mean: Optional[np.ndarray] = None
        self.sig_std: Optional[np.ndarray] = None
    
    def collect_source_data(self, n_samples: int = 200) -> None:
        """Collect cone signatures from source environment A."""
        print("Phase 1: Collecting cone signatures from env A...")
        env_A = self.env_factory_A(self.env_kwargs_A)
        
        self.source_dataset = collect_cone_dataset(
            env_A, self.actor_A,
            n_samples=n_samples,
            n_eval_episodes=self.config.n_eval_episodes,
            k_sigma=self.config.k_sigma,
            eval_horizon=self.config.eval_horizon,
        )
        
        # Compute normalization stats
        sigs = np.array([s.to_vector() for s in self.source_dataset])
        self.sig_mean = sigs.mean(axis=0)
        self.sig_std = sigs.std(axis=0) + 1e-8
        
        print(f"  Collected {len(self.source_dataset)} samples")
        print(f"  Signature stats: mean={self.sig_mean}, std={self.sig_std}")
    
    def _normalize(self, sig: np.ndarray) -> np.ndarray:
        """Normalize signature vector."""
        if self.config.normalize_signatures and self.sig_mean is not None:
            return (sig - self.sig_mean) / self.sig_std
        return sig
    
    def train_epoch(self) -> float:
        """
        Run one epoch of functor training using Evolution Strategies (ES).
        
        Since cone signature computation involves non-differentiable rollouts,
        we use ES to estimate gradients:
        1. Sample perturbations to functor parameters
        2. Evaluate loss for each perturbation
        3. Estimate gradient as weighted sum of perturbations
        """
        self.functor.train()
        env_B = self.env_factory_B(self.env_kwargs_B)
        device = self.actor_B.device
        
        # ES hyperparameters
        n_perturbations = 8  # Number of perturbations per batch
        sigma = 0.1  # Perturbation std
        
        # Shuffle indices
        indices = np.random.permutation(len(self.source_dataset))
        total_loss = 0.0
        n_batches = 0
        
        for start in range(0, len(indices), self.config.batch_size):
            batch_idx = indices[start:start + self.config.batch_size]
            
            # Get baseline loss
            baseline_loss = 0.0
            for idx in batch_idx:
                sample = self.source_dataset[idx]
                z_A = torch.tensor(sample.z, dtype=torch.float32, device=device).unsqueeze(0)
                sig_A = sample.to_vector()
                
                with torch.no_grad():
                    z_B = self.functor(z_A)
                
                sig_B_obj = get_cone_signature(
                    env_B, self.actor_B, z_B,
                    n_episodes=max(1, self.config.n_eval_episodes // 4),
                    k_sigma=self.config.k_sigma,
                    eval_horizon=self.config.eval_horizon,
                )
                sig_B = sig_B_obj.to_vector()
                
                sig_A_norm = self._normalize(sig_A)
                sig_B_norm = self._normalize(sig_B)
                loss = np.sum((sig_A_norm - sig_B_norm) ** 2)
                baseline_loss += loss
            
            baseline_loss /= len(batch_idx)
            total_loss += baseline_loss
            n_batches += 1
            
            # ES gradient estimation
            param_shapes = {name: p.shape for name, p in self.functor.named_parameters()}
            perturbations = []
            perturbed_losses = []
            
            for _ in range(n_perturbations):
                # Sample perturbation
                perturbation = {
                    name: torch.randn_like(p) * sigma
                    for name, p in self.functor.named_parameters()
                }
                perturbations.append(perturbation)
                
                # Apply perturbation
                with torch.no_grad():
                    for name, p in self.functor.named_parameters():
                        p.add_(perturbation[name])
                
                # Evaluate perturbed loss
                perturbed_loss = 0.0
                for idx in batch_idx[:2]:  # Subsample for speed
                    sample = self.source_dataset[idx]
                    z_A = torch.tensor(sample.z, dtype=torch.float32, device=device).unsqueeze(0)
                    sig_A = sample.to_vector()
                    
                    with torch.no_grad():
                        z_B = self.functor(z_A)
                    
                    sig_B_obj = get_cone_signature(
                        env_B, self.actor_B, z_B,
                        n_episodes=1,  # Single episode for speed
                        k_sigma=self.config.k_sigma,
                        eval_horizon=self.config.eval_horizon,
                    )
                    sig_B = sig_B_obj.to_vector()
                    
                    sig_A_norm = self._normalize(sig_A)
                    sig_B_norm = self._normalize(sig_B)
                    perturbed_loss += np.sum((sig_A_norm - sig_B_norm) ** 2)
                
                perturbed_loss /= 2
                perturbed_losses.append(perturbed_loss)
                
                # Remove perturbation
                with torch.no_grad():
                    for name, p in self.functor.named_parameters():
                        p.sub_(perturbation[name])
            
            # Compute ES gradient estimate
            perturbed_losses = np.array(perturbed_losses)
            advantages = -(perturbed_losses - baseline_loss)  # Negative because we minimize
            
            # Weighted gradient
            self.optimizer.zero_grad()
            
            for name, p in self.functor.named_parameters():
                grad = torch.zeros_like(p)
                for i, perturbation in enumerate(perturbations):
                    grad += perturbation[name] * advantages[i]
                grad /= (n_perturbations * sigma)
                p.grad = -grad  # Negative because optimizer minimizes
            
            self.optimizer.step()
        
        return total_loss / max(n_batches, 1)
    
    def train(self, n_epochs: Optional[int] = None) -> None:
        """Run full functor training."""
        n_epochs = n_epochs or self.config.n_epochs
        
        print("\nPhase 2: Training functor...")
        for epoch in range(n_epochs):
            loss = self.train_epoch()
            self.train_losses.append(loss)
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch}: loss={loss:.4f}")
        
        print(f"  Training complete. Final loss: {self.train_losses[-1]:.4f}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate functor quality.
        
        Computes:
        - Cone preservation (correlation between sig_A and sig_B)
        - Transfer delta vs native
        - Per-component errors
        """
        print("\nPhase 3: Evaluating functor...")
        self.functor.eval()
        
        env_A = self.env_factory_A(self.env_kwargs_A)
        env_B = self.env_factory_B(self.env_kwargs_B)
        device = self.actor_B.device
        
        # Collect held-out samples
        n_eval = min(50, len(self.source_dataset))
        eval_indices = np.random.choice(len(self.source_dataset), n_eval, replace=False)
        
        sig_A_list = []
        sig_B_functor_list = []
        sig_B_native_list = []
        z_A_list = []
        z_B_functor_list = []
        errors = []
        
        for idx in eval_indices:
            sample = self.source_dataset[idx]
            z_A = torch.tensor(sample.z, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get functor-mapped z_B
            with torch.no_grad():
                z_B_functor = self.functor(z_A)
            
            # Evaluate in env B with functor z
            sig_B_functor = get_cone_signature(
                env_B, self.actor_B, z_B_functor,
                n_episodes=self.config.n_eval_episodes,
                k_sigma=self.config.k_sigma,
                eval_horizon=self.config.eval_horizon,
            )
            
            # Get native z_B for comparison
            env_B.reset()
            obs_B = env_B.observe()
            s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device).unsqueeze(0)
            z_B_native, _, _ = self.actor_B.sample_z(s0_B)
            
            sig_B_native = get_cone_signature(
                env_B, self.actor_B, z_B_native,
                n_episodes=self.config.n_eval_episodes,
                k_sigma=self.config.k_sigma,
                eval_horizon=self.config.eval_horizon,
            )
            
            sig_A_list.append(sample.to_vector())
            sig_B_functor_list.append(sig_B_functor.to_vector())
            sig_B_native_list.append(sig_B_native.to_vector())
            z_A_list.append(sample.z)
            z_B_functor_list.append(z_B_functor.squeeze().cpu().numpy())
            
            # Compute error
            error = np.sum((self._normalize(sample.to_vector()) - 
                          self._normalize(sig_B_functor.to_vector())) ** 2)
            errors.append(error)
        
        sig_A_arr = np.array(sig_A_list)
        sig_B_functor_arr = np.array(sig_B_functor_list)
        sig_B_native_arr = np.array(sig_B_native_list)
        z_A_arr = np.array(z_A_list)
        z_B_functor_arr = np.array(z_B_functor_list)
        errors_arr = np.array(errors)
        
        # Compute correlations for each signature component
        from scipy import stats
        component_names = ["log_cone_vol", "E_T", "bind_rate", "lambda_bind"]
        correlations = {}
        
        for i, name in enumerate(component_names):
            if np.std(sig_A_arr[:, i]) > 1e-8 and np.std(sig_B_functor_arr[:, i]) > 1e-8:
                r, p = stats.pearsonr(sig_A_arr[:, i], sig_B_functor_arr[:, i])
                correlations[name] = {"pearson_r": float(r), "p_value": float(p)}
            else:
                correlations[name] = {"pearson_r": 0.0, "p_value": 1.0}
        
        # Compute transfer delta (functor vs native)
        transfer_delta = {
            "log_cone_vol": float(np.mean(sig_B_functor_arr[:, 0] - sig_B_native_arr[:, 0])),
            "E_T": float(np.mean(sig_B_functor_arr[:, 1] - sig_B_native_arr[:, 1])),
            "bind_rate": float(np.mean(sig_B_functor_arr[:, 2] - sig_B_native_arr[:, 2])),
        }
        
        results = {
            "correlations": correlations,
            "transfer_delta": transfer_delta,
            "mean_error": float(np.mean(errors_arr)),
            "std_error": float(np.std(errors_arr)),
            "sig_A": sig_A_arr.tolist(),
            "sig_B_functor": sig_B_functor_arr.tolist(),
            "sig_B_native": sig_B_native_arr.tolist(),
            "z_A": z_A_arr.tolist(),
            "z_B_functor": z_B_functor_arr.tolist(),
            "errors": errors_arr.tolist(),
        }
        
        self.eval_results.append(results)
        
        print(f"  Mean signature error: {results['mean_error']:.4f}")
        for name, corr in correlations.items():
            print(f"  {name}: r={corr['pearson_r']:.3f}")
        
        return results
    
    def plot_results(self) -> None:
        """Generate evaluation plots."""
        if not self.eval_results:
            print("No evaluation results to plot")
            return
        
        results = self.eval_results[-1]
        sig_A = np.array(results["sig_A"])
        sig_B_functor = np.array(results["sig_B_functor"])
        sig_B_native = np.array(results["sig_B_native"])
        z_A = np.array(results["z_A"])
        errors = np.array(results["errors"])
        
        component_names = ["Log Cone Vol (C)", "Expected Horizon (H)", "Bind Rate (r)", "λ_bind"]
        
        # Plot 1: Cone preservation scatter (sig_A vs sig_B_functor)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for i, (ax, name) in enumerate(zip(axes.flatten(), component_names)):
            ax.scatter(sig_A[:, i], sig_B_functor[:, i], alpha=0.6, label="Functor")
            ax.scatter(sig_A[:, i], sig_B_native[:, i], alpha=0.4, marker='x', label="Native")
            
            # Add diagonal
            lim_min = min(sig_A[:, i].min(), sig_B_functor[:, i].min())
            lim_max = max(sig_A[:, i].max(), sig_B_functor[:, i].max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label="Perfect")
            
            r = results["correlations"][["log_cone_vol", "E_T", "bind_rate", "lambda_bind"][i]]["pearson_r"]
            ax.set_xlabel(f"{name} (Env A)")
            ax.set_ylabel(f"{name} (Env B)")
            ax.set_title(f"{name}: r={r:.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Cone Signature Preservation: Env A → Env B")
        plt.tight_layout()
        fig.savefig(self.run_dir / "fig" / "cone_preservation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Plot 2: Transfer delta comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = list(results["transfer_delta"].keys())
        functor_means = [results["transfer_delta"][m] for m in metrics]
        
        x = np.arange(len(metrics))
        ax.bar(x, functor_means, alpha=0.7)
        ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Δ (Functor - Native)")
        ax.set_title("Transfer Delta: Functor z_B vs Native z_B")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.run_dir / "fig" / "transfer_delta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Plot 3: Error heatmap in z-space (using PCA for 2D projection)
        if z_A.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_A_2d = pca.fit_transform(z_A)
        else:
            z_A_2d = z_A[:, :2]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        scatter = ax.scatter(z_A_2d[:, 0], z_A_2d[:, 1], c=errors, cmap='hot', alpha=0.7, s=60)
        plt.colorbar(scatter, ax=ax, label="Functor Error")
        ax.set_xlabel("z PC1")
        ax.set_ylabel("z PC2")
        ax.set_title("Functor Error in z-space (source env)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.run_dir / "fig" / "error_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Plot 4: Training loss curve
        if self.train_losses:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(self.train_losses)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Functor Training Loss")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(self.run_dir / "fig" / "training_loss.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        
        print(f"  Plots saved to: {self.run_dir / 'fig'}")
    
    def save_results(self) -> None:
        """Save all results to disk."""
        # Save config
        config_dict = {
            "z_dim": self.config.z_dim,
            "functor_type": self.config.functor_type,
            "lr": self.config.lr,
            "batch_size": self.config.batch_size,
            "n_epochs": self.config.n_epochs,
        }
        with open(self.run_dir / "data" / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save training losses
        with open(self.run_dir / "data" / "training_losses.json", "w") as f:
            json.dump(self.train_losses, f)
        
        # Save evaluation results
        if self.eval_results:
            with open(self.run_dir / "data" / "eval_results.json", "w") as f:
                json.dump(self.eval_results[-1], f, indent=2)
        
        # Save functor weights
        torch.save(self.functor.state_dict(), self.run_dir / "data" / "functor.pt")
        
        # Save source dataset
        dataset_dict = [s.to_dict() for s in self.source_dataset]
        with open(self.run_dir / "data" / "source_dataset.json", "w") as f:
            json.dump(dataset_dict, f, indent=2)
        
        print(f"Results saved to: {self.run_dir}")
    
    def run(self, n_source_samples: int = 200) -> Dict[str, Any]:
        """Run complete functor learning pipeline."""
        print("=" * 60)
        print("Functor Learning: Cross-Environment Commitment Transfer")
        print(f"Run directory: {self.run_dir}")
        print("=" * 60)
        
        # Phase 1: Collect source data
        self.collect_source_data(n_samples=n_source_samples)
        
        # Phase 2: Train functor
        self.train()
        
        # Phase 3: Evaluate
        results = self.evaluate()
        
        # Phase 4: Plot and save
        self.plot_results()
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Functor learning complete!")
        print("=" * 60)
        
        return results


def run_functor_experiment(
    actor_A: Actor,
    actor_B: Actor,
    env_factory_A: Callable,
    env_factory_B: Callable,
    env_kwargs_A: Dict[str, Any],
    env_kwargs_B: Dict[str, Any],
    config: FunctorConfig,
    n_source_samples: int = 200,
) -> FunctorTrainer:
    """
    Convenience function to run a complete functor experiment.
    """
    functor = create_functor(config)
    
    trainer = FunctorTrainer(
        functor=functor,
        actor_A=actor_A,
        actor_B=actor_B,
        env_factory_A=env_factory_A,
        env_factory_B=env_factory_B,
        env_kwargs_A=env_kwargs_A,
        env_kwargs_B=env_kwargs_B,
        config=config,
    )
    
    trainer.run(n_source_samples=n_source_samples)
    return trainer
