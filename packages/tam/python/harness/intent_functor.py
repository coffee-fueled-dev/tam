"""
Intent Functor: Maps z_intent between environments.

V2 - Structural matching via relative ordering, not absolute values.

Key improvements:
1. Rank-based invariants: Match relative ordering, not absolute scale
2. Pairwise/triplet loss: "tighter than" relationship preserved
3. E[T] as primary intent coordinate: horizon is the canonical axis
4. Multiple equivalence classes: test rotation, scaling, permutation

Core insight: We don't need F(z) to produce the same cone volume.
We need F to preserve: "z1 is tighter than z2" → "F(z1) is tighter than F(z2)"
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


@dataclass
class TubeSignature:
    """
    Tube-only signature for an intent z_intent.
    
    Primary coordinate: E[T] (horizon) - the canonical intent axis
    Secondary: log_cone_vol - for ordering comparisons
    """
    z_intent: np.ndarray
    E_T: float  # Primary: expected horizon (canonical intent)
    log_cone_vol: float  # Secondary: for ordering
    
    def to_vector(self) -> np.ndarray:
        """Convert to 2D vector. E[T] first (primary)."""
        return np.array([self.E_T, self.log_cone_vol])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "z_intent": self.z_intent.tolist(),
            "E_T": self.E_T,
            "log_cone_vol": self.log_cone_vol,
        }


@dataclass
class PairedSample:
    """A paired sample from environments A and B."""
    state_dict_A: Dict
    obs_A: np.ndarray
    z_intent_A: np.ndarray
    sig_A: TubeSignature
    obs_B: np.ndarray
    # Ranks within dataset (computed after collection)
    rank_E_T: Optional[int] = None
    rank_cone: Optional[int] = None
    quantile_E_T: Optional[float] = None  # 0-1
    quantile_cone: Optional[float] = None  # 0-1


@dataclass 
class IntentFunctorConfig:
    """Configuration for intent functor learning."""
    z_intent_dim: int = 4
    hidden_dim: int = 64
    
    # Functor architecture
    functor_type: str = "affine"  # "affine", "mlp"
    
    # Training
    lr: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 50
    
    # Loss type
    loss_type: str = "ordering"  # "ordering", "triplet", "rank_corr"
    margin: float = 0.1  # margin for ordering/triplet loss
    
    # Evaluation
    k_sigma: float = 2.0
    eval_horizon: int = 16
    
    # Diversity
    intent_noise_std: float = 0.2
    
    # Output
    output_dir: Optional[Path] = None
    seed: int = 0


class AffineFunctor(nn.Module):
    """Pure affine functor: F(z) = Wz + b"""
    
    def __init__(self, z_intent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(z_intent_dim))
        self.b = nn.Parameter(torch.zeros(z_intent_dim))
    
    def forward(self, z_intent: torch.Tensor) -> torch.Tensor:
        return z_intent @ self.W.T + self.b


class MLPFunctor(nn.Module):
    """MLP functor with residual."""
    
    def __init__(self, z_intent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_intent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_intent_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, z_intent: torch.Tensor) -> torch.Tensor:
        return z_intent + self.net(z_intent)


def create_intent_functor(config: IntentFunctorConfig) -> nn.Module:
    if config.functor_type == "affine":
        return AffineFunctor(config.z_intent_dim)
    elif config.functor_type == "mlp":
        return MLPFunctor(config.z_intent_dim, config.hidden_dim)
    else:
        raise ValueError(f"Unknown functor type: {config.functor_type}")


@torch.no_grad()
def get_tube_signature(
    actor,
    z_intent: torch.Tensor,
    s0_t: torch.Tensor,
    eval_horizon: int = 16,
) -> TubeSignature:
    """Extract tube-only signature. E[T] is primary."""
    device = actor.device
    z_intent = z_intent.to(device)
    s0_t = s0_t.to(device)
    
    if z_intent.dim() == 1:
        z_intent = z_intent.unsqueeze(0)
    if s0_t.dim() == 1:
        s0_t = s0_t.unsqueeze(0)
    
    # Dummy z_real
    z_real = torch.zeros(z_intent.size(0), actor.z_real_dim, device=device)
    z = torch.cat([z_intent, z_real], dim=-1)
    
    muK, sigK, stop_logit = actor._tube_init(z, s0_t)
    p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
    
    T = eval_horizon
    _, log_var = actor._tube_traj(muK, sigK, T)
    std = torch.exp(0.5 * log_var)
    
    w, E_T = truncated_geometric_weights(p_stop, T)
    cone_vol_t = torch.prod(std, dim=-1)
    log_vol_t = torch.log(cone_vol_t + 1e-8)
    weighted_log_vol = float((w * log_vol_t).sum().item())
    
    return TubeSignature(
        z_intent=z_intent.squeeze().cpu().numpy(),
        E_T=float(E_T.item()),
        log_cone_vol=weighted_log_vol,
    )


def collect_paired_dataset(
    env_A,
    env_B,
    actor_A,
    n_samples: int = 200,
    eval_horizon: int = 16,
    intent_noise_std: float = 0.2,
    use_structured_diversity: bool = True,  # Use algebra operators for diversity
) -> List[PairedSample]:
    """
    Collect paired dataset and compute ranks/quantiles.
    
    If use_structured_diversity=True, we explicitly vary z_intent along
    different "intent axes" to ensure we get diverse commitments.
    """
    device = actor_A.device
    dataset = []
    
    # Get intent space dimensions
    z_intent_dim = actor_A.z_intent_dim
    
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  Collecting paired sample {i}/{n_samples}")
        
        obs_A = env_A.reset()
        state_dict_A = env_A.get_state_dict()
        
        if hasattr(env_B, 'reset_paired'):
            obs_B = env_B.reset_paired(state_dict_A)
        else:
            obs_B = env_B.reset_with_state(state_dict_A)
        
        s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get base z_intent
        z_intent_mu = actor_A.get_intent_embedding(s0_A)
        
        if use_structured_diversity:
            # Create structured diversity: vary along random directions with varying magnitudes
            # This ensures we get a spread of intents even if the actor didn't learn to vary
            
            # Random direction in z_intent space
            direction = torch.randn(1, z_intent_dim, device=device)
            direction = direction / (direction.norm() + 1e-8)
            
            # Random magnitude (uniform from -3 to +3)
            magnitude = (torch.rand(1, device=device) * 6.0 - 3.0).item()
            
            z_intent = z_intent_mu + magnitude * direction
        else:
            # Original: just add noise
            if intent_noise_std > 0:
                noise = torch.randn_like(z_intent_mu) * intent_noise_std
                z_intent = z_intent_mu + noise
            else:
                z_intent = z_intent_mu
        
        sig_A = get_tube_signature(actor_A, z_intent, s0_A, eval_horizon)
        
        dataset.append(PairedSample(
            state_dict_A=state_dict_A,
            obs_A=obs_A,
            z_intent_A=z_intent.squeeze().cpu().numpy(),
            sig_A=sig_A,
            obs_B=obs_B,
        ))
    
    # Compute ranks and quantiles
    E_Ts = np.array([s.sig_A.E_T for s in dataset])
    cones = np.array([s.sig_A.log_cone_vol for s in dataset])
    
    E_T_order = np.argsort(E_Ts)
    cone_order = np.argsort(cones)
    
    for rank, idx in enumerate(E_T_order):
        dataset[idx].rank_E_T = rank
        dataset[idx].quantile_E_T = rank / (len(dataset) - 1 + 1e-8)
    
    for rank, idx in enumerate(cone_order):
        dataset[idx].rank_cone = rank
        dataset[idx].quantile_cone = rank / (len(dataset) - 1 + 1e-8)
    
    return dataset


def compute_ordering_loss(
    sig_A_i: np.ndarray,
    sig_A_j: np.ndarray,
    sig_B_i: np.ndarray,
    sig_B_j: np.ndarray,
    margin: float = 0.1,
) -> float:
    """
    Pairwise ordering loss: preserve "i < j" relationship.
    
    For each dimension, if A_i < A_j, then we want B_i < B_j.
    Loss = sum of margin violations.
    """
    loss = 0.0
    
    for d in range(len(sig_A_i)):
        # Direction in A
        diff_A = sig_A_j[d] - sig_A_i[d]
        diff_B = sig_B_j[d] - sig_B_i[d]
        
        # Same sign = preserved ordering
        # We want: sign(diff_A) == sign(diff_B)
        # Loss if they have opposite signs
        if abs(diff_A) > 1e-6:  # Only count if there's a meaningful difference
            # Hinge loss: max(0, margin - diff_A * diff_B / |diff_A|)
            normalized_agreement = diff_A * diff_B / (abs(diff_A) + 1e-8)
            loss += max(0.0, margin - normalized_agreement)
    
    return loss


def compute_triplet_loss(
    sig_anchor: np.ndarray,
    sig_positive: np.ndarray,  # Closer in rank
    sig_negative: np.ndarray,  # Farther in rank
    margin: float = 0.1,
) -> float:
    """
    Triplet loss: anchor should be closer to positive than negative.
    
    This preserves local neighborhoods in the ordering.
    """
    dist_pos = np.sum((sig_anchor - sig_positive) ** 2)
    dist_neg = np.sum((sig_anchor - sig_negative) ** 2)
    
    # Want dist_pos < dist_neg by margin
    return max(0.0, dist_pos - dist_neg + margin)


def compute_rank_correlation_loss(
    ranks_A: np.ndarray,
    values_B: np.ndarray,
) -> float:
    """
    Rank correlation loss: maximize Spearman correlation.
    
    ranks_A: rank ordering in env A (what we want to preserve)
    values_B: values in env B (should have same ordering)
    """
    from scipy.stats import spearmanr
    
    if len(ranks_A) < 3:
        return 0.0
    
    # Rank the B values
    ranks_B = np.argsort(np.argsort(values_B))
    
    r, _ = spearmanr(ranks_A, ranks_B)
    if np.isnan(r):
        return 1.0
    
    # Loss = 1 - correlation (want to minimize)
    return 1.0 - r


class IntentFunctorTrainer:
    """
    Trains intent functor using ORDERING-based loss.
    
    Key insight: Preserve relative structure, not absolute values.
    """
    
    def __init__(
        self,
        functor: nn.Module,
        actor_A,
        actor_B,
        env_factory_A: Callable,
        env_factory_B: Callable,
        env_kwargs_A: Dict[str, Any],
        env_kwargs_B: Dict[str, Any],
        config: IntentFunctorConfig,
    ):
        self.functor = functor
        self.actor_A = actor_A
        self.actor_B = actor_B
        self.env_factory_A = env_factory_A
        self.env_factory_B = env_factory_B
        self.env_kwargs_A = env_kwargs_A
        self.env_kwargs_B = env_kwargs_B
        self.config = config
        
        for param in actor_A.parameters():
            param.requires_grad = False
        for param in actor_B.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(functor.parameters(), lr=config.lr)
        
        if config.output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path("runs") / f"intent_functor_{timestamp}"
        else:
            self.run_dir = Path(config.output_dir)
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "fig").mkdir(exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        
        self.paired_dataset: List[PairedSample] = []
        self.train_losses: List[float] = []
        self.eval_results: List[Dict[str, Any]] = []
    
    def collect_paired_data(self, n_samples: int = 200) -> None:
        print("Phase 1: Collecting PAIRED dataset with rank annotations...")
        env_A = self.env_factory_A(self.env_kwargs_A)
        env_B = self.env_factory_B(self.env_kwargs_B)
        
        self.paired_dataset = collect_paired_dataset(
            env_A, env_B, self.actor_A,
            n_samples=n_samples,
            eval_horizon=self.config.eval_horizon,
            intent_noise_std=self.config.intent_noise_std,
        )
        
        # Stats
        E_Ts = [s.sig_A.E_T for s in self.paired_dataset]
        cones = [s.sig_A.log_cone_vol for s in self.paired_dataset]
        
        print(f"  Collected {len(self.paired_dataset)} samples")
        print(f"  E[T] range: [{min(E_Ts):.2f}, {max(E_Ts):.2f}]")
        print(f"  Log cone vol range: [{min(cones):.2f}, {max(cones):.2f}]")
    
    def _get_sig_B(self, sample: PairedSample) -> TubeSignature:
        """Get signature in B for sample with current functor."""
        device = self.actor_B.device
        z_intent_A = torch.tensor(sample.z_intent_A, dtype=torch.float32, device=device).unsqueeze(0)
        s0_B = torch.tensor(sample.obs_B, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            z_intent_B = self.functor(z_intent_A)
        
        return get_tube_signature(self.actor_B, z_intent_B, s0_B, self.config.eval_horizon)
    
    def _compute_batch_loss(self, batch_samples: List[PairedSample]) -> float:
        """Compute loss for a batch using ordering/triplet/rank_corr."""
        if len(batch_samples) < 2:
            return 0.0
        
        # Get B signatures
        sigs_B = [self._get_sig_B(s) for s in batch_samples]
        
        if self.config.loss_type == "ordering":
            # Pairwise ordering loss
            total_loss = 0.0
            n_pairs = 0
            
            for i in range(len(batch_samples)):
                for j in range(i + 1, len(batch_samples)):
                    loss = compute_ordering_loss(
                        batch_samples[i].sig_A.to_vector(),
                        batch_samples[j].sig_A.to_vector(),
                        sigs_B[i].to_vector(),
                        sigs_B[j].to_vector(),
                        margin=self.config.margin,
                    )
                    total_loss += loss
                    n_pairs += 1
            
            return total_loss / max(n_pairs, 1)
        
        elif self.config.loss_type == "triplet":
            # Triplet loss based on E[T] rank
            total_loss = 0.0
            n_triplets = 0
            
            # Sort by rank
            sorted_samples = sorted(zip(batch_samples, sigs_B), key=lambda x: x[0].rank_E_T)
            
            for i in range(1, len(sorted_samples) - 1):
                anchor = sorted_samples[i]
                positive = sorted_samples[i - 1] if i > 0 else sorted_samples[i + 1]
                negative = sorted_samples[-1] if i < len(sorted_samples) - 1 else sorted_samples[0]
                
                loss = compute_triplet_loss(
                    anchor[1].to_vector(),
                    positive[1].to_vector(),
                    negative[1].to_vector(),
                    margin=self.config.margin,
                )
                total_loss += loss
                n_triplets += 1
            
            return total_loss / max(n_triplets, 1)
        
        elif self.config.loss_type == "rank_corr":
            # Rank correlation loss for E[T]
            ranks_A = np.array([s.rank_E_T for s in batch_samples])
            values_B = np.array([s.E_T for s in sigs_B])
            
            loss_ET = compute_rank_correlation_loss(ranks_A, values_B)
            
            # Also for cone vol
            ranks_A_cone = np.array([s.rank_cone for s in batch_samples])
            values_B_cone = np.array([s.log_cone_vol for s in sigs_B])
            
            loss_cone = compute_rank_correlation_loss(ranks_A_cone, values_B_cone)
            
            # Weight E[T] higher (primary coordinate)
            return 0.7 * loss_ET + 0.3 * loss_cone
        
        else:
            return 0.0
    
    def train_epoch(self) -> float:
        """Train one epoch using ES on ordering loss."""
        self.functor.train()
        
        n_perturbations = 8
        sigma = 0.05
        
        indices = np.random.permutation(len(self.paired_dataset))
        total_loss = 0.0
        n_batches = 0
        
        # Larger batches for ordering loss (need pairs/triplets)
        batch_size = max(self.config.batch_size, 16)
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_samples = [self.paired_dataset[i] for i in batch_idx]
            
            if len(batch_samples) < 4:
                continue
            
            # Baseline loss
            baseline_loss = self._compute_batch_loss(batch_samples)
            total_loss += baseline_loss
            n_batches += 1
            
            # ES gradient estimation
            perturbations = []
            perturbed_losses = []
            
            for _ in range(n_perturbations):
                perturbation = {
                    name: torch.randn_like(p) * sigma
                    for name, p in self.functor.named_parameters()
                }
                perturbations.append(perturbation)
                
                with torch.no_grad():
                    for name, p in self.functor.named_parameters():
                        p.add_(perturbation[name])
                
                # Evaluate on subset
                subset = batch_samples[:min(8, len(batch_samples))]
                perturbed_loss = self._compute_batch_loss(subset)
                perturbed_losses.append(perturbed_loss)
                
                with torch.no_grad():
                    for name, p in self.functor.named_parameters():
                        p.sub_(perturbation[name])
            
            perturbed_losses = np.array(perturbed_losses)
            advantages = -(perturbed_losses - baseline_loss)
            
            if np.std(advantages) > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            self.optimizer.zero_grad()
            
            for name, p in self.functor.named_parameters():
                grad = torch.zeros_like(p)
                for i, perturbation in enumerate(perturbations):
                    grad += perturbation[name] * advantages[i]
                grad /= (n_perturbations * sigma)
                p.grad = -grad
            
            torch.nn.utils.clip_grad_norm_(self.functor.parameters(), 1.0)
            self.optimizer.step()
        
        return total_loss / max(n_batches, 1)
    
    def train(self, n_epochs: Optional[int] = None) -> None:
        n_epochs = n_epochs or self.config.n_epochs
        
        print(f"\nPhase 2: Training with {self.config.loss_type} loss...")
        for epoch in range(n_epochs):
            loss = self.train_epoch()
            self.train_losses.append(loss)
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch}: loss={loss:.4f}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate: rank correlation preservation."""
        print("\nPhase 3: Evaluating rank preservation...")
        self.functor.eval()
        
        from scipy.stats import spearmanr, kendalltau
        
        n_eval = min(100, len(self.paired_dataset))
        eval_samples = self.paired_dataset[:n_eval]
        
        # Get all B signatures
        sigs_B_transported = [self._get_sig_B(s) for s in eval_samples]
        
        # Also get identity baseline
        sigs_B_identity = []
        device = self.actor_B.device
        for s in eval_samples:
            z_intent_A = torch.tensor(s.z_intent_A, dtype=torch.float32, device=device).unsqueeze(0)
            s0_B = torch.tensor(s.obs_B, dtype=torch.float32, device=device).unsqueeze(0)
            sig = get_tube_signature(self.actor_B, z_intent_A, s0_B, self.config.eval_horizon)
            sigs_B_identity.append(sig)
        
        # Extract values
        E_T_A = np.array([s.sig_A.E_T for s in eval_samples])
        cone_A = np.array([s.sig_A.log_cone_vol for s in eval_samples])
        
        E_T_B_trans = np.array([s.E_T for s in sigs_B_transported])
        cone_B_trans = np.array([s.log_cone_vol for s in sigs_B_transported])
        
        E_T_B_id = np.array([s.E_T for s in sigs_B_identity])
        cone_B_id = np.array([s.log_cone_vol for s in sigs_B_identity])
        
        # Compute rank correlations
        results = {
            "E_T": {
                "transported_spearman": float(spearmanr(E_T_A, E_T_B_trans)[0]) if not np.isnan(spearmanr(E_T_A, E_T_B_trans)[0]) else 0.0,
                "transported_kendall": float(kendalltau(E_T_A, E_T_B_trans)[0]) if not np.isnan(kendalltau(E_T_A, E_T_B_trans)[0]) else 0.0,
                "identity_spearman": float(spearmanr(E_T_A, E_T_B_id)[0]) if not np.isnan(spearmanr(E_T_A, E_T_B_id)[0]) else 0.0,
                "identity_kendall": float(kendalltau(E_T_A, E_T_B_id)[0]) if not np.isnan(kendalltau(E_T_A, E_T_B_id)[0]) else 0.0,
            },
            "cone": {
                "transported_spearman": float(spearmanr(cone_A, cone_B_trans)[0]) if not np.isnan(spearmanr(cone_A, cone_B_trans)[0]) else 0.0,
                "transported_kendall": float(kendalltau(cone_A, cone_B_trans)[0]) if not np.isnan(kendalltau(cone_A, cone_B_trans)[0]) else 0.0,
                "identity_spearman": float(spearmanr(cone_A, cone_B_id)[0]) if not np.isnan(spearmanr(cone_A, cone_B_id)[0]) else 0.0,
                "identity_kendall": float(kendalltau(cone_A, cone_B_id)[0]) if not np.isnan(kendalltau(cone_A, cone_B_id)[0]) else 0.0,
            },
            "raw_data": {
                "E_T_A": E_T_A.tolist(),
                "E_T_B_trans": E_T_B_trans.tolist(),
                "E_T_B_id": E_T_B_id.tolist(),
                "cone_A": cone_A.tolist(),
                "cone_B_trans": cone_B_trans.tolist(),
                "cone_B_id": cone_B_id.tolist(),
            }
        }
        
        # Compute improvement
        results["improvement"] = {
            "E_T_spearman": results["E_T"]["transported_spearman"] - results["E_T"]["identity_spearman"],
            "cone_spearman": results["cone"]["transported_spearman"] - results["cone"]["identity_spearman"],
        }
        
        self.eval_results.append(results)
        
        print(f"\n  Rank Preservation (Spearman ρ):")
        print(f"    E[T]:     Transported={results['E_T']['transported_spearman']:.3f}, Identity={results['E_T']['identity_spearman']:.3f}, Δ={results['improvement']['E_T_spearman']:+.3f}")
        print(f"    Cone Vol: Transported={results['cone']['transported_spearman']:.3f}, Identity={results['cone']['identity_spearman']:.3f}, Δ={results['improvement']['cone_spearman']:+.3f}")
        
        return results
    
    def plot_results(self) -> None:
        if not self.eval_results:
            return
        
        results = self.eval_results[-1]
        
        E_T_A = np.array(results["raw_data"]["E_T_A"])
        E_T_B_trans = np.array(results["raw_data"]["E_T_B_trans"])
        E_T_B_id = np.array(results["raw_data"]["E_T_B_id"])
        cone_A = np.array(results["raw_data"]["cone_A"])
        cone_B_trans = np.array(results["raw_data"]["cone_B_trans"])
        cone_B_id = np.array(results["raw_data"]["cone_B_id"])
        
        # Plot 1: Rank preservation scatter (E[T] primary)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # E[T]
        ax = axes[0]
        ax.scatter(E_T_A, E_T_B_trans, alpha=0.7, label=f"Transported (ρ={results['E_T']['transported_spearman']:.3f})", s=50)
        ax.scatter(E_T_A, E_T_B_id, alpha=0.4, marker='x', label=f"Identity (ρ={results['E_T']['identity_spearman']:.3f})", s=30)
        
        # Trend lines
        z_trans = np.polyfit(E_T_A, E_T_B_trans, 1)
        z_id = np.polyfit(E_T_A, E_T_B_id, 1)
        x_line = np.linspace(E_T_A.min(), E_T_A.max(), 100)
        ax.plot(x_line, np.polyval(z_trans, x_line), 'b--', alpha=0.5, label='Transported trend')
        ax.plot(x_line, np.polyval(z_id, x_line), 'orange', linestyle='--', alpha=0.5, label='Identity trend')
        
        ax.set_xlabel("E[T] in Env A")
        ax.set_ylabel("E[T] in Env B")
        ax.set_title(f"E[T] Preservation (PRIMARY)\nΔρ = {results['improvement']['E_T_spearman']:+.3f}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Cone vol
        ax = axes[1]
        ax.scatter(cone_A, cone_B_trans, alpha=0.7, label=f"Transported (ρ={results['cone']['transported_spearman']:.3f})", s=50)
        ax.scatter(cone_A, cone_B_id, alpha=0.4, marker='x', label=f"Identity (ρ={results['cone']['identity_spearman']:.3f})", s=30)
        
        ax.set_xlabel("Log Cone Vol in Env A")
        ax.set_ylabel("Log Cone Vol in Env B")
        ax.set_title(f"Cone Volume Preservation (secondary)\nΔρ = {results['improvement']['cone_spearman']:+.3f}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle("Rank Ordering Preservation: Does F preserve 'tighter than' / 'longer horizon'?")
        plt.tight_layout()
        fig.savefig(self.run_dir / "fig" / "rank_preservation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Plot 2: Bar chart comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ["E[T] (primary)", "Cone Vol"]
        transported = [results["E_T"]["transported_spearman"], results["cone"]["transported_spearman"]]
        identity = [results["E_T"]["identity_spearman"], results["cone"]["identity_spearman"]]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, transported, width, label='Transported F(z)', color='steelblue')
        bars2 = ax.bar(x + width/2, identity, width, label='Identity z', color='gray')
        
        ax.set_ylabel("Spearman ρ (rank correlation)")
        ax.set_title("Rank Preservation: Functor vs Identity Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars1, transported):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)
        for bar, val in zip(bars2, identity):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(self.run_dir / "fig" / "rank_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Plot 3: Training loss
        if self.train_losses:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(self.train_losses)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{self.config.loss_type.capitalize()} Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(self.run_dir / "fig" / "training_loss.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        
        # Plot 4: Functor weights
        if isinstance(self.functor, AffineFunctor):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            W = self.functor.W.detach().cpu().numpy()
            b = self.functor.b.detach().cpu().numpy()
            
            im = axes[0].imshow(W, cmap='coolwarm', aspect='auto', vmin=-2, vmax=2)
            axes[0].set_title("Functor W (deviation from identity shown)")
            plt.colorbar(im, ax=axes[0])
            
            # Show how far from identity
            identity = np.eye(W.shape[0])
            diff = W - identity
            axes[1].imshow(diff, cmap='coolwarm', aspect='auto', vmin=-0.5, vmax=0.5)
            axes[1].set_title("W - I (learned deviation)")
            
            plt.tight_layout()
            fig.savefig(self.run_dir / "fig" / "functor_weights.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        
        print(f"  Plots saved to: {self.run_dir / 'fig'}")
    
    def save_results(self) -> None:
        with open(self.run_dir / "data" / "config.json", "w") as f:
            json.dump({
                "z_intent_dim": self.config.z_intent_dim,
                "functor_type": self.config.functor_type,
                "lr": self.config.lr,
                "n_epochs": self.config.n_epochs,
                "loss_type": self.config.loss_type,
                "margin": self.config.margin,
            }, f, indent=2)
        
        with open(self.run_dir / "data" / "training_losses.json", "w") as f:
            json.dump(self.train_losses, f)
        
        if self.eval_results:
            with open(self.run_dir / "data" / "eval_results.json", "w") as f:
                json.dump(self.eval_results[-1], f, indent=2)
        
        torch.save(self.functor.state_dict(), self.run_dir / "data" / "intent_functor.pt")
        
        print(f"Results saved to: {self.run_dir}")
    
    def run(self, n_source_samples: int = 200) -> Dict[str, Any]:
        print("=" * 60)
        print("Intent Functor V2: Rank/Ordering Preservation")
        print(f"Loss type: {self.config.loss_type}")
        print(f"Primary coordinate: E[T] (horizon)")
        print(f"Run directory: {self.run_dir}")
        print("=" * 60)
        
        self.collect_paired_data(n_samples=n_source_samples)
        self.train()
        results = self.evaluate()
        self.plot_results()
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Complete!")
        print("=" * 60)
        
        return results


def run_intent_functor_experiment(
    actor_A,
    actor_B,
    env_factory_A: Callable,
    env_factory_B: Callable,
    env_kwargs_A: Dict[str, Any],
    env_kwargs_B: Dict[str, Any],
    config: IntentFunctorConfig,
    n_source_samples: int = 200,
) -> IntentFunctorTrainer:
    functor = create_intent_functor(config)
    
    trainer = IntentFunctorTrainer(
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


def run_multi_env_evaluation(
    actor_A,
    actor_B_dict: Dict[str, Any],  # {name: (actor_B, env_factory_B, env_kwargs_B)}
    env_factory_A: Callable,
    env_kwargs_A: Dict[str, Any],
    config: IntentFunctorConfig,
    n_samples: int = 200,
) -> Dict[str, Any]:
    """
    Evaluate functor across multiple target environments.
    
    Tests: mirrored, rotated, scaled, shifted-rules
    """
    results = {}
    
    for name, (actor_B, env_factory_B, env_kwargs_B) in actor_B_dict.items():
        print(f"\n{'='*60}")
        print(f"Testing on: {name}")
        print("=" * 60)
        
        config.output_dir = Path("runs") / f"intent_functor_{name}"
        
        trainer = run_intent_functor_experiment(
            actor_A=actor_A,
            actor_B=actor_B,
            env_factory_A=env_factory_A,
            env_factory_B=env_factory_B,
            env_kwargs_A=env_kwargs_A,
            env_kwargs_B=env_kwargs_B,
            config=config,
            n_source_samples=n_samples,
        )
        
        results[name] = trainer.eval_results[-1] if trainer.eval_results else {}
    
    return results
