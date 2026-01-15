"""
Reference Probe Gauge Alignment.

Uses a minimal set of shared reference probes to fix gauge across actors
without manual intervention. The probes are designed to:
- Excite the same basin structure (topology) across actors
- Be agnostic to each actor's gauge (rotation/permutation/sign)
- Be minimal (K+1 probes for K basins)
- Sufficient to fix gauge when training functors on probe-consistency

Key insight: boundary probes (midpoints between goals) reveal where "choice happens"
and a tie-break probe breaks remaining symmetries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .zmap import ZMap
from .utils_energy import sample_z_sphere, compute_energy


@dataclass
class ReferenceProbeConfig:
    """Configuration for reference probe experiment."""
    z_dim: int = 2
    K: int = 3  # Number of basins/modes
    n_probes: int = None  # Default: K+1
    tau: float = 0.05  # Temperature for basin softmax
    lambda_cyc: float = 1.0
    lambda_comp: float = 1.0
    lambda_probe: float = 2.0  # Probe consistency weight
    lr: float = 1e-3
    
    def __post_init__(self):
        if self.n_probes is None:
            self.n_probes = self.K + 1


def minimal_probe_set_from_goals(
    goals: np.ndarray,
    eps: float = 0.1,
    seed: int = 0,
    use_cyclic: bool = False,
    n_probes: int = None,
) -> np.ndarray:
    """
    Generate minimal probe set from goal vectors.
    
    Args:
        goals: (K, d) goal vectors in state space
        eps: Magnitude of tie-break offset
        seed: Random seed
        use_cyclic: If True, use cyclic midpoints instead of nearest-neighbor
        n_probes: Number of probes to generate (default: K+1)
                  If < K+1, skips tie-break and/or samples boundary probes
    
    Returns:
        probes: (n_probes, d) probe positions
    """
    rng = np.random.default_rng(seed)
    K, d = goals.shape
    
    if n_probes is None:
        n_probes = K + 1
    
    # 1) Boundary probes: midpoints
    boundary_probes = []
    for k in range(K):
        if use_cyclic:
            # Cyclic midpoint: between k and (k+1) mod K
            j = (k + 1) % K
        else:
            # Nearest neighbor midpoint
            diffs = goals - goals[k]
            dists = np.linalg.norm(diffs, axis=1)
            dists[k] = np.inf
            j = int(np.argmin(dists))
        
        boundary_probes.append(0.5 * (goals[k] + goals[j]))
    
    boundary_probes = np.stack(boundary_probes, axis=0)  # (K, d)
    
    # 2) Tie-break probe: centroid + small random offset
    g_bar = goals.mean(axis=0)
    u = rng.normal(size=d)
    u = u / (np.linalg.norm(u) + 1e-8)
    tie_probe = g_bar + eps * u
    
    # Combine all probes
    all_probes = np.concatenate([boundary_probes, tie_probe[None, :]], axis=0)  # (K+1, d)
    
    # Subsample if n_probes < K+1
    if n_probes < K + 1:
        # Prioritize: first n_probes-1 boundary probes, then tie-break if room
        if n_probes <= K:
            # Sample boundary probes only (no tie-break)
            indices = rng.choice(K, size=n_probes, replace=False)
            all_probes = boundary_probes[indices]
        else:
            # This shouldn't happen since n_probes < K+1 and n_probes > K is impossible
            all_probes = all_probes[:n_probes]
    elif n_probes > K + 1:
        # Add more random probes if requested
        extra = n_probes - (K + 1)
        extra_probes = []
        for _ in range(extra):
            # Random point near a random goal
            g_idx = rng.integers(0, K)
            offset = rng.normal(size=d) * eps * 2
            extra_probes.append(goals[g_idx] + offset)
        extra_probes = np.stack(extra_probes, axis=0)
        all_probes = np.concatenate([all_probes, extra_probes], axis=0)
    
    return all_probes.astype(np.float32)


def get_goals_from_env(env) -> np.ndarray:
    """
    Extract goal vectors from environment.
    
    Works with CMGEnv and CMGWorldVariant.
    """
    # Try to access goals through different interfaces
    if hasattr(env, 'base_env'):
        # CMGWorldVariant wraps CMGEnv
        base = env.base_env
        if hasattr(base, 'params') and hasattr(base.params, 'g'):
            return base.params.g.cpu().numpy() if torch.is_tensor(base.params.g) else base.params.g
    
    if hasattr(env, 'params') and hasattr(env.params, 'g'):
        g = env.params.g
        return g.cpu().numpy() if torch.is_tensor(g) else g
    
    # Fallback: generate random goals
    K = getattr(env, 'K', 3)
    d = getattr(env, 'state_dim', 3)
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    goals = np.stack([np.cos(angles), np.sin(angles), np.zeros(K)], axis=1)
    return goals.astype(np.float32)


def probes_to_observations(
    probes: np.ndarray,
    env,
) -> np.ndarray:
    """
    Convert probe positions to full observations.
    
    For CMGEnv, observation = [position, goal, time] or similar.
    """
    K_probes = probes.shape[0]
    obs_dim = getattr(env, 'obs_dim', probes.shape[1] * 2 + 1)
    d = probes.shape[1]  # State dimension
    
    # Get a sample goal from env to use as context
    goals = get_goals_from_env(env)
    default_goal = goals.mean(axis=0)  # Use centroid as "neutral" goal
    
    obs_list = []
    for i in range(K_probes):
        # Start with [position, goal]
        obs = np.concatenate([probes[i], default_goal])
        
        # Pad with zeros to match obs_dim (handles time, mode_id, etc.)
        if len(obs) < obs_dim:
            padding = np.zeros(obs_dim - len(obs), dtype=np.float32)
            obs = np.concatenate([obs, padding])
        elif len(obs) > obs_dim:
            obs = obs[:obs_dim]
        
        obs_list.append(obs)
    
    return np.stack(obs_list, axis=0).astype(np.float32)


@torch.no_grad()
def get_best_z_for_obs(
    actor,
    obs: torch.Tensor,
    n_candidates: int = 100,
) -> torch.Tensor:
    """
    Get best z commitment for an observation via CEM-like search.
    
    Args:
        actor: Actor model
        obs: (obs_dim,) observation
        n_candidates: Number of z candidates to evaluate
    
    Returns:
        best_z: (z_dim,) best commitment
    """
    device = obs.device
    z_dim = actor.z_dim
    
    # Sample candidates on unit sphere
    z_candidates = sample_z_sphere(n_candidates, z_dim, device)
    
    # Compute energy for each
    energies = compute_energy(actor, obs, z_candidates)
    
    # Return lowest energy z
    best_idx = energies.argmin()
    return z_candidates[best_idx]


class ReferenceProbeTrainer:
    """
    Trains functor maps using shared reference probes for gauge alignment.
    
    The key addition over TriadicFunctorTrainer: probe consistency loss
    that encourages functors to agree on probe responses.
    """
    
    def __init__(
        self,
        actor_A,
        actor_B,
        actor_C,
        env_A,
        env_B,
        env_C,
        config: ReferenceProbeConfig,
        device: torch.device,
    ):
        self.actor_A = actor_A
        self.actor_B = actor_B
        self.actor_C = actor_C
        self.env_A = env_A
        self.env_B = env_B
        self.env_C = env_C
        self.config = config
        self.device = device
        
        # Freeze actors
        for actor in [actor_A, actor_B, actor_C]:
            for param in actor.parameters():
                param.requires_grad = False
        
        # Generate probes from environment goals
        goals = get_goals_from_env(env_A)
        self.probe_positions = minimal_probe_set_from_goals(
            goals, seed=42, n_probes=config.n_probes
        )
        
        # Convert to observations for each environment
        self.probes_A = torch.tensor(
            probes_to_observations(self.probe_positions, env_A),
            dtype=torch.float32, device=device
        )
        self.probes_B = torch.tensor(
            probes_to_observations(self.probe_positions, env_B),
            dtype=torch.float32, device=device
        )
        self.probes_C = torch.tensor(
            probes_to_observations(self.probe_positions, env_C),
            dtype=torch.float32, device=device
        )
        
        # 6 functor maps
        self.F_AB = ZMap(config.z_dim).to(device)
        self.F_BA = ZMap(config.z_dim).to(device)
        self.F_AC = ZMap(config.z_dim).to(device)
        self.F_CA = ZMap(config.z_dim).to(device)
        self.F_BC = ZMap(config.z_dim).to(device)
        self.F_CB = ZMap(config.z_dim).to(device)
        
        all_params = (
            list(self.F_AB.parameters()) +
            list(self.F_BA.parameters()) +
            list(self.F_AC.parameters()) +
            list(self.F_CA.parameters()) +
            list(self.F_BC.parameters()) +
            list(self.F_CB.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=config.lr)
        
        self.history = []
        
        # Cache probe responses (updated periodically)
        self._probe_cache = None
        self._cache_step = 0
    
    def _get_probe_responses(self) -> Dict[str, torch.Tensor]:
        """
        Get best z for each probe for each actor.
        
        Returns dict with 'z_A', 'z_B', 'z_C' each of shape (n_probes, z_dim)
        """
        n_probes = self.probes_A.shape[0]
        z_dim = self.config.z_dim
        
        z_A = torch.zeros(n_probes, z_dim, device=self.device)
        z_B = torch.zeros(n_probes, z_dim, device=self.device)
        z_C = torch.zeros(n_probes, z_dim, device=self.device)
        
        for i in range(n_probes):
            z_A[i] = get_best_z_for_obs(self.actor_A, self.probes_A[i])
            z_B[i] = get_best_z_for_obs(self.actor_B, self.probes_B[i])
            z_C[i] = get_best_z_for_obs(self.actor_C, self.probes_C[i])
        
        return {'z_A': z_A, 'z_B': z_B, 'z_C': z_C}
    
    def _compute_probe_consistency_loss(
        self,
        probe_responses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute probe consistency loss.
        
        Encourages: F_AB(z_A*) ≈ z_B*, F_AC(z_A*) ≈ z_C*, etc.
        """
        z_A = probe_responses['z_A']
        z_B = probe_responses['z_B']
        z_C = probe_responses['z_C']
        
        # Map A's probe responses to B and C
        z_A_to_B = self.F_AB(z_A)
        z_A_to_C = self.F_AC(z_A)
        
        # Map B's probe responses to A and C
        z_B_to_A = self.F_BA(z_B)
        z_B_to_C = self.F_BC(z_B)
        
        # Map C's probe responses to A and B
        z_C_to_A = self.F_CA(z_C)
        z_C_to_B = self.F_CB(z_C)
        
        def cosine_loss(z1, z2):
            """1 - cos(z1, z2) averaged over samples."""
            z1_norm = F.normalize(z1, p=2, dim=-1)
            z2_norm = F.normalize(z2, p=2, dim=-1)
            return (1.0 - (z1_norm * z2_norm).sum(dim=-1)).mean()
        
        # Probe consistency: mapped responses should match target actor's responses
        loss = (
            cosine_loss(z_A_to_B, z_B) +
            cosine_loss(z_A_to_C, z_C) +
            cosine_loss(z_B_to_A, z_A) +
            cosine_loss(z_B_to_C, z_C) +
            cosine_loss(z_C_to_A, z_A) +
            cosine_loss(z_C_to_B, z_B)
        ) / 6.0
        
        return loss
    
    def train_step(self) -> Dict[str, float]:
        """
        Single training step with probe-based gauge alignment.
        """
        # Update probe cache periodically
        if self._probe_cache is None or self._cache_step % 50 == 0:
            self._probe_cache = self._get_probe_responses()
        self._cache_step += 1
        
        # Sample random z for cycle/composition losses
        n_samples = 32
        z_A = sample_z_sphere(n_samples, self.config.z_dim, self.device)
        z_B = sample_z_sphere(n_samples, self.config.z_dim, self.device)
        z_C = sample_z_sphere(n_samples, self.config.z_dim, self.device)
        
        # ========== CYCLE CONSISTENCY LOSS ==========
        z_ABA = self.F_BA(self.F_AB(z_A))
        z_ACA = self.F_CA(self.F_AC(z_A))
        z_BAB = self.F_AB(self.F_BA(z_B))
        z_BCB = self.F_CB(self.F_BC(z_B))
        z_CAC = self.F_AC(self.F_CA(z_C))
        z_CBC = self.F_BC(self.F_CB(z_C))
        
        def cycle_loss(z, z_cycle):
            sim = (F.normalize(z, dim=-1) * F.normalize(z_cycle, dim=-1)).sum(dim=-1)
            return (1.0 - sim).mean()
        
        L_cyc = (
            cycle_loss(z_A, z_ABA) +
            cycle_loss(z_A, z_ACA) +
            cycle_loss(z_B, z_BAB) +
            cycle_loss(z_B, z_BCB) +
            cycle_loss(z_C, z_CAC) +
            cycle_loss(z_C, z_CBC)
        ) / 6.0
        
        # ========== COMPOSITION CONSISTENCY LOSS ==========
        z_AC = self.F_AC(z_A)
        z_ABC = self.F_BC(self.F_AB(z_A))
        
        z_AB = self.F_AB(z_A)
        z_ACB = self.F_CB(self.F_AC(z_A))
        
        z_BC = self.F_BC(z_B)
        z_BAC = self.F_AC(self.F_BA(z_B))
        
        def comp_loss(z1, z2):
            sim = (F.normalize(z1, dim=-1) * F.normalize(z2, dim=-1)).sum(dim=-1)
            return (1.0 - sim).mean()
        
        L_comp = (
            comp_loss(z_AC, z_ABC) +
            comp_loss(z_AB, z_ACB) +
            comp_loss(z_BC, z_BAC)
        ) / 3.0
        
        # ========== PROBE CONSISTENCY LOSS ==========
        L_probe = self._compute_probe_consistency_loss(self._probe_cache)
        
        # ========== TOTAL LOSS ==========
        loss = (
            self.config.lambda_cyc * L_cyc +
            self.config.lambda_comp * L_comp +
            self.config.lambda_probe * L_probe
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Composition metric (for monitoring)
        with torch.no_grad():
            z_test = sample_z_sphere(100, self.config.z_dim, self.device)
            z_AC_test = self.F_AC(z_test)
            z_ABC_test = self.F_BC(self.F_AB(z_test))
            comp_sim = (F.normalize(z_AC_test, dim=-1) * F.normalize(z_ABC_test, dim=-1)).sum(dim=-1).mean()
        
        metrics = {
            'loss': loss.item(),
            'L_cyc': L_cyc.item(),
            'L_comp': L_comp.item(),
            'L_probe': L_probe.item(),
            'comp_sim': comp_sim.item(),
        }
        self.history.append(metrics)
        
        return metrics
    
    def train(
        self,
        n_epochs: int = 5000,
        print_every: int = 500,
    ) -> List[Dict[str, float]]:
        """Run full training loop."""
        for epoch in range(n_epochs):
            metrics = self.train_step()
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"loss={metrics['loss']:.3f}, "
                      f"probe={metrics['L_probe']:.3f}, "
                      f"comp_sim={metrics['comp_sim']:.3f}")
        
        return self.history
    
    def get_functors(self) -> Dict[str, ZMap]:
        """Return all trained functors."""
        return {
            'F_AB': self.F_AB,
            'F_BA': self.F_BA,
            'F_AC': self.F_AC,
            'F_CA': self.F_CA,
            'F_BC': self.F_BC,
            'F_CB': self.F_CB,
        }
    
    def _probe_similarity_permutation_aware(
        self,
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        z_mapped: torch.Tensor,
    ) -> float:
        """
        Compute probe similarity with permutation-aware matching.
        
        Uses Hungarian algorithm to find best basin assignment, since gauge
        is only defined up to basin permutation.
        
        Args:
            z_source: (n_probes, z_dim) source actor's probe responses
            z_target: (n_probes, z_dim) target actor's probe responses
            z_mapped: (n_probes, z_dim) mapped source responses (F(z_source))
        
        Returns:
            Best similarity after optimal permutation matching
        """
        from scipy.optimize import linear_sum_assignment
        
        n_probes = z_source.shape[0]
        
        # Compute similarity matrix: how well does mapped probe i match target probe j?
        z_mapped_norm = F.normalize(z_mapped, dim=-1)
        z_target_norm = F.normalize(z_target, dim=-1)
        sim_matrix = (z_mapped_norm @ z_target_norm.t()).detach().cpu().numpy()  # (n_probes, n_probes)
        
        # Hungarian matching: find best assignment
        # We want to maximize total similarity, so negate for minimization
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        
        # Average similarity under optimal assignment
        best_sim = sim_matrix[row_ind, col_ind].mean()
        
        return float(best_sim)
    
    def evaluate_probe_agreement(self) -> Dict[str, float]:
        """
        Evaluate how well functors agree on probe responses.
        
        Uses permutation-aware matching (Hungarian algorithm) since gauge
        is only defined up to basin permutation.
        
        Returns similarity for all 6 directed functor maps.
        """
        probe_responses = self._get_probe_responses()
        z_A = probe_responses['z_A']
        z_B = probe_responses['z_B']
        z_C = probe_responses['z_C']
        
        # Permutation-aware similarities
        sim_AB = self._probe_similarity_permutation_aware(z_A, z_B, self.F_AB(z_A))
        sim_AC = self._probe_similarity_permutation_aware(z_A, z_C, self.F_AC(z_A))
        sim_BA = self._probe_similarity_permutation_aware(z_B, z_A, self.F_BA(z_B))
        sim_BC = self._probe_similarity_permutation_aware(z_B, z_C, self.F_BC(z_B))
        sim_CA = self._probe_similarity_permutation_aware(z_C, z_A, self.F_CA(z_C))
        sim_CB = self._probe_similarity_permutation_aware(z_C, z_B, self.F_CB(z_C))
        
        all_sims = [sim_AB, sim_AC, sim_BA, sim_BC, sim_CA, sim_CB]
        
        return {
            'probe_sim_AB': sim_AB,
            'probe_sim_AC': sim_AC,
            'probe_sim_BA': sim_BA,
            'probe_sim_BC': sim_BC,
            'probe_sim_CA': sim_CA,
            'probe_sim_CB': sim_CB,
            'probe_sim_mean': float(np.mean(all_sims)),
            'probe_sim_min': float(np.min(all_sims)),
        }
    
    def _random_orthogonal_matrix(self, d: int) -> torch.Tensor:
        """Generate a random orthogonal matrix via QR decomposition."""
        # Sample random matrix and orthogonalize
        random_matrix = torch.randn(d, d, device=self.device)
        q, _ = torch.linalg.qr(random_matrix)
        return q
    
    def compute_gauge_metrics(self, n_bootstrap: int = 100, n_null_samples: int = 1000) -> Dict[str, float]:
        """
        Compute dimension/mode-invariant gauge metrics using statistical comparison.
        
        Composition: z-score against random baseline (works well).
        Probe: random orthogonal scramble null (can't be "repaired" by Hungarian matching).
        
        The key insight: Hungarian matching can undo index permutations, but it
        cannot undo a random rotation of the latent space. So we use random
        orthogonal matrices as the null "gauge scramble."
        
        Returns:
            - comp_zscore: How many stdevs above random the composition similarity is
            - probe_pvalue: Empirical p-value from orthogonal scramble test
            - probe_adv: Advantage over null (observed - null_mean)
            - gauge_fixed: True if comp_z > 5 AND probe_p < 0.001 AND null is valid
        """
        z_dim = self.config.z_dim
        
        # ========== COMPOSITION METRIC (z-score, works well) ==========
        z_test = sample_z_sphere(200, z_dim, self.device)
        with torch.no_grad():
            z_AC = self.F_AC(z_test)
            z_ABC = self.F_BC(self.F_AB(z_test))
            trained_comp = (F.normalize(z_AC, dim=-1) * F.normalize(z_ABC, dim=-1)).sum(dim=-1)
            trained_comp_mean = trained_comp.mean().item()
        
        # Bootstrap random baseline for composition
        random_comps = []
        for _ in range(n_bootstrap):
            z_rand1 = sample_z_sphere(200, z_dim, self.device)
            z_rand2 = sample_z_sphere(200, z_dim, self.device)
            sim = (F.normalize(z_rand1, dim=-1) * F.normalize(z_rand2, dim=-1)).sum(dim=-1)
            random_comps.append(sim.mean().item())
        
        random_comp_mean = float(np.mean(random_comps))
        random_comp_std = float(np.std(random_comps)) + 1e-8
        comp_zscore = (trained_comp_mean - random_comp_mean) / random_comp_std
        
        # ========== PROBE METRIC (random orthogonal scramble null) ==========
        # This null CANNOT be repaired by Hungarian matching because rotating
        # z-space destroys the gauge alignment in a way permutation can't fix.
        
        probe_responses = self._get_probe_responses()
        z_A = probe_responses['z_A']
        z_B = probe_responses['z_B']
        z_C = probe_responses['z_C']
        
        # Observed probe similarity (permutation-aware)
        probe_agreement = self.evaluate_probe_agreement()
        observed_probe_sim = probe_agreement['probe_sim_mean']
        
        # Null: apply random orthogonal transformation to target z's
        # This scrambles the gauge while preserving the geometry (unit sphere)
        null_probe_sims = []
        for _ in range(n_null_samples):
            # Random rotation for each target space
            R_B = self._random_orthogonal_matrix(z_dim)
            R_C = self._random_orthogonal_matrix(z_dim)
            
            # Scrambled targets (still on unit sphere)
            z_B_scrambled = (z_B @ R_B.t())
            z_C_scrambled = (z_C @ R_C.t())
            # Re-normalize to ensure unit vectors (QR should preserve, but be safe)
            z_B_scrambled = F.normalize(z_B_scrambled, dim=-1)
            z_C_scrambled = F.normalize(z_C_scrambled, dim=-1)
            
            # Compute similarity with scrambled targets (still permutation-aware)
            sim_AB = self._probe_similarity_permutation_aware(
                z_A, z_B_scrambled, self.F_AB(z_A)
            )
            sim_AC = self._probe_similarity_permutation_aware(
                z_A, z_C_scrambled, self.F_AC(z_A)
            )
            null_probe_sims.append((sim_AB + sim_AC) / 2)
        
        null_probe_sims = np.array(null_probe_sims)
        null_probe_mean = float(np.mean(null_probe_sims))
        null_probe_std = float(np.std(null_probe_sims))
        
        # Sanity check: is the null degenerate?
        null_is_valid = null_probe_std > 1e-4
        if not null_is_valid:
            print(f"WARNING: Degenerate null distribution (std={null_probe_std:.2e}). Test may be invalid.")
        
        # Empirical p-value: fraction of null samples >= observed
        n_above = np.sum(null_probe_sims >= observed_probe_sim)
        probe_pvalue = (1 + n_above) / (1 + n_null_samples)
        
        # Advantage over null (more interpretable than z-score when null has real variance)
        probe_adv = observed_probe_sim - null_probe_mean
        
        # Z-score for reference (but use with caution if std is small)
        probe_zscore = probe_adv / (null_probe_std + 1e-8)
        
        # Gauge is fixed if:
        # - Composition is very strong (z > 5)
        # - Probe p-value beats all null samples (p < 1/n_null_samples)
        # - Probe advantage is meaningful (adv > 0.05)
        # - Null is not degenerate
        comp_threshold = 5.0
        probe_p_threshold = 1.0 / (1 + n_null_samples)  # "beat all null samples"
        probe_adv_threshold = 0.05  # Minimum meaningful effect size
        
        gauge_fixed = (
            comp_zscore > comp_threshold and 
            probe_pvalue <= probe_p_threshold and  # <= because best possible p equals threshold
            probe_adv > probe_adv_threshold and
            null_is_valid
        )
        
        # P-value reporting: track if at resolution limit
        p_resolution = 1.0 / (1 + n_null_samples)
        p_at_resolution = (probe_pvalue <= p_resolution)
        
        return {
            'comp_sim': trained_comp_mean,
            'comp_zscore': comp_zscore,
            'comp_baseline_mean': random_comp_mean,
            'comp_baseline_std': random_comp_std,
            'probe_sim': observed_probe_sim,
            'probe_pvalue': probe_pvalue,
            'probe_pvalue_at_resolution': p_at_resolution,  # True if p hit the floor
            'probe_adv': probe_adv,  # Advantage over null
            'probe_zscore': probe_zscore,  # For reference only
            'probe_baseline_mean': null_probe_mean,
            'probe_baseline_std': null_probe_std,
            'null_is_valid': null_is_valid,
            'gauge_fixed': gauge_fixed,
            'comp_threshold': comp_threshold,
            'probe_p_threshold': probe_p_threshold,
            'probe_adv_threshold': probe_adv_threshold,
            'p_resolution': p_resolution,
        }


def plot_reference_probe_results(
    history: List[Dict[str, float]],
    probe_positions: np.ndarray,
    gauge_metrics: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot training results for reference probe alignment.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    epochs = np.arange(len(history))
    
    def smooth(x, window=50):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    # Top left: Loss components
    ax = axes[0, 0]
    ax.plot(smooth([h['L_cyc'] for h in history]), label='Cycle')
    ax.plot(smooth([h['L_comp'] for h in history]), label='Composition')
    ax.plot(smooth([h['L_probe'] for h in history]), label='Probe')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    
    # Top middle: Total loss
    ax = axes[0, 1]
    ax.plot(epochs, [h['loss'] for h in history], alpha=0.3)
    ax.plot(smooth([h['loss'] for h in history]), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    
    # Top right: Composition similarity with baseline
    ax = axes[0, 2]
    comp_sim = [h['comp_sim'] for h in history]
    ax.plot(epochs, comp_sim, alpha=0.3, color='blue')
    ax.plot(smooth(comp_sim), color='blue', linewidth=2, label='Trained')
    
    # Show baseline mean ± std
    baseline_mean = gauge_metrics['comp_baseline_mean']
    baseline_std = gauge_metrics['comp_baseline_std']
    ax.axhline(baseline_mean, color='red', linestyle='--', alpha=0.7, label=f'Random baseline')
    ax.axhspan(baseline_mean - 2*baseline_std, baseline_mean + 2*baseline_std, 
               color='red', alpha=0.1, label='±2σ')
    
    comp_z = gauge_metrics.get('comp_zscore', 0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Composition (z={comp_z:.1f}σ above random)')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='lower right')
    
    # Bottom left: Probe positions visualization (if 2D or 3D)
    ax = axes[1, 0]
    if probe_positions.shape[1] >= 2:
        ax.scatter(probe_positions[:-1, 0], probe_positions[:-1, 1], 
                   c='blue', s=100, label='Boundary probes', marker='o')
        ax.scatter(probe_positions[-1, 0], probe_positions[-1, 1], 
                   c='red', s=150, label='Tie-break probe', marker='*')
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        ax.set_title('Probe Positions (first 2 dims)')
        ax.legend()
        ax.axis('equal')
    else:
        ax.text(0.5, 0.5, f'{probe_positions.shape[0]} probes\n{probe_positions.shape[1]}D',
                ha='center', va='center', fontsize=14)
        ax.set_title('Probe Info')
    
    # Bottom middle: Probe loss over time
    ax = axes[1, 1]
    probe_loss = [h['L_probe'] for h in history]
    ax.plot(epochs, probe_loss, alpha=0.3)
    ax.plot(smooth(probe_loss), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Probe Consistency Loss')
    
    probe_p = gauge_metrics.get('probe_pvalue', 1.0)
    probe_adv = gauge_metrics.get('probe_adv', 0)
    ax.set_title(f'Probe Agreement (p={probe_p:.4f}, adv={probe_adv:+.3f})')
    ax.set_yscale('log')
    
    # Bottom right: Summary statistics with p-value
    ax = axes[1, 2]
    comp_threshold = gauge_metrics.get('comp_threshold', 5.0)
    probe_p_threshold = gauge_metrics.get('probe_p_threshold', 0.001)
    probe_adv_threshold = gauge_metrics.get('probe_adv_threshold', 0.05)
    null_is_valid = gauge_metrics.get('null_is_valid', True)
    gauge_fixed = gauge_metrics['gauge_fixed']
    null_std = gauge_metrics.get('probe_baseline_std', 0)
    
    # Format p-value as inequality when at resolution limit
    p_resolution = gauge_metrics.get('p_resolution', 0.001)
    p_at_resolution = gauge_metrics.get('probe_pvalue_at_resolution', False)
    p_val = gauge_metrics['probe_pvalue']
    p_str = f"<{p_resolution:.4f}" if p_at_resolution else f"{p_val:.4f}"
    
    validity_warning = "" if null_is_valid else "\n⚠️ DEGENERATE NULL (std≈0)"
    
    # Show which criteria passed/failed
    comp_pass = gauge_metrics['comp_zscore'] > comp_threshold
    p_pass = p_val <= probe_p_threshold
    adv_pass = probe_adv > probe_adv_threshold
    
    text = (
        f"Dimension-Invariant Gauge Metrics\n"
        f"{'='*38}\n"
        f"Composition:\n"
        f"  similarity: {gauge_metrics['comp_sim']:.3f}\n"
        f"  z-score:    {gauge_metrics['comp_zscore']:.1f}σ {'✓' if comp_pass else '✗'} (>{comp_threshold}σ)\n"
        f"\n"
        f"Probe Agreement (orthogonal scramble null):\n"
        f"  similarity:  {gauge_metrics['probe_sim']:.3f}\n"
        f"  null mean:   {gauge_metrics['probe_baseline_mean']:.3f}\n"
        f"  null std:    {null_std:.3f}\n"
        f"  advantage:   {probe_adv:+.3f} {'✓' if adv_pass else '✗'} (>{probe_adv_threshold})\n"
        f"  p-value:     {p_str} {'✓' if p_pass else '✗'} (<{probe_p_threshold})\n"
        f"\n"
        f"{'='*38}\n"
        f"Gauge Fixed: {'✓' if gauge_fixed else '✗'}{validity_warning}\n"
    )
    ax.text(0.05, 0.5, text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def run_reference_probe_test(
    output_dir: Path,
    n_actor_epochs: int = 500,
    n_functor_epochs: int = 5000,
    K: int = 3,
    z_dim: int = 2,
    d: int = None,  # State dimension (default: z_dim + 1)
    n_probes: int = None,  # Number of probes (default: K+1)
) -> Dict:
    """
    Run the reference probe gauge alignment experiment.
    """
    from v2.actors.actor import Actor
    from .world_variants import create_world_A, create_world_B, create_world_C, CMGWorldVariant
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default state dimension
    if d is None:
        d = max(3, z_dim + 1)  # At least 3D for visualization
    
    # Default probe count
    if n_probes is None:
        n_probes = K + 1
    
    print("============================================================")
    print("REFERENCE PROBE GAUGE ALIGNMENT")
    print("============================================================")
    print(f"K={K} basins, z_dim={z_dim}, d={d}, probes={n_probes}")
    print(f"Actor epochs: {n_actor_epochs}, Functor epochs: {n_functor_epochs}")
    print()
    
    # Phase 0: Create environments
    print("Phase 0: Creating environments...")
    config_A = create_world_A(K=K, d=d)
    config_B = create_world_B(K=K, d=d)
    config_C = create_world_C(K=K, d=d)
    
    env_A = CMGWorldVariant(config_A)
    env_B = CMGWorldVariant(config_B)
    env_C = CMGWorldVariant(config_C)
    
    # Phase 0: Train actors independently
    print("Phase 0: Training actors...")
    
    def train_actor(env, name):
        print(f"  Training actor {name}...")
        actor = Actor(
            obs_dim=env.obs_dim,
            pred_dim=env.state_dim,
            T=env.T,
            z_dim=z_dim,
            n_knots=5,
        ).to(device)
        
        optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        
        for epoch in range(n_actor_epochs):
            obs = env.reset()
            traj = [obs]
            done = False
            while not done:
                action = np.random.randn(env.state_dim).astype(np.float32) * 0.3
                obs, _, done, _ = env.step(action)
                traj.append(obs)
            
            traj_array = np.array(traj)
            traj_delta = torch.tensor(
                traj_array[1:, :env.state_dim] - traj_array[0, :env.state_dim],
                dtype=torch.float32, device=device
            )
            s0 = torch.tensor(traj[0], dtype=torch.float32, device=device)
            
            actor.train_step_wta(s0, traj_delta, optimizer)
            
            if (epoch + 1) % max(1, n_actor_epochs // 5) == 0:
                print(f"    Epoch {epoch+1}/{n_actor_epochs}")
        
        return actor
    
    actor_A = train_actor(env_A, "A")
    actor_B = train_actor(env_B, "B")
    actor_C = train_actor(env_C, "C")
    
    # Phase 1: Generate probes and train functors
    print()
    print("Phase 1: Training functors with reference probes...")
    
    config = ReferenceProbeConfig(
        z_dim=z_dim,
        K=K,
        n_probes=n_probes,
        tau=0.05,
        lambda_cyc=1.0,
        lambda_comp=1.0,
        lambda_probe=2.0,
        lr=1e-3,
    )
    
    trainer = ReferenceProbeTrainer(
        actor_A, actor_B, actor_C,
        env_A, env_B, env_C,
        config, device,
    )
    
    print(f"  Generated {trainer.probe_positions.shape[0]} probes")
    
    history = trainer.train(n_epochs=n_functor_epochs, print_every=500)
    
    # Evaluate final results using dimension-invariant metrics
    print()
    print("Evaluating final results with dimension-invariant metrics...")
    
    # Compute gauge metrics (z-scores against random baseline)
    gauge_metrics = trainer.compute_gauge_metrics(n_bootstrap=100)
    
    # Probe agreement details
    probe_agreement = trainer.evaluate_probe_agreement()
    
    # Final probe loss (from last 100 epochs of history)
    final_probe_loss = np.mean([h['L_probe'] for h in history[-100:]]) if len(history) >= 100 else np.mean([h['L_probe'] for h in history])
    
    results = {
        # Raw metrics
        'comp_sim': gauge_metrics['comp_sim'],
        'probe_sim': gauge_metrics['probe_sim'],
        'final_probe_loss': final_probe_loss,
        
        # Dimension-invariant metrics
        'comp_zscore': gauge_metrics['comp_zscore'],
        'probe_pvalue': gauge_metrics['probe_pvalue'],
        'probe_pvalue_at_resolution': gauge_metrics['probe_pvalue_at_resolution'],
        'probe_adv': gauge_metrics['probe_adv'],  # Advantage over null
        
        # Baselines
        'comp_baseline_mean': gauge_metrics['comp_baseline_mean'],
        'comp_baseline_std': gauge_metrics['comp_baseline_std'],
        'probe_baseline_mean': gauge_metrics['probe_baseline_mean'],
        'probe_baseline_std': gauge_metrics['probe_baseline_std'],
        
        # Null validity check
        'null_is_valid': gauge_metrics['null_is_valid'],
        
        # Final decision
        'gauge_fixed': gauge_metrics['gauge_fixed'],
        'comp_threshold': gauge_metrics.get('comp_threshold', 5.0),
        'probe_p_threshold': gauge_metrics.get('probe_p_threshold', 0.001),
        'probe_adv_threshold': gauge_metrics.get('probe_adv_threshold', 0.05),
        
        # Config
        'n_probes': trainer.probe_positions.shape[0],
        'K': K,
        'z_dim': z_dim,
        'd': d,
        'n_actor_epochs': n_actor_epochs,
        'n_functor_epochs': n_functor_epochs,
    }
    
    print()
    print("============================================================")
    print("RESULTS (dimension-invariant)")
    print("============================================================")
    print(f"Composition: {gauge_metrics['comp_sim']:.3f} (z-score: {gauge_metrics['comp_zscore']:.1f}σ)")
    
    # Format p-value as inequality when at resolution limit
    p_resolution = gauge_metrics.get('p_resolution', 0.001)
    p_at_resolution = gauge_metrics.get('probe_pvalue_at_resolution', False)
    p_val = gauge_metrics['probe_pvalue']
    p_str = f"<{p_resolution:.4f}" if p_at_resolution else f"{p_val:.4f}"
    
    probe_adv = gauge_metrics['probe_adv']
    null_std = gauge_metrics['probe_baseline_std']
    print(f"Probe agreement: {gauge_metrics['probe_sim']:.3f} (null mean: {gauge_metrics['probe_baseline_mean']:.3f}, std: {null_std:.3f})")
    print(f"Probe advantage: {probe_adv:+.3f}, p-value: {p_str}")
    
    if not gauge_metrics['null_is_valid']:
        print(f"⚠️ WARNING: Degenerate null (std={null_std:.2e}). Test may be invalid.")
    
    comp_thresh = gauge_metrics.get('comp_threshold', 5.0)
    probe_p_thresh = gauge_metrics.get('probe_p_threshold', 0.001)
    probe_adv_thresh = gauge_metrics.get('probe_adv_threshold', 0.05)
    print(f"Gauge fixed: {results['gauge_fixed']} (comp_z>{comp_thresh}σ AND p<{probe_p_thresh} AND adv>{probe_adv_thresh})")
    
    # Plot results
    plot_path = output_dir / "reference_probe_alignment.png"
    plot_reference_probe_results(history, trainer.probe_positions, gauge_metrics, plot_path)
    
    return results


# Standalone entry point
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-epochs", type=int, default=500)
    parser.add_argument("--functor-epochs", type=int, default=5000)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--z-dim", type=int, default=2)
    args = parser.parse_args()
    
    output_dir = Path("artifacts/functor_tests") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_reference_probe_test(
        output_dir,
        n_actor_epochs=args.actor_epochs,
        n_functor_epochs=args.functor_epochs,
        K=args.K,
        z_dim=args.z_dim,
    )
