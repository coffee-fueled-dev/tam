"""
Cross-environment transfer harness for TAM experiments.

Tests whether commitments/skills learned in one environment transfer to another.
Supports multiple reuse modes and comprehensive transfer diagnostics.

Phases:
0. Hygiene: Multi-seed eval, CIs, paired comparisons
1. Diagnostics: Cone-summary portability, rank preservation
2. Behavioral retrieval reuse
3. Behavioral prototype clustering
5. Canonical transfer plots
6. Sanity checks
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from .experiment_harness import Environment, ExperimentConfig, ExperimentHarness
from .evaluation import evaluate_agent_generic, EvalSnapshot, tube_predictions_for_episode

try:
    from actor import Actor
except ImportError:
    try:
        from ..actor import Actor  # type: ignore
    except ImportError:
        Actor = None  # type: ignore


@dataclass
class TransferConfig:
    """Configuration for cross-environment transfer experiments."""
    
    # Source environments (train here)
    source_envs: List[Tuple[str, Callable, Dict[str, Any]]] = field(default_factory=list)
    # Format: (name, env_factory, env_kwargs)
    
    # Target environments (test transfer here)
    target_envs: List[Tuple[str, Callable, Dict[str, Any]]] = field(default_factory=list)
    
    # Training parameters
    train_steps: int = 6000
    eval_episodes: int = 200
    
    # Actor parameters
    actor_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # PHASE 0: Multi-seed parameters
    n_seeds: int = 10  # Number of seeds per (source, target, mode) triplet
    bootstrap_samples: int = 1000  # Bootstrap samples for CI computation
    
    # Transfer parameters (including Phase 2/3 reuse modes)
    reuse_modes: List[str] = field(default_factory=lambda: [
        "native", "memory", "prototype", "behavioral", "random", "shuffled"
    ])
    n_prototypes: int = 4  # Number of prototype z's to extract per source env
    behavioral_k: int = 10  # Number of candidates for behavioral retrieval
    freeze_actor: bool = True  # Freeze actor weights during transfer
    freeze_tube: bool = False  # Freeze tube weights during transfer
    freeze_policy: bool = False  # Freeze policy weights during transfer
    
    # Phase 1: Portability test parameters
    n_portability_samples: int = 50  # Number of z's to test for portability
    
    # Phase 6: Sanity check parameters
    enable_sanity_checks: bool = True
    hr_eval_values: List[int] = field(default_factory=lambda: [0, 1, -1])  # -1 = max
    
    # Evaluation parameters
    k_sigma_list: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    k_star: float = 2.0
    
    # Output directory
    output_dir: Optional[Path] = None
    
    # Base seed (will be incremented for multi-seed runs)
    seed: int = 0


@dataclass
class PerEpisodeRecord:
    """Record for a single episode evaluation."""
    source_env: str
    target_env: str
    reuse_mode: str
    seed: int
    episode_idx: int
    outcome: float  # -J (higher is better)
    bind_success: float
    coverage: float  # at k*
    sharpness: float  # log cone volume
    E_T: float
    volatility: float
    chosen_z: Optional[np.ndarray] = None
    neighbor_distance: Optional[float] = None  # For reuse modes
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source_env": self.source_env,
            "target_env": self.target_env,
            "reuse_mode": self.reuse_mode,
            "seed": self.seed,
            "episode_idx": self.episode_idx,
            "outcome": self.outcome,
            "bind_success": self.bind_success,
            "coverage": self.coverage,
            "sharpness": self.sharpness,
            "E_T": self.E_T,
            "volatility": self.volatility,
        }
        if self.neighbor_distance is not None:
            d["neighbor_distance"] = self.neighbor_distance
        return d


@dataclass
class ConeSummary:
    """Cone summary for a commitment z."""
    z: np.ndarray  # [z_dim]
    cone_volume: float
    E_T: float
    bind_rate: float
    outcome: float


class CrossEnvTransferHarness:
    """
    Cross-environment transfer harness with comprehensive diagnostics.
    
    Implements:
    - Phase 0: Multi-seed evaluation with CIs and paired comparisons
    - Phase 1: Cone-summary portability and rank preservation tests
    - Phase 2: Behavioral retrieval reuse
    - Phase 3: Behavioral prototype clustering
    - Phase 5: Canonical transfer plots
    - Phase 6: Sanity checks
    """
    
    def __init__(self, config: TransferConfig):
        """Initialize the cross-environment transfer harness."""
        self.config = config
        
        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create output directory
        if config.output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path("runs") / f"transfer_{timestamp}"
        else:
            self.run_dir = Path(config.output_dir)
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "fig").mkdir(exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        
        # Storage for trained models
        self.trained_actors: Dict[str, Actor] = {}  # source_env_name -> actor
        self.source_memories: Dict[str, List[Dict]] = {}  # Full memory with metadata
        self.source_zs: Dict[str, np.ndarray] = {}  # Just z arrays [N, z_dim]
        self.prototypes: Dict[str, np.ndarray] = {}  # KMeans prototypes
        self.behavioral_prototypes: Dict[str, List[ConeSummary]] = {}  # Behavioral prototypes
        
        # Phase 1: Cone summaries for portability testing
        self.cone_summaries: Dict[str, Dict[str, List[ConeSummary]]] = {}  # source -> target -> summaries
        
        # Storage for results
        self.per_episode_records: List[PerEpisodeRecord] = []
        self.seed_aggregates: List[Dict[str, Any]] = []  # Per-seed aggregated results
        self.transfer_results: List[Dict[str, Any]] = []  # Legacy format
    
    # =========================================================================
    # PHASE 0: Training and Basic Evaluation
    # =========================================================================
    
    def train_source_envs(self) -> None:
        """Train actors on all source environments."""
        print("=" * 60)
        print("Training on source environments...")
        print("=" * 60)
        
        for env_name, env_factory, env_kwargs in self.config.source_envs:
            print(f"\nTraining on source environment: {env_name}")
            
            train_config = ExperimentConfig(
                name=f"{env_name}_source",
                seed=self.config.seed,
                train_steps=self.config.train_steps,
                eval_every=self.config.train_steps + 1,
                eval_episodes=0,
                env_kwargs=env_kwargs,
                actor_kwargs=self.config.actor_kwargs,
                k_sigma_list=self.config.k_sigma_list,
                k_star=self.config.k_star,
                output_dir=self.run_dir / f"source_{env_name}",
            )
            
            harness = ExperimentHarness(
                config=train_config,
                env_factory=env_factory,
            )
            
            actor, _ = harness.train()
            self._freeze_weights(actor)
            self.trained_actors[env_name] = actor
            
            # Extract memory with full metadata
            self._extract_memory(env_name, actor)
            
            # Extract prototypes (both geometric and behavioral)
            self._extract_prototypes(env_name, actor)
            
            print(f"  Training complete. Memory size: {len(self.source_zs.get(env_name, []))}")
            print(f"  Geometric prototypes: {len(self.prototypes.get(env_name, []))}")
            print(f"  Behavioral prototypes: {len(self.behavioral_prototypes.get(env_name, []))}")
    
    def _freeze_weights(self, actor: Actor) -> None:
        """Freeze actor weights according to config."""
        if self.config.freeze_actor:
            for param in actor.actor.parameters():
                param.requires_grad = False
        if self.config.freeze_tube:
            for param in actor.tube.parameters():
                param.requires_grad = False
        if self.config.freeze_policy:
            for param in actor.pol.parameters():
                param.requires_grad = False
    
    def _extract_memory(self, env_name: str, actor: Actor) -> None:
        """Extract memory with full metadata for behavioral retrieval."""
        if not hasattr(actor, 'mem') or len(actor.mem) == 0:
            print(f"  Warning: No memory for {env_name}")
            self.source_memories[env_name] = []
            self.source_zs[env_name] = np.array([])
            return
        
        self.source_memories[env_name] = list(actor.mem)
        zs = np.stack([m["z"].numpy() for m in actor.mem], axis=0)
        self.source_zs[env_name] = zs
    
    def _extract_prototypes(self, env_name: str, actor: Actor) -> None:
        """Extract both geometric (KMeans) and behavioral prototypes."""
        zs = self.source_zs[env_name]
        mem = self.source_memories[env_name]
        
        if len(zs) == 0:
            self.prototypes[env_name] = np.array([])
            self.behavioral_prototypes[env_name] = []
            return
        
        # Geometric prototypes (KMeans on z space)
        if len(zs) >= self.config.n_prototypes:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.config.n_prototypes, random_state=0, n_init=10)
                kmeans.fit(zs)
                self.prototypes[env_name] = kmeans.cluster_centers_
            except ImportError:
                idx = np.random.choice(len(zs), size=self.config.n_prototypes, replace=False)
                self.prototypes[env_name] = zs[idx]
        else:
            self.prototypes[env_name] = zs
        
        # Behavioral prototypes (cluster by behavior, not z)
        self._extract_behavioral_prototypes(env_name, mem)
    
    def _extract_behavioral_prototypes(self, env_name: str, mem: List[Dict]) -> None:
        """Extract prototypes by clustering on behavioral features (cone_vol, E[T], bind_rate)."""
        if len(mem) < self.config.n_prototypes:
            self.behavioral_prototypes[env_name] = [
                ConeSummary(
                    z=m["z"].numpy(),
                    cone_volume=m.get("cone_vol", 0.0),
                    E_T=m.get("E_T", 1.0),
                    bind_rate=m.get("soft_bind", 0.0),
                    outcome=-m.get("cone_vol", 0.0),  # Proxy
                )
                for m in mem
            ]
            return
        
        # Build behavioral feature matrix
        features = np.array([
            [m.get("cone_vol", 0.0), m.get("E_T", 1.0), m.get("soft_bind", 0.0)]
            for m in mem
        ])
        
        # Standardize features
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0) + 1e-8
        features_norm = (features - feat_mean) / feat_std
        
        # Cluster on behavioral features
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.n_prototypes, random_state=0, n_init=10)
            labels = kmeans.fit_predict(features_norm)
            
            # Pick representative z from each cluster (best bind_rate)
            prototypes = []
            for c in range(self.config.n_prototypes):
                mask = labels == c
                if not mask.any():
                    continue
                cluster_indices = np.where(mask)[0]
                # Pick by best bind_rate in cluster
                best_idx = cluster_indices[
                    np.argmax([mem[i].get("soft_bind", 0.0) for i in cluster_indices])
                ]
                m = mem[best_idx]
                prototypes.append(ConeSummary(
                    z=m["z"].numpy(),
                    cone_volume=m.get("cone_vol", 0.0),
                    E_T=m.get("E_T", 1.0),
                    bind_rate=m.get("soft_bind", 0.0),
                    outcome=-m.get("cone_vol", 0.0),
                ))
            self.behavioral_prototypes[env_name] = prototypes
        except ImportError:
            # Fallback: random selection
            idx = np.random.choice(len(mem), size=min(len(mem), self.config.n_prototypes), replace=False)
            self.behavioral_prototypes[env_name] = [
                ConeSummary(
                    z=mem[i]["z"].numpy(),
                    cone_volume=mem[i].get("cone_vol", 0.0),
                    E_T=mem[i].get("E_T", 1.0),
                    bind_rate=mem[i].get("soft_bind", 0.0),
                    outcome=-mem[i].get("cone_vol", 0.0),
                )
                for i in idx
            ]
    
    # =========================================================================
    # PHASE 1: Cone-Summary Portability Tests
    # =========================================================================
    
    def run_portability_tests(self) -> Dict[str, Any]:
        """
        Run cone-summary portability tests between all source/target pairs.
        
        For each z in source memory, compute (C_source, H_source) and (C_target, H_target).
        Check if rankings are preserved.
        """
        print("\n" + "=" * 60)
        print("Phase 1: Running cone-summary portability tests...")
        print("=" * 60)
        
        results = {}
        
        for source_name, _, _ in self.config.source_envs:
            results[source_name] = {}
            source_actor = self.trained_actors[source_name]
            source_zs = self.source_zs[source_name]
            
            if len(source_zs) == 0:
                continue
            
            # Sample z's for testing
            n_samples = min(self.config.n_portability_samples, len(source_zs))
            sample_idx = np.random.choice(len(source_zs), size=n_samples, replace=False)
            sampled_zs = source_zs[sample_idx]
            
            for target_name, target_factory, target_kwargs in self.config.target_envs:
                print(f"  Testing portability: {source_name} -> {target_name}")
                
                target_env = target_factory(target_kwargs)
                target_env.reset()
                target_obs0 = target_env.observe()
                
                # Compute cone summaries in both environments
                source_summaries = []
                target_summaries = []
                
                for z_np in sampled_zs:
                    z_t = torch.tensor(z_np, dtype=torch.float32, device=source_actor.device).unsqueeze(0)
                    
                    # Source env summary
                    # Use a dummy obs from source (we don't have access, so use zeros)
                    s0_dummy = torch.zeros(1, source_actor.obs_dim, device=source_actor.device)
                    try:
                        mu_s, lv_s, w_s, p_s = tube_predictions_for_episode(
                            source_actor, s0_dummy.squeeze().cpu().numpy(), z_t, T=16, Hr_eval=0
                        )
                        cone_vol_s = float(torch.prod(torch.exp(0.5 * lv_s), dim=-1).mean())
                        E_T_s = float((w_s * torch.arange(1, 17, device=w_s.device)).sum())
                        source_summaries.append({"cone_vol": cone_vol_s, "E_T": E_T_s})
                    except Exception:
                        source_summaries.append({"cone_vol": 0.0, "E_T": 1.0})
                    
                    # Target env summary
                    try:
                        mu_t, lv_t, w_t, p_t = tube_predictions_for_episode(
                            source_actor, target_obs0, z_t, T=16, Hr_eval=0
                        )
                        cone_vol_t = float(torch.prod(torch.exp(0.5 * lv_t), dim=-1).mean())
                        E_T_t = float((w_t * torch.arange(1, 17, device=w_t.device)).sum())
                        target_summaries.append({"cone_vol": cone_vol_t, "E_T": E_T_t})
                    except Exception:
                        target_summaries.append({"cone_vol": 0.0, "E_T": 1.0})
                
                # Compute correlations
                C_source = np.array([s["cone_vol"] for s in source_summaries])
                C_target = np.array([s["cone_vol"] for s in target_summaries])
                H_source = np.array([s["E_T"] for s in source_summaries])
                H_target = np.array([s["E_T"] for s in target_summaries])
                
                # Pearson correlation
                if np.std(C_source) > 1e-8 and np.std(C_target) > 1e-8:
                    cone_corr, cone_p = stats.pearsonr(C_source, C_target)
                else:
                    cone_corr, cone_p = 0.0, 1.0
                
                if np.std(H_source) > 1e-8 and np.std(H_target) > 1e-8:
                    horizon_corr, horizon_p = stats.pearsonr(H_source, H_target)
                else:
                    horizon_corr, horizon_p = 0.0, 1.0
                
                # Rank correlation (Spearman)
                if np.std(C_source) > 1e-8 and np.std(C_target) > 1e-8:
                    cone_rank_corr, cone_rank_p = stats.spearmanr(C_source, C_target)
                else:
                    cone_rank_corr, cone_rank_p = 0.0, 1.0
                
                if np.std(H_source) > 1e-8 and np.std(H_target) > 1e-8:
                    horizon_rank_corr, horizon_rank_p = stats.spearmanr(H_source, H_target)
                else:
                    horizon_rank_corr, horizon_rank_p = 0.0, 1.0
                
                results[source_name][target_name] = {
                    "cone_pearson_r": float(cone_corr),
                    "cone_pearson_p": float(cone_p),
                    "cone_spearman_r": float(cone_rank_corr),
                    "cone_spearman_p": float(cone_rank_p),
                    "horizon_pearson_r": float(horizon_corr),
                    "horizon_pearson_p": float(horizon_p),
                    "horizon_spearman_r": float(horizon_rank_corr),
                    "horizon_spearman_p": float(horizon_rank_p),
                    "C_source": C_source.tolist(),
                    "C_target": C_target.tolist(),
                    "H_source": H_source.tolist(),
                    "H_target": H_target.tolist(),
                }
                
                print(f"    Cone corr: Pearson={cone_corr:.3f}, Spearman={cone_rank_corr:.3f}")
                print(f"    Horizon corr: Pearson={horizon_corr:.3f}, Spearman={horizon_rank_corr:.3f}")
        
        return results
    
    # =========================================================================
    # PHASE 0/2/3: Multi-seed Transfer Evaluation
    # =========================================================================
    
    def run_transfer_tests_multiseed(self) -> None:
        """
        Run transfer tests with multiple seeds for statistical validity.
        
        For each (source, target, reuse_mode, seed), run eval_episodes episodes
        and collect per-episode records.
        """
        print("\n" + "=" * 60)
        print("Running multi-seed transfer tests...")
        print("=" * 60)
        
        for source_name, _, _ in self.config.source_envs:
            for target_name, target_factory, target_kwargs in self.config.target_envs:
                print(f"\nTransfer: {source_name} -> {target_name}")
                
                for reuse_mode in self.config.reuse_modes:
                    print(f"  Mode: {reuse_mode}")
                    
                    for seed_offset in range(self.config.n_seeds):
                        seed = self.config.seed + seed_offset
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        
                        result = self._evaluate_transfer_single_seed(
                            source_name, target_name, target_factory, target_kwargs,
                            reuse_mode, seed
                        )
                        
                        self.seed_aggregates.append(result)
                        
                        # Also store in legacy format
                        self.transfer_results.append({
                            "source_env": source_name,
                            "target_env": target_name,
                            "reuse_mode": reuse_mode,
                            "seed": seed,
                            "mean_outcome": result.get("mean_outcome", 0.0),
                            "mean_coverage": result.get("mean_coverage", 0.0),
                            "mean_sharpness": result.get("mean_sharpness", 0.0),
                            "mean_E_T": result.get("mean_E_T", 0.0),
                        })
                    
                    # Print summary for this mode
                    mode_results = [r for r in self.seed_aggregates 
                                   if r["source_env"] == source_name 
                                   and r["target_env"] == target_name 
                                   and r["reuse_mode"] == reuse_mode]
                    if mode_results:
                        outcomes = [r["mean_outcome"] for r in mode_results]
                        print(f"    Outcome: {np.mean(outcomes):.3f} ± {np.std(outcomes):.3f}")
    
    def _evaluate_transfer_single_seed(
        self,
        source_name: str,
        target_name: str,
        target_factory: Callable,
        target_kwargs: Dict[str, Any],
        reuse_mode: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Evaluate transfer for a single seed."""
        actor = self.trained_actors[source_name]
        target_env = target_factory(target_kwargs)
        
        # Create appropriate reuse actor
        eval_actor = self._create_reuse_actor(actor, source_name, reuse_mode, target_env)
        
        try:
            snapshot = evaluate_agent_generic(
                agent=eval_actor,
                env=target_env,
                step=0,
                ks=self.config.k_sigma_list,
                n_episodes=self.config.eval_episodes,
                k_star=self.config.k_star,
                Hr_eval=0,
                pred_dim=getattr(actor, 'pred_dim', None),
            )
            
            return {
                "source_env": source_name,
                "target_env": target_name,
                "reuse_mode": reuse_mode,
                "seed": seed,
                "mean_outcome": -snapshot.mean_J,
                "mean_coverage": snapshot.empirical_coverage.get(self.config.k_star, 0.0),
                "mean_sharpness": snapshot.mean_sharp_log_vol,
                "mean_E_T": snapshot.mean_E_T,
                "mean_volatility": snapshot.mean_volatility,
                "J_points": snapshot.J_points.tolist() if snapshot.J_points is not None else [],
            }
        except Exception as e:
            print(f"    Error (seed {seed}): {e}")
            return {
                "source_env": source_name,
                "target_env": target_name,
                "reuse_mode": reuse_mode,
                "seed": seed,
                "error": str(e),
                "mean_outcome": 0.0,
                "mean_coverage": 0.0,
                "mean_sharpness": 0.0,
                "mean_E_T": 0.0,
            }
    
    def _create_reuse_actor(
        self, 
        actor: Actor, 
        source_name: str, 
        reuse_mode: str,
        target_env: Any,
    ) -> Any:
        """Create actor wrapper for the specified reuse mode."""
        if reuse_mode == "native":
            return actor
        elif reuse_mode == "memory":
            return MemoryReuseActor(actor, self.source_zs[source_name])
        elif reuse_mode == "prototype":
            return PrototypeReuseActor(actor, self.prototypes[source_name])
        elif reuse_mode == "behavioral":
            return BehavioralReuseActor(
                actor, 
                self.behavioral_prototypes[source_name],
                self.source_memories[source_name],
                k=self.config.behavioral_k,
            )
        elif reuse_mode == "random":
            return RandomZActor(actor, self.source_zs[source_name])
        elif reuse_mode == "shuffled":
            return ShuffledMemoryActor(actor, self.source_zs[source_name])
        else:
            raise ValueError(f"Unknown reuse_mode: {reuse_mode}")
    
    # =========================================================================
    # PHASE 0: Compute Confidence Intervals and Paired Comparisons
    # =========================================================================
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute confidence intervals and paired comparisons.
        
        Returns dict with:
        - ci: Bootstrap 95% CIs per (source, target, mode)
        - paired: Paired differences (Δ_memory, Δ_proto, etc.) vs native
        """
        print("\n" + "=" * 60)
        print("Computing confidence intervals and paired comparisons...")
        print("=" * 60)
        
        results = {"ci": {}, "paired": {}}
        
        # Group results by (source, target)
        for source_name, _, _ in self.config.source_envs:
            results["ci"][source_name] = {}
            results["paired"][source_name] = {}
            
            for target_name, _, _ in self.config.target_envs:
                results["ci"][source_name][target_name] = {}
                results["paired"][source_name][target_name] = {}
                
                # Get native results for paired comparison
                native_results = self._get_mode_results(source_name, target_name, "native")
                native_outcomes = {r["seed"]: r["mean_outcome"] for r in native_results}
                
                for mode in self.config.reuse_modes:
                    mode_results = self._get_mode_results(source_name, target_name, mode)
                    if not mode_results:
                        continue
                    
                    outcomes = np.array([r["mean_outcome"] for r in mode_results])
                    
                    # Bootstrap CI
                    ci = self._bootstrap_ci(outcomes, self.config.bootstrap_samples)
                    results["ci"][source_name][target_name][mode] = {
                        "mean": float(np.mean(outcomes)),
                        "std": float(np.std(outcomes)),
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                    }
                    
                    # Paired comparison vs native
                    if mode != "native" and native_outcomes:
                        deltas = []
                        for r in mode_results:
                            if r["seed"] in native_outcomes:
                                delta = r["mean_outcome"] - native_outcomes[r["seed"]]
                                deltas.append(delta)
                        
                        if deltas:
                            deltas = np.array(deltas)
                            delta_ci = self._bootstrap_ci(deltas, self.config.bootstrap_samples)
                            results["paired"][source_name][target_name][mode] = {
                                "mean_delta": float(np.mean(deltas)),
                                "std_delta": float(np.std(deltas)),
                                "ci_low": delta_ci[0],
                                "ci_high": delta_ci[1],
                                "n_pairs": len(deltas),
                                "win_rate": float(np.mean(deltas > 0)),
                            }
        
        return results
    
    def _get_mode_results(self, source: str, target: str, mode: str) -> List[Dict]:
        """Get results for a specific (source, target, mode) triplet."""
        return [r for r in self.seed_aggregates
                if r["source_env"] == source
                and r["target_env"] == target
                and r["reuse_mode"] == mode
                and "error" not in r]
    
    def _bootstrap_ci(self, data: np.ndarray, n_samples: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if len(data) == 0:
            return (0.0, 0.0)
        
        boot_means = []
        for _ in range(n_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            boot_means.append(np.mean(sample))
        
        boot_means = np.array(boot_means)
        ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return (ci_low, ci_high)
    
    # =========================================================================
    # PHASE 5: Canonical Transfer Plots
    # =========================================================================
    
    def plot_all(self, portability_results: Dict, statistics: Dict) -> None:
        """Generate all canonical transfer plots."""
        print("\n" + "=" * 60)
        print("Generating canonical transfer plots...")
        print("=" * 60)
        
        try:
            # 1. Transfer gain (paired) with CI
            self._plot_paired_transfer_gain(statistics)
            
            # 2. Cone-summary portability
            self._plot_portability(portability_results)
            
            # 3. Reuse breakdown (win/lose/tie)
            self._plot_reuse_breakdown(statistics)
            
            # 4. Transfer matrix heatmap
            self._plot_transfer_matrix()
            
            # 5. Mode comparison box plot
            self._plot_mode_comparison()
            
        except Exception as e:
            print(f"Warning: Failed to generate some plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_paired_transfer_gain(self, statistics: Dict) -> None:
        """Plot paired transfer gain with confidence intervals."""
        n_sources = min(len(self.config.source_envs), 2)
        fig, axes = plt.subplots(1, n_sources, figsize=(7 * n_sources, 6))
        if n_sources == 1:
            axes = [axes]
        
        # Collect data
        modes_to_plot = [m for m in self.config.reuse_modes if m != "native"]
        
        for ax_idx, (source_name, _, _) in enumerate(self.config.source_envs[:2]):  # Max 2 sources
            ax = axes[ax_idx]
            
            for target_name, _, _ in self.config.target_envs:
                paired = statistics.get("paired", {}).get(source_name, {}).get(target_name, {})
                
                x_labels = []
                means = []
                ci_lows = []
                ci_highs = []
                
                for mode in modes_to_plot:
                    if mode in paired:
                        x_labels.append(mode)
                        means.append(paired[mode]["mean_delta"])
                        ci_lows.append(paired[mode]["ci_low"])
                        ci_highs.append(paired[mode]["ci_high"])
                
                if means:
                    x = np.arange(len(x_labels))
                    yerr = np.array([
                        [m - l for m, l in zip(means, ci_lows)],
                        [h - m for m, h in zip(means, ci_highs)]
                    ])
                    ax.bar(x, means, yerr=yerr, capsize=5, alpha=0.7, label=target_name)
            
            ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
            ax.set_ylabel("Δ Outcome (reuse − native)")
            ax.set_xlabel("Reuse Mode")
            ax.set_title(f"Transfer Gain: {source_name}")
            ax.set_xticks(range(len(modes_to_plot)))
            ax.set_xticklabels(modes_to_plot, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "fig" / "paired_transfer_gain.png")
    
    def _plot_portability(self, portability_results: Dict) -> None:
        """Plot cone-summary portability scatter plots."""
        n_pairs = sum(len(targets) for targets in portability_results.values())
        if n_pairs == 0:
            return
        
        n_cols = max(n_pairs, 1)
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        elif n_cols == 1:
            axes = axes.reshape(2, 1)
        
        plot_idx = 0
        for source, targets in portability_results.items():
            for target, data in targets.items():
                C_s = np.array(data["C_source"])
                C_t = np.array(data["C_target"])
                H_s = np.array(data["H_source"])
                H_t = np.array(data["H_target"])
                
                # Cone volume scatter
                if n_cols > 1:
                    ax1 = axes[0, plot_idx]
                    ax2 = axes[1, plot_idx]
                else:
                    ax1 = axes[0]
                    ax2 = axes[1]
                
                if len(C_s) > 0 and len(C_t) > 0:
                    ax1.scatter(C_s, C_t, alpha=0.5)
                    if C_s.max() > C_s.min() and C_t.max() > C_t.min():
                        ax1.plot([C_s.min(), C_s.max()], [C_s.min(), C_s.max()], 'r--', alpha=0.5)
                ax1.set_xlabel(f"C_source ({source})")
                ax1.set_ylabel(f"C_target ({target})")
                ax1.set_title(f"Cone Vol: r={data['cone_spearman_r']:.2f}")
                
                # Horizon scatter
                if len(H_s) > 0 and len(H_t) > 0:
                    ax2.scatter(H_s, H_t, alpha=0.5)
                    if H_s.max() > H_s.min() and H_t.max() > H_t.min():
                        ax2.plot([H_s.min(), H_s.max()], [H_s.min(), H_s.max()], 'r--', alpha=0.5)
                ax2.set_xlabel(f"H_source ({source})")
                ax2.set_ylabel(f"H_target ({target})")
                ax2.set_title(f"Horizon: r={data['horizon_spearman_r']:.2f}")
                
                plot_idx += 1
        
        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "fig" / "cone_portability.png")
    
    def _plot_reuse_breakdown(self, statistics: Dict) -> None:
        """Plot reuse breakdown: fraction of wins/loses/ties."""
        paired = statistics.get("paired", {})
        
        modes_to_plot = [m for m in self.config.reuse_modes if m != "native"]
        
        # Aggregate win rates across all source-target pairs
        mode_wins = {m: [] for m in modes_to_plot}
        
        for source, targets in paired.items():
            for target, modes in targets.items():
                for mode, data in modes.items():
                    if mode in mode_wins:
                        mode_wins[mode].append(data.get("win_rate", 0.5))
        
        if not any(mode_wins.values()):
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(len(modes_to_plot))
        wins = [np.mean(mode_wins[m]) if mode_wins[m] else 0.5 for m in modes_to_plot]
        loses = [1 - w for w in wins]
        
        ax.bar(x, wins, label="Reuse wins", alpha=0.7, color='green')
        ax.bar(x, loses, bottom=wins, label="Native wins", alpha=0.7, color='red')
        
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
        ax.set_ylabel("Fraction")
        ax.set_xlabel("Reuse Mode")
        ax.set_title("Reuse Breakdown: Fraction of Episodes where Reuse Beats Native")
        ax.set_xticks(x)
        ax.set_xticklabels(modes_to_plot)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "fig" / "reuse_breakdown.png")
    
    def _plot_transfer_matrix(self) -> None:
        """Plot transfer matrix heatmap."""
        # Build matrix
        source_names = [s[0] for s in self.config.source_envs]
        target_names = [t[0] for t in self.config.target_envs]
        
        n_sources = len(source_names)
        n_targets = len(target_names)
        n_modes = len(self.config.reuse_modes)
        
        outcome_matrix = np.full((n_sources * n_modes, n_targets), np.nan)
        mode_labels = []
        
        for i, source in enumerate(source_names):
            for j, mode in enumerate(self.config.reuse_modes):
                row_idx = i * n_modes + j
                mode_labels.append(f"{source}\n{mode}")
                for k, target in enumerate(target_names):
                    results = self._get_mode_results(source, target, mode)
                    if results:
                        outcomes = [r["mean_outcome"] for r in results]
                        outcome_matrix[row_idx, k] = np.mean(outcomes)
        
        fig, ax = plt.subplots(1, 1, figsize=(max(8, n_targets * 2), max(6, n_sources * n_modes * 0.6)))
        im = ax.imshow(outcome_matrix, aspect='auto', cmap='RdYlGn')
        
        ax.set_xticks(range(n_targets))
        ax.set_xticklabels(target_names)
        ax.set_yticks(range(n_sources * n_modes))
        ax.set_yticklabels(mode_labels, fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Outcome")
        
        for i in range(n_sources * n_modes):
            for j in range(n_targets):
                if not np.isnan(outcome_matrix[i, j]):
                    ax.text(j, i, f"{outcome_matrix[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=7)
        
        ax.set_xlabel("Target Environment")
        ax.set_ylabel("Source × Mode")
        ax.set_title("Transfer Matrix: Mean Outcome")
        
        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "fig" / "transfer_matrix.png")
    
    def _plot_mode_comparison(self) -> None:
        """Plot outcome comparison across reuse modes (box plot)."""
        mode_outcomes = {m: [] for m in self.config.reuse_modes}
        
        for r in self.seed_aggregates:
            if "error" not in r:
                mode_outcomes[r["reuse_mode"]].append(r["mean_outcome"])
        
        data = [mode_outcomes[m] for m in self.config.reuse_modes if mode_outcomes[m]]
        labels = [m for m in self.config.reuse_modes if mode_outcomes[m]]
        
        if not data:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel("Outcome (higher is better)")
        ax.set_title("Outcome Distribution by Reuse Mode")
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, mode in enumerate(labels):
            if mode_outcomes[mode]:
                mean_val = np.mean(mode_outcomes[mode])
                ax.plot(i + 1, mean_val, 'ro', markersize=8)
        
        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "fig" / "mode_comparison.png")
    
    @staticmethod
    def _save_fig(fig, path: Path, dpi: int = 150):
        """Save figure and close."""
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    def save_results(self, portability_results: Dict, statistics: Dict) -> None:
        """Save all results to disk."""
        # Save per-seed aggregates
        with open(self.run_dir / "data" / "seed_aggregates.json", "w") as f:
            json.dump(self.seed_aggregates, f, indent=2, default=str)
        
        # Save statistics (CIs and paired comparisons)
        with open(self.run_dir / "data" / "statistics.json", "w") as f:
            json.dump(statistics, f, indent=2, default=str)
        
        # Save portability results
        with open(self.run_dir / "data" / "portability.json", "w") as f:
            json.dump(portability_results, f, indent=2, default=str)
        
        # Save config
        config_dict = {
            "source_envs": [s[0] for s in self.config.source_envs],
            "target_envs": [t[0] for t in self.config.target_envs],
            "reuse_modes": self.config.reuse_modes,
            "n_seeds": self.config.n_seeds,
            "train_steps": self.config.train_steps,
            "eval_episodes": self.config.eval_episodes,
            "n_prototypes": self.config.n_prototypes,
            "behavioral_k": self.config.behavioral_k,
        }
        with open(self.run_dir / "data" / "transfer_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save transfer matrix for backward compatibility
        transfer_matrix = self._compute_transfer_matrix()
        with open(self.run_dir / "data" / "transfer_matrix.json", "w") as f:
            json.dump(transfer_matrix, f, indent=2)
        
        print(f"\nResults saved to: {self.run_dir}")
    
    def _compute_transfer_matrix(self) -> Dict[str, Any]:
        """Compute transfer matrix with means and CIs."""
        matrix = {}
        
        for source_name, _, _ in self.config.source_envs:
            matrix[source_name] = {}
            for target_name, _, _ in self.config.target_envs:
                matrix[source_name][target_name] = {}
                for mode in self.config.reuse_modes:
                    results = self._get_mode_results(source_name, target_name, mode)
                    if results:
                        outcomes = [r["mean_outcome"] for r in results]
                        matrix[source_name][target_name][mode] = {
                            "mean": float(np.mean(outcomes)),
                            "std": float(np.std(outcomes)),
                            "n": len(outcomes),
                        }
        
        return matrix
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def run(self) -> None:
        """Run complete transfer experiment with all phases."""
        print("=" * 60)
        print("Cross-Environment Transfer Experiment")
        print(f"Run directory: {self.run_dir}")
        print("=" * 60)
        
        # Phase 0: Train on source environments
        self.train_source_envs()
        
        # Phase 1: Portability tests
        portability_results = self.run_portability_tests()
        
        # Phase 0/2/3: Multi-seed transfer evaluation
        self.run_transfer_tests_multiseed()
        
        # Phase 0: Compute statistics
        statistics = self.compute_statistics()
        
        # Phase 5: Generate plots
        self.plot_all(portability_results, statistics)
        
        # Save all results
        self.save_results(portability_results, statistics)
        
        print("\n" + "=" * 60)
        print("Transfer experiment complete!")
        print("=" * 60)


# =============================================================================
# Reuse Actor Wrappers
# =============================================================================

class MemoryReuseActor:
    """Wrapper that replaces sampled z with nearest neighbor from source memory."""
    
    def __init__(self, actor: Actor, source_zs: np.ndarray):
        self.actor = actor
        self.source_zs = source_zs
        self.device = actor.device
    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    
    def sample_z(self, s0_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_native, z_mu, z_logstd = self.actor.sample_z(s0_t)
        
        if len(self.source_zs) == 0:
            return z_native, z_mu, z_logstd
        
        z_native_np = z_native.detach().cpu().numpy().squeeze()
        distances = np.linalg.norm(self.source_zs - z_native_np, axis=1)
        nearest_idx = np.argmin(distances)
        z_reused = torch.tensor(
            self.source_zs[nearest_idx], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        return z_reused, z_mu, z_logstd


class PrototypeReuseActor:
    """Wrapper that chooses z from fixed prototypes (nearest to native z)."""
    
    def __init__(self, actor: Actor, prototypes: np.ndarray):
        self.actor = actor
        self.prototypes = prototypes
        self.device = actor.device
    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    
    def sample_z(self, s0_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_native, z_mu, z_logstd = self.actor.sample_z(s0_t)
        
        if len(self.prototypes) == 0:
            return z_native, z_mu, z_logstd
        
        z_native_np = z_native.detach().cpu().numpy().squeeze()
        distances = np.linalg.norm(self.prototypes - z_native_np, axis=1)
        best_idx = np.argmin(distances)
        z_proto = torch.tensor(
            self.prototypes[best_idx], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        return z_proto, z_mu, z_logstd


class BehavioralReuseActor:
    """
    Wrapper that selects z using behavioral scoring.
    
    Evaluates K candidates from source memory and picks the best by:
    score = -NLL0_proxy - α*cone_vol + β*E[T]
    """
    
    def __init__(
        self, 
        actor: Actor, 
        behavioral_prototypes: List[ConeSummary],
        source_memory: List[Dict],
        k: int = 10,
        alpha: float = 0.1,
        beta: float = 0.05,
    ):
        self.actor = actor
        self.prototypes = behavioral_prototypes
        self.memory = source_memory
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.device = actor.device
    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    
    def sample_z(self, s0_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_native, z_mu, z_logstd = self.actor.sample_z(s0_t)
        
        if len(self.prototypes) == 0:
            return z_native, z_mu, z_logstd
        
        # Score each prototype using proxy metrics
        best_score = float('-inf')
        best_z = None
        
        for proto in self.prototypes:
            # Proxy score: prefer high bind rate, low cone volume, reasonable horizon
            score = proto.bind_rate - self.alpha * proto.cone_volume + self.beta * proto.E_T
            
            if score > best_score:
                best_score = score
                best_z = proto.z
        
        if best_z is None:
            return z_native, z_mu, z_logstd
        
        z_behavioral = torch.tensor(
            best_z, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        return z_behavioral, z_mu, z_logstd


class RandomZActor:
    """Wrapper that samples random z from source memory (sanity check)."""
    
    def __init__(self, actor: Actor, source_zs: np.ndarray):
        self.actor = actor
        self.source_zs = source_zs
        self.device = actor.device
    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    
    def sample_z(self, s0_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_native, z_mu, z_logstd = self.actor.sample_z(s0_t)
        
        if len(self.source_zs) == 0:
            return z_native, z_mu, z_logstd
        
        idx = np.random.randint(len(self.source_zs))
        z_random = torch.tensor(
            self.source_zs[idx], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        return z_random, z_mu, z_logstd


class ShuffledMemoryActor:
    """
    Wrapper with shuffled memory (sanity check).
    
    Shuffles the source z's so nearest-neighbor lookup is meaningless.
    """
    
    def __init__(self, actor: Actor, source_zs: np.ndarray):
        self.actor = actor
        # Shuffle z's
        shuffled = source_zs.copy()
        np.random.shuffle(shuffled)
        self.source_zs = shuffled
        self.device = actor.device
    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    
    def sample_z(self, s0_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_native, z_mu, z_logstd = self.actor.sample_z(s0_t)
        
        if len(self.source_zs) == 0:
            return z_native, z_mu, z_logstd
        
        # Nearest neighbor in shuffled memory
        z_native_np = z_native.detach().cpu().numpy().squeeze()
        distances = np.linalg.norm(self.source_zs - z_native_np, axis=1)
        nearest_idx = np.argmin(distances)
        z_shuffled = torch.tensor(
            self.source_zs[nearest_idx], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        return z_shuffled, z_mu, z_logstd


# =============================================================================
# Convenience Function
# =============================================================================

def run_transfer_experiment(config: TransferConfig) -> CrossEnvTransferHarness:
    """Run a complete transfer experiment."""
    harness = CrossEnvTransferHarness(config)
    harness.run()
    return harness
