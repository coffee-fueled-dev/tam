"""
Example script showing how to use the cross-environment transfer harness.

This example demonstrates:
- Phase 0: Multi-seed evaluation with CIs
- Phase 1: Cone-summary portability tests
- Phase 2: Behavioral retrieval reuse
- Phase 3: Behavioral prototype clustering
- Phase 5: Canonical transfer plots
- Phase 6: Sanity checks (random, shuffled baselines)
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from harness.cross_env_runner import TransferConfig, run_transfer_experiment

# Import environments
try:
    from envs.latent_rule_gridworld import LatentRuleGridworld
    from envs.hidden_regime_fault import HiddenRegimeFaultEnv
except ImportError:
    LatentRuleGridworld = None  # type: ignore
    HiddenRegimeFaultEnv = None  # type: ignore


def make_gridworld_env(kwargs):
    """Factory function for gridworld environment."""
    if LatentRuleGridworld is None:
        raise ImportError("LatentRuleGridworld not available")
    return LatentRuleGridworld(**kwargs)


def make_pendulum_env(kwargs):
    """Factory function for pendulum environment."""
    if HiddenRegimeFaultEnv is None:
        raise ImportError("HiddenRegimeFaultEnv not available")
    return HiddenRegimeFaultEnv(**kwargs)


def main():
    """Example: Run a cross-environment transfer experiment with all phases."""
    
    # Create transfer configuration with full diagnostic settings
    config = TransferConfig(
        # Source environments (train here)
        source_envs=[
            ("gridworld_s0", make_gridworld_env, {"seed": 0}),
            # Can add more source envs for cross-environment testing:
            # ("gridworld_s1", make_gridworld_env, {"seed": 42}),
        ],
        
        # Target environments (test transfer here)
        target_envs=[
            ("gridworld_t1", make_gridworld_env, {"seed": 1}),  # Different seed = different rules
            ("gridworld_t2", make_gridworld_env, {"seed": 2}),  # Another variant
        ],
        
        # Training parameters
        train_steps=2000,  # Increase for real experiments
        eval_episodes=100,  # Episodes per (source, target, mode, seed)
        
        # Phase 0: Multi-seed evaluation
        n_seeds=5,  # Run each condition with 5 seeds for CIs (use 10+ for real experiments)
        bootstrap_samples=1000,  # Bootstrap samples for CI computation
        
        # Actor parameters
        actor_kwargs={
            "obs_dim": 9,  # gridworld observation dimension
            "pred_dim": 2,  # predicts normalized (x, y)
            "action_dim": 4,  # discrete actions
            "z_dim": 8,
            "maxH": 64,
            "minT": 2,
            "M": 16,
            "k_sigma": 2.0,
            "bind_success_frac": 0.85,
            "lambda_h": 0.002,
            "beta_kl": 3e-4,
            "halt_bias": -1.0,
            "reasoning_mode": "fixed",
        },
        
        # Transfer parameters (all reuse modes including sanity checks)
        reuse_modes=[
            "native",      # Phase 0: Baseline (z ~ q(z|s0))
            "memory",      # Phase 2: Nearest-neighbor in z-space
            "prototype",   # Phase 3: KMeans prototypes
            "behavioral",  # Phase 3: Behavioral prototypes (cluster by outcome)
            "random",      # Phase 6: Random z from memory (sanity check)
            "shuffled",    # Phase 6: Nearest-neighbor in shuffled memory (sanity check)
        ],
        n_prototypes=4,  # Number of prototypes for prototype/behavioral modes
        behavioral_k=10,  # K candidates for behavioral retrieval
        
        # Freeze settings
        freeze_actor=True,  # Freeze actor encoder during transfer
        freeze_tube=False,  # Allow tube to adapt (set True for strict transfer)
        freeze_policy=False,
        
        # Phase 1: Portability test settings
        n_portability_samples=50,  # Number of z's to test for portability
        
        # Evaluation parameters
        k_sigma_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        k_star=2.0,
        
        seed=0,
    )
    
    # Run transfer experiment (executes all phases)
    print("=" * 60)
    print("Cross-Environment Transfer Experiment")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Source envs: {[s[0] for s in config.source_envs]}")
    print(f"  Target envs: {[t[0] for t in config.target_envs]}")
    print(f"  Reuse modes: {config.reuse_modes}")
    print(f"  Seeds per condition: {config.n_seeds}")
    print(f"  Episodes per seed: {config.eval_episodes}")
    print(f"  Total evaluations: {len(config.source_envs) * len(config.target_envs) * len(config.reuse_modes) * config.n_seeds}")
    print()
    
    harness = run_transfer_experiment(config)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nResults saved to: {harness.run_dir}")
    print("\nKey outputs:")
    print(f"  - Statistics (CIs, paired comparisons): data/statistics.json")
    print(f"  - Portability test results: data/portability.json")
    print(f"  - Per-seed aggregates: data/seed_aggregates.json")
    print(f"  - Transfer matrix: data/transfer_matrix.json")
    print("\nKey plots:")
    print(f"  - Paired transfer gain: fig/paired_transfer_gain.png")
    print(f"  - Cone portability: fig/cone_portability.png")
    print(f"  - Reuse breakdown: fig/reuse_breakdown.png")
    print(f"  - Transfer matrix: fig/transfer_matrix.png")
    print(f"  - Mode comparison: fig/mode_comparison.png")


if __name__ == "__main__":
    main()
