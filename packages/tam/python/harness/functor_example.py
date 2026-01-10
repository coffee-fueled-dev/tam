"""
Example script for functor learning between environments.

This demonstrates learning a mapping Φ: Z_A → Z_B such that
commitments from environment A induce equivalent cone behavior in B.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch

from harness.functor import (
    FunctorConfig,
    FunctorTrainer,
    create_functor,
    run_functor_experiment,
)
from harness.experiment_harness import ExperimentConfig, ExperimentHarness

# Import environments
try:
    from envs.latent_rule_gridworld import LatentRuleGridworld
    from envs.equivalent_envs import (
        RotatedGridworld,
        MirroredGridworld,
        ScaledGridworld,
        make_standard_gridworld,
        make_rotated_gridworld,
        make_mirrored_gridworld,
    )
except ImportError as e:
    print(f"Import error: {e}")
    LatentRuleGridworld = None


def train_actor_on_env(
    env_factory,
    env_kwargs,
    actor_kwargs,
    train_steps: int = 3000,
    seed: int = 0,
    name: str = "actor",
) -> "Actor":
    """Train an actor on a given environment."""
    from harness.experiment_harness import ExperimentConfig, ExperimentHarness
    
    config = ExperimentConfig(
        name=name,
        seed=seed,
        train_steps=train_steps,
        eval_every=train_steps + 1,  # Skip eval during training
        eval_episodes=0,
        env_kwargs=env_kwargs,
        actor_kwargs=actor_kwargs,
        output_dir=Path("runs") / f"functor_pretrain_{name}",
    )
    
    harness = ExperimentHarness(config=config, env_factory=env_factory)
    actor, _ = harness.train()
    
    return actor


def main():
    """Run functor learning experiment."""
    print("=" * 60)
    print("Functor Learning: Cross-Environment Commitment Transfer")
    print("=" * 60)
    
    # Common actor configuration
    actor_kwargs = {
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
    }
    
    # Environment A: Standard gridworld
    env_kwargs_A = {"seed": 0}
    
    # Environment B: Mirrored gridworld (same uncertainty, different coordinates)
    env_kwargs_B = {"seed": 0, "mirror_x": True}
    
    print("\nStep 1: Training actor on Environment A (standard gridworld)...")
    actor_A = train_actor_on_env(
        env_factory=make_standard_gridworld,
        env_kwargs=env_kwargs_A,
        actor_kwargs=actor_kwargs,
        train_steps=2000,
        seed=0,
        name="env_A",
    )
    
    print("\nStep 2: Training actor on Environment B (mirrored gridworld)...")
    actor_B = train_actor_on_env(
        env_factory=make_mirrored_gridworld,
        env_kwargs=env_kwargs_B,
        actor_kwargs=actor_kwargs,
        train_steps=2000,
        seed=42,
        name="env_B",
    )
    
    print("\nStep 3: Learning functor Φ: Z_A → Z_B...")
    
    # Configure functor learning
    functor_config = FunctorConfig(
        z_dim=8,
        functor_type="linear",  # Start with linear
        lr=1e-3,
        batch_size=16,
        n_epochs=30,
        n_eval_episodes=10,  # Reduce for faster testing
        k_sigma=2.0,
        eval_horizon=16,
        seed=0,
    )
    
    # Run functor learning
    trainer = run_functor_experiment(
        actor_A=actor_A,
        actor_B=actor_B,
        env_factory_A=make_standard_gridworld,
        env_factory_B=make_mirrored_gridworld,
        env_kwargs_A=env_kwargs_A,
        env_kwargs_B=env_kwargs_B,
        config=functor_config,
        n_source_samples=100,  # Reduce for faster testing
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("FUNCTOR LEARNING COMPLETE")
    print("=" * 60)
    
    if trainer.eval_results:
        results = trainer.eval_results[-1]
        print("\nCone Signature Correlations (A → B):")
        for name, corr in results["correlations"].items():
            r = corr["pearson_r"]
            status = "✓" if r > 0.5 else "○" if r > 0.2 else "✗"
            print(f"  {status} {name}: r = {r:.3f}")
        
        print("\nTransfer Delta (Functor - Native):")
        for name, delta in results["transfer_delta"].items():
            print(f"  {name}: Δ = {delta:+.3f}")
        
        print(f"\nMean Signature Error: {results['mean_error']:.4f}")
    
    print(f"\nResults saved to: {trainer.run_dir}")
    print("\nKey outputs:")
    print("  - fig/cone_preservation.png: Scatter plots of sig_A vs sig_B")
    print("  - fig/transfer_delta.png: Functor vs native comparison")
    print("  - fig/error_heatmap.png: Functor error in z-space")
    print("  - data/functor.pt: Learned functor weights")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print("""
If correlations are high (r > 0.5):
  → Intent geometry IS transportable between these environments
  → The functor successfully maps commitments while preserving semantics

If correlations are low (r < 0.3):
  → z encodes environment-specific information
  → Commitments are not directly transferable
  → May need: nonlinear functor, environment embeddings, or shared training
""")


def run_all_env_pairs():
    """Run functor learning on all equivalent environment pairs."""
    from envs.equivalent_envs import EQUIVALENT_ENV_PAIRS
    
    print("=" * 60)
    print("Running functor learning on all environment pairs")
    print("=" * 60)
    
    actor_kwargs = {
        "obs_dim": 9,
        "pred_dim": 2,
        "action_dim": 4,
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
    }
    
    results_summary = []
    
    for pair in EQUIVALENT_ENV_PAIRS:
        name, factory_A, kwargs_A, factory_B, kwargs_B, description = pair
        
        print(f"\n{'=' * 60}")
        print(f"Pair: {name}")
        print(f"Description: {description}")
        print("=" * 60)
        
        try:
            # Train actors
            actor_A = train_actor_on_env(
                env_factory=factory_A,
                env_kwargs=kwargs_A,
                actor_kwargs=actor_kwargs,
                train_steps=1500,
                seed=0,
                name=f"{name}_A",
            )
            
            actor_B = train_actor_on_env(
                env_factory=factory_B,
                env_kwargs=kwargs_B,
                actor_kwargs=actor_kwargs,
                train_steps=1500,
                seed=42,
                name=f"{name}_B",
            )
            
            # Learn functor
            functor_config = FunctorConfig(
                z_dim=8,
                functor_type="linear",
                n_epochs=20,
                n_eval_episodes=5,
                output_dir=Path("runs") / f"functor_{name}",
            )
            
            trainer = run_functor_experiment(
                actor_A=actor_A,
                actor_B=actor_B,
                env_factory_A=factory_A,
                env_factory_B=factory_B,
                env_kwargs_A=kwargs_A,
                env_kwargs_B=kwargs_B,
                config=functor_config,
                n_source_samples=50,
            )
            
            if trainer.eval_results:
                results = trainer.eval_results[-1]
                results_summary.append({
                    "pair": name,
                    "correlations": results["correlations"],
                    "mean_error": results["mean_error"],
                })
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Functor Learning Across Environment Pairs")
    print("=" * 60)
    
    for result in results_summary:
        print(f"\n{result['pair']}:")
        for comp, corr in result["correlations"].items():
            r = corr["pearson_r"]
            print(f"  {comp}: r = {r:.3f}")
        print(f"  Mean error: {result['mean_error']:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_env_pairs()
    else:
        main()
