#!/usr/bin/env python3
"""
Universal evaluation suite runner.
Works with any actor and environment that supports the interface.
"""

import sys
from pathlib import Path

# Add v2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
from v2.eval import Runner, RunConfig
from v2.eval.suites import UniversalSuite
from v2.actors.knot_v2 import GeometricKnotActor
from v2.environments.cmg import CMGEnv, CMGConfig


def main():
    parser = argparse.ArgumentParser(description="Run universal evaluation suite")
    
    # Actor args
    parser.add_argument("--z-dim", type=int, default=None, dest="z_dim",
                       help="Latent dimension (defaults to d if not specified)")
    parser.add_argument("--T", type=int, default=20, help="Trajectory length")
    
    # Environment args
    parser.add_argument("--d", type=int, default=3, help="State dimensionality")
    parser.add_argument("--K", type=int, default=4, help="Number of modes")
    parser.add_argument("--t-gate", type=int, default=5, dest="t_gate",
                       help="Gating window")
    
    # Training args
    parser.add_argument("--train-epochs", type=int, default=2000, dest="train_epochs",
                       help="Total training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=200, dest="warmup_epochs",
                       help="Encoder warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size",
                       help="Batch size")
    
    # Evaluation args
    parser.add_argument("--test-episodes", type=int, default=50, dest="test_episodes",
                       help="Number of test episodes")
    
    # Output args
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir",
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Determine z_dim
    z_dim = args.z_dim if args.z_dim is not None else args.d
    
    # Create config
    config = RunConfig(
        actor_cls=GeometricKnotActor,
        actor_kwargs={
            'z_dim': z_dim,
            'T': args.T,
            'pred_dim': args.d,
        },
        env_cls=CMGEnv,
        env_kwargs={
            'config': CMGConfig(
                d=args.d,
                K=args.K,
                T=args.T,
                t_gate=args.t_gate,
            )
        },
        train_epochs=args.train_epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        test_episodes=args.test_episodes,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed,
    )
    
    # Run training and evaluation
    runner = Runner(config)
    results = runner.run()
    
    # Run universal suite
    suite = UniversalSuite(runner.output_dir)
    suite_results = suite.run(
        runner.actor,
        runner.env,
        training_history=runner.training_history,
        eval_results=results['eval'],
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
