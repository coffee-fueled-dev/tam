"""
Example script showing how to use the experiment harness.

Run from the parent directory:
    python3 harness/example.py
Or:
    cd packages/tam/python && python3 harness/example.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from harness and envs
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from harness import ExperimentConfig, run_experiment

# Import environment
try:
    from envs.latent_rule_gridworld import LatentRuleGridworld
except ImportError:
    LatentRuleGridworld = None  # type: ignore


def make_gridworld_env(kwargs):
    """Factory function for gridworld environment."""
    if LatentRuleGridworld is None:
        raise ImportError("LatentRuleGridworld not available")
    return LatentRuleGridworld(**kwargs)


def main():
    """Example: Run a gridworld experiment using the harness."""
    
    # Create configuration
    config = ExperimentConfig(
        name="gridworld_harness_example",
        seed=0,
        train_steps=6000,  # Short for example
        eval_every=500,
        eval_episodes=50,
        env_kwargs={
            "seed": 0,
        },
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
    )
    
    # Run experiment
    actor, env, harness = run_experiment(
        env_factory=make_gridworld_env,
        config=config,
    )
    
    print(f"Experiment complete!")
    print(f"Final actor history length: {len(actor.history['step'])}")
    print(f"Memory size: {len(actor.mem)}")


if __name__ == "__main__":
    main()
