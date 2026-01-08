# Experiment Harness

A unified framework for running TAM experiments with different environments and actors.

## Overview

The harness provides a clean separation between:
- **Experiment configuration** (`ExperimentConfig`)
- **Experiment execution** (`ExperimentHarness`)
- **Environment and actor factories** (provided by user)

## Basic Usage

```python
from harness import ExperimentConfig, run_experiment
from envs.latent_rule_gridworld import LatentRuleGridworld

# Define environment factory
def make_env(kwargs):
    return LatentRuleGridworld(**kwargs)

# Create config
config = ExperimentConfig(
    name="gridworld_test",
    seed=0,
    train_steps=6000,
    eval_every=1000,
    eval_episodes=200,
    env_kwargs={"seed": 0},
    actor_kwargs={
        "obs_dim": 9,
        "pred_dim": 2,
        "action_dim": 4,
        "z_dim": 8,
        "maxH": 64,
    },
)

# Run experiment
actor, env, harness = run_experiment(
    env_factory=make_env,
    config=config,
)
```

## Components

### `ExperimentConfig`

Configuration dataclass containing:
- Experiment metadata (name, seed)
- Training parameters (steps, eval frequency)
- Actor and environment kwargs
- Evaluation parameters

### `ExperimentHarness`

Main experiment runner that:
- Manages training loop
- Handles periodic evaluation
- Saves results and visualizations
- Manages experiment state

### `run_experiment()`

Convenience function that creates a harness, runs training, and saves results.

## Extending the Harness

To add custom evaluation or visualizations:

1. **Subclass `ExperimentHarness`** and override `_evaluate()`:
```python
class CustomHarness(ExperimentHarness):
    def _evaluate(self, step: int):
        # Custom evaluation logic
        pass
```

2. **Pass visualization functions** to `save_results()`:
```python
def my_viz(actor, env, run_dir):
    # Create visualizations
    pass

harness.save_results(visualizations=[my_viz])
```

## Environment Protocol

Environments must implement:
- `reset() -> None`
- `observe() -> np.ndarray`
- `rollout(policy_fn, horizon) -> (obs_seq, state_seq, actions, info)`

See `Environment` protocol in `experiment_harness.py` for details.
