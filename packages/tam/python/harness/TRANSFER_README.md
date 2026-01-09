# Cross-Environment Transfer Harness

This harness implements comprehensive cross-environment transfer testing for TAM (Theory of Action and Memory) experiments. It tests whether commitments/skills learned in one environment transfer to another.

## Overview

The transfer harness implements all phases from the TODO list:

- **Phase 0**: Multi-seed evaluation with confidence intervals and paired comparisons
- **Phase 1**: Cone-summary portability and rank preservation tests
- **Phase 2**: Behavioral retrieval reuse (selects z by behavioral score, not distance)
- **Phase 3**: Behavioral prototype clustering (clusters by outcome, not z-space)
- **Phase 5**: Canonical transfer plots (paired gain, portability, breakdown, matrix)
- **Phase 6**: Sanity checks (random z, shuffled memory baselines)

## Quick Start

```python
from harness.cross_env_runner import TransferConfig, run_transfer_experiment

config = TransferConfig(
    source_envs=[
        ("gridworld_s0", make_gridworld_env, {"seed": 0}),
    ],
    target_envs=[
        ("gridworld_t1", make_gridworld_env, {"seed": 1}),
    ],
    train_steps=2000,
    eval_episodes=100,
    n_seeds=10,  # Run each condition with 10 seeds for statistical validity
    reuse_modes=["native", "memory", "prototype", "behavioral", "random", "shuffled"],
    actor_kwargs={...},
)

harness = run_transfer_experiment(config)
```

## Reuse Modes

1. **`native`**: Baseline - z ~ q(z|s0) using current env observation
2. **`memory`**: Nearest-neighbor z from source env memory (z-space distance)
3. **`prototype`**: KMeans prototypes from source env (geometric clustering)
4. **`behavioral`**: Behavioral prototypes (clustered by cone_vol, E[T], bind_rate)
5. **`random`**: Random z from source memory (sanity check)
6. **`shuffled`**: Nearest-neighbor in shuffled memory (sanity check)

## Output Files

### Data Files (`data/`)

- **`seed_aggregates.json`**: Per-seed results for all (source, target, mode, seed) combinations
- **`statistics.json`**: Confidence intervals and paired comparisons
- **`portability.json`**: Cone-summary portability test results (Pearson/Spearman correlations)
- **`transfer_matrix.json`**: Transfer matrix with means and standard deviations
- **`transfer_config.json`**: Experiment configuration

### Plots (`fig/`)

- **`paired_transfer_gain.png`**: Paired transfer gain (Δ = reuse - native) with 95% CIs
- **`cone_portability.png`**: Scatter plots of C_source vs C_target and H_source vs H_target
- **`reuse_breakdown.png`**: Fraction of episodes where reuse beats native
- **`transfer_matrix.png`**: Heatmap of mean outcomes by source→target→mode
- **`mode_comparison.png`**: Box plots comparing outcome distributions across modes

## Key Features

### Multi-Seed Evaluation (Phase 0)

Each (source, target, reuse_mode) combination is evaluated with `n_seeds` different random seeds. This enables:
- Bootstrap confidence intervals (95% CI)
- Paired comparisons (Δ_memory, Δ_proto vs native)
- Statistical significance testing

### Cone-Summary Portability (Phase 1)

Tests whether z commitments have shared semantics across environments:
- Samples z's from source memory
- Computes (C_source, H_source) and (C_target, H_target) for each z
- Computes Pearson and Spearman correlations
- If correlations ≈ 0, z is not portable → reuse via distance is meaningless

### Behavioral Reuse (Phase 2/3)

Instead of selecting z by distance in z-space, behavioral reuse:
- Scores candidates by: `score = bind_rate - α*cone_vol + β*E[T]`
- Selects best-scoring z from K candidates
- Tests whether behavioral similarity transfers better than geometric similarity

### Sanity Checks (Phase 6)

- **Random z**: Should perform worse than memory/prototype
- **Shuffled memory**: Nearest-neighbor in shuffled memory should destroy transfer benefit
- If shuffled performs as well as memory → distance-based reuse is meaningless

## Configuration Parameters

```python
TransferConfig(
    # Environments
    source_envs=[...],  # List of (name, factory, kwargs) tuples
    target_envs=[...],  # List of (name, factory, kwargs) tuples
    
    # Training
    train_steps=6000,  # Steps per source env
    eval_episodes=200,  # Episodes per (source, target, mode, seed)
    
    # Statistical validity
    n_seeds=10,  # Seeds per condition
    bootstrap_samples=1000,  # Bootstrap samples for CI
    
    # Reuse modes
    reuse_modes=["native", "memory", "prototype", "behavioral", "random", "shuffled"],
    n_prototypes=4,  # Number of prototypes
    behavioral_k=10,  # K candidates for behavioral retrieval
    
    # Freezing
    freeze_actor=True,  # Freeze actor encoder during transfer
    freeze_tube=False,  # Allow tube to adapt
    freeze_policy=False,
    
    # Portability tests
    n_portability_samples=50,  # z's to test for portability
    
    # Evaluation
    k_sigma_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    k_star=2.0,
    
    seed=0,  # Base seed
)
```

## Interpreting Results

### Decision Gates (from TODO)

Only proceed to next phases if:

1. **Phase 1** shows partial portability of (C, H) → Spearman r > 0.3
2. **Phase 2** beats native in ≥1 env with CI → mean_delta > 0 and CI excludes 0
3. **Phase 3** outperforms nearest-z → behavioral > memory in paired comparison

If none hold:
> z currently encodes environment-specific commitments, not transferable concepts — which tells you exactly what to fix next.

### Key Metrics

- **Transfer gain**: `Δ = outcome(reuse) - outcome(native)`
  - Positive Δ → reuse helps
  - Negative Δ → reuse hurts
  - Check if 95% CI excludes 0 (statistically significant)

- **Win rate**: Fraction of seeds/episodes where reuse beats native
  - > 0.5 → reuse generally helpful
  - < 0.5 → reuse generally harmful

- **Portability correlation**: Spearman r for cone volume and horizon
  - r > 0.5 → strong portability
  - r < 0.3 → weak/no portability

## Example Output Interpretation

```
Transfer Gain: gridworld_s0
  Mode: memory
    Outcome: -10.763 ± 0.123
    Δ vs native: +0.014 ± 0.045 (CI: [-0.075, +0.103])
    Win rate: 0.52

  Mode: behavioral
    Outcome: -10.701 ± 0.098
    Δ vs native: +0.076 ± 0.038 (CI: [+0.002, +0.150])  ← Significant!
    Win rate: 0.68
```

Interpretation:
- Memory reuse: Small positive gain, not statistically significant (CI includes 0)
- Behavioral reuse: Significant positive gain (CI excludes 0), wins 68% of the time
- Conclusion: Behavioral reuse transfers better than memory reuse

## See Also

- `transfer_example.py`: Complete example script
- `experiment_harness.py`: Single-environment harness (used as subroutine)
- `evaluation.py`: Generic evaluation functions
