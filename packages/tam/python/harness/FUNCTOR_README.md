# Functor Learning for Cross-Environment Commitment Transfer

This module implements functor learning between environments, testing whether intent geometry (commitment semantics) is transportable.

## Core Idea

Learn a mapping Φ: Z_A → Z_B such that commitments from environment A, when translated via Φ, induce equivalent **cone behavior** in environment B.

**Key principle**: Match behavioral invariants (cone semantics), not actions or trajectories.

## Cone Signature

The cone signature captures commitment semantics:

| Symbol | Meaning                           |
| ------ | --------------------------------- |
| C      | Weighted log cone volume          |
| H      | Expected horizon E[T]             |
| r      | Soft bind rate                    |
| λ      | Dual pressure (reliability price) |

```
cone_sig = [C, H, r, λ_bind]
```

## Architecture

```
Environment A (frozen)              Environment B (frozen)
─────────────────────              ─────────────────────
s0_A ──► z_A ──► tube_A ──► cone_A   s0_B ──► Φ(z_A) ──► tube_B ──► cone_B
           │                                    ▲
           └──────────── invariants ────────────┘
```

The functor only sees:

- z_A (commitment from env A)
- cone statistics from A
- cone statistics from B

## Functor Types

1. **Linear**: `Φ(z) = Wz + b` — Start here. If this fails, nonlinear won't save you.
2. **Affine**: Learned scale + shift + linear
3. **MLP**: Nonlinear with residual connection
4. **Residual**: Low-rank `z + UV^T z`

## Training Procedure

### Phase 0: Prerequisites

- Train TAM fully in env A
- Train TAM fully in env B
- Freeze everything

### Phase 1: Data Collection

```python
D = []
for episode:
    z_A = sample_z_A(s0_A)
    sig_A = cone_stats(env_A, z_A)
    D.append(z_A, sig_A)
```

### Phase 2: Functor Fitting

Uses Evolution Strategies (ES) since cone signature computation is non-differentiable:

```python
for each batch:
    # 1. Sample perturbations to Φ parameters
    # 2. Evaluate loss for each perturbation
    # 3. Estimate gradient as weighted sum
    # 4. Update Φ
```

Loss: `L = ||normalize(sig_A) - normalize(sig_B)||²`

## Quick Start

```python
from harness.functor import FunctorConfig, run_functor_experiment
from envs.equivalent_envs import make_standard_gridworld, make_mirrored_gridworld

# Train actors on both environments first
actor_A = train_actor_on_env(make_standard_gridworld, ...)
actor_B = train_actor_on_env(make_mirrored_gridworld, ...)

# Learn functor
config = FunctorConfig(
    z_dim=8,
    functor_type="linear",
    n_epochs=30,
)

trainer = run_functor_experiment(
    actor_A=actor_A,
    actor_B=actor_B,
    env_factory_A=make_standard_gridworld,
    env_factory_B=make_mirrored_gridworld,
    env_kwargs_A={"seed": 0},
    env_kwargs_B={"seed": 0, "mirror_x": True},
    config=config,
)
```

## Evaluation

### Plot 1: Cone Preservation Scatter

- x-axis: sig_A component
- y-axis: sig_B component (after functor)
- Perfect functor → diagonal

### Plot 2: Transfer Delta vs Native

Compare in env B:

- Native z_B ~ q_B(z|s)
- Functor z_B = Φ(z_A)

Functor success ≠ outperform native
Functor success = matches structure

### Plot 3: Error Heatmap in z-space

- z_A colored by functor error
- Shows whether equivalence exists globally or only on a submanifold

## Equivalent Environment Pairs

Pre-defined pairs for testing:

| Pair                 | Description               |
| -------------------- | ------------------------- |
| standard_to_rotated  | Coordinate rotation (45°) |
| standard_to_mirrored | Reflection transformation |
| standard_to_scaled   | Velocity scaling (1.5x)   |
| standard_to_shifted  | Rule permutation          |

## Interpreting Results

### If correlations are high (r > 0.5):

- ✓ Intent geometry IS transportable
- ✓ Functor successfully maps commitments
- → Concepts are not clusters, but transportable structures

### If correlations are low (r < 0.3):

- ✗ z encodes environment-specific information
- ✗ Commitments not directly transferable
- → May need: nonlinear functor, environment embeddings, shared training

## What Success Would Mean

If a linear functor works:

1. Intent geometry is environment-invariant
2. Concepts are transportable structures
3. TAM commitments are semantic commitments, not "skills"

That's a big deal — and very falsifiable.

## Files

- `functor.py`: Core functor module and training
- `functor_example.py`: Example script
- `envs/equivalent_envs.py`: Topologically equivalent environments

## Output Structure

```
runs/functor_YYYYMMDD_HHMMSS/
├── fig/
│   ├── cone_preservation.png    # Scatter: sig_A vs sig_B
│   ├── transfer_delta.png       # Functor vs native comparison
│   ├── error_heatmap.png        # Functor error in z-space
│   └── training_loss.png        # Training curve
├── data/
│   ├── config.json
│   ├── training_losses.json
│   ├── eval_results.json
│   ├── functor.pt               # Learned weights
│   └── source_dataset.json      # Collected cone signatures
```

## Command Line

```bash
# Run single experiment
python3 harness/functor_example.py

# Run all environment pairs
python3 harness/functor_example.py --all
```
