# V2 Migration - New Structure

This directory contains the refactored TAM codebase following the structure outlined in `todo.md`.

## Structure

```
v2/
├── actors/              # Actor implementations
│   └── actor.py      # Unsupervised Actor with multimodal router
├── environments/        # Environment implementations
│   └── cmg/            # CMG-specific environment
│       ├── env.py      # CMGEnv, CMGConfig
│       ├── episode.py  # generate_episode, rollout_with_forced_mode
│       └── diagnostics.py  # CMG topology tests (TODO)
├── eval/               # Universal evaluation harness
│   ├── runner.py       # Training/eval orchestration (TODO)
│   ├── dataset.py      # Episode buffers, sampling (TODO)
│   ├── metrics.py      # ✅ Universal metrics (label-free)
│   ├── plots/          # Universal plotting
│   │   ├── latent.py  # ✅ Latent space plots
│   │   └── geometry.py # ✅ Tube/energy/trajectory plots
│   └── suites/         # Suite definitions
│       └── universal.py # Universal suite (TODO)
├── tests/              # Fast pytest tests
│   ├── smoke/          # Smoke tests (TODO)
│   └── universal/      # Universal contract tests (TODO)
└── scripts/             # CLI entry points
    ├── run_universal_suite.py  # Universal suite runner (TODO)
    └── run_cmg_suite.py         # CMG-specific suite (TODO)
```

## Completed ✅

### Universal Components (Label-Free)

1. **`eval/metrics.py`**

   - `compute_latent_metrics()` - Pairwise cosine, effective rank, variance (always available)
   - Optional supervised metrics (silhouette, CH, DB) when labels provided
   - `compute_contract_metrics()` - Leak, volume, start error aggregations

2. **`eval/plots/latent.py`**

   - `plot_latent_scatter()` - Universal latent scatter (2D polar/Cartesian, higher-D PCA)
   - Optional label coloring, or color by reward/volume/head
   - `plot_pairwise_cosine_histogram()` - Label-free cosine similarity distribution

3. **`eval/plots/geometry.py`**

   - `plot_tube_overlay()` - Single tube with optional actual trajectory
   - `plot_multi_start_tubes()` - Multiple tubes from different z (measures implicit ports)
   - `plot_energy_landscape()` - CEM energy basins (actor-only, universal)
   - `plot_trajectory_pca()` - PCA of tube spines (label-free, optional coloring)

4. **`environments/cmg/`**
   - `env.py` - CMGEnv and CMGConfig (copied from original)
   - `episode.py` - generate_episode, rollout_with_forced_mode

## In Progress 🚧

### Remaining Components

1. **`eval/runner.py`** - Main training/eval orchestration

   - `make_run_dir()` - Output directory creation
   - `seed_all()` - Reproducibility
   - `train_encoder_stage()` - Encoder-only warmup
   - `train_actor_stage()` - Actor/tube training
   - `evaluate()` - Evaluation loop
   - `save_summary()` - JSON summary generation

2. **`eval/dataset.py`** - Episode buffers and sampling

   - `EpisodeBuffer` - FIFO buffer for episodes
   - `EpisodeDataset` - Iterator for training
   - Mode-balanced sampling (CMG-specific, but can be generalized)

3. **`environments/cmg/diagnostics.py`** - CMG topology tests

   - `fork_separability_test()` - Measures if environment forces disjoint futures
   - `commitment_regret_test()` - Volume penalty for shared z
   - `gating_irreversibility_test()` - Late-switching penalty

4. **`eval/suites/universal.py`** - Universal suite definition

   - Learning curves
   - Episode rollouts
   - Tube overlap matrix
   - Energy landscape
   - Latent health metrics

5. **Scripts** - CLI entry points
   - `scripts/run_universal_suite.py` - Universal suite runner
   - `scripts/run_cmg_suite.py` - CMG-specific suite runner

## Key Design Principles

1. **Label-Free First**: All universal components work without labels
2. **Optional Labels**: Supervised metrics/coloring available when labels provided
3. **Actor-Only**: Universal components don't assume environment internals
4. **Environment-Specific**: CMG diagnostics live with CMG environment
5. **Separation of Concerns**: Plots are pure rendering, metrics are pure computation

## Usage Example

```python
from v2.eval import Runner, RunConfig
from v2.actors.actor import Actor
from v2.environments.cmg import CMGEnv, CMGConfig

config = RunConfig(
    actor_cls=Actor,
    env_cls=CMGEnv,
    env_config=CMGConfig(d=3, K=4, T=20),
    train_epochs=2000,
    warmup_epochs=200,
)

runner = Runner(config)
results = runner.run()
```

## Migration Notes

- Original `tests/test_knot_v2.py` contains ~1800 lines
- New structure splits into ~10 focused modules
- Universal components are ~200-300 lines each
- CMG-specific components are ~100-200 lines each
- Total code size similar, but much more maintainable
