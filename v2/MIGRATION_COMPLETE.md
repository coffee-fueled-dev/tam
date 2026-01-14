# Migration Complete ✅

## Summary

The codebase has been successfully migrated to the new `v2/` structure following the design outlined in `todo.md`. The new structure separates:

1. **Universal components** (actor-agnostic, env-agnostic)
2. **CMG-specific components** (environment topology diagnostics)
3. **Reusable evaluation harness** (training/eval orchestration)

## New Structure

```
v2/
├── actors/                    # Actor implementations
│   ├── knot_v2.py           # Supervised GeometricKnotActor
│   └── knot_v3.py           # Unsupervised Actor with multimodal router
├── environments/
│   └── cmg/                  # CMG-specific environment
│       ├── env.py           # CMGEnv, CMGConfig
│       ├── episode.py       # generate_episode, rollout_with_forced_mode
│       └── diagnostics.py   # ✅ Fork, regret, irreversibility tests
├── eval/                     # Universal evaluation harness
│   ├── runner.py            # ✅ Training/eval orchestration
│   ├── dataset.py           # ✅ Episode buffers, sampling
│   ├── metrics.py           # ✅ Universal metrics (label-free)
│   ├── plots/               # Universal plotting
│   │   ├── latent.py       # ✅ Latent space plots
│   │   └── geometry.py     # ✅ Tube/energy/trajectory plots
│   └── suites/
│       └── universal.py    # ✅ Universal suite definition
├── scripts/                  # CLI entry points
│   ├── run_universal_suite.py  # ✅ Universal suite runner
│   └── run_cmg_suite.py        # ✅ CMG suite runner (with topology tests)
└── tests/                    # Fast pytest tests (TODO: add smoke tests)
```

## Key Features

### Universal Components (Label-Free)

- **Metrics**: Pairwise cosine, effective rank, variance (always available)
- **Plots**: Latent scatter, energy landscape, trajectory PCA (optional label coloring)
- **Runner**: Two-stage training (encoder warmup → actor training)
- **Suite**: Learning curves, episode rollouts, tube overlap, energy landscape, latent health

### CMG-Specific Components

- **Diagnostics**: Fork separability, commitment regret, gating irreversibility
- **Episode Generation**: Mode-balanced sampling, forced-mode rollouts

## Usage

### Universal Suite (Any Actor + Any Env)

```bash
python v2/scripts/run_universal_suite.py \
    --d 3 --K 4 --z-dim 2 \
    --train-epochs 2000 --warmup-epochs 200 \
    --test-episodes 50
```

### CMG Suite (With Topology Tests)

```bash
python v2/scripts/run_cmg_suite.py \
    --d 3 --K 4 --z-dim 2 \
    --train-epochs 2000 --warmup-epochs 200 \
    --test-episodes 50
```

### Programmatic Usage

```python
from v2.eval import Runner, RunConfig
from v2.actors.knot_v2 import GeometricKnotActor
from v2.environments.cmg import CMGEnv, CMGConfig

config = RunConfig(
    actor_cls=GeometricKnotActor,
    actor_kwargs={'z_dim': 2, 'T': 20, 'pred_dim': 3},
    env_cls=CMGEnv,
    env_kwargs={'config': CMGConfig(d=3, K=4, T=20)},
    train_epochs=2000,
    warmup_epochs=200,
)

runner = Runner(config)
results = runner.run()
```

## Migration Checklist

- [x] Directory structure created
- [x] CMG environment organized
- [x] Universal metrics (label-free)
- [x] Universal plots (label-free)
- [x] Universal geometry plots
- [x] Eval runner with training stages
- [x] Dataset/buffer utilities
- [x] CMG diagnostics extracted
- [x] Universal suite definition
- [x] CLI scripts created
- [ ] Update imports in existing code (if needed)
- [ ] Add smoke tests (optional)

## Next Steps

1. **Test the new structure**: Run the scripts to verify everything works
2. **Update existing code**: If any code imports from old locations, update to `v2/`
3. **Add smoke tests**: Create fast pytest tests in `v2/tests/smoke/`
4. **Documentation**: Add docstrings and usage examples

## Design Principles Achieved

✅ **Label-Free First**: Universal components work without labels  
✅ **Optional Labels**: Supervised metrics available when labels provided  
✅ **Actor-Only**: Universal components don't assume environment internals  
✅ **Environment-Specific**: CMG diagnostics live with CMG environment  
✅ **Separation of Concerns**: Plots are rendering, metrics are computation, runner is orchestration
