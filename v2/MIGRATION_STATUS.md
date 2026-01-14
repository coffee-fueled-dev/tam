# Migration Status

## Completed ✅

1. **Directory Structure**: Created `v2/` with proper subdirectories
2. **CMG Environment**: Copied and organized into `v2/environments/cmg/`
   - `env.py` - CMGEnv and CMGConfig
   - `episode.py` - generate_episode, rollout_with_forced_mode
3. **Universal Metrics**: Created `v2/eval/metrics.py`
   - Label-free latent metrics (pairwise cosine, effective rank, variance)
   - Optional supervised metrics (silhouette, CH, DB) when labels provided
   - Contract metrics (leak, volume, start error, etc.)
4. **Universal Plots**: Created `v2/eval/plots/latent.py`
   - Label-free latent scatter plots
   - Optional label coloring
   - Pairwise cosine histogram

## Completed ✅

5. **Geometry Plots**: ✅ Extracted to `v2/eval/plots/geometry.py`

   - Tube overlay plots
   - Energy landscape
   - Trajectory PCA

6. **Eval Runner**: ✅ Created `v2/eval/runner.py`

   - Main orchestration with two-stage training
   - `v2/eval/dataset.py` - Episode buffers and sampling

7. **CMG Diagnostics**: ✅ Extracted to `v2/environments/cmg/diagnostics.py`

   - Fork separability
   - Commitment regret
   - Gating irreversibility

8. **Scripts**: ✅ Created CLI entry points

   - `v2/scripts/run_universal_suite.py`
   - `v2/scripts/run_cmg_suite.py`

9. **Universal Suite**: ✅ Created `v2/eval/suites/universal.py`
   - Learning curves
   - Episode rollouts
   - Tube overlap matrix
   - Energy landscape
   - Latent health

## Next Steps

1. Extract geometry plotting functions (tube overlay, energy landscape, trajectory PCA)
2. Create universal eval runner with training stages
3. Create dataset/buffer utilities
4. Extract CMG diagnostics to `environments/cmg/diagnostics.py`
5. Create universal suite definition
6. Create CLI scripts
7. Update imports in existing code

## Files to Migrate

From `tests/test_knot_v2.py`:

- `tube_intersection_test` → `eval/plots/geometry.py` (rewrite as multi-start, label-free)
- `cem_energy_landscape` → `eval/plots/geometry.py` (already universal)
- `mode_conditional_volume_curves` → CMG-specific or universal with reward/volume coloring
- `trajectory_space_pca` → `eval/plots/geometry.py` (label-free version)
- `fork_separability_test` → `environments/cmg/diagnostics.py`
- `commitment_regret_test` → `environments/cmg/diagnostics.py`
- `gating_irreversibility_test` → `environments/cmg/diagnostics.py`
- Training loop → `eval/runner.py`
- Episode generation → `eval/dataset.py`
