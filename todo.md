The goal is to split this into:

1. **a reusable training/eval harness** (actor-agnostic, env-agnostic where possible)
2. **a library of universal tests/plots** (contract geometry + latent diagnostics that don’t assume “modes”)
3. **environment-specific suites** (CMG topological necessity, forced-mode rollouts, etc.) that live _with the env_

Below is a concrete file structure + a “keep vs move” classification for every major thing in your script.

---

## Proposed file structure

### `tam/` (or your project root package)

```
tam/
  actors/
    knot_v2.py
    knot_v3.py               # your new online unsupervised actor variant (optional)
  environments/
    cmg/
      __init__.py
      env.py                     # CMGEnv, CMGConfig
      episode.py                 # generate_episode, rollout utilities
      diagnostics.py             # CMG-only diagnostics (fork, regret, irreversibility)
      tests/
        test_topology.py         # pytest unit tests for CMG topology (fast)
        suite_topology.py        # runnable “produce plots” script (slow)
  eval/
    __init__.py
    runner.py                    # run loop: train/eval, saving, seeding, logging
    dataset.py                   # buffers, episode sampling, iterators
    metrics.py                   # generic metric helpers
    plots/
      __init__.py
      latent.py                  # universal latent plots (no labels assumed)
      geometry.py                # tube plots, energy landscape, trajectory PCA
    suites/
      __init__.py
      universal.py               # universal suite definition (what to run, in what order)
  tests/
    conftest.py                  # shared fixtures: temp output dir, seeds
    smoke/
      test_train_step.py         # tiny “does it run” tests
      test_inference.py
    universal/
      test_contract.py           # numerical contract checks (no plots)
      test_latent.py             # invariants for latent (norm, variance, collapse)
      test_geometry.py           # tube overlap/hausdorff (actor-only)
  scripts/
    run_universal_suite.py       # CLI for universal suite (actor + any env that supports interface)
    run_cmg_suite.py             # CLI for CMG-specific suite
  artifacts/
    runs/                        # outputs
```

### Why this structure works

- `eval/runner.py` becomes the single place that knows about:
  - output dirs, timestamps, configs, JSON summaries
  - training stages (encoder SSL, actor training, eval)
- `eval/plots/*` are purely rendering utilities (take arrays, save fig)
- `tests/universal/*` are **pytest** tests that run fast and fail loudly
- `environments/cmg/diagnostics.py` is where all “forced mode”, “fork index”, “irreversibility”, etc. live

---

## Define a minimal interface between runner ⇄ env ⇄ actor

To keep universal stuff universal, you want small “capability” interfaces.

### Environment protocol (universal)

```python
class EpisodeEnv:
    obs_dim: int
    state_dim: int
    T: int

    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...
    def generate_episode(self, policy_mode: str) -> dict: ...
```

### Optional capability: “has discrete modes”

Only CMG needs it.

```python
class HasModes:
    K: int
    def rollout_with_forced_mode(self, k: int, policy: str) -> dict: ...
```

Your universal suite should not import `K`, `k`, “mode centroids”, etc. It should treat them as optional.

---

## What to keep as universal vs move to CMG-specific

Here’s a crisp rule:

- **Universal** = does not assume labeled modes, forced-mode rollouts, known goals, or env internals.
- **CMG-specific** = uses `k`, `K`, forced modes, explicit goals `g`, `t_gate` semantics, etc.

### Universal tests/plots to keep

These survive across any environment where you can produce episodes and where the actor produces tubes.

#### A) Contract + tube quality (actor-centric, no env modes needed)

Keep:

- **Leakage / over-estimation / volume / start-anchor / direction / endpoint** distributions
- “tube vs realized trajectory” overlay (for a few episodes)
- **tube intersection / overlap / hausdorff** _but not “from mode centroids”_:
  - instead: pick multiple inferred z’s for the _same s0_ (multi-start), compare tubes

Where it goes:

- `eval/metrics.py` for aggregation
- `eval/plots/geometry.py` for plotting
- `tests/universal/test_contract.py` for assertions (e.g., leak below threshold in trained model)

#### B) Latent health (unsupervised, online friendly)

Keep:

- **z norm** stats (should be ~1 if you normalize)
- collapse metrics: variance / pairwise cosine similarity / effective rank
- head utilization entropy (if using multimodal router)
- energy landscape (basins) for a fixed s0 (does not require env modes)

Where it goes:

- `eval/plots/latent.py`
- `tests/universal/test_latent.py`

#### C) Energy landscape plot (CEM energy)

Keep your `cem_energy_landscape`, but make it **actor-only**:

- don’t reference `config.K` as “expected basin count”
- just report: `n_basins`, `energy_range`, `topk_z`

Where it goes:

- `eval/plots/geometry.py` or `eval/plots/latent.py` (either is fine)

#### D) Trajectory space PCA (spines μ)

Keep, but make it:

- sample episodes, infer z, compute μ, PCA them
- no labeling/coloring by mode (unless env provides labels)
- optionally color by reward, volume, leak, or head index

Where it goes:

- `eval/plots/geometry.py`

---

### CMG-specific tests/plots to move (environment topology / necessity)

These should live in `environments/cmg/diagnostics.py` and `environments/cmg/tests/`.

Move:

- `fork_separability_test` (uses forced modes + CMG topology)
- `commitment_regret_test` (currently uses mode centroids + forced-mode rollouts)
- `gating_irreversibility_test` (explicitly uses `t_gate` + switching logic)
- anything referencing:
  - `env.params.g`, goals array, `env.k`, `rollout_with_forced_mode`
  - “ports required votes” logic
  - “expected ≥K basins” interpretation

Keep the **idea** of “necessity testing”, but the implementation is CMG’s.

---

## What to delete or demote (diagnostic-only, not worth keeping long-term)

These are useful while developing but become noise later:

- **Silhouette score on z with true labels** (this is explicitly supervised / env-specific)
- ARI computed as `adjusted_rand_score(labels, labels)` (this is always 1 and should be removed)
- “mode-balanced sampling” / stratified buffers keyed on mode labels
- “mode centroids” computed from labels (for unsupervised/online actor, centroids should come from router heads or clustering)

Instead, universal replacements:

- pairwise cosine similarity histogram
- head usage entropy (router)
- regret proxy distribution (WTA gap)
- tube overlap for multiple inferred z’s at same s0

---

## How to split your current file into modules (mapping)

### 1) Runner / orchestration

Create `eval/runner.py`:

- `make_run_dir()`
- `seed_all()`
- `train_encoder_stage(...)`
- `train_actor_stage(...)`
- `evaluate(...)`
- `save_summary(...)`
- calls suites (universal + env-specific)

Your current `run_test(...)` becomes ~30 lines: assemble config, call runner, call suites.

### 2) Universal plots

- `plot_results` becomes `eval/plots/latent.py::plot_latent_scatter(...)`
  - but: don’t assume modes
  - allow optional `labels` for coloring if provided
  - default: color by head index / reward quartile / volume

### 3) Universal metrics

- `compute_clustering_metrics` should be rewritten to not require labels.
  - Keep CH/DB/silhouette only when labels exist.
  - Add always-available metrics:
    - mean pairwise cosine sim
    - effective rank
    - covariance eigenvalues / condition number
    - utilization entropy

Put this in `eval/metrics.py`.

### 4) Universal geometry diagnostics

Move:

- tube intersection (rewrite to be label-free)
- energy landscape
- trajectory PCA (label-free)

into `eval/suites/universal.py` + `eval/plots/geometry.py`.

### 5) CMG topology suite

Move those 3 tests into:

- `environments/cmg/diagnostics.py`
- `environments/cmg/tests/test_topology.py` (fast versions)
- `scripts/run_cmg_suite.py` (slow plot-producing version)

---

## Recommended “universal suite” (minimal but high signal)

If I had to keep only a handful that remain useful forever:

### Universal suite outputs (per run)

1. **learning curves** (train loss terms over time): leak, vol, start_err, dir_loss, end_err
2. **episode rollouts** (N=8): overlay actual trajectory and tube (μ±σ) in first 2 dims or PCA
3. **tube overlap matrix** for multi-start z at fixed s0 (measures “implicit ports”)
4. **energy landscape** basins for fixed s0 (2D: exact, >2D: PCA projection + top-k points)
5. **latent health**: pairwise cosine histogram + effective rank + head usage entropy (if applicable)

Everything else can be opt-in.

---

## Recommended CMG suite (keep near env)

1. fork separability
2. irreversibility knee
3. regret ratio (shared vs per-mode) — but rewrite it so “per-mode” comes from _forced-mode trajectories_ and “shared” comes from a _single z_ you choose (e.g., best z at s0 or router max-weight head), rather than label centroids

---

## Concrete naming for tests vs suites

- **pytest tests**: fast, deterministic, no plotting

  - `tests/universal/test_contract.py`
  - `environments/cmg/tests/test_topology.py`

- **suites**: produce plots + JSON summaries, can be slow
  - `scripts/run_universal_suite.py`
  - `scripts/run_cmg_suite.py`

That split will keep CI fast and your research plots accessible.

---

## Quick call on each plot you currently generate

| Plot / Output                             |       Keep universal? | Where                                                                                                          |
| ----------------------------------------- | --------------------: | -------------------------------------------------------------------------------------------------------------- |
| `latent_clustering.png` (colored by mode) |            ❌ (as-is) | Replace with label-free latent plot in `eval/plots/latent.py`; CMG-mode coloring version can stay in env suite |
| `tube_intersection.png` (mode centroids)  |            ⚠️ rewrite | Universal version: multi-start tubes at same s0; CMG version can remain too                                    |
| `cem_energy_landscape.png`                |                    ✅ | Universal (`eval/plots/geometry.py`)                                                                           |
| `mode_volume_curves.png`                  | ❌ (mode-conditional) | CMG-specific or replace with “volume vs episode quantiles/reward” universal                                    |
| `trajectory_pca.png` (mode labels)        |          ✅ (rewrite) | Universal label-free PCA (color by reward/volume/head)                                                         |
| `fork_separability.png`                   |                    ❌ | CMG env diagnostics                                                                                            |
| `commitment_regret.png`                   |                    ❌ | CMG env diagnostics                                                                                            |
| `gating_irreversibility.png`              |                    ❌ | CMG env diagnostics                                                                                            |
