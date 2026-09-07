# Stage 1B — held-out contextual transfer

This package is the first substantive Stage 1 experiment. It asks whether
conditioning expectations on an observable context supports transfer to unseen
coordinates.

A pass supports contextual statistical modeling. It does **not** establish
TAM-specific novelty over Bayesian, robust-control, or conformal methods.

`minimal/` and `stage1a/` remain unchanged runnable baselines.

## World

- Unbounded integer grid
- Observable terrain: `plain` if `(x + y) % 2 == 0`, else `rotated`
- Opaque ports `n`, `e`, `s`, `w`, `stay`
- On `plain`, ports use the ordinary cardinal deltas
- On `rotated`, the same ports apply a fixed 90° clockwise remapping
- No obstacles, delayed effects, hidden regimes, or learned representations

## Predictors (identical interaction budgets)

1. **unconditional** — one distribution per port
2. **exact_state** — one distribution per `(x, y, port)` (memorization, no transfer)
3. **context_pooled** — one distribution per `(terrain, port)`

## Protocol

Preregistered before running:

- Seeds `0–19`
- Disjoint train/eval cells (see `experiment.TRAIN_CELLS` / `EVAL_CELLS`)
- Train near the origin; eval/nav in a distant band so paths do not re-enter
  train coordinates
- Train: 20 forced outcomes per `(terrain, port)` on train cells only, with
  terrains interleaved per port so bounded unconditional histories keep a
  balanced mixture
- Freeze all predictors
- Probes: 20 held-out forced outcomes per `(terrain, port)` (no learning)
- Navigation: 32 held-out start/target trials, 24-step budget (no learning)
- Matched target streams; separate selection streams per controller
- References: full-domain cone, random policy, oracle controller
- Path efficiency is secondary only; L1 is not the checkerboard geodesic
- `mean_interactions_to_first_hit` averages successful trials; use
  `censored_rate` with it

## Evidence targets

1. Context-pooled held-out probe coverage ≥ 0.90, cone cardinality ≤ 1.25,
   Brier ≤ 0.03
2. Context-pooled held-out navigation success ≥ 0.90
3. Context-pooled navigation success exceeds unconditional by ≥ 0.20
4. Context-pooled navigation success exceeds exact-state by ≥ 0.20
5. Context-pooled held-out Brier improves over unconditional by ≥ 0.20
6. Context-pooled held-out Brier improves over exact-state by ≥ 0.20

Report unmet criteria without retuning.

## Run

```sh
python3 -m stage1b.experiment --output artifacts/stage1b
python3 -m unittest discover -s stage1b -t .
```

Artifacts: `steps.jsonl`, `summary.json`, `model.json`.
