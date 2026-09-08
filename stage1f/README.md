# Stage 1F — Geometric Cone Refinement

This package treats cones as **literal geometric objects**: angular sectors over
unit 2D one-step trajectories. A realized trajectory is either inside or outside
the saved pre-action `ConeSet`. An outside realization is a contradiction;
refinement must **widen** an existing cone or **add** a new one so the observed
trajectory is contained. The goal is tight cones that contain reality.

There is **no** control, safety, reward, narrowing, or TAM-superiority claim.
Earlier stages remain unchanged runnable baselines. Categorical cones in prior
stages are finite-set approximations of this geometric role.

## Geometry

- Trajectory angle θ ∈ [0, 360) degrees (integer draws)
- Cone: `{r (cos θ, sin θ) : r ≥ 0, circular_distance(θ, center) ≤ half_width}`
- `ConeSet`: union of cones; binding succeeds iff the realized angle is in the union
- Measure: total angular measure of the union (degrees)

## Learners

1. **single_widen** — one connected sector; widen the minimal covering arc on every miss
2. **multi_cone** — widen the nearest cone when exterior distance ≤ `split_gap`; otherwise add a point cone; merge overlaps
3. **full_circle** — vacuous full-circle baseline

## Scenarios

| Scenario | Support |
|----------|---------|
| `unimodal` | one 30° sector centered at 0° |
| `bimodal` | two 20° sectors at 0° and 180° |
| `emergence` | only mode A for 200 steps, then A∪B |

Shared angle streams across learners. Containment is scored on the immutable
pre-update commitment; refinement follows.

## Evidence targets

See `protocol.json` → `evidence`. Key points:

- Every contradiction has post-update containment
- Held-out pre-update coverage ≥ 0.95 after burn-in
- Unimodal: one tight cone for both adaptive learners
- Bimodal / emergence: `multi_cone` ends with two cones and substantially less
  union angle than `single_widen`
- First emerging-mode realization is a contradiction and is represented immediately

## Run

```sh
python3 -m stage1f.experiment --output artifacts/stage1f
python3 -m unittest discover -s stage1f -t .
```

Artifacts: `protocol.snapshot.json`, `summary.json`, per-seed run records under
`runs/`.
