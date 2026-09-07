# Stage 1C — TAM commitment ablation

This package keeps the Stage 1B context-pooled `(terrain, port)` predictor
fixed and ablates commitment/selection rules. It asks whether 90%-mass saved
cones improve held-out calibration or control beyond ordinary uses of the same
predictive distribution.

A pass can support keeping TAM commitment machinery, keeping cones as
measurement only, or rejecting TAM selection novelty on this task. It does
**not** establish neural necessity, proliferation, temporal belief, or
learning-to-learn.

`minimal/`, `stage1a/`, and `stage1b/` remain unchanged runnable baselines.

## Controllers (identical frozen predictor)

1. **tam_cone** — smallest ≥90% cone; select by expected squared distance, then
   smaller cone, then RNG
2. **expected_value** — select by expected squared distance only; post-hoc 90%
   set is reported for measurement, not used in selection
3. **point_map** — commit to the single MAP outcome; probability ties broken by
   displacement order `(-1,0), (0,-1), (0,0), (0,1), (1,0)`, then port order
4. **full_domain** — always commit to all five outcomes; expected-distance
   selection only

References: oracle controller and random policy.

## World

- Unbounded integer grid
- Observable terrain: `plain` if `(x + y) % 2 == 0`, else `rotated`
- Opaque ports `n`, `e`, `s`, `w`, `stay`
- On `plain`, ordinary cardinal deltas; on `rotated`, fixed 90° clockwise remap
- Moving ports succeed with probability **0.8**, otherwise stay; `stay` is
  deterministic
- Train near the origin; eval/nav in a distant band (same cells as Stage 1B)

## Protocol

Preregistered before running:

- Seeds `0–19`
- Train: 40 interleaved forced outcomes per `(terrain, port)` on train cells
- Update one shared context-pooled predictor only during training (`history_size`
  64 so all training samples are retained)
- Freeze the predictor
- Probes: 20 held-out forced outcomes per `(terrain, port)`, scored under every
  commitment rule with no learning
- Navigation: 32 matched held-out start/target trials, 30-step budget, no learning
- Shared target stream; per-controller selection and noise streams
- `commitment_brier` renormalizes predictive mass onto the committed cone

## Evidence targets

1. **Calibration usefulness:** `tam_cone` probe coverage ≥ 0.90 and mean cone
   cardinality ≤ 2.25
2. **Not vacuous:** `tam_cone` mean cone cardinality ≥ 0.25 below `full_domain`
   while coverage ≥ 0.90
3. **Control competence:** `tam_cone` held-out navigation success ≥ 0.80
4. **Advantage over point prediction:** navigation success exceeds `point_map`
   by ≥ 0.10, **or** commitment Brier is better by ≥ 0.05 when success is within
   0.05
5. **Advantage over EV:** navigation success exceeds `expected_value` by ≥ 0.05,
   **or** specificity improves by ≥ 0.25 mean cardinality at matched coverage
   within 0.02; otherwise record **no TAM selection advantage**
6. **Full-domain is not enough:** navigation success exceeds `full_domain` by
   ≥ 0.05, **or** matches within 0.05 while using strictly narrower cones

Decision:

- Keep TAM commitment machinery only if targets 1–3 pass and there is a
  **navigation** gain over MAP (≥0.10) or EV (target 5)
- Keep cones as measurement only if calibration/specificity pass but selection
  matches EV (no TAM selection advantage)
- Commitment-Brier gains over MAP alone support calibrated commitments, not
  selector novelty
- Target 5's specificity arm is inactive by construction: EV reports the same
  post-hoc 90% set as `tam_cone` for measurement; only navigation can show a
  selection advantage
- Do not claim TAM novelty if MAP or EV match `tam_cone` on control and no
  calibrated-commitment gain remains meaningful

Report unmet criteria without retuning.

## Run

```sh
python3 -m stage1c.experiment --output artifacts/stage1c
python3 -m unittest discover -s stage1c -t .
```

Artifacts: `steps.jsonl`, `summary.json`, `model.json`.
