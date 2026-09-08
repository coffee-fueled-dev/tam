# Stage 1G — Situational Cone Volumes

Stage 1F showed that widen/add always restores containment. Indefinite widening
is still a failure mode of specificity: wide cones contain reality while
predicting nothing specific.

This package adds one mechanism: **situation-conditioned cone volume**. The same
port claims different geometric `ConeSet` measures in different observable
situations. There is **no** control, safety, narrowing, or TAM-superiority claim.

## Situations and supports

Observable label `s ∈ {tight, wide, shifted}`:

| Situation | Center | Half-width | Measure |
|-----------|--------|------------|---------|
| `tight` | 0° | 10° | 20° |
| `wide` | 0° | 45° | 90° |
| `shifted` | 180° | 15° | 30° |

## Learners

1. **`unconditional_multi`** — one global multi-cone over all situations
2. **`situation_multi`** — independent multi-cone per observed situation
3. **`unconditional_single`** — one global connected widen
4. **`full_circle`** — vacuous baseline

Train: interleaved situations. Eval: held-out per situation with learning frozen.
Diagnostic (not a pass gate): `tight` later emits `wide` support without
relabeling, to document no-narrowing over-width.

## Evidence targets

See `protocol.json` → `evidence`. Key points:

- Every contradiction has post-update containment
- `situation_multi` held-out coverage ≥ 0.95 in every situation
- In `tight`, `situation_multi` stays ≤ 25° while `unconditional_multi` is
  contaminated (≥ 80°) or the volume gap is ≥ 50°
- In `shifted`, `situation_multi` keeps a local cone without inheriting `wide`
- `full_circle` remains 360°

## Run

```sh
python3 -m stage1g.experiment --output artifacts/stage1g
python3 -m unittest discover -s stage1g -t .
```
