# Stage 1L — Continuous Edge-Gap Drift

Stage 1K showed fixed `split_gap=30°` works for **stationary** separate and
overlap regimes, with a known boundary limitation. Real-ish tasks move the
separation through that space.

This package keeps Stage 1J sticky hysteresis (`W=48`, `T=64`, `split_gap=30`)
and adds one fixed mechanism: **situation-binned stores** keyed by a coarse
observable feature of a continuous situation that sets the second mode center.

There is **no** adaptive-split, neural embedding, control, safety, or
TAM-superiority claim.

## Continuous situation

Mode A is fixed at `0°±10°`. Mode B center is set by edge gap
`g = center_B − 20°` with `g ∈ [5°, 70°]` (`center_B ∈ [25°, 90°]`).

Observable situation feature: the continuous edge gap (or its plateau label).
Binning:

| Bin | Edge gap |
|-----|----------|
| `separate` | `g ≥ 50°` |
| `boundary` | `15° < g < 50°` |
| `overlap` | `g ≤ 15°` |

## Learners

1. **`binned_hysteresis`** — independent sticky hysteresis per regime bin
2. **`pooled_hysteresis`** — one sticky store across all situations
3. **`pooled_single`** — one covering cone
4. **`full_circle`** — vacuous baseline

## Scenarios

- **`plateau_tour`**: hold separate → boundary → overlap; evaluate last
  `eval_tail` steps for settled binned claims, and the first
  `overlap_transition` steps for pooled over-claim after the switch.
- **`slow_sweep`**: linearly sweep edge gap `70° → 5°`; score windows by
  instantaneous bin; compare pooled vs binned excess.

## Evidence targets

See `protocol.json` → `evidence`. Boundary plateau/window is diagnostic only.

## Run

```sh
python3 -m stage1l.experiment --output artifacts/stage1l
python3 -m unittest discover -s stage1l -t .
```
