# Stage 1K — Overlapping / Near-Merge Modes

Stages 1I–1J handled well-separated modes. Clustering still uses a fixed
`split_gap` (primary `30°`). This package **characterizes** when that rule:

- keeps two tight cones (well-separated)
- correctly merges overlapping supports
- avoids false-splitting a wide unimodal arc
- becomes unstable near the edge-gap boundary

There is **no** adaptive-split, control, safety, or TAM-superiority claim.

## Edge gap

For two modes with half-width `10°`:

`edge_gap = center_separation − 20°`

| Scenario | Centers | Edge gap | Expectation (gap=30) |
|----------|---------|----------|----------------------|
| `separate70` | 0°, 90° | 70° | 2 cones, measure ≤ 45° |
| `boundary30` | 0°, 50° | 30° | diagnostic / unstable |
| `overlap5` | 0°, 25° | 5° | 1 cone, measure ≤ 55° |
| `wide_unimodal` | 0°±40° | n/a | 1 cone, measure ≤ 85° |

## Learners

1. **`hysteresis_gap30`** — Stage 1J sticky hysteresis, `split_gap=30`, `W=48`, `T=64`
2. **`window_single`** — one covering cone
3. **`full_circle`** — vacuous baseline
4. Diagnostics: `hysteresis_gap15`, `hysteresis_gap45`

## Evidence targets

See `protocol.json` → `evidence`. `boundary30` is diagnostic only.

## Run

```sh
python3 -m stage1k.experiment --output artifacts/stage1k
python3 -m unittest discover -s stage1k -t .
```
