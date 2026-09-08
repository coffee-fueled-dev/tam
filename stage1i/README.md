# Stage 1I — Multimodal Cone Pruning

Stage 1F required **add** for separated modes. Stage 1H required **narrow** for
unimodal volume change. Neither tested mode disappearance: when support is
`A∪B` and then becomes `A`, a single connected window cone bridges empty space,
while a cumulative multi-cone keeps a dead mode.

This package adds **window multi-cone reconstruction**: retain the latest `W`
angles, cluster by circular gaps above `split_gap`, and rebuild one minimal cone
per cluster. Vanished modes leave the window and are pruned.

Success is **minimum union measure at matched future coverage** after a mode
disappears, plus correct cone count.

There is **no** control, safety, continuous-state, port-proliferation, or
TAM-superiority claim.

## Modes

| Mode | Center | Half-width |
|------|--------|------------|
| A | 0° | 10° |
| B | 180° | 10° |

Regimes: `AB` (equal mix), `A`, `B`. Situation label `site` is unchanged.

## Learners

1. **`window_multi`** — rolling history → gap-cluster → multi-cone (candidate)
2. **`window_single`** — Stage 1H single covering cone (bridges)
3. **`cumulative_multi`** — Stage 1F multi-cone over all history
4. **`full_circle`** — vacuous baseline

## Scenarios

- `stationary_AB` (900)
- `hidden_AB_to_A` (`AB=300`, `A=300`) — primary prune probe
- `hidden_AB_to_B` — symmetric diagnostic only

## Evidence targets

See `protocol.json` → `evidence`. Key points:

- Adaptive post-contradiction containment 100%
- Stationary AB: `window_multi` ≈ 2 cones, measure ≤ 45°; `window_single` ≥ 150°
- After `AB→A`: prune to 1 cone ≤ 25° with p90 delay ≤ 80
- Final post block: coverage ≥ 0.95, measure ≤ 22°, count ≤ 1.05
- Cumulative keeps ≥ 1.8 cones / measure ≥ 35° after drop
- Single-window limitation is the AB bridge (measure ≥ 150°), not permanent
  post-drop overclaim (the window also forgets; cumulative retains dead modes)

## Run

```sh
python3 -m stage1i.experiment --output artifacts/stage1i
python3 -m unittest discover -s stage1i -t .
```
