# Stage 1J — Mode Churn / Hysteresis

Stage 1I’s `window_multi` prunes as soon as a mode leaves the rolling window.
That is correct for sustained `AB→A`, but under rapid on/off flicker it chatters
(repeated add/prune) and can under-cover a mode that briefly returns.

This package adds **sticky hysteresis**: add a separated mode immediately when
the window supports it; prune a sticky cone only after `T` consecutive steps
with no matching window cluster.

- Window `W=48`, grace `T=64`, `split_gap=30°`
  (fast A-gaps of 56 steps: plain window loses B by gap end; hysteresis retains)
- Modes A (`0°±10`) and B (`180°±10`); situation label `site` unchanged

There is **no** control, safety, overlapping-mode, continuous-state, or
TAM-superiority claim.

## Learners

1. **`window_multi`** — Stage 1I rebuild-from-window (chatter baseline)
2. **`hysteresis_multi`** — sticky cones; immediate add; prune after miss streak ≥ `T`
3. **`cumulative_multi`** — never prunes
4. **`full_circle`** — vacuous baseline

## Scenarios

| Scenario | Schedule |
|----------|----------|
| `stationary_AB` | AB × 600 |
| `sustained_drop` | AB 200 → A 200 |
| `fast_churn` | AB 128, then 12× (A 56 / AB 24) |
| `slow_churn` | AB 128, then 6× (A 96 / AB 96) — diagnostic |

## Evidence targets

See `protocol.json` → `evidence`. Key points:

- Adaptive post-contradiction containment 100%
- Stationary AB: adaptive learners coverage ≥ 0.95, ~2 cones, measure ≤ 45°
- Fast churn: hysteresis end-of-gap count ≥ 1.8, B coverage ≥ 0.90 on AB
  on-blocks, prune rate ≤ half of `window_multi`
- `window_multi` end-of-gap count ≤ 1.1 (drops B)
- Sustained drop: hysteresis ends ≤ 30° / ≤ 1.2 cones; p90 delay ≤ 130
- Cumulative stays ≥ 1.8 cones / ≥ 35° after sustained drop

## Run

```sh
python3 -m stage1j.experiment --output artifacts/stage1j
python3 -m unittest discover -s stage1j -t .
```
