# Stage 1H — Situational Cone Narrowing

Stage 1G showed that observable situations keep cone volumes specific across
contexts, and that fixed-label `tight → wide` expansion works. It did **not**
test contraction: cumulative cones cannot shrink when the same situation returns
to a tight regime.

This package tests whether a **rolling-window** geometric cone can widen under
hidden support expansion and later narrow under return to tight support, while a
cumulative cone remains permanently wide.

Success is **minimum geometric volume at matched future coverage**, not
permanent containment of every historical trajectory.

There is **no** control, safety, multimodal contraction, continuous-state, or
TAM-superiority claim.

## World

- One forced port; one unchanged observable situation label `site`
- Hidden regime support:
  - **A**: center `0°`, half-width `10°` (measure 20°)
  - **B**: center `0°`, half-width `45°` (measure 90°)
- Scenarios: `stationary_A`, `stationary_B`, `hidden_ABA` (`A1=300`, `B=300`, `A2=300`)
- Seeds `0–99`; shared angle streams across learners

## Learners

1. **`window64`** — retain latest 64 angles; rebuild minimal covering cone
2. **`cumulative`** — all observations; cannot narrow
3. **`frozen_a1`** — learn in A1, freeze at first transition
4. **`full_circle`** — vacuous 360° baseline

Windows 32 and 128 are sensitivity diagnostics only.

## Evidence targets

See `protocol.json` → `evidence`. Key points:

- Adaptive post-contradiction containment is 100%
- Stationary: `window64` exact and empirical coverage ≥ 0.95 with zero excess
- Expansion: in B, rolling exact coverage ≥ 0.95 with p90 delay ≤ 100
- Contraction: in A2, rolling exact coverage ≥ 0.95 and measure ≤ 25° with p90 delay ≤ 80
- Final A2: empirical coverage ≥ 0.95 and mean measure ≤ 22°
- Cumulative remains wide in final A2 (measure ≥ 85°, excess ≥ 60°)
- Frozen fails in B (exact coverage ≤ 0.35) while A2 exact coverage ≥ 0.95

## Run

```sh
python3 -m stage1h.experiment --output artifacts/stage1h
python3 -m unittest discover -s stage1h -t .
```
