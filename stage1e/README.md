# Stage 1E — Commitment Ledger and Drift Audit

This package treats cones as:

1. explicit pre-action commitments
2. empirically coverage-audited predictive summaries
3. binding-failure signals
4. replayable accountability records

It removes action selection. Ports follow a forced alternating schedule. There is
**no** control, safety, reward, or TAM-superiority claim.

Call cones **empirically coverage-audited predictive summaries**, not inherently
calibrated explanations.

Earlier stages remain unchanged runnable baselines.

## Protocol snapshot

Immutable constants live in `protocol.json` and are snapshotted into
`artifacts/stage1e/protocol.snapshot.json` before each run.

- Outcomes `{-1, 0, +1}`; ports `a`, `b`
- Regime A: `a={+1:.80, 0:.15, -1:.05}`; `b` mirrors signs
- Regime B: reverse each port’s modal signs
- Regime D: inside-cone drift `a={+1:.55, 0:.40, -1:.05}`; `b` mirrors
- Seeds `0–99`; calibrate detectors on `0–49`; evaluate evidence on `50–99`
- Scenarios: `stationary_A` (1500), `hidden_ABA` (500/500/500), `hidden_ADA`
- Burn-in: first 200 A interactions
- Primary learner: bounded window 64, α=0.5, smallest ≥90% cone

## Variants

- `window64` — primary
- `cumulative` — no forgetting
- `frozen200` — freeze after burn-in
- `full_domain` — vacuous cone over all outcomes
- `oracle` — evaluator-only probabilities for Brier regret (not an actor variant)

## Detectors

Offline on saved pre-action records only:

1. `cone_miss` — failures in last 10 steps
2. `nll` — sum of `-log P(observed)` over 10 steps
3. `raw_two_window` — TV distance between previous/latest 20-step histograms

Thresholds chosen on calibration seeds for ≤0.1 false alarms / 1000 stationary
interactions, then frozen for evaluation.

## Evidence targets

See `protocol.json` → `evidence`. Key points:

- Stationary coverage ≥0.90 with mean cardinality ≤2.10 and set-calibration gap ≤0.03
- Reversal detection by `cone_miss` within 20 steps in ≥95% of eval seeds
- Inside-cone drift: `cone_miss` detection ≤20% within 100 steps, while NLL or
  raw detector ≥90% — a documented **limitation**, not an implementation failure
- Hash-chain replay and tamper rejection
- Hindsight reconstruction from final A2 model disagrees with saved B labels on
  ≥30% of late-B interactions

## Seal trust boundary

`seal.json` / `seal.txt` prove consistency only relative to an externally
retained seal. A party that rewrites every artifact can forge a new coherent
chain. Keep a copy of `seal.txt` outside `artifacts/`.

## Run

```sh
python3 -m stage1e.experiment --output artifacts/stage1e
python3 -m stage1e.replay --ledger artifacts/stage1e/runs/stationary_A/50.jsonl
python3 -m unittest discover -s stage1e -t .
```

Artifacts: `protocol.snapshot.json`, `runs/<scenario>/<seed>.jsonl`,
`final_states.json`, `summary.json`, `seal.json`, `seal.txt`.
