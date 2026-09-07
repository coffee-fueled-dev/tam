# Minimal TAM

This package is a standard-library-only Stage 0 implementation of the
Trajectory-Affordance Model. It keeps the world separate from the actor and
implements the cycle:

`observation → port selection → saved commitment → binding → observation → refinement`

The actor learns a bounded outcome distribution for three opaque ports. A
commitment is the smallest predicted outcome set with at least 90% probability
mass. Selection mostly minimizes expected squared distance to a supplied target,
with 10% random exploration.

## Run

From the repository root:

```sh
python3 -m minimal.experiment --output artifacts/minimal
python3 -m unittest discover -s minimal -t .
```

The experiment uses seeds 0–19 and writes:

- `steps.jsonl`: every warm-up and control interaction, including the immutable
  pre-action prediction, binding result, Brier score, and cone refinement.
- `summary.json`: configuration, per-seed and aggregate measurements, controls,
  and predeclared evidence-target results.
- `model.json`: final bounded histories for every scenario, variant, and seed.

The stationary, hidden-reversal, and noisy scenarios compare continuously
learning (`online`), post-warm-up fixed (`frozen`), and per-decision cleared
(`amnesic`) agents. Target sequences are matched across variants; target,
environment-noise, and selection randomness use separate streams.

Prediction measurements are computed before model updates. Target success is
separate from binding coverage. Interactions to first hit are averaged over
successful trials only, and a target remains fixed after it is first reached.
Late metrics use the final 200 control interactions. Reversal metrics are also
reported in non-overlapping 100-interaction blocks.

The numerical evidence criteria are reported as pass or fail without changing
the declared protocol. They test this minimal instance; they do not establish
that TAM is better than conventional model-based control.
