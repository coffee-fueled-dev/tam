# Stage 1A — 2D engineering gate

This package is a **correctness gate**, not a research claim. It verifies that
the Stage 0 commitment loop works with tuple-valued observations, joint 2D
displacements, Euclidean selection, and JSON serialization.

It does **not** establish scalability, contextual transfer, or TAM-specific
novelty. Context-dependent dynamics and held-out transfer are Stage 1B.

## Protocol

- Unbounded integer grid
- Five opaque ports: `n`, `e`, `s`, `w`, `stay`
- Deterministic context-independent displacements only
- Seeds `0–4`
- 25 cyclic warm-up interactions
- 16 balanced target trials of 20 interactions
- Targets: cardinal `(±6,0)`, `(0,±6)` and diagonal `(±4,±4)`

Gate criteria (correctness, not science):

1. Late moving-port cones identify the deterministic outcome (cardinality 1,
   coverage ≥ 0.9).
2. Target success rate ≥ 0.9.
3. Every cardinal and diagonal target type is reached at least once across seeds.

## Run

```sh
python3 -m stage1a.experiment --output artifacts/stage1a
python3 -m unittest discover -s stage1a -t .
```

Artifacts:

- `steps.jsonl`
- `summary.json`
- `model.json`
