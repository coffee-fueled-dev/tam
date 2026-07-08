# Atlas Core

`atlas/` is a small standalone package for testing the chart-first atlas
mechanics described in `formulation/architecture.md`.

This package intentionally implements only the most fundamental pieces:

- `SituationEncoder`
- `AtlasStore`
- `ChartRetriever`
- `ChartProjector`
- a minimal runtime loop with spawn-on-miss

It does not yet implement:

- full LLM/agent integration
- mature split/merge semantics
- benchmark integration with `v4`
- production vector search backends

## Main Idea

Charts are the persistent objects.

- the encoder maps raw context into `situation_latent` and `query_key`
- retrieval returns the top-k chart candidates
- projection scores how well each chart supports the current situation
- one bindable `PortView` is derived per chart
- if no candidate fits well enough, the runtime spawns a new chart

## Files

- `core.py`: stable chart and situation dataclasses
- `encoder.py`: reference and identity encoders
- `store.py`: simple persistent chart registry
- `retrieval.py`: top-k retrieval over chart keys
- `projector.py`: chart-local projection and contradiction
- `runtime.py`: minimal self-organizing loop
- `worlds.py`: tiny synthetic recurring regimes
- `benchmark/`: stress-test worlds, metrics, runner, and reporting
- `example.py`: end-to-end demo

## Run The Example

```bash
python3 -m atlas.example
```

## Run The Tests

```bash
python3 -m unittest discover -s atlas/tests
```

## Run The Benchmark

```bash
python3 -m atlas.benchmark.report
```

## Dump Benchmark JSONL

```bash
python3 -m atlas.benchmark.report --output tmp/atlas-benchmark.jsonl
```
