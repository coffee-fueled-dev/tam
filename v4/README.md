# TAM v4

`v4` is a theory-first rewrite built around one explicit cycle:

1. Build a `Situation`
2. Parameterize port fibers from that situation
3. Freeze a bind-time claim
4. Let the world return an episode
5. Interpret the episode into latent transition space
6. Measure contradiction between the claim and the realized transition
7. Build the next situation

The main modules are:

- `theory.py`: typed implementation rules and glossary
- `core.py`: theory objects and geometry
- `context.py`: deterministic context selection
- `world.py`: world interfaces and context history
- `ports.py`: port abstractions
- `inference.py`: latent-state and episode interpretation helpers
- `runtime_torch.py`: PyTorch adapter for situation encoding and port fibers
- `training.py`: contradiction-driven training loop
- `worlds.py`: simple reference 2D world
- `benchmark/`: structured corridor benchmark, observation adapter, metrics, and runner
- `tests/`: theory conformance tests

## Run The Example

```bash
python3 -m v4.example
```

## Run The Tests

```bash
python3 -m unittest discover -s v4/tests
```

## Run The Benchmark

```bash
python3 -c "from v4.benchmark import run_benchmark_suite; print(len(run_benchmark_suite()))"
```

## Dump Benchmark JSONL

```bash
python3 -m v4.benchmark.dump_jsonl --output tmp/v4-benchmark.jsonl --include-scorecard
```
