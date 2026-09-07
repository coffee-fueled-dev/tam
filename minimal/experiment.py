"""Reproducible Stage 0 experiments for the minimal TAM."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .model import Agent, Observation, OutcomeModel, PORTS, brier_score

SCENARIOS = ("stationary", "reversal", "noisy")
VARIANTS = ("online", "frozen", "amnesic")


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(20))
    warmup_interactions: int = 36
    trials: int = 60
    interactions_per_trial: int = 20
    reversal_interaction: int = 600


class World:
    """A 1D world that reveals observations, not its dynamics."""

    def __init__(self, scenario: str, noise_rng: random.Random) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        self._scenario = scenario
        self._noise_rng = noise_rng
        self._position = 0
        self._target = 0
        self._reversed = False

    @property
    def observation(self) -> Observation:
        return Observation(self._position, self._target)

    def reset(self, target: int) -> Observation:
        self._position = 0
        self._target = target
        return self.observation

    def reverse(self) -> None:
        if self._scenario == "reversal":
            self._reversed = True

    def step(self, port: str) -> Observation:
        direction = {"a": 1, "b": -1, "c": 0}[port]
        if self._reversed:
            direction = -direction
        if self._scenario == "noisy" and direction and self._noise_rng.random() >= 0.8:
            direction = 0
        self._position += direction
        return self.observation


def _stream_seed(seed: int, scenario: str, stream: int) -> int:
    return seed * 10_000 + SCENARIOS.index(scenario) * 100 + stream


def _probabilities(commitment) -> dict[str, float]:
    return {str(outcome): probability for outcome, probability in commitment.probabilities.items()}


def _interaction(
    agent: Agent,
    world: World,
    port: str,
    commitment,
    update: bool,
    metadata: dict[str, object],
) -> dict[str, object]:
    before = world.observation
    after = world.step(port)
    displacement = after.position - before.position
    score = brier_score(commitment.probabilities, displacement)
    binding = displacement in commitment.cone
    if update:
        agent.observe(port, displacement)
    refined = agent.model.predict(port)
    return {
        **metadata,
        "observation_before": asdict(before),
        "observation_after": asdict(after),
        "port": port,
        "probabilities": _probabilities(commitment),
        "cone": list(commitment.cone),
        "trajectory": [before.position, after.position],
        "displacement": displacement,
        "brier": score,
        "binding_success": binding,
        "goal_achieved": after.position == after.target,
        "cone_added": sorted(set(refined.cone) - set(commitment.cone)),
        "cone_removed": sorted(set(commitment.cone) - set(refined.cone)),
    }


def _prediction_metrics(records: Iterable[dict[str, object]]) -> dict[str, float | None]:
    rows = list(records)
    if not rows:
        return {"brier": None, "coverage": None, "cone_cardinality": None}
    return {
        "brier": mean(float(row["brier"]) for row in rows),
        "coverage": mean(bool(row["binding_success"]) for row in rows),
        "cone_cardinality": mean(len(row["cone"]) for row in rows),
    }


def _target_metrics(trials: list[dict[str, object]]) -> dict[str, float | None]:
    hits = [
        int(trial["first_hit"])
        for trial in trials
        if trial["first_hit"] is not None
    ]
    return {
        "success_rate": mean(bool(trial["success"]) for trial in trials) if trials else None,
        "mean_interactions_to_first_hit": mean(hits) if hits else None,
    }


def _summarize_run(
    records: list[dict[str, object]],
    trials: list[dict[str, object]],
    config: Config,
    scenario: str,
) -> dict[str, object]:
    control = [row for row in records if row["phase"] == "control"]
    total = config.trials * config.interactions_per_trial
    late = [row for row in control if int(row["control_interaction"]) >= total - 200]
    late_trials = [
        trial for trial in trials
        if int(trial["control_start"]) >= total - 200
    ]
    summary: dict[str, object] = {
        "prediction": _prediction_metrics(control),
        "prediction_by_port": {
            port: _prediction_metrics(row for row in control if row["port"] == port)
            for port in PORTS
        },
        "target": _target_metrics(trials),
        "late_last_200": {
            "prediction": _prediction_metrics(late),
            "moving_ports": _prediction_metrics(
                row for row in late if row["port"] in ("a", "b")
            ),
            "target": _target_metrics(late_trials),
        },
    }
    if scenario == "reversal":
        blocks = []
        for start in range(0, total, 100):
            block_rows = [
                row for row in control
                if start <= int(row["control_interaction"]) < start + 100
            ]
            block_trials = [
                trial for trial in trials
                if start <= int(trial["control_start"]) < start + 100
            ]
            blocks.append({
                "start": start,
                "end": min(start + 100, total),
                "prediction": _prediction_metrics(block_rows),
                "target": _target_metrics(block_trials),
            })
        summary["reversal_blocks"] = blocks
    return summary


def _run(
    seed: int,
    scenario: str,
    variant: str,
    config: Config,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    target_rng = random.Random(seed * 10_000 + 1)
    noise_rng = random.Random(_stream_seed(seed, scenario, 2))
    selection_rng = random.Random(_stream_seed(seed, scenario, 3))
    agent = Agent()
    world = World(scenario, noise_rng)
    records: list[dict[str, object]] = []

    for index in range(config.warmup_interactions):
        port = PORTS[index % len(PORTS)]
        commitment = agent.model.predict(port)
        records.append(_interaction(
            agent,
            world,
            port,
            commitment,
            True,
            {"phase": "warmup", "interaction": index},
        ))

    trials: list[dict[str, object]] = []
    control_index = 0
    for trial_index in range(config.trials):
        target = target_rng.choice((-5, 5))
        world.reset(target)
        first_hit = None
        for step in range(config.interactions_per_trial):
            if scenario == "reversal" and control_index == config.reversal_interaction:
                world.reverse()
            if variant == "amnesic":
                agent.model.clear()
            port, commitment = agent.select(world.observation, selection_rng)
            update = variant != "frozen"
            row = _interaction(
                agent,
                world,
                port,
                commitment,
                update,
                {
                    "phase": "control",
                    "interaction": config.warmup_interactions + control_index,
                    "control_interaction": control_index,
                    "trial": trial_index,
                    "trial_step": step + 1,
                },
            )
            records.append(row)
            if first_hit is None and row["goal_achieved"]:
                first_hit = step + 1
            control_index += 1
        trials.append({
            "trial": trial_index,
            "target": target,
            "control_start": trial_index * config.interactions_per_trial,
            "success": first_hit is not None,
            "first_hit": first_hit,
        })

    return records, _summarize_run(records, trials, config, scenario), agent.model.to_dict()


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _average_metrics(items: list[dict[str, float | None]]) -> dict[str, float | None]:
    return {
        key: _average(item[key] for item in items)
        for key in items[0]
    }


def _aggregate(run_summaries: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for scenario in SCENARIOS:
        aggregate[scenario] = {}
        for variant in VARIANTS:
            runs = [
                item["metrics"] for item in run_summaries
                if item["scenario"] == scenario and item["variant"] == variant
            ]
            group = {
                "prediction": _average_metrics([run["prediction"] for run in runs]),
                "prediction_by_port": {
                    port: _average_metrics([
                        run["prediction_by_port"][port] for run in runs
                    ])
                    for port in PORTS
                },
                "target": _average_metrics([run["target"] for run in runs]),
                "late_last_200": {
                    "prediction": _average_metrics([
                        run["late_last_200"]["prediction"] for run in runs
                    ]),
                    "moving_ports": _average_metrics([
                        run["late_last_200"]["moving_ports"] for run in runs
                    ]),
                    "target": _average_metrics([
                        run["late_last_200"]["target"] for run in runs
                    ]),
                },
            }
            if scenario == "reversal":
                group["reversal_blocks"] = [
                    {
                        "start": block_index * 100,
                        "end": (block_index + 1) * 100,
                        "prediction": _average_metrics([
                            run["reversal_blocks"][block_index]["prediction"]
                            for run in runs
                        ]),
                        "target": _average_metrics([
                            run["reversal_blocks"][block_index]["target"]
                            for run in runs
                        ]),
                    }
                    for block_index in range(len(runs[0]["reversal_blocks"]))
                ]
            aggregate[scenario][variant] = group
    return aggregate


def _evidence(aggregate: dict[str, object]) -> dict[str, object]:
    stationary = aggregate["stationary"]
    reversal = aggregate["reversal"]
    noisy = aggregate["noisy"]
    stationary_cone = stationary["online"]["late_last_200"]["moving_ports"]["cone_cardinality"]
    stationary_coverage = stationary["online"]["late_last_200"]["moving_ports"]["coverage"]
    online_advantage = (
        stationary["online"]["target"]["success_rate"]
        - stationary["amnesic"]["target"]["success_rate"]
    )
    reversal_advantage = (
        reversal["online"]["late_last_200"]["target"]["success_rate"]
        - reversal["frozen"]["late_last_200"]["target"]["success_rate"]
    )
    noisy_cone = noisy["online"]["late_last_200"]["moving_ports"]["cone_cardinality"]
    noisy_coverage = noisy["online"]["late_last_200"]["moving_ports"]["coverage"]
    return {
        "stationary_specificity_and_coverage": {
            "met": stationary_cone <= 1.5 and stationary_coverage >= 0.9,
            "cone_cardinality": stationary_cone,
            "coverage": stationary_coverage,
            "criterion": "cone_cardinality <= 1.5 and coverage >= 0.9",
        },
        "online_over_amnesic_target_success": {
            "met": online_advantage >= 0.2,
            "difference": online_advantage,
            "criterion": "stationary target-success difference >= 0.2",
        },
        "online_reversal_recovery_over_frozen": {
            "met": reversal_advantage >= 0.2,
            "difference": reversal_advantage,
            "criterion": "final-200 target-success difference >= 0.2",
        },
        "noisy_cones_broader_with_coverage": {
            "met": noisy_cone > stationary_cone and noisy_coverage >= 0.9,
            "noisy_cone_cardinality": noisy_cone,
            "deterministic_cone_cardinality": stationary_cone,
            "noisy_coverage": noisy_coverage,
            "criterion": "noisy cardinality > deterministic and noisy coverage >= 0.9",
        },
    }


def run_experiment(output: str | Path, config: Config = Config()) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, object]] = []
    models: dict[str, object] = {}
    with (output_path / "steps.jsonl").open("w", encoding="utf-8") as steps:
        for scenario in SCENARIOS:
            for variant in VARIANTS:
                for seed in config.seeds:
                    run_id = f"{scenario}:{variant}:{seed}"
                    records, metrics, model = _run(seed, scenario, variant, config)
                    for record in records:
                        steps.write(json.dumps({
                            "run_id": run_id,
                            "seed": seed,
                            "scenario": scenario,
                            "variant": variant,
                            **record,
                        }, sort_keys=True) + "\n")
                    run_summaries.append({
                        "run_id": run_id,
                        "seed": seed,
                        "scenario": scenario,
                        "variant": variant,
                        "metrics": metrics,
                    })
                    models[run_id] = model

    aggregate = _aggregate(run_summaries)
    summary = {
        "configuration": {
            **asdict(config),
            "seeds": list(config.seeds),
            "ports": list(PORTS),
            "exploration_probability": 0.1,
            "commitment_mass": 0.9,
            "history_size": 32,
        },
        "seeds": list(config.seeds),
        "per_seed": run_summaries,
        "aggregate": aggregate,
        "references": {
            "full_domain_cone": {"coverage": 1.0, "cone_cardinality": 3},
            "constant_stay": {"target_success": 0.0},
        },
        "evidence_targets": _evidence(aggregate),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "model.json").write_text(
        json.dumps(models, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/minimal")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    for name, result in summary["evidence_targets"].items():
        print(f"{'PASS' if result['met'] else 'FAIL'}  {name}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
