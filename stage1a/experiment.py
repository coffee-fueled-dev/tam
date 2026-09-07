"""Stage 1A engineering gate: context-independent 2D control."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .model import (
    MOVING_PORTS,
    OUTCOMES,
    PORT_DELTAS,
    PORTS,
    Agent,
    Observation,
    brier_score,
)

TARGETS = (
    (6, 0),
    (-6, 0),
    (0, 6),
    (0, -6),
    (4, 4),
    (4, -4),
    (-4, 4),
    (-4, -4),
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(5))
    warmup_interactions: int = 25
    trials: int = 16
    interactions_per_trial: int = 20


class World:
    """Opaque deterministic 2D world with context-independent effects."""

    def __init__(self) -> None:
        self._position = (0, 0)
        self._target = (0, 0)

    @property
    def observation(self) -> Observation:
        return Observation(self._position, self._target)

    def reset(self, target: tuple[int, int]) -> Observation:
        self._position = (0, 0)
        self._target = target
        return self.observation

    def step(self, port: str) -> Observation:
        dx, dy = PORT_DELTAS[port]
        self._position = (self._position[0] + dx, self._position[1] + dy)
        return self.observation


def balanced_targets(seed: int, count: int) -> list[tuple[int, int]]:
    rng = random.Random(seed * 10_000 + 1)
    cycle = list(TARGETS)
    rng.shuffle(cycle)
    targets: list[tuple[int, int]] = []
    while len(targets) < count:
        block = list(cycle)
        rng.shuffle(block)
        targets.extend(block)
    return targets[:count]


def _probabilities(commitment) -> dict[str, float]:
    return {
        f"{dx},{dy}": probability
        for (dx, dy), probability in commitment.probabilities.items()
    }


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
    displacement = (
        after.position[0] - before.position[0],
        after.position[1] - before.position[1],
    )
    score = brier_score(commitment.probabilities, displacement)
    binding = displacement in commitment.cone
    if update:
        agent.observe(port, displacement)
    refined = agent.model.predict(port)
    return {
        **metadata,
        "observation_before": {
            "position": list(before.position),
            "target": list(before.target),
        },
        "observation_after": {
            "position": list(after.position),
            "target": list(after.target),
        },
        "port": port,
        "probabilities": _probabilities(commitment),
        "cone": [list(outcome) for outcome in commitment.cone],
        "trajectory": [list(before.position), list(after.position)],
        "displacement": list(displacement),
        "brier": score,
        "binding_success": binding,
        "goal_achieved": after.position == after.target,
        "cone_added": sorted(
            set(refined.cone) - set(commitment.cone),
            key=lambda outcome: outcome,
        ),
        "cone_removed": sorted(
            set(commitment.cone) - set(refined.cone),
            key=lambda outcome: outcome,
        ),
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
        "mean_path_efficiency": (
            mean(float(trial["path_efficiency"]) for trial in trials if trial["success"])
            if any(trial["success"] for trial in trials)
            else None
        ),
    }


def _summarize_run(
    records: list[dict[str, object]],
    trials: list[dict[str, object]],
    config: Config,
) -> dict[str, object]:
    control = [row for row in records if row["phase"] == "control"]
    total = config.trials * config.interactions_per_trial
    late = [row for row in control if int(row["control_interaction"]) >= total // 2]
    return {
        "prediction": _prediction_metrics(control),
        "prediction_by_port": {
            port: _prediction_metrics(row for row in control if row["port"] == port)
            for port in PORTS
        },
        "target": _target_metrics(trials),
        "late": {
            "prediction": _prediction_metrics(late),
            "moving_ports": _prediction_metrics(
                row for row in late if row["port"] in MOVING_PORTS
            ),
        },
        "targets_reached": sorted({
            tuple(trial["target"]) for trial in trials if trial["success"]
        }),
    }


def _run(seed: int, config: Config) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    selection_rng = random.Random(seed * 10_000 + 3)
    agent = Agent()
    world = World()
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
    targets = balanced_targets(seed, config.trials)
    control_index = 0
    for trial_index, target in enumerate(targets):
        world.reset(target)
        first_hit = None
        steps_taken = 0
        for step in range(config.interactions_per_trial):
            port, commitment = agent.select(world.observation, selection_rng)
            row = _interaction(
                agent,
                world,
                port,
                commitment,
                True,
                {
                    "phase": "control",
                    "interaction": config.warmup_interactions + control_index,
                    "control_interaction": control_index,
                    "trial": trial_index,
                    "trial_step": step + 1,
                },
            )
            records.append(row)
            steps_taken += 1
            if first_hit is None and row["goal_achieved"]:
                first_hit = step + 1
            control_index += 1
        manhattan = abs(target[0]) + abs(target[1])
        trials.append({
            "trial": trial_index,
            "target": list(target),
            "control_start": trial_index * config.interactions_per_trial,
            "success": first_hit is not None,
            "first_hit": first_hit,
            "path_efficiency": (
                manhattan / first_hit if first_hit is not None else None
            ),
            "steps_taken": steps_taken,
        })

    return records, _summarize_run(records, trials, config), agent.model.to_dict()


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _gate(aggregate: dict[str, object], run_summaries: list[dict[str, object]]) -> dict[str, object]:
    late_moving = aggregate["late"]["moving_ports"]
    target = aggregate["target"]
    all_targets = {
        tuple(target)
        for item in run_summaries
        for target in item["metrics"]["targets_reached"]
    }
    return {
        "late_moving_cone_identifies_outcome": {
            "met": (
                late_moving["cone_cardinality"] is not None
                and late_moving["cone_cardinality"] <= 1.0 + 1e-9
                and late_moving["coverage"] is not None
                and late_moving["coverage"] >= 0.9
            ),
            "cone_cardinality": late_moving["cone_cardinality"],
            "coverage": late_moving["coverage"],
            "criterion": "late moving-port cone_cardinality == 1 and coverage >= 0.9",
        },
        "target_success": {
            "met": target["success_rate"] is not None and target["success_rate"] >= 0.9,
            "success_rate": target["success_rate"],
            "criterion": "target success_rate >= 0.9",
        },
        "all_target_types_reachable": {
            "met": set(TARGETS).issubset(all_targets),
            "reached": sorted(all_targets),
            "required": [list(target) for target in TARGETS],
            "criterion": "every cardinal and diagonal target type reached at least once",
        },
    }


def run_experiment(output: str | Path, config: Config = Config()) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, object]] = []
    models: dict[str, object] = {}
    with (output_path / "steps.jsonl").open("w", encoding="utf-8") as steps:
        for seed in config.seeds:
            run_id = f"deterministic:{seed}"
            records, metrics, model = _run(seed, config)
            for record in records:
                steps.write(json.dumps({
                    "run_id": run_id,
                    "seed": seed,
                    **record,
                }, sort_keys=True) + "\n")
            run_summaries.append({
                "run_id": run_id,
                "seed": seed,
                "metrics": metrics,
            })
            models[run_id] = model

    aggregate = {
        "prediction": {
            key: _average(item["metrics"]["prediction"][key] for item in run_summaries)
            for key in ("brier", "coverage", "cone_cardinality")
        },
        "prediction_by_port": {
            port: {
                key: _average(
                    item["metrics"]["prediction_by_port"][port][key]
                    for item in run_summaries
                )
                for key in ("brier", "coverage", "cone_cardinality")
            }
            for port in PORTS
        },
        "target": {
            key: _average(item["metrics"]["target"][key] for item in run_summaries)
            for key in (
                "success_rate",
                "mean_interactions_to_first_hit",
                "mean_path_efficiency",
            )
        },
        "late": {
            "prediction": {
                key: _average(
                    item["metrics"]["late"]["prediction"][key]
                    for item in run_summaries
                )
                for key in ("brier", "coverage", "cone_cardinality")
            },
            "moving_ports": {
                key: _average(
                    item["metrics"]["late"]["moving_ports"][key]
                    for item in run_summaries
                )
                for key in ("brier", "coverage", "cone_cardinality")
            },
        },
    }
    summary = {
        "configuration": {
            **asdict(config),
            "seeds": list(config.seeds),
            "ports": list(PORTS),
            "outcomes": [list(outcome) for outcome in OUTCOMES],
            "targets": [list(target) for target in TARGETS],
            "exploration_probability": 0.1,
            "commitment_mass": 0.9,
            "history_size": 32,
            "purpose": "engineering_gate",
        },
        "seeds": list(config.seeds),
        "per_seed": run_summaries,
        "aggregate": aggregate,
        "gate": _gate(aggregate, run_summaries),
        "references": {
            "full_domain_cone": {
                "coverage": 1.0,
                "cone_cardinality": len(OUTCOMES),
            },
            "constant_stay": {"target_success": 0.0},
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "model.json").write_text(
        json.dumps(models, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/stage1a")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    for name, result in summary["gate"].items():
        print(f"{'PASS' if result['met'] else 'FAIL'}  {name}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
