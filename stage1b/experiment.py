"""Stage 1B: held-out contextual transfer under observable terrain."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Iterable

from .model import (
    OUTCOMES,
    PORT_DELTAS,
    PORTS,
    PREDICTORS,
    ROTATED_DELTAS,
    TERRAINS,
    Agent,
    Commitment,
    Observation,
    brier_score,
    make_predictor,
)

# Preregistered disjoint coordinate split. Terrain is an observable formula.
# Train stays near the origin; eval/nav use a distant band so paths do not
# re-enter memorized train coordinates.
TRAIN_CELLS: dict[str, tuple[tuple[int, int], ...]] = {
    "plain": (
        (0, 0), (1, 1), (2, 0), (0, 2), (-1, -1),
        (-2, 0), (0, -2), (1, -1), (-1, 1), (2, 2),
    ),
    "rotated": (
        (1, 0), (0, 1), (-1, 0), (0, -1), (2, 1),
        (1, 2), (-2, -1), (-1, -2), (3, -2), (-2, 3),
    ),
}
EVAL_CELLS: dict[str, tuple[tuple[int, int], ...]] = {
    "plain": (
        (12, 12), (12, 14), (14, 12), (14, 14),
        (16, 10), (10, 16), (16, 12), (12, 16),
    ),
    "rotated": (
        (12, 13), (13, 12), (14, 13), (13, 14),
        (16, 11), (11, 16), (15, 12), (12, 15),
    ),
}

NAV_STARTS = tuple(
    cell for terrain in TERRAINS for cell in EVAL_CELLS[terrain]
)
NAV_TARGETS = (
    (18, 18), (18, 20), (20, 18), (20, 20),
    (22, 16), (16, 22), (22, 18), (18, 22),
    (18, 19), (19, 18), (20, 19), (19, 20),
    (22, 17), (17, 22), (21, 18), (18, 21),
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(20))
    train_outcomes_per_pair: int = 20
    probe_outcomes_per_pair: int = 20
    navigation_trials: int = 32
    interactions_per_trial: int = 24
    history_size: int = 32


def terrain_at(position: tuple[int, int]) -> str:
    """Observable terrain: plain when x+y is even, rotated when odd."""
    return "plain" if (position[0] + position[1]) % 2 == 0 else "rotated"


def _validate_split() -> None:
    train = {cell for cells in TRAIN_CELLS.values() for cell in cells}
    eval_cells = {cell for cells in EVAL_CELLS.values() for cell in cells}
    if train & eval_cells:
        raise ValueError("train and eval cells must be disjoint")
    for terrain, cells in TRAIN_CELLS.items():
        for cell in cells:
            if terrain_at(cell) != terrain:
                raise ValueError(f"train cell {cell} is not {terrain}")
    for terrain, cells in EVAL_CELLS.items():
        for cell in cells:
            if terrain_at(cell) != terrain:
                raise ValueError(f"eval cell {cell} is not {terrain}")


_validate_split()


class World:
    """Opaque unbounded 2D world with observable terrain-dependent effects."""

    def __init__(self) -> None:
        self._position = (0, 0)
        self._target = (0, 0)

    @property
    def observation(self) -> Observation:
        return Observation(self._position, self._target, terrain_at(self._position))

    def teleport(
        self,
        position: tuple[int, int],
        target: tuple[int, int],
    ) -> Observation:
        self._position = position
        self._target = target
        return self.observation

    def step(self, port: str) -> Observation:
        deltas = (
            PORT_DELTAS if terrain_at(self._position) == "plain"
            else ROTATED_DELTAS
        )
        dx, dy = deltas[port]
        self._position = (self._position[0] + dx, self._position[1] + dy)
        return self.observation


def _probabilities(commitment) -> dict[str, float]:
    return {
        f"{dx},{dy}": probability
        for (dx, dy), probability in commitment.probabilities.items()
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
    efficiencies = [
        float(trial["path_efficiency"])
        for trial in trials
        if trial["path_efficiency"] is not None
    ]
    return {
        "success_rate": (
            mean(bool(trial["success"]) for trial in trials) if trials else None
        ),
        "mean_interactions_to_first_hit": mean(hits) if hits else None,
        "mean_path_efficiency": mean(efficiencies) if efficiencies else None,
        "censored_rate": (
            mean(trial["first_hit"] is None for trial in trials) if trials else None
        ),
    }


def _record_step(
    observation_before: Observation,
    observation_after: Observation,
    port: str,
    commitment,
    refined_commitment,
    metadata: dict[str, object],
) -> dict[str, object]:
    displacement = (
        observation_after.position[0] - observation_before.position[0],
        observation_after.position[1] - observation_before.position[1],
    )
    return {
        **metadata,
        "observation_before": {
            "position": list(observation_before.position),
            "target": list(observation_before.target),
            "terrain": observation_before.terrain,
        },
        "observation_after": {
            "position": list(observation_after.position),
            "target": list(observation_after.target),
            "terrain": observation_after.terrain,
        },
        "port": port,
        "probabilities": _probabilities(commitment),
        "cone": [list(outcome) for outcome in commitment.cone],
        "displacement": list(displacement),
        "brier": brier_score(commitment.probabilities, displacement),
        "binding_success": displacement in commitment.cone,
        "goal_achieved": observation_after.position == observation_after.target,
        "cone_added": [
            list(outcome)
            for outcome in sorted(set(refined_commitment.cone) - set(commitment.cone))
        ],
        "cone_removed": [
            list(outcome)
            for outcome in sorted(set(commitment.cone) - set(refined_commitment.cone))
        ],
    }


def _train(
    agents: dict[str, Agent],
    world: World,
    config: Config,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed * 10_000 + 11)
    records: list[dict[str, object]] = []
    # Interleave terrains inside each port so bounded unconditional histories
    # retain a balanced mixture rather than a last-terrain-only window.
    for port in PORTS:
        plain_cells = list(TRAIN_CELLS["plain"])
        rotated_cells = list(TRAIN_CELLS["rotated"])
        rng.shuffle(plain_cells)
        rng.shuffle(rotated_cells)
        for index in range(config.train_outcomes_per_pair):
            for terrain, cells in (
                ("plain", plain_cells),
                ("rotated", rotated_cells),
            ):
                position = cells[index % len(cells)]
                before = world.teleport(position, position)
                commitments = {
                    kind: agent.predictor.predict(before, port)
                    for kind, agent in agents.items()
                }
                after = world.step(port)
                displacement = (
                    after.position[0] - before.position[0],
                    after.position[1] - before.position[1],
                )
                for kind, agent in agents.items():
                    agent.observe(before, port, displacement)
                    refined = agent.predictor.predict(before, port)
                    records.append(_record_step(
                        before,
                        after,
                        port,
                        commitments[kind],
                        refined,
                        {
                            "phase": "train",
                            "predictor": kind,
                            "terrain": terrain,
                            "pair_index": index,
                        },
                    ))
    return records


def _probes(
    agents: dict[str, Agent],
    world: World,
    config: Config,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed * 10_000 + 13)
    records: list[dict[str, object]] = []
    for terrain in TERRAINS:
        cells = list(EVAL_CELLS[terrain])
        for port in PORTS:
            for index in range(config.probe_outcomes_per_pair):
                position = cells[rng.randrange(len(cells))]
                before = world.teleport(position, position)
                after = world.step(port)
                for kind, agent in agents.items():
                    commitment = agent.predictor.predict(before, port)
                    records.append(_record_step(
                        before,
                        after,
                        port,
                        commitment,
                        commitment,
                        {
                            "phase": "probe",
                            "predictor": kind,
                            "terrain": terrain,
                            "pair_index": index,
                        },
                    ))
    return records


def _navigation_pairs(
    seed: int,
    count: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rng = random.Random(seed * 10_000 + 17)
    starts = list(NAV_STARTS)
    targets = list(NAV_TARGETS)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(pairs) < count:
        start = starts[rng.randrange(len(starts))]
        target = targets[rng.randrange(len(targets))]
        if start != target:
            pairs.append((start, target))
    return pairs


def _oracle_port(observation: Observation) -> str:
    deltas = PORT_DELTAS if observation.terrain == "plain" else ROTATED_DELTAS
    best = "stay"
    best_loss = None
    for port in PORTS:
        dx, dy = deltas[port]
        nxt = (observation.position[0] + dx, observation.position[1] + dy)
        loss = (
            (nxt[0] - observation.target[0]) ** 2
            + (nxt[1] - observation.target[1]) ** 2
        )
        if best_loss is None or loss < best_loss or (
            loss == best_loss and port < best
        ):
            best = port
            best_loss = loss
    return best


def _full_domain_commitment() -> Commitment:
    return Commitment(
        MappingProxyType({outcome: 1 / len(OUTCOMES) for outcome in OUTCOMES}),
        OUTCOMES,
    )


def _navigate_one(
    kind: str,
    agent: Agent | None,
    world: World,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    config: Config,
    seed: int,
    stream: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selection_rng = random.Random(seed * 10_000 + stream)
    records: list[dict[str, object]] = []
    trials: list[dict[str, object]] = []
    for trial_index, (start, target) in enumerate(pairs):
        world.teleport(start, target)
        first_hit = None
        path_length = abs(start[0] - target[0]) + abs(start[1] - target[1])
        for step in range(config.interactions_per_trial):
            observation = world.observation
            if kind == "oracle":
                port = _oracle_port(observation)
                commitment = _full_domain_commitment()
            elif kind == "random":
                port = selection_rng.choice(PORTS)
                commitment = _full_domain_commitment()
            else:
                assert agent is not None
                port, commitment = agent.select(observation, selection_rng)
            after = world.step(port)
            records.append(_record_step(
                observation,
                after,
                port,
                commitment,
                commitment,
                {
                    "phase": "navigation",
                    "predictor": kind,
                    "trial": trial_index,
                    "trial_step": step + 1,
                },
            ))
            if first_hit is None and after.position == after.target:
                first_hit = step + 1
                break
        trials.append({
            "trial": trial_index,
            "start": list(start),
            "target": list(target),
            "success": first_hit is not None,
            "first_hit": first_hit,
            "path_efficiency": (
                path_length / first_hit if first_hit is not None and first_hit > 0
                else None
            ),
        })
    return records, trials


def _navigate(
    agents: dict[str, Agent],
    world: World,
    config: Config,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    pairs = _navigation_pairs(seed, config.navigation_trials)
    records: list[dict[str, object]] = []
    trials: dict[str, list[dict[str, object]]] = {}
    controllers: list[tuple[str, Agent | None, int]] = [
        (kind, agents[kind], 19 + index)
        for index, kind in enumerate(PREDICTORS)
    ]
    controllers.append(("oracle", None, 50))
    controllers.append(("random", None, 51))
    for kind, agent, stream in controllers:
        kind_records, kind_trials = _navigate_one(
            kind, agent, world, pairs, config, seed, stream
        )
        records.extend(kind_records)
        trials[kind] = kind_trials
    return records, trials


def _run_seed(
    seed: int,
    config: Config,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    world = World()
    agents = {
        kind: Agent(make_predictor(kind, config.history_size), exploration=0.1)
        for kind in PREDICTORS
    }
    records: list[dict[str, object]] = []
    records.extend(_train(agents, world, config, seed))
    records.extend(_probes(agents, world, config, seed))
    nav_records, nav_trials = _navigate(agents, world, config, seed)
    records.extend(nav_records)

    probe_rows = [row for row in records if row["phase"] == "probe"]
    metrics = {
        "probes": {
            kind: {
                "overall": _prediction_metrics(
                    row for row in probe_rows if row["predictor"] == kind
                ),
                "by_terrain": {
                    terrain: _prediction_metrics(
                        row for row in probe_rows
                        if row["predictor"] == kind and row["terrain"] == terrain
                    )
                    for terrain in TERRAINS
                },
            }
            for kind in PREDICTORS
        },
        "navigation": {
            kind: _target_metrics(nav_trials[kind])
            for kind in list(PREDICTORS) + ["oracle", "random"]
        },
        "stored_samples": {
            kind: agents[kind].predictor.sample_count() for kind in PREDICTORS
        },
    }
    models = {kind: agents[kind].predictor.to_dict() for kind in PREDICTORS}
    return records, metrics, models


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _aggregate(run_summaries: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {
        "probes": {},
        "navigation": {},
        "stored_samples": {},
    }
    for kind in PREDICTORS:
        aggregate["probes"][kind] = {
            "overall": {
                key: _average(
                    item["metrics"]["probes"][kind]["overall"][key]
                    for item in run_summaries
                )
                for key in ("brier", "coverage", "cone_cardinality")
            },
            "by_terrain": {
                terrain: {
                    key: _average(
                        item["metrics"]["probes"][kind]["by_terrain"][terrain][key]
                        for item in run_summaries
                    )
                    for key in ("brier", "coverage", "cone_cardinality")
                }
                for terrain in TERRAINS
            },
        }
        aggregate["stored_samples"][kind] = _average(
            item["metrics"]["stored_samples"][kind] for item in run_summaries
        )
    for kind in list(PREDICTORS) + ["oracle", "random"]:
        aggregate["navigation"][kind] = {
            key: _average(
                item["metrics"]["navigation"][kind][key] for item in run_summaries
            )
            for key in (
                "success_rate",
                "mean_interactions_to_first_hit",
                "mean_path_efficiency",
                "censored_rate",
            )
        }
    return aggregate


def _evidence(aggregate: dict[str, object]) -> dict[str, object]:
    pooled = aggregate["probes"]["context_pooled"]["overall"]
    unconditional = aggregate["probes"]["unconditional"]["overall"]
    exact = aggregate["probes"]["exact_state"]["overall"]
    nav_pooled = aggregate["navigation"]["context_pooled"]["success_rate"]
    nav_uncond = aggregate["navigation"]["unconditional"]["success_rate"]
    nav_exact = aggregate["navigation"]["exact_state"]["success_rate"]
    return {
        "context_pooled_probe_quality": {
            "met": (
                pooled["coverage"] is not None
                and pooled["coverage"] >= 0.9
                and pooled["cone_cardinality"] is not None
                and pooled["cone_cardinality"] <= 1.25
                and pooled["brier"] is not None
                and pooled["brier"] <= 0.03
            ),
            "coverage": pooled["coverage"],
            "cone_cardinality": pooled["cone_cardinality"],
            "brier": pooled["brier"],
            "criterion": "coverage >= 0.9, cone_cardinality <= 1.25, brier <= 0.03",
        },
        "context_pooled_navigation_success": {
            "met": nav_pooled is not None and nav_pooled >= 0.9,
            "success_rate": nav_pooled,
            "criterion": "held-out navigation success >= 0.9",
        },
        "navigation_over_unconditional": {
            "met": (
                nav_pooled is not None
                and nav_uncond is not None
                and nav_pooled - nav_uncond >= 0.2
            ),
            "difference": (
                None if nav_pooled is None or nav_uncond is None
                else nav_pooled - nav_uncond
            ),
            "criterion": "navigation success advantage over unconditional >= 0.2",
        },
        "navigation_over_exact_state": {
            "met": (
                nav_pooled is not None
                and nav_exact is not None
                and nav_pooled - nav_exact >= 0.2
            ),
            "difference": (
                None if nav_pooled is None or nav_exact is None
                else nav_pooled - nav_exact
            ),
            "criterion": "navigation success advantage over exact-state >= 0.2",
        },
        "brier_over_unconditional": {
            "met": (
                pooled["brier"] is not None
                and unconditional["brier"] is not None
                and unconditional["brier"] - pooled["brier"] >= 0.2
            ),
            "improvement": (
                None if pooled["brier"] is None or unconditional["brier"] is None
                else unconditional["brier"] - pooled["brier"]
            ),
            "criterion": "held-out Brier improvement over unconditional >= 0.2",
        },
        "brier_over_exact_state": {
            "met": (
                pooled["brier"] is not None
                and exact["brier"] is not None
                and exact["brier"] - pooled["brier"] >= 0.2
            ),
            "improvement": (
                None if pooled["brier"] is None or exact["brier"] is None
                else exact["brier"] - pooled["brier"]
            ),
            "criterion": "held-out Brier improvement over exact-state >= 0.2",
        },
    }


def run_experiment(output: str | Path, config: Config = Config()) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, object]] = []
    models: dict[str, object] = {}
    started = time.perf_counter()
    with (output_path / "steps.jsonl").open("w", encoding="utf-8") as steps:
        for seed in config.seeds:
            records, metrics, model = _run_seed(seed, config)
            for record in records:
                steps.write(json.dumps({"seed": seed, **record}, sort_keys=True) + "\n")
            run_summaries.append({"seed": seed, "metrics": metrics})
            models[str(seed)] = model
    elapsed = time.perf_counter() - started
    aggregate = _aggregate(run_summaries)
    summary = {
        "configuration": {
            **asdict(config),
            "seeds": list(config.seeds),
            "ports": list(PORTS),
            "terrains": list(TERRAINS),
            "predictors": list(PREDICTORS),
            "train_cells": {
                terrain: [list(cell) for cell in TRAIN_CELLS[terrain]]
                for terrain in TERRAINS
            },
            "eval_cells": {
                terrain: [list(cell) for cell in EVAL_CELLS[terrain]]
                for terrain in TERRAINS
            },
            "terrain_rule": "plain if (x+y)%2==0 else rotated",
            "commitment_mass": 0.9,
            "exploration_probability": 0.1,
            "purpose": "held_out_contextual_transfer",
        },
        "seeds": list(config.seeds),
        "per_seed": run_summaries,
        "aggregate": aggregate,
        "evidence_targets": _evidence(aggregate),
        "references": {
            "full_domain_cone": {
                "coverage": 1.0,
                "cone_cardinality": len(OUTCOMES),
            },
            "oracle_navigation": aggregate["navigation"]["oracle"],
            "random_navigation": aggregate["navigation"]["random"],
        },
        "runtime_seconds": elapsed,
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
    parser.add_argument("--output", default="artifacts/stage1b")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    for name, result in summary["evidence_targets"].items():
        print(f"{'PASS' if result['met'] else 'FAIL'}  {name}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
