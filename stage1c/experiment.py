"""Stage 1C: TAM commitment ablation under noisy contextual dynamics."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .model import (
    CONTROLLERS,
    OUTCOMES,
    PORT_DELTAS,
    PORTS,
    ROTATED_DELTAS,
    TERRAINS,
    Commitment,
    ContextPooledPredictor,
    Observation,
    brier_score,
    build_commitment,
    commitment_brier,
    select_port,
)

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
NAV_STARTS = tuple(cell for terrain in TERRAINS for cell in EVAL_CELLS[terrain])
NAV_TARGETS = (
    (18, 18), (18, 20), (20, 18), (20, 20),
    (22, 16), (16, 22), (22, 18), (18, 22),
    (18, 19), (19, 18), (20, 19), (19, 20),
    (22, 17), (17, 22), (21, 18), (18, 21),
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(20))
    train_outcomes_per_pair: int = 40
    probe_outcomes_per_pair: int = 20
    navigation_trials: int = 32
    interactions_per_trial: int = 30
    history_size: int = 64
    move_success_probability: float = 0.8


def terrain_at(position: tuple[int, int]) -> str:
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
    """Opaque noisy 2D world with observable terrain-dependent effects."""

    def __init__(self, noise_rng: random.Random, move_success: float = 0.8) -> None:
        self._noise_rng = noise_rng
        self._move_success = move_success
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
        if (dx, dy) != (0, 0) and self._noise_rng.random() >= self._move_success:
            dx, dy = 0, 0
        self._position = (self._position[0] + dx, self._position[1] + dy)
        return self.observation


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
        if best_loss is None or loss < best_loss or (loss == best_loss and port < best):
            best = port
            best_loss = loss
    return best


def _probabilities(commitment: Commitment) -> dict[str, float]:
    return {
        f"{dx},{dy}": probability
        for (dx, dy), probability in commitment.probabilities.items()
    }


def _record(
    before: Observation,
    after: Observation,
    port: str,
    commitment: Commitment,
    metadata: dict[str, object],
) -> dict[str, object]:
    displacement = (
        after.position[0] - before.position[0],
        after.position[1] - before.position[1],
    )
    return {
        **metadata,
        "observation_before": {
            "position": list(before.position),
            "target": list(before.target),
            "terrain": before.terrain,
        },
        "observation_after": {
            "position": list(after.position),
            "target": list(after.target),
            "terrain": after.terrain,
        },
        "port": port,
        "probabilities": _probabilities(commitment),
        "cone": [list(outcome) for outcome in commitment.cone],
        "displacement": list(displacement),
        "brier": brier_score(commitment.probabilities, displacement),
        "commitment_brier": commitment_brier(commitment, displacement),
        "binding_success": displacement in commitment.cone,
        "goal_achieved": after.position == after.target,
    }


def _prediction_metrics(records: Iterable[dict[str, object]]) -> dict[str, float | None]:
    rows = list(records)
    if not rows:
        return {
            "brier": None,
            "commitment_brier": None,
            "coverage": None,
            "cone_cardinality": None,
        }
    return {
        "brier": mean(float(row["brier"]) for row in rows),
        "commitment_brier": mean(float(row["commitment_brier"]) for row in rows),
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


def _train(
    predictor: ContextPooledPredictor,
    config: Config,
    seed: int,
) -> list[dict[str, object]]:
    noise_rng = random.Random(seed * 10_000 + 11)
    world = World(noise_rng, config.move_success_probability)
    cell_rng = random.Random(seed * 10_000 + 12)
    records: list[dict[str, object]] = []
    for port in PORTS:
        plain_cells = list(TRAIN_CELLS["plain"])
        rotated_cells = list(TRAIN_CELLS["rotated"])
        cell_rng.shuffle(plain_cells)
        cell_rng.shuffle(rotated_cells)
        for index in range(config.train_outcomes_per_pair):
            for terrain, cells in (
                ("plain", plain_cells),
                ("rotated", rotated_cells),
            ):
                position = cells[index % len(cells)]
                before = world.teleport(position, position)
                commitment = build_commitment(
                    predictor.probabilities(before, port),
                    "tam_cone",
                )
                after = world.step(port)
                displacement = (
                    after.position[0] - before.position[0],
                    after.position[1] - before.position[1],
                )
                predictor.observe(before, port, displacement)
                records.append(_record(
                    before,
                    after,
                    port,
                    commitment,
                    {
                        "phase": "train",
                        "controller": "shared_predictor",
                        "terrain": terrain,
                        "pair_index": index,
                    },
                ))
    return records


def _probes(
    predictor: ContextPooledPredictor,
    config: Config,
    seed: int,
) -> list[dict[str, object]]:
    # One shared noise stream so every controller scores the same transitions.
    noise_rng = random.Random(seed * 10_000 + 13)
    world = World(noise_rng, config.move_success_probability)
    cell_rng = random.Random(seed * 10_000 + 14)
    records: list[dict[str, object]] = []
    for terrain in TERRAINS:
        cells = list(EVAL_CELLS[terrain])
        for port in PORTS:
            for index in range(config.probe_outcomes_per_pair):
                position = cells[cell_rng.randrange(len(cells))]
                before = world.teleport(position, position)
                probs = predictor.probabilities(before, port)
                after = world.step(port)
                for controller in CONTROLLERS:
                    commitment = build_commitment(probs, controller)
                    records.append(_record(
                        before,
                        after,
                        port,
                        commitment,
                        {
                            "phase": "probe",
                            "controller": controller,
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


def _navigate_controller(
    kind: str,
    predictor: ContextPooledPredictor | None,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    config: Config,
    seed: int,
    selection_stream: int,
    noise_stream: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selection_rng = random.Random(seed * 10_000 + selection_stream)
    noise_rng = random.Random(seed * 10_000 + noise_stream)
    world = World(noise_rng, config.move_success_probability)
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
                commitment = build_commitment(
                    {outcome: 1 / len(OUTCOMES) for outcome in OUTCOMES},
                    "full_domain",
                )
            elif kind == "random":
                port = selection_rng.choice(PORTS)
                commitment = build_commitment(
                    {outcome: 1 / len(OUTCOMES) for outcome in OUTCOMES},
                    "full_domain",
                )
            else:
                assert predictor is not None
                port_probabilities = {
                    port_name: predictor.probabilities(observation, port_name)
                    for port_name in PORTS
                }
                port, commitment = select_port(
                    observation,
                    port_probabilities,
                    kind,
                    selection_rng,
                )
            after = world.step(port)
            records.append(_record(
                observation,
                after,
                port,
                commitment,
                {
                    "phase": "navigation",
                    "controller": kind,
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


def _run_seed(
    seed: int,
    config: Config,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    predictor = ContextPooledPredictor(config.history_size)
    records: list[dict[str, object]] = []
    records.extend(_train(predictor, config, seed))
    records.extend(_probes(predictor, config, seed))
    pairs = _navigation_pairs(seed, config.navigation_trials)
    nav_trials: dict[str, list[dict[str, object]]] = {}
    controllers = list(CONTROLLERS) + ["oracle", "random"]
    for index, kind in enumerate(controllers):
        nav_records, trials = _navigate_controller(
            kind,
            predictor if kind in CONTROLLERS else None,
            pairs,
            config,
            seed,
            selection_stream=19 + index,
            noise_stream=40 + index,
        )
        records.extend(nav_records)
        nav_trials[kind] = trials

    probe_rows = [row for row in records if row["phase"] == "probe"]
    metrics = {
        "probes": {
            kind: {
                "overall": _prediction_metrics(
                    row for row in probe_rows if row["controller"] == kind
                ),
                "by_terrain": {
                    terrain: _prediction_metrics(
                        row for row in probe_rows
                        if row["controller"] == kind and row["terrain"] == terrain
                    )
                    for terrain in TERRAINS
                },
            }
            for kind in CONTROLLERS
        },
        "navigation": {
            kind: _target_metrics(nav_trials[kind])
            for kind in controllers
        },
        "stored_samples": predictor.sample_count(),
        "action_frequency": {
            kind: {
                port: mean(
                    row["port"] == port
                    for row in records
                    if row["phase"] == "navigation" and row["controller"] == kind
                )
                for port in PORTS
            }
            for kind in CONTROLLERS
        },
    }
    return records, metrics, predictor.to_dict()


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _aggregate(run_summaries: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {
        "probes": {},
        "navigation": {},
        "action_frequency": {},
        "stored_samples": _average(
            item["metrics"]["stored_samples"] for item in run_summaries
        ),
    }
    for kind in CONTROLLERS:
        aggregate["probes"][kind] = {
            "overall": {
                key: _average(
                    item["metrics"]["probes"][kind]["overall"][key]
                    for item in run_summaries
                )
                for key in ("brier", "commitment_brier", "coverage", "cone_cardinality")
            },
            "by_terrain": {
                terrain: {
                    key: _average(
                        item["metrics"]["probes"][kind]["by_terrain"][terrain][key]
                        for item in run_summaries
                    )
                    for key in (
                        "brier",
                        "commitment_brier",
                        "coverage",
                        "cone_cardinality",
                    )
                }
                for terrain in TERRAINS
            },
        }
        aggregate["action_frequency"][kind] = {
            port: _average(
                item["metrics"]["action_frequency"][kind][port]
                for item in run_summaries
            )
            for port in PORTS
        }
    for kind in list(CONTROLLERS) + ["oracle", "random"]:
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
    tam = aggregate["probes"]["tam_cone"]["overall"]
    ev = aggregate["probes"]["expected_value"]["overall"]
    point = aggregate["probes"]["point_map"]["overall"]
    full = aggregate["probes"]["full_domain"]["overall"]
    nav_tam = aggregate["navigation"]["tam_cone"]["success_rate"]
    nav_ev = aggregate["navigation"]["expected_value"]["success_rate"]
    nav_map = aggregate["navigation"]["point_map"]["success_rate"]
    nav_full = aggregate["navigation"]["full_domain"]["success_rate"]

    map_success_gap = (
        None if nav_tam is None or nav_map is None else nav_tam - nav_map
    )
    map_brier_gap = (
        None
        if tam["commitment_brier"] is None or point["commitment_brier"] is None
        else point["commitment_brier"] - tam["commitment_brier"]
    )
    map_success_close = (
        map_success_gap is not None and abs(map_success_gap) <= 0.05
    )
    advantage_over_map = (
        (map_success_gap is not None and map_success_gap >= 0.10)
        or (map_success_close and map_brier_gap is not None and map_brier_gap >= 0.05)
    )

    ev_success_gap = (
        None if nav_tam is None or nav_ev is None else nav_tam - nav_ev
    )
    coverage_matched = (
        tam["coverage"] is not None
        and ev["coverage"] is not None
        and abs(tam["coverage"] - ev["coverage"]) <= 0.02
    )
    specificity_gain = (
        None
        if tam["cone_cardinality"] is None or ev["cone_cardinality"] is None
        else ev["cone_cardinality"] - tam["cone_cardinality"]
    )
    # expected_value reports the same post-hoc 90% cone for measurement, so
    # cardinality matches by construction; selection advantage is nav-only.
    advantage_over_ev = (
        (ev_success_gap is not None and ev_success_gap >= 0.05)
        or (
            coverage_matched
            and specificity_gain is not None
            and specificity_gain >= 0.25
        )
    )

    full_success_gap = (
        None if nav_tam is None or nav_full is None else nav_tam - nav_full
    )
    full_domain_ok = (
        (full_success_gap is not None and full_success_gap >= 0.05)
        or (
            full_success_gap is not None
            and abs(full_success_gap) <= 0.05
            and tam["cone_cardinality"] is not None
            and full["cone_cardinality"] is not None
            and tam["cone_cardinality"] < full["cone_cardinality"]
        )
    )

    return {
        "calibration_usefulness": {
            "met": (
                tam["coverage"] is not None
                and tam["coverage"] >= 0.9
                and tam["cone_cardinality"] is not None
                and tam["cone_cardinality"] <= 2.25
            ),
            "coverage": tam["coverage"],
            "cone_cardinality": tam["cone_cardinality"],
            "criterion": "tam_cone coverage >= 0.9 and cone_cardinality <= 2.25",
        },
        "not_vacuous": {
            "met": (
                tam["coverage"] is not None
                and tam["coverage"] >= 0.9
                and tam["cone_cardinality"] is not None
                and full["cone_cardinality"] is not None
                and full["cone_cardinality"] - tam["cone_cardinality"] >= 0.25
            ),
            "tam_cone_cardinality": tam["cone_cardinality"],
            "full_domain_cardinality": full["cone_cardinality"],
            "coverage": tam["coverage"],
            "criterion": (
                "tam_cone cardinality >= 0.25 below full_domain with coverage >= 0.9"
            ),
        },
        "control_competence": {
            "met": nav_tam is not None and nav_tam >= 0.8,
            "success_rate": nav_tam,
            "criterion": "tam_cone navigation success >= 0.8",
        },
        "advantage_over_point_map": {
            "met": advantage_over_map,
            "success_difference": map_success_gap,
            "commitment_brier_improvement": map_brier_gap,
            "criterion": (
                "nav success +0.10 over point_map, or commitment_brier +0.05 "
                "when success within 0.05"
            ),
        },
        "advantage_over_expected_value": {
            "met": advantage_over_ev,
            "success_difference": ev_success_gap,
            "specificity_gain": specificity_gain,
            "no_tam_selection_advantage": not advantage_over_ev,
            "criterion": (
                "nav success +0.05 over EV, or +0.25 narrower cones at matched coverage"
            ),
        },
        "full_domain_not_enough": {
            "met": full_domain_ok,
            "success_difference": full_success_gap,
            "tam_cone_cardinality": tam["cone_cardinality"],
            "full_domain_cardinality": full["cone_cardinality"],
            "criterion": (
                "nav success +0.05 over full_domain, or match within 0.05 with "
                "narrower cones"
            ),
        },
    }


def _decision(evidence: dict[str, object]) -> dict[str, object]:
    calibration_ok = (
        evidence["calibration_usefulness"]["met"]
        and evidence["not_vacuous"]["met"]
        and evidence["control_competence"]["met"]
    )
    map_success_gap = evidence["advantage_over_point_map"]["success_difference"]
    nav_gain_over_map = map_success_gap is not None and map_success_gap >= 0.10
    nav_gain_over_ev = evidence["advantage_over_expected_value"]["met"]
    if calibration_ok and (nav_gain_over_map or nav_gain_over_ev):
        verdict = "keep_tam_commitment_machinery"
    elif (
        evidence["calibration_usefulness"]["met"]
        and evidence["not_vacuous"]["met"]
        and not nav_gain_over_ev
    ):
        # Selection matches EV; cones remain useful as calibrated commitments
        # (e.g. vs point_map coverage/commitment_brier) but not as a selector.
        verdict = "keep_cones_as_measurement_only"
    else:
        verdict = "do_not_claim_tam_novelty"
    return {
        "verdict": verdict,
        "calibration_and_control_ok": calibration_ok,
        "nav_gain_over_map_or_ev": nav_gain_over_map or nav_gain_over_ev,
        "commitment_quality_over_point_map": evidence["advantage_over_point_map"]["met"],
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
    evidence = _evidence(aggregate)
    summary = {
        "configuration": {
            **asdict(config),
            "seeds": list(config.seeds),
            "ports": list(PORTS),
            "terrains": list(TERRAINS),
            "controllers": list(CONTROLLERS),
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
            "purpose": "tam_commitment_ablation",
        },
        "seeds": list(config.seeds),
        "per_seed": run_summaries,
        "aggregate": aggregate,
        "evidence_targets": evidence,
        "decision": _decision(evidence),
        "references": {
            "oracle_navigation": aggregate["navigation"]["oracle"],
            "random_navigation": aggregate["navigation"]["random"],
            "full_domain_cone": {
                "coverage": 1.0,
                "cone_cardinality": len(OUTCOMES),
            },
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
    parser.add_argument("--output", default="artifacts/stage1c")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    for name, result in summary["evidence_targets"].items():
        print(f"{'PASS' if result['met'] else 'FAIL'}  {name}")
    print(f"DECISION  {summary['decision']['verdict']}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
