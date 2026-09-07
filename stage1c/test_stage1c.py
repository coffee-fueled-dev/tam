from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from .experiment import (
    EVAL_CELLS,
    TRAIN_CELLS,
    Config,
    World,
    run_experiment,
    terrain_at,
)
from .model import (
    ContextPooledPredictor,
    Observation,
    build_commitment,
    cone_90,
    map_outcome,
    select_port,
)


class SplitTests(unittest.TestCase):
    def test_train_eval_are_disjoint_and_terrain_consistent(self) -> None:
        train = {cell for cells in TRAIN_CELLS.values() for cell in cells}
        eval_cells = {cell for cells in EVAL_CELLS.values() for cell in cells}
        self.assertFalse(train & eval_cells)
        for terrain, cells in TRAIN_CELLS.items():
            for cell in cells:
                self.assertEqual(terrain_at(cell), terrain)
        for terrain, cells in EVAL_CELLS.items():
            for cell in cells:
                self.assertEqual(terrain_at(cell), terrain)


class CommitmentTests(unittest.TestCase):
    def test_cone_90_and_map_builders(self) -> None:
        probabilities = {
            (0, 1): 0.72,
            (0, 0): 0.20,
            (1, 0): 0.04,
            (0, -1): 0.02,
            (-1, 0): 0.02,
        }
        self.assertEqual(cone_90(probabilities), ((0, 1), (0, 0)))
        self.assertEqual(map_outcome(probabilities), (0, 1))
        tam = build_commitment(probabilities, "tam_cone")
        point = build_commitment(probabilities, "point_map")
        full = build_commitment(probabilities, "full_domain")
        self.assertEqual(tam.cone, ((0, 1), (0, 0)))
        self.assertEqual(point.cone, ((0, 1),))
        self.assertEqual(len(full.cone), 5)

    def test_tam_uses_cone_width_tiebreak(self) -> None:
        observation = Observation((0, 0), (0, 1), "plain")
        # Both ports have identical expected distance toward (0,1), but different cones.
        port_probabilities = {
            "n": {
                (0, 1): 0.92,
                (0, 0): 0.02,
                (1, 0): 0.02,
                (0, -1): 0.02,
                (-1, 0): 0.02,
            },
            "e": {
                (0, 1): 0.50,
                (0, 0): 0.42,
                (1, 0): 0.04,
                (0, -1): 0.02,
                (-1, 0): 0.02,
            },
            "s": {
                (0, -1): 1.0,
                (0, 0): 0.0,
                (1, 0): 0.0,
                (0, 1): 0.0,
                (-1, 0): 0.0,
            },
            "w": {
                (-1, 0): 1.0,
                (0, 0): 0.0,
                (1, 0): 0.0,
                (0, 1): 0.0,
                (0, -1): 0.0,
            },
            "stay": {
                (0, 0): 1.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (0, -1): 0.0,
                (-1, 0): 0.0,
            },
        }
        rng = random.Random(0)
        port, commitment = select_port(
            observation, port_probabilities, "tam_cone", rng
        )
        self.assertEqual(port, "n")
        self.assertEqual(commitment.cone, ((0, 1),))

    def test_point_map_selects_by_map_displacement(self) -> None:
        observation = Observation((0, 0), (1, 0), "plain")
        port_probabilities = {
            "n": {
                (0, 0): 0.51,
                (1, 0): 0.49,
                (0, 1): 0.0,
                (0, -1): 0.0,
                (-1, 0): 0.0,
            },
            "e": {
                (1, 0): 0.51,
                (-1, 0): 0.49,
                (0, 0): 0.0,
                (0, 1): 0.0,
                (0, -1): 0.0,
            },
            "s": {
                (0, -1): 1.0,
                (0, 0): 0.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (-1, 0): 0.0,
            },
            "w": {
                (-1, 0): 1.0,
                (0, 0): 0.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (0, -1): 0.0,
            },
            "stay": {
                (0, 0): 1.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (0, -1): 0.0,
                (-1, 0): 0.0,
            },
        }
        rng = random.Random(0)
        map_port, map_commitment = select_port(
            observation, port_probabilities, "point_map", rng
        )
        ev_port, _ = select_port(
            observation, port_probabilities, "expected_value", rng
        )
        self.assertEqual(map_port, "e")
        self.assertEqual(map_commitment.cone, ((1, 0),))
        self.assertEqual(ev_port, "n")

    def test_expected_value_ignores_cone_width(self) -> None:
        observation = Observation((0, 0), (0, 0), "plain")
        port_probabilities = {
            "stay": {
                (0, 0): 1.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (0, -1): 0.0,
                (-1, 0): 0.0,
            },
            "n": {
                (0, 1): 0.5,
                (0, -1): 0.5,
                (0, 0): 0.0,
                (1, 0): 0.0,
                (-1, 0): 0.0,
            },
            "e": {
                (1, 0): 1.0,
                (0, 0): 0.0,
                (0, 1): 0.0,
                (0, -1): 0.0,
                (-1, 0): 0.0,
            },
            "s": {
                (0, -1): 1.0,
                (0, 0): 0.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (-1, 0): 0.0,
            },
            "w": {
                (-1, 0): 1.0,
                (0, 0): 0.0,
                (0, 1): 0.0,
                (1, 0): 0.0,
                (0, -1): 0.0,
            },
        }
        rng = random.Random(0)
        port, _ = select_port(
            observation, port_probabilities, "expected_value", rng
        )
        self.assertEqual(port, "stay")


class WorldTests(unittest.TestCase):
    def test_noise_can_zero_moving_ports(self) -> None:
        world = World(random.Random(0), move_success=0.0)
        world.teleport((0, 0), (0, 0))
        after = world.step("n")
        self.assertEqual(after.position, (0, 0))

    def test_observation_surface(self) -> None:
        world = World(random.Random(0))
        self.assertEqual(
            set(vars(world.observation)),
            {"position", "target", "terrain"},
        )


class ExperimentTests(unittest.TestCase):
    config = Config(
        seeds=(0,),
        train_outcomes_per_pair=4,
        probe_outcomes_per_pair=2,
        navigation_trials=4,
        interactions_per_trial=6,
        history_size=64,
    )

    def test_complete_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            for filename in ("steps.jsonl", "summary.json", "model.json"):
                self.assertTrue((Path(directory) / filename).is_file())
            self.assertEqual(len(summary["evidence_targets"]), 6)
            self.assertIn("verdict", summary["decision"])

    def test_evaluation_does_not_update_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            # 4 outcomes * 2 terrains * 5 ports = 40 training samples.
            self.assertEqual(summary["per_seed"][0]["metrics"]["stored_samples"], 40)

    def test_probe_noise_stream_is_shared_across_controllers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_experiment(directory, self.config)
            rows = [
                json.loads(line)
                for line in (Path(directory) / "steps.jsonl").read_text().splitlines()
                if '"phase": "probe"' in line
            ]
            by_key: dict[tuple, list] = {}
            for row in rows:
                key = (
                    row["terrain"],
                    row["port"],
                    row["pair_index"],
                    tuple(row["observation_before"]["position"]),
                )
                by_key.setdefault(key, []).append(row["displacement"])
            for displacements in by_key.values():
                self.assertEqual(len(set(tuple(item) for item in displacements)), 1)

    def test_runs_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_experiment(first, self.config)
            run_experiment(second, self.config)
            self.assertEqual(
                (Path(first) / "steps.jsonl").read_bytes(),
                (Path(second) / "steps.jsonl").read_bytes(),
            )
            left = json.loads((Path(first) / "summary.json").read_text())
            right = json.loads((Path(second) / "summary.json").read_text())
            left.pop("runtime_seconds", None)
            right.pop("runtime_seconds", None)
            self.assertEqual(left, right)

    def test_predictor_serialization(self) -> None:
        model = ContextPooledPredictor()
        obs = Observation((0, 0), (0, 0), "plain")
        model.observe(obs, "n", (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(ContextPooledPredictor.load(path).to_dict(), model.to_dict())


if __name__ == "__main__":
    unittest.main()
