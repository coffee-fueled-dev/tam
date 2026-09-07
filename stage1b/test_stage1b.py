from __future__ import annotations

import json
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
    ExactStatePredictor,
    Observation,
    UnconditionalPredictor,
    load_predictor,
    make_predictor,
    save_predictor,
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


class PredictorTests(unittest.TestCase):
    def test_unconditional_ignores_terrain(self) -> None:
        model = UnconditionalPredictor()
        plain = Observation((0, 0), (0, 0), "plain")
        rotated = Observation((1, 0), (0, 0), "rotated")
        for _ in range(18):
            model.observe(plain, "n", (0, 1))
        self.assertEqual(model.predict(plain, "n").cone, ((0, 1),))
        self.assertEqual(model.predict(rotated, "n").cone, ((0, 1),))

    def test_exact_state_does_not_transfer(self) -> None:
        model = ExactStatePredictor()
        train = Observation((0, 0), (0, 0), "plain")
        held_out = Observation((3, 3), (0, 0), "plain")
        for _ in range(18):
            model.observe(train, "n", (0, 1))
        self.assertEqual(model.predict(train, "n").cone, ((0, 1),))
        self.assertEqual(len(model.predict(held_out, "n").cone), 5)

    def test_context_pooled_transfers_across_coordinates(self) -> None:
        model = ContextPooledPredictor()
        train = Observation((0, 0), (0, 0), "plain")
        held_out = Observation((3, 3), (0, 0), "plain")
        for _ in range(18):
            model.observe(train, "n", (0, 1))
        self.assertEqual(model.predict(held_out, "n").cone, ((0, 1),))

    def test_context_pooled_keeps_terrains_separate(self) -> None:
        model = ContextPooledPredictor()
        plain = Observation((0, 0), (0, 0), "plain")
        rotated = Observation((1, 0), (0, 0), "rotated")
        for _ in range(18):
            model.observe(plain, "n", (0, 1))
            model.observe(rotated, "n", (1, 0))
        self.assertEqual(model.predict(plain, "n").cone, ((0, 1),))
        self.assertEqual(model.predict(rotated, "n").cone, ((1, 0),))

    def test_serialization_round_trip(self) -> None:
        for kind in ("unconditional", "exact_state", "context_pooled"):
            model = make_predictor(kind)
            obs = Observation((0, 0), (1, 0), "plain")
            model.observe(obs, "e", (1, 0))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{kind}.json"
                save_predictor(model, path)
                restored = load_predictor(path)
                self.assertEqual(restored.to_dict(), model.to_dict())


class WorldTests(unittest.TestCase):
    def test_rotated_terrain_remaps_ports(self) -> None:
        world = World()
        world.teleport((1, 0), (0, 0))
        self.assertEqual(world.observation.terrain, "rotated")
        after = world.step("n")
        self.assertEqual(after.position, (2, 0))

    def test_plain_terrain_uses_ordinary_deltas(self) -> None:
        world = World()
        world.teleport((0, 0), (0, 0))
        self.assertEqual(world.observation.terrain, "plain")
        after = world.step("n")
        self.assertEqual(after.position, (0, 1))

    def test_observation_exposes_only_position_target_terrain(self) -> None:
        world = World()
        self.assertEqual(
            set(vars(world.observation)),
            {"position", "target", "terrain"},
        )


class ExperimentTests(unittest.TestCase):
    config = Config(
        seeds=(0,),
        train_outcomes_per_pair=2,
        probe_outcomes_per_pair=2,
        navigation_trials=4,
        interactions_per_trial=6,
        history_size=32,
    )

    def test_complete_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            for filename in ("steps.jsonl", "summary.json", "model.json"):
                self.assertTrue((Path(directory) / filename).is_file())
            self.assertEqual(len(summary["per_seed"]), 1)
            self.assertEqual(len(summary["evidence_targets"]), 6)
            first = json.loads(
                (Path(directory) / "steps.jsonl").read_text().splitlines()[0]
            )
            self.assertIn("predictor", first)
            self.assertEqual(first["phase"], "train")

    def test_evaluation_does_not_update_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            # Training writes 2 outcomes * 2 terrains * 5 ports = 20 samples
            # into each unconditional port history, but exact-state stores by cell.
            models = summary["per_seed"][0]["metrics"]["stored_samples"]
            self.assertEqual(models["unconditional"], 20)
            self.assertEqual(models["context_pooled"], 20)
            # Probe/navigation must not increase these further; smoke config only trains.
            self.assertGreater(models["exact_state"], 0)

    def test_runs_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_experiment(first, self.config)
            run_experiment(second, self.config)
            for filename in ("steps.jsonl", "model.json"):
                self.assertEqual(
                    (Path(first) / filename).read_bytes(),
                    (Path(second) / filename).read_bytes(),
                )
            left = json.loads((Path(first) / "summary.json").read_text())
            right = json.loads((Path(second) / "summary.json").read_text())
            left.pop("runtime_seconds", None)
            right.pop("runtime_seconds", None)
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
