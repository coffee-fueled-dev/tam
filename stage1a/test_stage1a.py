from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from .experiment import Config, World, balanced_targets, run_experiment
from .model import Agent, Observation, OutcomeModel, PORTS


class ModelTests(unittest.TestCase):
    def test_initial_prediction_is_uniform_and_full_domain(self) -> None:
        prediction = OutcomeModel().predict("n")
        self.assertEqual(len(prediction.cone), 5)
        for probability in prediction.probabilities.values():
            self.assertAlmostEqual(probability, 0.2)

    def test_repeated_outcomes_narrow_the_cone(self) -> None:
        model = OutcomeModel()
        for _ in range(18):
            model.observe("n", (0, 1))
        self.assertEqual(model.predict("n").cone, ((0, 1),))

    def test_history_evicts_the_oldest_sample(self) -> None:
        model = OutcomeModel(history_size=3)
        for outcome in ((0, 1), (1, 0), (0, -1), (0, -1)):
            model.observe("e", outcome)
        self.assertEqual(
            model.to_dict()["histories"]["e"],
            ["1,0", "0,-1", "0,-1"],
        )

    def test_commitments_are_immutable_snapshots(self) -> None:
        model = OutcomeModel()
        commitment = model.predict("n")
        model.observe("n", (0, 1))
        self.assertAlmostEqual(commitment.probabilities[(0, 1)], 0.2)
        with self.assertRaises(TypeError):
            commitment.probabilities[(0, 1)] = 1.0

    def test_prediction_does_not_mutate_model(self) -> None:
        model = OutcomeModel()
        before = model.to_dict()
        model.predict("n")
        self.assertEqual(model.to_dict(), before)

    def test_serialization_round_trip(self) -> None:
        model = OutcomeModel()
        for outcome in ((0, 1), (0, 1), (0, 0)):
            model.observe("n", outcome)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(OutcomeModel.load(path).to_dict(), model.to_dict())


class LoopTests(unittest.TestCase):
    def test_select_minimizes_2d_distance(self) -> None:
        agent = Agent(exploration=0.0)
        for _ in range(12):
            agent.observe("n", (0, 1))
            agent.observe("e", (1, 0))
            agent.observe("s", (0, -1))
            agent.observe("w", (-1, 0))
            agent.observe("stay", (0, 0))
        rng = random.Random(0)
        port, _ = agent.select(Observation((0, 0), (0, 5)), rng)
        self.assertEqual(port, "n")
        port, _ = agent.select(Observation((0, 0), (5, 0)), rng)
        self.assertEqual(port, "e")

    def test_prediction_independent_of_position(self) -> None:
        model = OutcomeModel()
        for _ in range(8):
            model.observe("n", (0, 1))
        first = model.predict("n")
        second = model.predict("n")
        self.assertEqual(first.cone, second.cone)
        self.assertEqual(dict(first.probabilities), dict(second.probabilities))

    def test_world_reset_preserves_learning(self) -> None:
        agent = Agent()
        agent.observe("n", (0, 1))
        world = World()
        world.step("n")
        observation = world.reset((-4, 4))
        self.assertEqual(observation, Observation((0, 0), (-4, 4)))
        self.assertEqual(agent.model.to_dict()["histories"]["n"], ["0,1"])

    def test_observation_does_not_reveal_hidden_world_state(self) -> None:
        world = World()
        self.assertEqual(set(vars(world.observation)), {"position", "target"})

    def test_balanced_targets_cover_all_types(self) -> None:
        targets = balanced_targets(0, 16)
        self.assertEqual(len(targets), 16)
        self.assertEqual(len(set(targets)), 8)


class ExperimentTests(unittest.TestCase):
    config = Config(
        seeds=(0,),
        warmup_interactions=5,
        trials=8,
        interactions_per_trial=4,
    )

    def test_complete_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            for filename in ("steps.jsonl", "summary.json", "model.json"):
                self.assertTrue((Path(directory) / filename).is_file())
            self.assertEqual(len(summary["per_seed"]), 1)
            self.assertEqual(len(summary["gate"]), 3)

    def test_runs_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_experiment(first, self.config)
            run_experiment(second, self.config)
            for filename in ("steps.jsonl", "summary.json", "model.json"):
                self.assertEqual(
                    (Path(first) / filename).read_bytes(),
                    (Path(second) / filename).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
