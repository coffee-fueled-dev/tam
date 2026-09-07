from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from .experiment import Config, World, _interaction, run_experiment
from .model import Agent, Observation, OutcomeModel


class ModelTests(unittest.TestCase):
    def test_initial_prediction_is_uniform_and_full_domain(self) -> None:
        prediction = OutcomeModel().predict("a")
        self.assertEqual(prediction.cone, (-1, 0, 1))
        for probability in prediction.probabilities.values():
            self.assertAlmostEqual(probability, 1 / 3)

    def test_repeated_outcomes_narrow_the_cone(self) -> None:
        model = OutcomeModel()
        for _ in range(9):
            model.observe("a", 1)
        self.assertEqual(model.predict("a").cone, (1,))

    def test_mixed_outcomes_retain_both_likely_values(self) -> None:
        model = OutcomeModel()
        for outcome in (-1, 1) * 16:
            model.observe("a", outcome)
        self.assertEqual(model.predict("a").cone, (-1, 1))

    def test_history_evicts_the_oldest_sample(self) -> None:
        model = OutcomeModel(history_size=3)
        for outcome in (-1, 0, 1, 1):
            model.observe("a", outcome)
        self.assertEqual(model.to_dict()["histories"]["a"], [0, 1, 1])

    def test_commitments_are_immutable_snapshots(self) -> None:
        model = OutcomeModel()
        commitment = model.predict("a")
        model.observe("a", 1)
        self.assertAlmostEqual(commitment.probabilities[1], 1 / 3)
        with self.assertRaises(TypeError):
            commitment.probabilities[1] = 1.0

    def test_prediction_does_not_mutate_model(self) -> None:
        model = OutcomeModel()
        before = model.to_dict()
        model.predict("a")
        self.assertEqual(model.to_dict(), before)

    def test_serialization_round_trip(self) -> None:
        model = OutcomeModel()
        for outcome in (1, 1, 0):
            model.observe("a", outcome)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(OutcomeModel.load(path).to_dict(), model.to_dict())


class LoopTests(unittest.TestCase):
    def test_score_and_binding_use_pre_update_commitment(self) -> None:
        agent = Agent()
        world = World("stationary", random.Random(0))
        commitment = agent.model.predict("a")
        row = _interaction(agent, world, "a", commitment, True, {"phase": "test"})
        self.assertAlmostEqual(row["brier"], 2 / 3)
        self.assertTrue(row["binding_success"])
        self.assertEqual(row["cone"], [-1, 0, 1])

    def test_world_reset_does_not_reset_learning(self) -> None:
        agent = Agent()
        agent.observe("a", 1)
        world = World("stationary", random.Random(0))
        world.step("a")
        observation = world.reset(-5)
        self.assertEqual(observation, Observation(0, -5))
        self.assertEqual(agent.model.to_dict()["histories"]["a"], [1])

    def test_observation_does_not_reveal_hidden_world_state(self) -> None:
        world = World("reversal", random.Random(0))
        self.assertEqual(set(vars(world.observation)), {"position", "target"})
        world.reverse()
        self.assertEqual(set(vars(world.step("a"))), {"position", "target"})


class ExperimentTests(unittest.TestCase):
    config = Config(
        seeds=(0,),
        warmup_interactions=3,
        trials=2,
        interactions_per_trial=4,
        reversal_interaction=4,
    )

    def test_complete_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            for filename in ("steps.jsonl", "summary.json", "model.json"):
                self.assertTrue((Path(directory) / filename).is_file())
            self.assertEqual(len(summary["per_seed"]), 9)
            self.assertEqual(len(summary["evidence_targets"]), 4)
            first = json.loads(
                (Path(directory) / "steps.jsonl").read_text().splitlines()[0]
            )
            self.assertIn("probabilities", first)
            self.assertIn("cone_added", first)

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
