"""Tests for Stage 1G situational cone volumes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .experiment import (
    generate_stream,
    load_protocol,
    run_experiment,
    run_learner_on_stream,
    sample_angle,
)
from .model import SituationalLearner


class LearnerTests(unittest.TestCase):
    def test_situation_routing_no_cross_leakage(self) -> None:
        learner = SituationalLearner("situation_multi", split_gap=30.0)
        learner.observe("tight", 0.0)
        learner.observe("tight", 10.0)
        self.assertTrue(learner.cone_set("tight").contains(0.0))
        self.assertFalse(learner.cone_set("wide").contains(0.0))
        self.assertEqual(learner.cone_set("wide").measure, 0.0)

    def test_unconditional_shares_store(self) -> None:
        learner = SituationalLearner("unconditional_multi", split_gap=30.0)
        learner.observe("wide", 40.0)
        self.assertTrue(learner.predict("tight", 0).contains(40.0))

    def test_commitment_immutable(self) -> None:
        learner = SituationalLearner("situation_multi")
        commitment = learner.predict("tight", 0)
        before = commitment.to_dict()
        learner.observe("tight", 12.0)
        self.assertEqual(commitment.to_dict(), before)
        self.assertFalse(commitment.contains(12.0))

    def test_contradiction_contained_after_update(self) -> None:
        learner = SituationalLearner("situation_multi", split_gap=30.0)
        for situation, angle in (("tight", 0), ("wide", 40), ("shifted", 180)):
            commitment = learner.predict(situation, 0)
            inside = commitment.contains(angle)
            refinement = learner.observe(situation, angle)
            if not inside:
                self.assertTrue(refinement.after.contains(angle))
                self.assertIn(refinement.operation, ("widen", "add"))

    def test_freeze_blocks_learning(self) -> None:
        learner = SituationalLearner("situation_multi")
        learner.observe("tight", 0.0)
        learner.freeze()
        before = learner.cone_set("tight").to_dict()
        refinement = learner.observe("tight", 10.0)
        self.assertEqual(refinement.operation, "noop")
        self.assertEqual(learner.cone_set("tight").to_dict(), before)


class ExperimentTests(unittest.TestCase):
    def test_stream_deterministic(self) -> None:
        protocol = load_protocol()
        self.assertEqual(generate_stream(protocol, 3), generate_stream(protocol, 3))

    def test_sample_in_support(self) -> None:
        import random

        rng = random.Random(0)
        support = {"center": 0, "half_width": 10, "weight": 1.0}
        for _ in range(50):
            angle = sample_angle(support, rng)
            delta = min(abs(angle - 0), 360 - abs(angle - 0))
            self.assertLessEqual(delta, 10)

    def test_shared_stream_across_learners(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, 1)
        results = {
            policy: run_learner_on_stream(policy, stream, protocol)
            for policy in protocol["learners"]
        }
        reference = [(row["situation"], row["angle"]) for row in stream]
        for result in results.values():
            seen = [(row["situation"], row["angle"]) for row in result["records"]]
            self.assertEqual(seen, reference)

    def test_unconditional_overclaims_tight(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, 0)
        unc = run_learner_on_stream("unconditional_multi", stream, protocol)
        sit = run_learner_on_stream("situation_multi", stream, protocol)
        unc_measure = unc["freeze_snapshot"]["tight"]["measure"]
        sit_measure = sit["freeze_snapshot"]["tight"]["measure"]
        self.assertGreaterEqual(unc_measure, 80.0)
        self.assertLessEqual(sit_measure, 25.0)
        self.assertGreaterEqual(unc_measure - sit_measure, 50.0)

    def test_runs_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_experiment(first_dir)
            second = run_experiment(second_dir)
            first["runtime_seconds"] = 0
            second["runtime_seconds"] = 0
            self.assertEqual(
                json.dumps(first["aggregates"], sort_keys=True),
                json.dumps(second["aggregates"], sort_keys=True),
            )
            self.assertEqual(
                json.dumps(first["checks"], sort_keys=True),
                json.dumps(second["checks"], sort_keys=True),
            )
            summary_a = json.loads(
                (Path(first_dir) / "summary.json").read_text(encoding="utf-8")
            )
            summary_b = json.loads(
                (Path(second_dir) / "summary.json").read_text(encoding="utf-8")
            )
            summary_a.pop("runtime_seconds", None)
            summary_b.pop("runtime_seconds", None)
            self.assertEqual(summary_a, summary_b)


if __name__ == "__main__":
    unittest.main()
