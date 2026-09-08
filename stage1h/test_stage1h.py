"""Tests for Stage 1H situational cone narrowing."""

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
)
from .model import (
    AngularCone,
    ConeSet,
    NarrowingLearner,
    classify_operation,
    make_learner,
    rebuild_from_history,
)


class ModelTests(unittest.TestCase):
    def test_window_eviction_narrows(self) -> None:
        learner = NarrowingLearner("window64", window_size=4)
        for angle in (0.0, 40.0, 0.0, 40.0):
            learner.observe(angle)
        self.assertGreater(learner.cone_set.measure, 20.0)
        # Evict the wide samples with tight ones.
        for angle in (0.0, 5.0, 10.0, -5.0):
            refinement = learner.observe(angle)
        self.assertLessEqual(learner.cone_set.measure, 20.0)
        self.assertIn(refinement.operation, ("narrow", "noop", "shift", "widen"))

    def test_wraparound_reconstruction(self) -> None:
        cone = rebuild_from_history([350.0, 10.0])
        self.assertAlmostEqual(cone.measure, 20.0, places=6)
        self.assertTrue(cone.contains(0.0))

    def test_commitment_immutable(self) -> None:
        learner = make_learner("window64")
        commitment = learner.predict(0)
        before = commitment.to_dict()
        learner.observe(25.0)
        self.assertEqual(commitment.to_dict(), before)
        self.assertFalse(commitment.contains(25.0))

    def test_current_observation_contained_after_miss(self) -> None:
        learner = make_learner("window64")
        for angle in (0.0, 180.0, 90.0, 270.0):
            commitment = learner.predict(0)
            inside = commitment.contains(angle)
            refinement = learner.observe(angle)
            if not inside:
                self.assertTrue(refinement.after.contains(angle))

    def test_cumulative_does_not_narrow(self) -> None:
        learner = make_learner("cumulative")
        learner.observe(0.0)
        learner.observe(40.0)
        wide = learner.cone_set.measure
        for _ in range(20):
            refinement = learner.observe(0.0)
            self.assertNotEqual(refinement.operation, "narrow")
        self.assertGreaterEqual(learner.cone_set.measure, wide - 1e-9)

    def test_frozen_stops_learning(self) -> None:
        learner = make_learner("frozen_a1")
        learner.observe(0.0)
        learner.freeze()
        before = learner.cone_set.to_dict()
        refinement = learner.observe(40.0)
        self.assertEqual(refinement.operation, "noop")
        self.assertEqual(learner.cone_set.to_dict(), before)

    def test_classify_narrow(self) -> None:
        before = ConeSet((AngularCone(0.0, 45.0),))
        after = ConeSet((AngularCone(0.0, 10.0),))
        self.assertEqual(classify_operation(before, after), "narrow")


class ExperimentTests(unittest.TestCase):
    def test_situation_label_unchanged(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "hidden_ABA", 0)
        labels = {row["situation"] for row in stream}
        self.assertEqual(labels, {protocol["situation_label"]})
        regimes = {row["regime"] for row in stream}
        self.assertEqual(regimes, {"A", "B"})

    def test_stream_deterministic_and_shared(self) -> None:
        protocol = load_protocol()
        a = generate_stream(protocol, "hidden_ABA", 2)
        b = generate_stream(protocol, "hidden_ABA", 2)
        self.assertEqual(a, b)
        results = {
            policy: run_learner_on_stream(policy, a, protocol)
            for policy in protocol["learners"]
        }
        reference = [row["angle"] for row in a]
        for result in results.values():
            self.assertEqual([row["angle"] for row in result["records"]], reference)

    def test_window_contracts_after_return(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "hidden_ABA", 0)
        result = run_learner_on_stream("window64", stream, protocol)
        a2 = [row for row in result["records"] if row["phase"] == "A2"]
        self.assertLessEqual(a2[-1]["pre_measure"], 25.0)
        cumulative = run_learner_on_stream("cumulative", stream, protocol)
        cum_a2 = [row for row in cumulative["records"] if row["phase"] == "A2"]
        self.assertGreaterEqual(cum_a2[-1]["pre_measure"], 85.0)

    def test_runs_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_experiment(first_dir)
            second = run_experiment(second_dir)
            first["runtime_seconds"] = 0
            second["runtime_seconds"] = 0
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
            self.assertEqual(summary_a["decision"], summary_b["decision"])
            self.assertEqual(summary_a["checks"], summary_b["checks"])


if __name__ == "__main__":
    unittest.main()
