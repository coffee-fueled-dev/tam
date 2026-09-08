"""Tests for Stage 1F geometric cone refinement."""

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
    sample_from_modes,
)
from .model import (
    CIRCLE,
    AngularCone,
    ConeSet,
    GeometricLearner,
    circular_distance,
    minimal_covering_cone,
    normalize_angle,
)


class GeometryTests(unittest.TestCase):
    def test_normalize_wraps(self) -> None:
        self.assertEqual(normalize_angle(360), 0.0)
        self.assertEqual(normalize_angle(-90), 270.0)

    def test_circular_distance(self) -> None:
        self.assertEqual(circular_distance(10, 350), 20.0)
        self.assertEqual(circular_distance(0, 180), 180.0)

    def test_boundary_inclusion(self) -> None:
        cone = AngularCone(0.0, 15.0)
        self.assertTrue(cone.contains(15.0))
        self.assertTrue(cone.contains(345.0))
        self.assertFalse(cone.contains(16.0))

    def test_wraparound_cone(self) -> None:
        cone = AngularCone(0.0, 20.0)
        self.assertTrue(cone.contains(350.0))
        self.assertTrue(cone.contains(10.0))
        self.assertFalse(cone.contains(180.0))

    def test_minimal_covering_is_tight(self) -> None:
        cone = minimal_covering_cone([350.0, 10.0])
        self.assertAlmostEqual(cone.measure, 20.0, places=6)
        self.assertTrue(cone.contains(350.0))
        self.assertTrue(cone.contains(0.0))
        self.assertTrue(cone.contains(10.0))

    def test_disjoint_union_measure(self) -> None:
        cones = ConeSet(
            (
                AngularCone(0.0, 10.0),
                AngularCone(180.0, 10.0),
            )
        )
        self.assertAlmostEqual(cones.measure, 40.0, places=6)

    def test_overlapping_union_measure(self) -> None:
        cones = ConeSet(
            (
                AngularCone(0.0, 20.0),
                AngularCone(10.0, 20.0),
            )
        )
        self.assertAlmostEqual(cones.measure, 50.0, places=6)


class LearnerTests(unittest.TestCase):
    def test_empty_add_then_widen(self) -> None:
        learner = GeometricLearner("single_widen")
        first = learner.observe(0.0)
        self.assertEqual(first.operation, "add")
        self.assertTrue(first.after.contains(0.0))
        second = learner.observe(20.0)
        self.assertEqual(second.operation, "widen")
        self.assertTrue(second.after.contains(20.0))
        self.assertAlmostEqual(second.after.measure, 20.0, places=6)

    def test_hit_is_noop(self) -> None:
        learner = GeometricLearner("single_widen")
        learner.observe(0.0)
        learner.observe(10.0)
        refinement = learner.observe(5.0)
        self.assertEqual(refinement.operation, "noop")

    def test_commitment_immutable(self) -> None:
        learner = GeometricLearner("single_widen")
        commitment = learner.predict(0)
        before = commitment.to_dict()
        learner.observe(40.0)
        self.assertEqual(commitment.to_dict(), before)
        self.assertFalse(commitment.contains(40.0))

    def test_multi_adds_when_separated(self) -> None:
        learner = GeometricLearner("multi_cone", split_gap=30.0)
        learner.observe(0.0)
        learner.observe(10.0)
        refinement = learner.observe(180.0)
        self.assertEqual(refinement.operation, "add")
        self.assertEqual(len(refinement.after.cones), 2)

    def test_multi_widens_when_nearby(self) -> None:
        learner = GeometricLearner("multi_cone", split_gap=30.0)
        learner.observe(0.0)
        refinement = learner.observe(25.0)
        self.assertEqual(refinement.operation, "widen")
        self.assertEqual(len(refinement.after.cones), 1)

    def test_contradiction_always_contained_after(self) -> None:
        learner = GeometricLearner("multi_cone", split_gap=30.0)
        angles = [0, 5, 180, 175, 350, 90]
        for angle in angles:
            commitment = learner.predict(0)
            inside = commitment.contains(angle)
            refinement = learner.observe(angle)
            if not inside:
                self.assertTrue(refinement.after.contains(angle))
                self.assertIn(refinement.operation, ("widen", "add"))

    def test_full_circle_vacuous(self) -> None:
        learner = GeometricLearner("full_circle")
        self.assertAlmostEqual(learner.cone_set.measure, CIRCLE, places=6)
        refinement = learner.observe(123.0)
        self.assertEqual(refinement.operation, "noop")
        self.assertTrue(refinement.after.contains(123.0))


class ExperimentTests(unittest.TestCase):
    def test_stream_deterministic(self) -> None:
        protocol = load_protocol()
        a = generate_stream(protocol, "bimodal", 7)
        b = generate_stream(protocol, "bimodal", 7)
        self.assertEqual(a, b)

    def test_sample_stays_in_mode(self) -> None:
        import random

        rng = random.Random(0)
        mode = {"center": 0, "half_width": 10, "weight": 1.0}
        for _ in range(100):
            angle = sample_from_modes([mode], rng)
            delta = min(abs(angle - 0), 360 - abs(angle - 0))
            self.assertLessEqual(delta, 10)

    def test_shared_stream_across_learners(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "unimodal", 0)
        split = float(protocol["split_gap_degrees"])
        results = {
            policy: run_learner_on_stream(policy, stream, split)
            for policy in protocol["learners"]
        }
        for policy, result in results.items():
            angles = [row["angle"] for row in result["records"]]
            self.assertEqual(angles, list(stream))

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
