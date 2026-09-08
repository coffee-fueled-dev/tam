"""Tests for Stage 1K overlapping / near-merge mode characterization."""

from __future__ import annotations

import unittest

from stage1i.model import cluster_angles, rebuild_multi_from_history

from .experiment import (
    generate_stream,
    load_protocol,
    run_learner_on_stream,
)
from .model import edge_gap, make_learner


class ClusteringTests(unittest.TestCase):
    def test_merges_when_edge_gap_much_less_than_split(self) -> None:
        # centers 0 and 25, hw 10 → edge gap 5 ≪ 30
        angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
        clusters = cluster_angles(angles, split_gap=30.0)
        self.assertEqual(len(clusters), 1)

    def test_splits_when_edge_gap_much_greater_than_split(self) -> None:
        # centers 0 and 90, hw 10 → edge gap 70 ≫ 30
        angles = list(range(0, 11)) + list(range(80, 101))
        clusters = cluster_angles([float(a) for a in angles], split_gap=30.0)
        self.assertEqual(len(clusters), 2)

    def test_wide_arc_stays_one_cluster(self) -> None:
        angles = [float(a) for a in range(-40, 41)]
        clusters = cluster_angles(angles, split_gap=30.0)
        self.assertEqual(len(clusters), 1)
        cones = rebuild_multi_from_history(angles, split_gap=30.0)
        self.assertEqual(len(cones.cones), 1)
        self.assertLessEqual(cones.measure, 85.0)

    def test_edge_gap_helper(self) -> None:
        self.assertAlmostEqual(edge_gap(0.0, 90.0, 10.0), 70.0)
        self.assertAlmostEqual(edge_gap(0.0, 50.0, 10.0), 30.0)
        self.assertAlmostEqual(edge_gap(0.0, 25.0, 10.0), 5.0)


class ModelTests(unittest.TestCase):
    def test_hysteresis_containment_on_miss(self) -> None:
        learner = make_learner("hysteresis_gap30", window_size=16, grace_t=8)
        for angle in (0.0, 90.0, 25.0, 180.0):
            commitment = learner.predict(0)
            inside = commitment.contains(angle)
            refinement = learner.observe(angle)
            if not inside:
                self.assertTrue(refinement.after.contains(angle))

    def test_separate_keeps_two_cones(self) -> None:
        learner = make_learner("hysteresis_gap30", window_size=48, grace_t=64)
        for _ in range(80):
            learner.observe(0.0)
            learner.observe(90.0)
        self.assertEqual(len(learner.cone_set.cones), 2)
        self.assertLessEqual(learner.cone_set.measure, 45.0)

    def test_overlap_merges_to_one(self) -> None:
        learner = make_learner("hysteresis_gap30", window_size=48, grace_t=64)
        for _ in range(80):
            learner.observe(0.0)
            learner.observe(25.0)
        self.assertEqual(len(learner.cone_set.cones), 1)
        self.assertLessEqual(learner.cone_set.measure, 55.0)


class ExperimentTests(unittest.TestCase):
    def test_shared_streams(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "separate70", 3)
        results = {
            policy: run_learner_on_stream(policy, stream, protocol)
            for policy in protocol["learners"]
        }
        reference = [row["angle"] for row in stream]
        for result in results.values():
            self.assertEqual([row["angle"] for row in result["records"]], reference)

    def test_runs_are_reproducible(self) -> None:
        protocol = load_protocol()

        def subset() -> dict:
            from .experiment import aggregate_scenario, evaluate_evidence, summarize_seed

            summaries = {name: [] for name in ("separate70", "overlap5", "wide_unimodal", "boundary30")}
            for scenario in summaries:
                for seed in range(3):
                    stream = generate_stream(protocol, scenario, seed)
                    policies = list(protocol["learners"])
                    if scenario in ("separate70", "overlap5"):
                        policies = policies + list(protocol["diagnostic_learners"])
                    results = {
                        policy: run_learner_on_stream(policy, stream, protocol)
                        for policy in policies
                    }
                    summaries[scenario].append(
                        summarize_seed(protocol, scenario, seed, results)
                    )
            aggregates = {
                scenario: aggregate_scenario(
                    protocol,
                    scenario,
                    rows,
                    policies=list(rows[0]["learners"].keys()),
                )
                for scenario, rows in summaries.items()
            }
            return evaluate_evidence(protocol, aggregates)

        self.assertEqual(subset(), subset())


if __name__ == "__main__":
    unittest.main()
