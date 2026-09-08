"""Tests for Stage 1I multimodal cone pruning."""

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
    cluster_angles,
    make_learner,
    rebuild_multi_from_history,
    rebuild_single_from_history,
)


class ModelTests(unittest.TestCase):
    def test_cluster_separates_modes(self) -> None:
        clusters = cluster_angles([0.0, 5.0, 180.0, 175.0], split_gap=30.0)
        self.assertEqual(len(clusters), 2)

    def test_rebuild_multi_no_bridge(self) -> None:
        cones = rebuild_multi_from_history(
            [0.0, 10.0, 180.0, 170.0],
            split_gap=30.0,
        )
        self.assertEqual(len(cones.cones), 2)
        self.assertLessEqual(cones.measure, 45.0)

    def test_rebuild_single_bridges(self) -> None:
        cones = rebuild_single_from_history([0.0, 10.0, 180.0, 170.0])
        self.assertEqual(len(cones.cones), 1)
        self.assertGreaterEqual(cones.measure, 150.0)

    def test_window_multi_prunes_vanished_mode(self) -> None:
        learner = make_learner("window_multi", window_size=8, split_gap=30.0)
        for angle in (0.0, 180.0, 5.0, 175.0, 10.0, 170.0, 0.0, 180.0):
            learner.observe(angle)
        self.assertEqual(len(learner.cone_set.cones), 2)
        for angle in (0.0, 5.0, 10.0, -5.0, 0.0, 8.0, 2.0, -8.0):
            refinement = learner.observe(angle)
        self.assertEqual(len(learner.cone_set.cones), 1)
        self.assertLessEqual(learner.cone_set.measure, 25.0)
        self.assertIn(refinement.operation, ("prune", "narrow", "noop", "shift", "widen"))

    def test_cumulative_retains_dead_mode(self) -> None:
        learner = make_learner("cumulative_multi", split_gap=30.0)
        for angle in (0.0, 10.0, 180.0, 170.0):
            learner.observe(angle)
        self.assertGreaterEqual(len(learner.cone_set.cones), 2)
        before = learner.cone_set.measure
        for _ in range(40):
            learner.observe(0.0)
        self.assertGreaterEqual(len(learner.cone_set.cones), 2)
        self.assertGreaterEqual(learner.cone_set.measure, before - 1e-9)
        self.assertGreaterEqual(learner.cone_set.measure, 20.0)

    def test_commitment_immutable(self) -> None:
        learner = make_learner("window_multi")
        commitment = learner.predict(0)
        before = commitment.to_dict()
        learner.observe(40.0)
        self.assertEqual(commitment.to_dict(), before)

    def test_miss_contained_after_update(self) -> None:
        learner = make_learner("window_multi", window_size=16)
        for angle in (0.0, 180.0, 90.0):
            commitment = learner.predict(0)
            inside = commitment.contains(angle)
            refinement = learner.observe(angle)
            if not inside:
                self.assertTrue(refinement.after.contains(angle))


class ExperimentTests(unittest.TestCase):
    def test_situation_label_fixed(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "hidden_AB_to_A", 0)
        self.assertEqual({row["situation"] for row in stream}, {protocol["situation_label"]})

    def test_shared_stream(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "stationary_AB", 1)
        results = {
            policy: run_learner_on_stream(policy, stream, protocol)
            for policy in protocol["learners"]
        }
        reference = [row["angle"] for row in stream]
        for result in results.values():
            self.assertEqual([row["angle"] for row in result["records"]], reference)

    def test_window_multi_beats_baselines_on_drop(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "hidden_AB_to_A", 0)
        multi = run_learner_on_stream("window_multi", stream, protocol)
        single = run_learner_on_stream("window_single", stream, protocol)
        cumulative = run_learner_on_stream("cumulative_multi", stream, protocol)
        post_multi = [row for row in multi["records"] if row["phase"] == "post"]
        ab_single = [row for row in single["records"] if row["phase"] == "AB"]
        post_cum = [row for row in cumulative["records"] if row["phase"] == "post"]
        self.assertLessEqual(post_multi[-1]["pre_measure"], 25.0)
        self.assertLessEqual(post_multi[-1]["pre_count"], 1)
        self.assertGreaterEqual(
            sum(row["pre_measure"] for row in ab_single) / len(ab_single),
            150.0,
        )
        self.assertGreaterEqual(post_cum[-1]["pre_count"], 2)

    def test_runs_are_byte_reproducible(self) -> None:
        protocol = load_protocol()
        # Full double experiment is expensive; verify deterministic streams and
        # identical aggregated checks over a fixed seed subset twice.
        def subset_checks() -> dict:
            from .experiment import (
                aggregate_scenario,
                evaluate_evidence,
                summarize_seed,
            )

            all_summaries = {
                "stationary_AB": [],
                "hidden_AB_to_A": [],
                "hidden_AB_to_B": [],
            }
            for scenario in all_summaries:
                for seed in range(5):
                    stream = generate_stream(protocol, scenario, seed)
                    results = {
                        policy: run_learner_on_stream(policy, stream, protocol)
                        for policy in protocol["learners"]
                    }
                    all_summaries[scenario].append(
                        summarize_seed(protocol, scenario, seed, results)
                    )
            aggregates = {
                scenario: aggregate_scenario(protocol, scenario, summaries)
                for scenario, summaries in all_summaries.items()
            }
            # evaluate_evidence expects sensitivity key optional via .get
            aggregates["sensitivity"] = {}
            return evaluate_evidence(protocol, aggregates)

        self.assertEqual(
            json.dumps(subset_checks(), sort_keys=True),
            json.dumps(subset_checks(), sort_keys=True),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(directory)
            summary = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["decision"]["verdict"], summary["decision"]["verdict"])
            self.assertIn(result["decision"]["verdict"], ("PASS", "FAIL"))


if __name__ == "__main__":
    unittest.main()
