"""Tests for Stage 1J mode churn hysteresis."""

from __future__ import annotations

import json
import unittest

from .experiment import (
    generate_stream,
    load_protocol,
    run_learner_on_stream,
)
from .model import HysteresisLearner, make_learner


class ModelTests(unittest.TestCase):
    def test_hysteresis_retains_across_short_gap(self) -> None:
        learner = HysteresisLearner(window_size=48, split_gap=30.0, grace_t=64)
        for _ in range(40):
            learner.observe(0.0)
            learner.observe(180.0)
        self.assertEqual(len(learner.cone_set.cones), 2)
        for _ in range(40):
            learner.observe(0.0)
        self.assertGreaterEqual(len(learner.cone_set.cones), 2)

    def test_hysteresis_prunes_after_grace(self) -> None:
        learner = HysteresisLearner(window_size=16, split_gap=30.0, grace_t=10)
        for _ in range(20):
            learner.observe(0.0)
            learner.observe(180.0)
        for _ in range(40):
            learner.observe(0.0)
        self.assertEqual(len(learner.cone_set.cones), 1)
        self.assertLessEqual(learner.cone_set.measure, 25.0)

    def test_immediate_add_on_new_mode(self) -> None:
        learner = HysteresisLearner(window_size=16, split_gap=30.0, grace_t=48)
        learner.observe(0.0)
        before = len(learner.cone_set.cones)
        refinement = learner.observe(180.0)
        self.assertGreaterEqual(len(learner.cone_set.cones), before)
        self.assertTrue(refinement.after.contains(180.0))
        self.assertIn(refinement.operation, ("add", "widen", "shift", "noop"))

    def test_current_observation_contained(self) -> None:
        learner = make_learner("hysteresis_multi", window_size=16, grace_t=8)
        for angle in (0.0, 180.0, 90.0, 270.0):
            commitment = learner.predict(0)
            inside = commitment.contains(angle)
            refinement = learner.observe(angle)
            if not inside:
                self.assertTrue(refinement.after.contains(angle))

    def test_window_multi_drops_on_long_gap(self) -> None:
        learner = make_learner("window_multi", window_size=16)
        for _ in range(20):
            learner.observe(0.0)
            learner.observe(180.0)
        for _ in range(30):
            learner.observe(0.0)
        self.assertEqual(len(learner.cone_set.cones), 1)


class ExperimentTests(unittest.TestCase):
    def test_shared_stream(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "fast_churn", 1)
        results = {
            policy: run_learner_on_stream(policy, stream, protocol)
            for policy in protocol["learners"]
        }
        reference = [row["angle"] for row in stream]
        for result in results.values():
            self.assertEqual([row["angle"] for row in result["records"]], reference)

    def test_fast_churn_hysteresis_keeps_b(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "fast_churn", 0)
        hyst = run_learner_on_stream("hysteresis_multi", stream, protocol)
        win = run_learner_on_stream("window_multi", stream, protocol)
        end_steps = int(protocol["evidence"]["fast_churn"]["gap_end_steps"])

        def gap_end_counts(records):
            by_gap = {}
            for row in records:
                phase = str(row["phase"])
                if phase.startswith("gap_"):
                    by_gap.setdefault(phase, []).append(row)
            values = []
            for rows in by_gap.values():
                values.extend(r["pre_count"] for r in rows[-end_steps:])
            return values

        self.assertGreaterEqual(sum(gap_end_counts(hyst["records"])) / len(gap_end_counts(hyst["records"])), 1.8)
        self.assertLessEqual(sum(gap_end_counts(win["records"])) / len(gap_end_counts(win["records"])), 1.1)

    def test_runs_are_reproducible(self) -> None:
        protocol = load_protocol()

        def subset() -> dict:
            from .experiment import aggregate_scenario, evaluate_evidence, summarize_seed

            summaries = {name: [] for name in protocol["scenarios"]}
            for scenario in summaries:
                for seed in range(3):
                    stream = generate_stream(protocol, scenario, seed)
                    results = {
                        policy: run_learner_on_stream(policy, stream, protocol)
                        for policy in protocol["learners"]
                    }
                    summaries[scenario].append(
                        summarize_seed(protocol, scenario, seed, results)
                    )
            aggregates = {
                scenario: aggregate_scenario(protocol, scenario, rows)
                for scenario, rows in summaries.items()
            }
            return evaluate_evidence(protocol, aggregates)

        self.assertEqual(
            json.dumps(subset(), sort_keys=True),
            json.dumps(subset(), sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
