"""Tests for Stage 1L continuous edge-gap drift."""

from __future__ import annotations

import unittest

from .experiment import (
    generate_stream,
    load_protocol,
    run_learner_on_stream,
)
from .model import (
    classify_regime,
    edge_gap_between,
    make_learner,
    modes_for_edge_gap,
)


class GeometryTests(unittest.TestCase):
    def test_modes_edge_gap_roundtrip(self) -> None:
        modes = modes_for_edge_gap(70.0)
        self.assertAlmostEqual(
            edge_gap_between(modes[0]["center"], modes[1]["center"], 10.0),
            70.0,
        )
        modes = modes_for_edge_gap(5.0)
        self.assertAlmostEqual(
            edge_gap_between(modes[0]["center"], modes[1]["center"], 10.0),
            5.0,
        )

    def test_regime_bins(self) -> None:
        protocol = load_protocol()
        bins = protocol["regime_bins"]
        self.assertEqual(classify_regime(70.0, bins), "separate")
        self.assertEqual(classify_regime(30.0, bins), "boundary")
        self.assertEqual(classify_regime(5.0, bins), "overlap")


class ModelTests(unittest.TestCase):
    def test_binned_stores_are_independent(self) -> None:
        learner = make_learner("binned_hysteresis", window_size=32, grace_t=16)
        for _ in range(40):
            learner.observe("separate", 0.0)
            learner.observe("separate", 90.0)
        for _ in range(40):
            learner.observe("overlap", 0.0)
            learner.observe("overlap", 25.0)
        self.assertEqual(len(learner.cone_set("separate").cones), 2)
        self.assertEqual(len(learner.cone_set("overlap").cones), 1)
        self.assertLessEqual(learner.cone_set("separate").measure, 45.0)
        self.assertLessEqual(learner.cone_set("overlap").measure, 55.0)

    def test_pooled_retains_dead_mode_after_jump(self) -> None:
        learner = make_learner("pooled_hysteresis", window_size=48, grace_t=64)
        for _ in range(40):
            learner.observe("separate", 0.0)
            learner.observe("separate", 90.0)
        for _ in range(20):
            learner.observe("overlap", 0.0)
            learner.observe("overlap", 25.0)
        # Grace keeps the unmatched distant mode; settled geometry has not yet pruned.
        self.assertGreaterEqual(len(learner.cone_set("overlap").cones), 2)

    def test_containment_on_miss(self) -> None:
        for policy in ("binned_hysteresis", "pooled_hysteresis", "pooled_single"):
            learner = make_learner(policy, window_size=16, grace_t=8)
            for regime, angle in (
                ("separate", 0.0),
                ("separate", 90.0),
                ("overlap", 25.0),
                ("boundary", 50.0),
            ):
                commitment = learner.predict(regime, 0)
                inside = commitment.contains(angle)
                refinement = learner.observe(regime, angle)
                if not inside:
                    self.assertTrue(refinement.after.contains(angle), policy)


class ExperimentTests(unittest.TestCase):
    def test_shared_streams(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "plateau_tour", 2)
        results = {
            policy: run_learner_on_stream(policy, stream, protocol)
            for policy in protocol["learners"]
        }
        reference = [row["angle"] for row in stream]
        for result in results.values():
            self.assertEqual([row["angle"] for row in result["records"]], reference)

    def test_slow_sweep_edge_gap_monotone(self) -> None:
        protocol = load_protocol()
        stream = generate_stream(protocol, "slow_sweep", 0)
        gaps = [row["edge_gap"] for row in stream]
        self.assertGreater(gaps[0], gaps[-1])
        self.assertTrue(all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1)))

    def test_runs_are_reproducible(self) -> None:
        protocol = load_protocol()

        def subset() -> dict:
            from .experiment import aggregate_scenario, evaluate_evidence, summarize_seed

            summaries = {name: [] for name in protocol["scenarios"]}
            for scenario in summaries:
                for seed in range(2):
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

        self.assertEqual(subset(), subset())


if __name__ == "__main__":
    unittest.main()
