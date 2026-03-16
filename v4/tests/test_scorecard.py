"""Tests for the benchmark scorecard layer."""

from __future__ import annotations

import unittest

from v4.benchmark.runner import BenchmarkConfig, BenchmarkVariant, run_benchmark_suite
from v4.benchmark.scorecard import score_benchmark_results
from v4.benchmark.worlds import CorridorWorldConfig


class ScorecardTests(unittest.TestCase):
    def test_scorecard_rows_are_emitted(self) -> None:
        config = BenchmarkConfig(
            name="shared_fiber",
            encoder_mode="shared",
            geometry_mode="fiber",
            steps=5,
        )
        raw = BenchmarkConfig(
            name="raw_fiber",
            encoder_mode="raw",
            geometry_mode="fiber",
            steps=5,
        )
        point = BenchmarkConfig(
            name="shared_point",
            encoder_mode="shared",
            geometry_mode="point",
            steps=5,
        )
        diagonal = BenchmarkConfig(
            name="shared_diagonal",
            encoder_mode="shared",
            geometry_mode="diagonal",
            steps=5,
        )
        variant = BenchmarkVariant(
            name="smoke_2d",
            world_config=CorridorWorldConfig(state_dim=2),
        )
        results = run_benchmark_suite(configs=[config, raw, point, diagonal], variants=[variant])
        rows = score_benchmark_results(results)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].benchmark_family, "smoke_2d")
        self.assertIn(rows[0].overall, ("meaningful", "not yet meaningful", "strongly meaningful"))

    def test_scorecard_rationale_mentions_claim_vs_task(self) -> None:
        structural = BenchmarkConfig(
            name="shared_fiber",
            encoder_mode="shared",
            geometry_mode="fiber",
            steps=5,
        )
        raw = BenchmarkConfig(
            name="raw_fiber",
            encoder_mode="raw",
            geometry_mode="fiber",
            steps=5,
        )
        point = BenchmarkConfig(
            name="shared_point",
            encoder_mode="shared",
            geometry_mode="point",
            steps=5,
        )
        diagonal = BenchmarkConfig(
            name="shared_diagonal",
            encoder_mode="shared",
            geometry_mode="diagonal",
            steps=5,
        )
        variant = BenchmarkVariant(
            name="corridor_2d_hidden",
            world_config=CorridorWorldConfig(state_dim=2, hidden_mode=1),
        )
        results = run_benchmark_suite(configs=[structural, raw, point, diagonal], variants=[variant])
        row = score_benchmark_results(results)[0]
        self.assertIn("task_success", row.rationale)
        self.assertIn("claim_success", row.rationale)


if __name__ == "__main__":
    unittest.main()
