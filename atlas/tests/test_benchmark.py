import unittest

from atlas.benchmark.report import format_result_line
from atlas.benchmark.runner import BenchmarkConfig, run_benchmark_suite
from atlas.benchmark.worlds import (
    aliasing_world,
    distractor_delay_world,
    feature_corruption_world,
    many_small_regimes_world,
    overlap_boundary_world,
    recurrence_world,
)


class AtlasBenchmarkTests(unittest.TestCase):
    def test_recurrence_world_reuses_charts(self) -> None:
        config = BenchmarkConfig(
            name="identity_l2_tight",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        )
        results = run_benchmark_suite(configs=[config], variants=[recurrence_world()])
        summary = results[0].summary
        self.assertEqual(summary.family, "recurrence")
        self.assertGreater(summary.spawn.reuse_rate, 0.5)
        self.assertLessEqual(summary.spawn.false_spawn_rate, 0.05)
        self.assertGreaterEqual(summary.spawn.chart_count, 2)

    def test_benchmark_runs_are_deterministic(self) -> None:
        config = BenchmarkConfig(
            name="reference_l2_strict",
            encoder_kind="reference",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=0.75,
            spawn_rank=2,
            default_radius=0.2,
            seed=11,
        )
        first = run_benchmark_suite(configs=[config], variants=[feature_corruption_world()])[0]
        second = run_benchmark_suite(configs=[config], variants=[feature_corruption_world()])[0]
        self.assertEqual(first.summary.spawn.chart_count, second.summary.spawn.chart_count)
        self.assertAlmostEqual(
            first.summary.contradiction.mean_contradiction,
            second.summary.contradiction.mean_contradiction,
            places=6,
        )

    def test_aliasing_world_reduces_chart_purity(self) -> None:
        config = BenchmarkConfig(
            name="identity_l2_tight",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        )
        result = run_benchmark_suite(configs=[config], variants=[aliasing_world()])[0]
        self.assertLess(result.summary.purity.chart_purity, 1.0)
        self.assertTrue(result.summary.purity.single_chart_collapse)

    def test_delayed_recurrence_reuses_under_strict_config(self) -> None:
        config = BenchmarkConfig(
            name="identity_l2_strict",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=0.75,
            spawn_rank=2,
            default_radius=0.2,
        )
        result = run_benchmark_suite(configs=[config], variants=[distractor_delay_world()])[0]
        self.assertEqual(result.summary.delay.delayed_false_spawn_rate, 0.0)
        self.assertGreater(result.summary.delay.delayed_reuse_rate, 0.0)

    def test_corruption_world_distinguishes_broad_from_strict(self) -> None:
        strict = BenchmarkConfig(
            name="identity_l2_strict",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=0.75,
            spawn_rank=2,
            default_radius=0.2,
        )
        broad = BenchmarkConfig(
            name="identity_cosine_broad",
            encoder_kind="identity",
            retriever_metric="cosine",
            top_k=3,
            spawn_threshold=1.3,
            spawn_rank=2,
            default_radius=0.55,
        )
        strict_result = run_benchmark_suite(configs=[strict], variants=[feature_corruption_world()])[0]
        broad_result = run_benchmark_suite(configs=[broad], variants=[feature_corruption_world()])[0]
        self.assertGreaterEqual(
            strict_result.summary.corruption.corruption_spawn_rate,
            broad_result.summary.corruption.corruption_spawn_rate,
        )

    def test_many_small_regimes_reference_no_longer_collapses(self) -> None:
        config = BenchmarkConfig(
            name="reference_l2_strict",
            encoder_kind="reference",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=0.75,
            spawn_rank=2,
            default_radius=0.2,
        )
        result = run_benchmark_suite(configs=[config], variants=[many_small_regimes_world()])[0]
        self.assertFalse(result.summary.purity.single_chart_collapse)
        self.assertEqual(result.summary.purity.undersegmentation_ratio, 0.0)

    def test_report_line_contains_core_fields(self) -> None:
        config = BenchmarkConfig(
            name="identity_l2_tight",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        )
        result = run_benchmark_suite(configs=[config], variants=[overlap_boundary_world()])[0]
        line = format_result_line(result)
        self.assertIn("contr=", line)
        self.assertIn("spawn=", line)
        self.assertIn("purity=", line)
        self.assertIn("ambig=", line)
        self.assertIn("trust=", line)
        self.assertIn("calib=", line)
        self.assertIn("samp=", line)
        self.assertIn("gctr=", line)
        self.assertIn("grad=", line)
        self.assertIn("collapse=", line)
        self.assertIn("charts=", line)

    def test_overlap_boundary_remains_hard(self) -> None:
        config = BenchmarkConfig(
            name="identity_l2_tight",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        )
        result = run_benchmark_suite(configs=[config], variants=[overlap_boundary_world()])[0]
        self.assertLess(result.summary.purity.chart_purity, 1.0)

    def test_recurrence_has_trusted_updates_and_geometry_motion(self) -> None:
        config = BenchmarkConfig(
            name="identity_l2_tight",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        )
        result = run_benchmark_suite(configs=[config], variants=[recurrence_world()])[0]
        self.assertGreater(result.summary.decision.trusted_calibration_update_rate, 0.0)
        self.assertGreater(result.summary.geometry.geometry_update_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
