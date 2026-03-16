"""Smoke tests for the benchmark package."""

from __future__ import annotations

import unittest

import torch

from v4.benchmark.observation import dimension_feature_matrix, feature_order
from v4.benchmark.runner import BenchmarkConfig, BenchmarkVariant, run_benchmark_suite
from v4.benchmark.worlds import CorridorWorldConfig, DoorRegion, DriftZone, StructuredCorridorWorld


class BenchmarkTests(unittest.TestCase):
    def test_structured_world_emits_structured_observation(self) -> None:
        world = StructuredCorridorWorld(
            CorridorWorldConfig(
                state_dim=2,
                doors=[DoorRegion(x_center=3.0, half_width=0.5, opening_radius=0.4)],
                drift_zones=[
                    DriftZone(
                        start_x=1.0,
                        end_x=2.0,
                        drift=torch.tensor([0.0, 0.1], dtype=torch.float32),
                    )
                ],
            )
        )
        atom = world.observe(world.initial_observed_state())
        self.assertEqual(atom.kind, "structured_observation")
        self.assertIn("frame", atom.metadata)
        self.assertIn("physical_state", atom.metadata)

    def test_observation_builds_per_dimension_feature_matrix(self) -> None:
        world = StructuredCorridorWorld(CorridorWorldConfig(state_dim=3))
        atom = world.observe(world.initial_observed_state())
        frame = atom.metadata["frame"]
        matrix, names = dimension_feature_matrix(frame)
        self.assertEqual(matrix.shape[0], 3)
        self.assertGreater(matrix.shape[1], 0)
        self.assertEqual(len(names), matrix.shape[1])
        self.assertGreater(len(feature_order(frame)), 0)

    def test_benchmark_suite_runs_single_variant(self) -> None:
        variant = BenchmarkVariant(
            name="smoke_2d",
            world_config=CorridorWorldConfig(state_dim=2),
        )
        config = BenchmarkConfig(
            name="smoke_shared",
            encoder_mode="shared",
            geometry_mode="fiber",
            steps=5,
        )
        results = run_benchmark_suite(configs=[config], variants=[variant])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].summary.variant_name, "smoke_2d")
        self.assertEqual(results[0].summary.ablation_name, "smoke_shared")


if __name__ == "__main__":
    unittest.main()
