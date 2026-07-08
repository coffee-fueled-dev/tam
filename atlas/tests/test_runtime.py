import unittest

import torch

from atlas.encoder import IdentitySituationEncoder
from atlas.projector import ChartProjector
from atlas.retrieval import ChartRetriever
from atlas.runtime import AtlasRuntime
from atlas.store import AtlasStore


class AtlasRuntimeTests(unittest.TestCase):
    def make_runtime(self) -> AtlasRuntime:
        return AtlasRuntime(
            encoder=IdentitySituationEncoder(),
            store=AtlasStore(),
            retriever=ChartRetriever(metric="l2"),
            projector=ChartProjector(default_threshold=1.0),
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        )

    def test_retrieval_returns_nearest_chart(self) -> None:
        runtime = self.make_runtime()
        runtime.store.create_chart(
            retrieval_key=torch.tensor([0.0, 0.0]),
            projection_basis=torch.eye(2),
            local_center=torch.tensor([0.0, 0.0]),
            local_radius=torch.tensor([0.5, 0.5]),
        )
        near = runtime.store.create_chart(
            retrieval_key=torch.tensor([1.0, 1.0]),
            projection_basis=torch.eye(2),
            local_center=torch.tensor([1.0, 1.0]),
            local_radius=torch.tensor([0.5, 0.5]),
        )
        matches = runtime.retriever.retrieve(torch.tensor([1.1, 0.9]), runtime.store.all_charts(), top_k=1)
        self.assertEqual(matches[0].chart_id, near.chart_id)

    def test_runtime_spawns_then_reuses_chart(self) -> None:
        runtime = self.make_runtime()
        first = runtime.step(torch.tensor([1.0, 1.0]))
        second = runtime.step(torch.tensor([1.1, 0.95]))
        self.assertIsNotNone(first.spawned_chart_id)
        self.assertIsNone(second.spawned_chart_id)
        self.assertEqual(len(runtime.store), 1)
        self.assertEqual(second.chosen_port.chart_id, first.chosen_port.chart_id)

    def test_runtime_spawns_new_chart_for_far_observation(self) -> None:
        runtime = self.make_runtime()
        runtime.step(torch.tensor([1.0, 1.0]))
        result = runtime.step(torch.tensor([4.0, 4.0]))
        self.assertIsNotNone(result.spawned_chart_id)
        self.assertEqual(len(runtime.store), 2)

    def test_runtime_reuse_updates_calibration_history(self) -> None:
        runtime = self.make_runtime()
        first = runtime.step(torch.tensor([1.0, 1.0]))
        self.assertIsNotNone(first.spawned_chart_id)
        runtime.step(torch.tensor([1.05, 0.95]))
        runtime.step(torch.tensor([1.04, 0.96]))
        runtime.step(torch.tensor([1.03, 0.97]))
        chart = runtime.store.get_chart(first.chosen_port.chart_id)
        self.assertGreaterEqual(chart.calibration.sample_count, 3)
        self.assertTrue(chart.calibration.ready)

    def test_runtime_trusted_reuse_adapts_geometry(self) -> None:
        runtime = self.make_runtime()
        first = runtime.step(torch.tensor([1.0, 1.0]))
        chart = runtime.store.get_chart(first.chosen_port.chart_id)
        before_center = chart.local_center.detach().clone()
        before_radius = chart.local_radius.detach().clone()
        result = runtime.step(torch.tensor([1.05, 0.95]))
        chart = runtime.store.get_chart(first.chosen_port.chart_id)
        self.assertEqual(result.decision, "reuse")
        self.assertTrue(result.trusted_calibration_update)
        self.assertGreater(result.geometry_center_shift, 0.0)
        self.assertGreater(result.geometry_radius_shift, 0.0)
        self.assertFalse(torch.equal(chart.local_center, before_center))
        self.assertFalse(torch.equal(chart.local_radius, before_radius))

    def test_runtime_spawns_when_calibrated_chart_is_out_of_support(self) -> None:
        runtime = self.make_runtime()
        chart = runtime.store.create_chart(
            retrieval_key=torch.tensor([1.0, 1.0]),
            projection_basis=torch.eye(2),
            local_center=torch.tensor([1.0, 1.0]),
            local_radius=torch.tensor([0.35, 0.35]),
        )
        runtime.store.update_calibration(chart.chart_id, 0.02)
        runtime.store.update_calibration(chart.chart_id, 0.04)
        runtime.store.update_calibration(chart.chart_id, 0.05)
        result = runtime.step(torch.tensor([2.0, 2.0]))
        self.assertIsNotNone(result.spawned_chart_id)
        self.assertEqual(len(runtime.store), 2)

    def test_runtime_marks_ambiguous_without_spawning_or_trusting_calibration(self) -> None:
        runtime = self.make_runtime()
        query = runtime.encoder.infer(torch.tensor([1.0, 1.0])).situation_latent
        runtime.store.create_chart(
            retrieval_key=query,
            projection_basis=torch.eye(2),
            local_center=query + torch.tensor([0.01, 0.0]),
            local_radius=torch.tensor([0.2, 0.2]),
        )
        runtime.store.create_chart(
            retrieval_key=query,
            projection_basis=torch.eye(2),
            local_center=query + torch.tensor([-0.01, 0.0]),
            local_radius=torch.tensor([0.2, 0.2]),
        )
        result = runtime.step(torch.tensor([1.0, 1.0]))
        chart = runtime.store.get_chart(result.chosen_port.chart_id)
        self.assertEqual(result.decision, "ambiguous")
        self.assertIsNone(result.spawned_chart_id)
        self.assertFalse(result.trusted_calibration_update)
        self.assertEqual(chart.calibration.sample_count, 0)
        self.assertEqual(chart.calibration.observed_sample_count, 1)


if __name__ == "__main__":
    unittest.main()
