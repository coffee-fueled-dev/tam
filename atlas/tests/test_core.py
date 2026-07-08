import tempfile
import unittest
from pathlib import Path

import torch

from atlas.core import ChartRecord
from atlas.projector import local_contradiction
from atlas.store import AtlasStore


class AtlasCoreTests(unittest.TestCase):
    def test_chart_record_normalizes_shapes(self) -> None:
        chart = ChartRecord(
            chart_id="chart_0",
            retrieval_key=torch.tensor([[1.0, 2.0]]),
            projection_basis=torch.tensor([1.0, 0.0, 0.0, 1.0]),
            local_center=torch.tensor([[0.0, 0.0]]),
            local_radius=torch.tensor([0.5]),
        )
        self.assertEqual(tuple(chart.retrieval_key.shape), (2,))
        self.assertEqual(tuple(chart.projection_basis.shape), (2, 2))
        self.assertEqual(tuple(chart.local_center.shape), (2,))
        self.assertEqual(tuple(chart.local_radius.shape), (2,))

    def test_local_contradiction_prefers_matching_point(self) -> None:
        center = torch.tensor([0.0, 0.0])
        radius = torch.tensor([0.5, 0.5])
        near = local_contradiction(torch.tensor([0.1, -0.1]), center, radius)
        far = local_contradiction(torch.tensor([2.0, 2.0]), center, radius)
        self.assertLess(float(near.item()), float(far.item()))

    def test_store_save_and_load_preserves_identity(self) -> None:
        store = AtlasStore()
        chart = store.create_chart(
            retrieval_key=torch.tensor([1.0, 1.0]),
            projection_basis=torch.eye(2),
            local_center=torch.tensor([1.0, 1.0]),
            local_radius=torch.tensor([0.5, 0.5]),
            parent_id="seed",
            created_step=3,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "atlas.pt"
            store.save(path)
            loaded = AtlasStore.load(path)
        loaded_chart = loaded.get_chart(chart.chart_id)
        self.assertEqual(loaded_chart.chart_id, chart.chart_id)
        self.assertEqual(loaded_chart.lineage.parent_id, "seed")
        self.assertEqual(loaded_chart.lineage.created_step, 3)

    def test_store_save_and_load_preserves_calibration(self) -> None:
        store = AtlasStore()
        chart = store.create_chart(
            retrieval_key=torch.tensor([1.0, 1.0]),
            projection_basis=torch.eye(2),
            local_center=torch.tensor([1.0, 1.0]),
            local_radius=torch.tensor([0.5, 0.5]),
        )
        store.observe_calibration(chart.chart_id, 0.05)
        store.update_calibration(chart.chart_id, 0.1)
        store.update_calibration(chart.chart_id, 0.2)
        store.adapt_chart_geometry(chart.chart_id, torch.tensor([1.1, 0.9]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "atlas.pt"
            store.save(path)
            loaded = AtlasStore.load(path)
        loaded_chart = loaded.get_chart(chart.chart_id)
        self.assertEqual(loaded_chart.calibration.sample_count, 2)
        self.assertEqual(loaded_chart.calibration.observed_sample_count, 3)
        self.assertAlmostEqual(loaded_chart.calibration.residual_history[-1], 0.2, places=6)
        self.assertNotEqual(float(loaded_chart.local_center[0].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
