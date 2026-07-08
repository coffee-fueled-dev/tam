"""Persistent chart registry for the atlas runtime."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from atlas.core import ChartCalibration, ChartLineage, ChartRecord, ChartStats

Tensor = torch.Tensor


def _chart_to_state(chart: ChartRecord) -> dict:
    return {
        "chart_id": chart.chart_id,
        "retrieval_key": chart.retrieval_key.detach().cpu(),
        "projection_basis": chart.projection_basis.detach().cpu(),
        "local_center": chart.local_center.detach().cpu(),
        "local_radius": chart.local_radius.detach().cpu(),
        "support_threshold": chart.support_threshold,
        "stats": asdict(chart.stats),
        "calibration": asdict(chart.calibration),
        "lineage": asdict(chart.lineage),
        "support_examples": list(chart.support_examples),
    }


def _chart_from_state(state: dict) -> ChartRecord:
    return ChartRecord(
        chart_id=state["chart_id"],
        retrieval_key=state["retrieval_key"],
        projection_basis=state["projection_basis"],
        local_center=state["local_center"],
        local_radius=state["local_radius"],
        support_threshold=float(state.get("support_threshold", 0.0)),
        stats=ChartStats(**state.get("stats", {})),
        calibration=ChartCalibration(**state.get("calibration", {})),
        lineage=ChartLineage(**state.get("lineage", {})),
        support_examples=list(state.get("support_examples", [])),
    )


class AtlasStore:
    """Simple persistent registry of charts."""

    def __init__(self):
        self._charts: Dict[str, ChartRecord] = {}
        self._next_chart_index = 0

    def __len__(self) -> int:
        return len(self._charts)

    def chart_ids(self) -> list[str]:
        return sorted(self._charts.keys())

    def all_charts(self) -> list[ChartRecord]:
        return [self._charts[chart_id] for chart_id in self.chart_ids()]

    def get_chart(self, chart_id: str) -> ChartRecord:
        return self._charts[chart_id]

    def add_chart(self, chart: ChartRecord) -> ChartRecord:
        self._charts[chart.chart_id] = chart
        self._next_chart_index = max(self._next_chart_index, self._index_from_id(chart.chart_id) + 1)
        return chart

    def create_chart(
        self,
        retrieval_key: Tensor,
        projection_basis: Tensor,
        local_center: Tensor,
        local_radius: Tensor,
        support_threshold: float = 0.0,
        created_step: int = 0,
        parent_id: Optional[str] = None,
        split_from: Optional[str] = None,
        support_examples: Optional[list[str]] = None,
    ) -> ChartRecord:
        chart = ChartRecord(
            chart_id=self.next_chart_id(),
            retrieval_key=retrieval_key,
            projection_basis=projection_basis,
            local_center=local_center,
            local_radius=local_radius,
            support_threshold=support_threshold,
            lineage=ChartLineage(
                parent_id=parent_id,
                split_from=split_from,
                created_step=created_step,
            ),
            support_examples=list(support_examples or []),
        )
        return self.add_chart(chart)

    def update_chart(self, chart: ChartRecord) -> None:
        if chart.chart_id not in self._charts:
            raise KeyError(f"Unknown chart_id: {chart.chart_id}")
        self._charts[chart.chart_id] = chart

    def update_stats(
        self,
        chart_id: str,
        support_score: float,
        contradiction: float,
        support_example: Optional[str] = None,
    ) -> None:
        chart = self.get_chart(chart_id)
        chart.stats.update(support_score=support_score, contradiction=contradiction)
        if support_example is not None:
            chart.support_examples.append(support_example)

    def increment_spawn(self, chart_id: str) -> None:
        chart = self.get_chart(chart_id)
        chart.stats.spawn_count += 1

    def observe_calibration(self, chart_id: str, residual: float) -> None:
        chart = self.get_chart(chart_id)
        chart.calibration.add_observed(residual)

    def update_calibration(self, chart_id: str, residual: float) -> None:
        chart = self.get_chart(chart_id)
        chart.calibration.add_residual(residual)

    def adapt_chart_geometry(
        self,
        chart_id: str,
        local_coords: Tensor,
        center_ema: float = 0.25,
        radius_ema: float = 0.25,
        radius_floor: float = 0.05,
    ) -> tuple[float, float]:
        chart = self.get_chart(chart_id)
        return chart.adapt_geometry(
            local_coords,
            center_ema=center_ema,
            radius_ema=radius_ema,
            radius_floor=radius_floor,
        )

    def calibration_threshold(self, chart_id: str, fallback_threshold: float) -> float:
        chart = self.get_chart(chart_id)
        return chart.calibration.threshold(fallback_threshold)

    def retrieval_strength(self, chart_id: str, distance: float) -> float:
        """Convert retrieval distance into a bounded prior."""
        del chart_id
        return 1.0 / (1.0 + max(distance, 0.0))

    def stability_prior(self, chart_id: str) -> float:
        """Estimate whether a chart has historically been reliable enough to reuse."""
        chart = self.get_chart(chart_id)
        stats = chart.stats
        usage_term = min(stats.usage_count / 8.0, 1.0)
        support_term = (stats.support_count / max(stats.usage_count, 1)) if stats.usage_count > 0 else 0.0
        contradiction_term = max(0.0, 1.0 - stats.contradiction_ema)
        spawn_penalty = min(stats.spawn_count / 4.0, 1.0)
        prior = (0.4 * usage_term) + (0.35 * support_term) + (0.35 * contradiction_term) - (0.2 * spawn_penalty)
        return max(0.0, min(prior, 1.0))

    def next_chart_id(self) -> str:
        chart_id = f"chart_{self._next_chart_index}"
        self._next_chart_index += 1
        return chart_id

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "next_chart_index": self._next_chart_index,
                "charts": [_chart_to_state(chart) for chart in self.all_charts()],
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AtlasStore":
        state = torch.load(Path(path), map_location="cpu")
        store = cls()
        store._next_chart_index = int(state.get("next_chart_index", 0))
        for chart_state in state.get("charts", []):
            store.add_chart(_chart_from_state(chart_state))
        return store

    def _index_from_id(self, chart_id: str) -> int:
        prefix = "chart_"
        if not chart_id.startswith(prefix):
            return -1
        suffix = chart_id[len(prefix) :]
        return int(suffix) if suffix.isdigit() else -1
