"""Chart-local projection and contradiction scoring."""

from __future__ import annotations

import torch

from atlas.core import ChartRecord, InferredSituation, PortView, ProjectedChart, ensure_1d

Tensor = torch.Tensor


def project_local_coords(situation_latent: Tensor, chart: ChartRecord) -> Tensor:
    """Project a latent situation into one chart's local coordinates."""
    latent = ensure_1d(situation_latent).to(chart.projection_basis.device)
    basis = chart.projection_basis
    if basis.numel() == 0 or basis.shape[-1] == 0:
        return torch.zeros(0, dtype=torch.float32, device=latent.device)
    return torch.linalg.pinv(basis) @ latent


def local_contradiction(local_coords: Tensor, local_center: Tensor, local_radius: Tensor, eps: float = 1e-6) -> Tensor:
    """Measure mismatch in one chart's local coordinate system."""
    if local_coords.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=local_center.device)
    residual = local_coords - local_center
    scaled = residual / (local_radius + eps)
    return torch.sqrt(torch.mean(scaled**2))


class ChartProjector:
    """Project retrieved charts into local coordinates and score fit."""

    def __init__(self, default_threshold: float = 1.0, confidence_scale: float = 6.0):
        self.default_threshold = default_threshold
        self.confidence_scale = confidence_scale

    def project(self, inferred: InferredSituation, chart: ChartRecord) -> ProjectedChart:
        local_coords = project_local_coords(inferred.situation_latent, chart)
        contradiction = local_contradiction(local_coords, chart.local_center, chart.local_radius)
        fallback_threshold = chart.support_threshold if chart.support_threshold > 0.0 else self.default_threshold
        threshold = chart.calibration.threshold(fallback_threshold)
        threshold_tensor = torch.tensor(threshold, dtype=torch.float32, device=contradiction.device)
        margin = threshold_tensor - contradiction
        p_value = chart.calibration.p_value(float(contradiction.item()))
        used_calibration = chart.calibration.ready
        p_value_tensor = torch.tensor(p_value, dtype=torch.float32, device=contradiction.device)
        support_score = torch.clamp(margin, min=0.0)
        support_confidence = (
            p_value_tensor
            if used_calibration
            else torch.sigmoid(self.confidence_scale * margin)
        )
        port_view = PortView(
            chart_id=chart.chart_id,
            port_name=chart.chart_id,
            support_score=support_score,
            contradiction=contradiction,
            raw_residual=contradiction,
            local_coords=local_coords,
            local_center=chart.local_center,
            local_radius=chart.local_radius,
            support_confidence=support_confidence,
            calibrated_threshold=threshold_tensor,
            calibration_margin=margin,
            calibration_p_value=p_value_tensor,
            calibration_sample_count=chart.calibration.sample_count,
            observed_calibration_sample_count=chart.calibration.observed_sample_count,
            used_calibration=used_calibration,
        )
        return ProjectedChart(
            chart=chart,
            local_coords=local_coords,
            support_score=support_score,
            contradiction=contradiction,
            raw_residual=contradiction,
            support_confidence=support_confidence,
            calibrated_threshold=threshold_tensor,
            calibration_margin=margin,
            calibration_p_value=p_value_tensor,
            calibration_sample_count=chart.calibration.sample_count,
            observed_calibration_sample_count=chart.calibration.observed_sample_count,
            used_calibration=used_calibration,
            port_view=port_view,
        )
