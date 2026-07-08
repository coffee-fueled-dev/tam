"""Core objects for the atlas runtime.

The atlas package keeps one idea central:

- charts are the persistent objects
- ports are per-situation bindable views derived from charts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch

Tensor = torch.Tensor
AtlasDecision = Literal["reuse", "ambiguous", "spawn"]


def ensure_1d(tensor: Tensor) -> Tensor:
    """Normalize a tensor to a flat float vector."""
    return tensor.reshape(-1).to(dtype=torch.float32)


def ensure_2d(tensor: Tensor, rows: int) -> Tensor:
    """Normalize a tensor to `(rows, cols)` form."""
    tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() == 0:
        return torch.zeros(rows, 0, dtype=torch.float32, device=tensor.device)
    if tensor.dim() == 1:
        return tensor.view(rows, -1)
    return tensor


def empirical_quantile(values: List[float], quantile: float) -> float:
    """Compute a stable empirical quantile over a small residual buffer."""
    if not values:
        return 0.0
    clipped = min(max(quantile, 0.0), 1.0)
    if len(values) == 1:
        return float(values[0])
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, clipped).item())


@dataclass
class InferredSituation:
    """Encoder output used for retrieval and local chart geometry."""

    observed_context: Tensor
    situation_latent: Tensor
    query_key: Tensor
    novelty_hint: Optional[Tensor] = None
    confidence_hint: Optional[Tensor] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observed_context = ensure_1d(self.observed_context)
        self.situation_latent = ensure_1d(self.situation_latent)
        self.query_key = ensure_1d(self.query_key)
        if self.novelty_hint is not None:
            self.novelty_hint = ensure_1d(self.novelty_hint)
        if self.confidence_hint is not None:
            self.confidence_hint = ensure_1d(self.confidence_hint)


@dataclass
class ChartLineage:
    """Persistent structural identity information for one chart."""

    parent_id: Optional[str] = None
    split_from: Optional[str] = None
    merged_into: Optional[str] = None
    created_step: int = 0


@dataclass
class ChartStats:
    """Usage and quality statistics used by the atlas lifecycle."""

    usage_count: int = 0
    support_count: int = 0
    spawn_count: int = 0
    contradiction_ema: float = 0.0
    last_support_score: float = 0.0
    last_contradiction: float = 0.0

    def update(self, support_score: float, contradiction: float, ema_rate: float = 0.1) -> None:
        self.usage_count += 1
        if support_score > 0.0:
            self.support_count += 1
        self.last_support_score = support_score
        self.last_contradiction = contradiction
        self.contradiction_ema = (
            contradiction
            if self.usage_count == 1
            else ((1.0 - ema_rate) * self.contradiction_ema) + (ema_rate * contradiction)
        )


@dataclass
class ChartCalibration:
    """Empirical support calibration state for one chart."""

    residual_history: List[float] = field(default_factory=list)
    observed_residual_history: List[float] = field(default_factory=list)
    max_samples: int = 32
    min_samples: int = 3
    alpha: float = 0.1
    calibrated_threshold: float = 0.0

    def _append(self, history: List[float], residual: float) -> None:
        history.append(float(residual))
        if len(history) > self.max_samples:
            del history[:-self.max_samples]

    def add_observed(self, residual: float) -> None:
        self._append(self.observed_residual_history, residual)

    def add_residual(self, residual: float) -> None:
        self.add_observed(residual)
        self._append(self.residual_history, residual)

    @property
    def sample_count(self) -> int:
        return len(self.residual_history)

    @property
    def observed_sample_count(self) -> int:
        return len(self.observed_residual_history)

    @property
    def ready(self) -> bool:
        return self.sample_count >= self.min_samples

    def threshold(self, fallback_threshold: float) -> float:
        if not self.ready:
            return float(fallback_threshold)
        self.calibrated_threshold = empirical_quantile(self.residual_history, 1.0 - self.alpha)
        return self.calibrated_threshold

    def p_value(self, residual: float) -> float:
        if not self.residual_history:
            return 1.0
        larger = sum(1 for item in self.residual_history if item >= residual)
        return float((larger + 1) / (self.sample_count + 1))


@dataclass
class ChartRecord:
    """One persistent local chart in the atlas."""

    chart_id: str
    retrieval_key: Tensor
    projection_basis: Tensor
    local_center: Tensor
    local_radius: Tensor
    support_threshold: float = 0.0
    stats: ChartStats = field(default_factory=ChartStats)
    calibration: ChartCalibration = field(default_factory=ChartCalibration)
    lineage: ChartLineage = field(default_factory=ChartLineage)
    support_examples: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.retrieval_key = ensure_1d(self.retrieval_key)
        latent_dim = int(self.retrieval_key.shape[-1])
        self.projection_basis = ensure_2d(self.projection_basis, latent_dim)
        local_dim = int(self.projection_basis.shape[-1])
        if local_dim == 0:
            self.local_center = torch.zeros(0, dtype=torch.float32, device=self.retrieval_key.device)
            self.local_radius = torch.zeros(0, dtype=torch.float32, device=self.retrieval_key.device)
        else:
            self.local_center = ensure_1d(self.local_center)
            self.local_radius = ensure_1d(self.local_radius)
            if self.local_radius.numel() == 1:
                self.local_radius = self.local_radius.repeat(local_dim)

    def adapt_geometry(
        self,
        local_coords: Tensor,
        center_ema: float = 0.25,
        radius_ema: float = 0.25,
        radius_floor: float = 0.05,
    ) -> tuple[float, float]:
        """Update local geometry toward a trusted reuse observation."""
        coords = ensure_1d(local_coords).to(self.local_center.device)
        if coords.numel() == 0:
            return 0.0, 0.0
        old_center = self.local_center.detach().clone()
        old_radius = self.local_radius.detach().clone()
        new_center = ((1.0 - center_ema) * old_center) + (center_ema * coords)
        residual = torch.abs(coords - new_center)
        # Keep radius adaptation conservative: expand toward new evidence, but only shrink slowly.
        target_radius = torch.clamp(1.5 * residual, min=radius_floor)
        ema_radius = ((1.0 - radius_ema) * old_radius) + (radius_ema * target_radius)
        min_radius = torch.clamp(old_radius * 0.95, min=radius_floor)
        new_radius = torch.maximum(ema_radius, min_radius)
        new_radius = torch.clamp(new_radius, min=radius_floor)
        self.local_center = new_center
        self.local_radius = new_radius
        center_shift = float(torch.norm(new_center - old_center).item())
        radius_shift = float(torch.mean(torch.abs(new_radius - old_radius)).item())
        return center_shift, radius_shift


@dataclass
class PortView:
    """One bindable port derived from an active chart."""

    chart_id: str
    port_name: str
    support_score: Tensor
    contradiction: Tensor
    raw_residual: Tensor
    local_coords: Tensor
    local_center: Tensor
    local_radius: Tensor
    support_confidence: Optional[Tensor] = None
    calibrated_threshold: Optional[Tensor] = None
    calibration_margin: Optional[Tensor] = None
    calibration_p_value: Optional[Tensor] = None
    calibration_sample_count: int = 0
    observed_calibration_sample_count: int = 0
    used_calibration: bool = False
    decision: AtlasDecision = "reuse"
    trusted_for_calibration: bool = False
    selection_score: Optional[Tensor] = None
    ambiguity: Optional[Tensor] = None

    def is_supported(self) -> bool:
        return float(self.support_score.item()) > 0.0


@dataclass
class ProjectedChart:
    """A chart projected into the current situation."""

    chart: ChartRecord
    local_coords: Tensor
    support_score: Tensor
    contradiction: Tensor
    raw_residual: Tensor
    support_confidence: Optional[Tensor]
    port_view: PortView
    calibrated_threshold: Optional[Tensor] = None
    calibration_margin: Optional[Tensor] = None
    calibration_p_value: Optional[Tensor] = None
    calibration_sample_count: int = 0
    observed_calibration_sample_count: int = 0
    used_calibration: bool = False
    trusted_for_calibration: bool = False
    selection_score: Optional[Tensor] = None


@dataclass
class RetrievalMatch:
    """One chart returned by retrieval."""

    chart_id: str
    distance: float
    score: float
