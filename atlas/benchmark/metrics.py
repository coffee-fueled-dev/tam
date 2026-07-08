"""Internal metrics for atlas benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

Tensor = torch.Tensor


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass
class ContradictionMetrics:
    mean_contradiction: float
    support_rate: float
    support_score_mean: float


@dataclass
class SpawnMetrics:
    spawn_rate: float
    false_spawn_rate: float
    chart_count: int
    chart_churn: float
    reuse_rate: float


@dataclass
class RetrievalMetrics:
    top_k_retrieval_hit: float
    nearest_chart_confusion: float


@dataclass
class CompactnessMetrics:
    chart_separability: float
    within_chart_coherence: float


@dataclass
class PurityMetrics:
    regime_purity: float
    chart_purity: float
    regime_fragmentation: float
    chart_overloading: float
    undersegmentation_ratio: float
    single_chart_collapse: bool


@dataclass
class GapMetrics:
    chosen_second_gap: float
    chosen_canonical_gap: float


@dataclass
class DelayMetrics:
    delayed_reuse_rate: float
    delayed_false_spawn_rate: float
    first_reuse_after_gap_latency: float


@dataclass
class CorruptionMetrics:
    corruption_spawn_rate: float
    same_regime_duplicate_chart_rate: float
    retrieval_stability_under_corruption: float


@dataclass
class CalibrationMetrics:
    mean_threshold: float
    mean_sample_count: float
    calibrated_decision_rate: float
    warmup_reuse_rate: float


@dataclass
class DecisionMetrics:
    ambiguous_rate: float
    trusted_calibration_update_rate: float


@dataclass
class GeometryMetrics:
    mean_center_shift: float
    mean_radius_shift: float
    geometry_update_rate: float


@dataclass
class AtlasBenchmarkSummary:
    contradiction: ContradictionMetrics
    spawn: SpawnMetrics
    retrieval: RetrievalMetrics
    compactness: CompactnessMetrics
    purity: PurityMetrics
    gaps: GapMetrics
    delay: DelayMetrics
    corruption: CorruptionMetrics
    calibration: CalibrationMetrics
    decision: DecisionMetrics
    geometry: GeometryMetrics
    family: str
    config_name: str


def summarize_contradiction(contradictions: Sequence[float], support_scores: Sequence[float]) -> ContradictionMetrics:
    support_rate = _mean(1.0 if score > 0.0 else 0.0 for score in support_scores)
    return ContradictionMetrics(
        mean_contradiction=_mean(contradictions),
        support_rate=support_rate,
        support_score_mean=_mean(support_scores),
    )


def summarize_spawn(
    spawned_flags: Sequence[bool],
    false_spawn_flags: Sequence[bool],
    regime_reuse_flags: Sequence[bool],
    chart_usage_counts: Dict[str, int],
    chart_count: int,
) -> SpawnMetrics:
    churn = 0.0
    if chart_usage_counts:
        churn = _mean(1.0 if count <= 1 else 0.0 for count in chart_usage_counts.values())
    return SpawnMetrics(
        spawn_rate=_mean(1.0 if flag else 0.0 for flag in spawned_flags),
        false_spawn_rate=_mean(1.0 if flag else 0.0 for flag in false_spawn_flags),
        chart_count=chart_count,
        chart_churn=churn,
        reuse_rate=_mean(1.0 if flag else 0.0 for flag in regime_reuse_flags),
    )


def summarize_retrieval(top_k_hits: Sequence[bool], nearest_confusions: Sequence[bool]) -> RetrievalMetrics:
    return RetrievalMetrics(
        top_k_retrieval_hit=_mean(1.0 if flag else 0.0 for flag in top_k_hits),
        nearest_chart_confusion=_mean(1.0 if flag else 0.0 for flag in nearest_confusions),
    )


def summarize_compactness(observations_by_chart: Dict[str, List[Tensor]]) -> CompactnessMetrics:
    means: List[Tensor] = []
    coherence: List[float] = []
    for observations in observations_by_chart.values():
        if not observations:
            continue
        stacked = torch.stack(observations, dim=0)
        mean_obs = stacked.mean(dim=0)
        means.append(mean_obs)
        coherence.append(float(torch.mean(torch.norm(stacked - mean_obs, dim=-1)).item()))

    separability = 0.0
    if len(means) > 1:
        distances: List[float] = []
        for first_idx in range(len(means)):
            for second_idx in range(first_idx + 1, len(means)):
                distances.append(float(torch.norm(means[first_idx] - means[second_idx]).item()))
        separability = _mean(distances)

    return CompactnessMetrics(
        chart_separability=separability,
        within_chart_coherence=_mean(coherence),
    )


def summarize_purity(
    regime_chart_counts: Dict[str, Dict[str, int]],
    chart_regime_counts: Dict[str, Dict[str, int]],
    chart_count: int,
) -> PurityMetrics:
    regime_purities: List[float] = []
    regime_fragmentation: List[float] = []
    for chart_counts in regime_chart_counts.values():
        total = sum(chart_counts.values())
        if total <= 0:
            continue
        dominant = max(chart_counts.values())
        regime_purities.append(dominant / total)
        regime_fragmentation.append(float(len(chart_counts)))

    chart_purities: List[float] = []
    chart_overloading: List[float] = []
    for regime_counts in chart_regime_counts.values():
        total = sum(regime_counts.values())
        if total <= 0:
            continue
        dominant = max(regime_counts.values())
        chart_purities.append(dominant / total)
        chart_overloading.append(float(max(len(regime_counts) - 1, 0)))

    regime_count = len(regime_chart_counts)
    undersegmentation_ratio = 0.0
    if regime_count > 0:
        undersegmentation_ratio = max(regime_count - chart_count, 0) / regime_count

    return PurityMetrics(
        regime_purity=_mean(regime_purities),
        chart_purity=_mean(chart_purities),
        regime_fragmentation=_mean(regime_fragmentation),
        chart_overloading=_mean(chart_overloading),
        undersegmentation_ratio=undersegmentation_ratio,
        single_chart_collapse=bool(chart_count == 1 and regime_count > 1),
    )


def summarize_gaps(
    chosen_second_gaps: Sequence[float],
    chosen_canonical_gaps: Sequence[float],
) -> GapMetrics:
    return GapMetrics(
        chosen_second_gap=_mean(chosen_second_gaps),
        chosen_canonical_gap=_mean(chosen_canonical_gaps),
    )


def summarize_delay(
    delayed_reuse_flags: Sequence[bool],
    delayed_false_spawn_flags: Sequence[bool],
    first_reuse_latencies: Sequence[float],
) -> DelayMetrics:
    return DelayMetrics(
        delayed_reuse_rate=_mean(1.0 if flag else 0.0 for flag in delayed_reuse_flags),
        delayed_false_spawn_rate=_mean(1.0 if flag else 0.0 for flag in delayed_false_spawn_flags),
        first_reuse_after_gap_latency=_mean(first_reuse_latencies),
    )


def summarize_corruption(
    corruption_spawn_flags: Sequence[bool],
    corrupted_regime_chart_counts: Dict[str, Dict[str, int]],
    retrieval_stability_flags: Sequence[bool],
) -> CorruptionMetrics:
    duplicate_rates: List[float] = []
    for chart_counts in corrupted_regime_chart_counts.values():
        duplicate_rates.append(float(max(len(chart_counts) - 1, 0)))
    return CorruptionMetrics(
        corruption_spawn_rate=_mean(1.0 if flag else 0.0 for flag in corruption_spawn_flags),
        same_regime_duplicate_chart_rate=_mean(duplicate_rates),
        retrieval_stability_under_corruption=_mean(
            1.0 if flag else 0.0 for flag in retrieval_stability_flags
        ),
    )


def summarize_calibration(
    thresholds: Sequence[float],
    sample_counts: Sequence[float],
    calibrated_flags: Sequence[bool],
    warmup_reuse_flags: Sequence[bool],
) -> CalibrationMetrics:
    return CalibrationMetrics(
        mean_threshold=_mean(thresholds),
        mean_sample_count=_mean(sample_counts),
        calibrated_decision_rate=_mean(1.0 if flag else 0.0 for flag in calibrated_flags),
        warmup_reuse_rate=_mean(1.0 if flag else 0.0 for flag in warmup_reuse_flags),
    )


def summarize_decision(
    ambiguous_flags: Sequence[bool],
    trusted_update_flags: Sequence[bool],
) -> DecisionMetrics:
    return DecisionMetrics(
        ambiguous_rate=_mean(1.0 if flag else 0.0 for flag in ambiguous_flags),
        trusted_calibration_update_rate=_mean(1.0 if flag else 0.0 for flag in trusted_update_flags),
    )


def summarize_geometry(
    center_shifts: Sequence[float],
    radius_shifts: Sequence[float],
) -> GeometryMetrics:
    return GeometryMetrics(
        mean_center_shift=_mean(center_shifts),
        mean_radius_shift=_mean(radius_shifts),
        geometry_update_rate=_mean(
            1.0 if (center > 0.0 or radius > 0.0) else 0.0
            for center, radius in zip(center_shifts, radius_shifts)
        ),
    )
