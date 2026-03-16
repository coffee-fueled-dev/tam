"""Metrics for the v4 benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from v4.benchmark.observation import feature_order_robustness

Tensor = torch.Tensor


@dataclass
class GeometryMetrics:
    mean_contradiction: float
    mean_radius: float
    coverage: float
    confidence_calibration: float


@dataclass
class PortMetrics:
    port_reuse_consistency: float
    inter_port_separability: float
    intra_port_coherence: float


@dataclass
class TaskMetrics:
    success_rate: float
    claim_success_rate: float
    final_progress: float
    sample_efficiency: float


@dataclass
class RepresentationMetrics:
    mean_feature_variation: float
    feature_order_robustness: float


@dataclass
class BenchmarkSummary:
    geometry: GeometryMetrics
    ports: PortMetrics
    task: TaskMetrics
    representation: RepresentationMetrics
    ablation_name: str
    variant_name: str


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def summarize_geometry(
    contradictions: List[float],
    radii: List[float],
    claim_successes: List[bool],
    confidences: List[float],
) -> GeometryMetrics:
    coverage = _mean(1.0 if success else 0.0 for success in claim_successes)
    confidence_calibration = 0.0
    if confidences and contradictions:
        confidence_tensor = torch.tensor(confidences, dtype=torch.float32)
        contradiction_tensor = torch.tensor(contradictions, dtype=torch.float32)
        if confidence_tensor.numel() > 1:
            confidence_calibration = float(
                torch.corrcoef(torch.stack([confidence_tensor, -contradiction_tensor]))[0, 1].item()
            )
            if not torch.isfinite(torch.tensor(confidence_calibration)):
                confidence_calibration = 0.0
    return GeometryMetrics(
        mean_contradiction=_mean(contradictions),
        mean_radius=_mean(radii),
        coverage=coverage,
        confidence_calibration=confidence_calibration,
    )


def summarize_ports(selected_ports: List[str], port_centers: Dict[str, List[Tensor]]) -> PortMetrics:
    total = len(selected_ports)
    if total == 0:
        return PortMetrics(0.0, 0.0, 0.0)

    counts: Dict[str, int] = {}
    for port in selected_ports:
        counts[port] = counts.get(port, 0) + 1

    port_reuse_consistency = max(counts.values()) / total

    coherence_scores: List[float] = []
    center_means: List[Tensor] = []
    for centers in port_centers.values():
        if not centers:
            continue
        stacked = torch.stack(centers, dim=0)
        mean_center = stacked.mean(dim=0)
        center_means.append(mean_center)
        coherence_scores.append(float(torch.mean(torch.norm(stacked - mean_center, dim=-1)).item()))

    inter_port_separability = 0.0
    if len(center_means) > 1:
        distances: List[float] = []
        for i in range(len(center_means)):
            for j in range(i + 1, len(center_means)):
                distances.append(float(torch.norm(center_means[i] - center_means[j]).item()))
        inter_port_separability = _mean(distances)

    return PortMetrics(
        port_reuse_consistency=port_reuse_consistency,
        inter_port_separability=inter_port_separability,
        intra_port_coherence=_mean(coherence_scores),
    )


def summarize_task(
    task_success_flags: List[bool],
    claim_success_flags: List[bool],
    progresses: List[float],
) -> TaskMetrics:
    success_rate = _mean(1.0 if flag else 0.0 for flag in task_success_flags)
    claim_success_rate = _mean(1.0 if flag else 0.0 for flag in claim_success_flags)
    final_progress = progresses[-1] if progresses else 0.0
    sample_efficiency = 0.0
    if task_success_flags:
        first_success_index = next(
            (idx for idx, flag in enumerate(task_success_flags) if flag),
            len(task_success_flags),
        )
        sample_efficiency = 1.0 - (first_success_index / max(len(task_success_flags), 1))
    return TaskMetrics(
        success_rate=success_rate,
        claim_success_rate=claim_success_rate,
        final_progress=final_progress,
        sample_efficiency=sample_efficiency,
    )


def summarize_representation(
    feature_matrices: List[Tensor],
    ordered_feature_names: List[List[str]],
) -> RepresentationMetrics:
    variation_values: List[float] = []
    for first, second in zip(feature_matrices, feature_matrices[1:]):
        if first.shape == second.shape:
            variation_values.append(float(torch.mean(torch.abs(first - second)).item()))

    return RepresentationMetrics(
        mean_feature_variation=_mean(variation_values),
        feature_order_robustness=feature_order_robustness(ordered_feature_names),
    )
