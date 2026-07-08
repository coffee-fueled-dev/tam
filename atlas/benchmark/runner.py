"""Runner for atlas stress-test benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import torch

from atlas.benchmark.metrics import (
    AtlasBenchmarkSummary,
    summarize_calibration,
    summarize_corruption,
    summarize_compactness,
    summarize_contradiction,
    summarize_decision,
    summarize_delay,
    summarize_geometry,
    summarize_gaps,
    summarize_purity,
    summarize_retrieval,
    summarize_spawn,
)
from atlas.benchmark.worlds import BenchmarkWorld, default_variants
from atlas.encoder import IdentitySituationEncoder, ReferenceSituationEncoder
from atlas.projector import ChartProjector
from atlas.retrieval import ChartRetriever
from atlas.runtime import AtlasRuntime
from atlas.store import AtlasStore
from atlas.training import train_reference_encoder_on_events

EncoderKind = Literal["identity", "reference"]


@dataclass
class BenchmarkConfig:
    name: str
    encoder_kind: EncoderKind
    retriever_metric: str
    top_k: int
    spawn_threshold: float
    spawn_rank: int
    default_radius: float
    seed: int = 7


@dataclass
class BenchmarkResult:
    family: str
    config: BenchmarkConfig
    summary: AtlasBenchmarkSummary


def default_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            name="identity_l2_tight",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        ),
        BenchmarkConfig(
            name="identity_l2_strict",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=0.75,
            spawn_rank=2,
            default_radius=0.2,
        ),
        BenchmarkConfig(
            name="identity_l2_wide_topk",
            encoder_kind="identity",
            retriever_metric="l2",
            top_k=5,
            spawn_threshold=1.0,
            spawn_rank=2,
            default_radius=0.35,
        ),
        BenchmarkConfig(
            name="identity_cosine_broad",
            encoder_kind="identity",
            retriever_metric="cosine",
            top_k=3,
            spawn_threshold=1.3,
            spawn_rank=2,
            default_radius=0.55,
        ),
        BenchmarkConfig(
            name="reference_l2_strict",
            encoder_kind="reference",
            retriever_metric="l2",
            top_k=3,
            spawn_threshold=0.75,
            spawn_rank=2,
            default_radius=0.2,
        ),
    ]


def _build_runtime(config: BenchmarkConfig, input_dim: int, events) -> AtlasRuntime:
    torch.manual_seed(config.seed)
    if config.encoder_kind == "identity":
        encoder = IdentitySituationEncoder()
    else:
        encoder = ReferenceSituationEncoder(
            input_dim=input_dim,
            latent_dim=input_dim,
            query_dim=input_dim,
            feature_dropout=0.05,
        )
        train_reference_encoder_on_events(encoder, events, seed=config.seed)
    return AtlasRuntime(
        encoder=encoder,
        store=AtlasStore(),
        retriever=ChartRetriever(metric=config.retriever_metric),
        projector=ChartProjector(default_threshold=config.spawn_threshold),
        top_k=config.top_k,
        spawn_threshold=config.spawn_threshold,
        spawn_rank=config.spawn_rank,
        default_radius=config.default_radius,
    )


def _run_one(world: BenchmarkWorld, config: BenchmarkConfig) -> BenchmarkResult:
    events = world.sequence()
    if not events:
        raise ValueError("Benchmark world emitted no events")

    input_dim = int(events[0].observation.reshape(-1).shape[-1])
    runtime = _build_runtime(config, input_dim=input_dim, events=events)

    contradictions: List[float] = []
    support_scores: List[float] = []
    spawned_flags: List[bool] = []
    false_spawn_flags: List[bool] = []
    regime_reuse_flags: List[bool] = []
    top_k_hits: List[bool] = []
    nearest_confusions: List[bool] = []
    chosen_second_gaps: List[float] = []
    chosen_canonical_gaps: List[float] = []
    delayed_reuse_flags: List[bool] = []
    delayed_false_spawn_flags: List[bool] = []
    first_reuse_latencies: List[float] = []
    corruption_spawn_flags: List[bool] = []
    corruption_retrieval_stability: List[bool] = []
    calibration_thresholds: List[float] = []
    calibration_sample_counts: List[float] = []
    calibrated_decision_flags: List[bool] = []
    warmup_reuse_flags: List[bool] = []
    ambiguous_flags: List[bool] = []
    trusted_update_flags: List[bool] = []
    geometry_center_shifts: List[float] = []
    geometry_radius_shifts: List[float] = []
    observations_by_chart: Dict[str, List[torch.Tensor]] = {}
    chart_usage_counts: Dict[str, int] = {}
    canonical_chart_by_regime: Dict[str, str] = {}
    regime_chart_counts: Dict[str, Dict[str, int]] = {}
    chart_regime_counts: Dict[str, Dict[str, int]] = {}
    corrupted_regime_chart_counts: Dict[str, Dict[str, int]] = {}
    pending_delay_by_regime: Dict[str, int] = {}

    for event_idx, event in enumerate(events):
        result = runtime.step(event.observation, support_example=f"{world.family}_{event_idx}")
        chart_id = result.chosen_port.chart_id
        chart_usage_counts[chart_id] = chart_usage_counts.get(chart_id, 0) + 1
        observations_by_chart.setdefault(chart_id, []).append(event.observation.reshape(-1).to(dtype=torch.float32))
        regime_chart_counts.setdefault(event.regime_name, {})
        regime_chart_counts[event.regime_name][chart_id] = regime_chart_counts[event.regime_name].get(chart_id, 0) + 1
        chart_regime_counts.setdefault(chart_id, {})
        chart_regime_counts[chart_id][event.regime_name] = chart_regime_counts[chart_id].get(event.regime_name, 0) + 1

        contradiction = float(result.chosen_port.contradiction.item())
        support_score = float(result.chosen_port.support_score.item())
        contradictions.append(contradiction)
        support_scores.append(support_score)
        if result.chosen_port.calibrated_threshold is not None:
            calibration_thresholds.append(float(result.chosen_port.calibrated_threshold.item()))
        calibration_sample_counts.append(float(result.chosen_port.calibration_sample_count))
        calibrated_decision_flags.append(result.chosen_port.used_calibration)
        ambiguous_flags.append(result.decision == "ambiguous")
        trusted_update_flags.append(result.trusted_calibration_update)
        geometry_center_shifts.append(result.geometry_center_shift)
        geometry_radius_shifts.append(result.geometry_radius_shift)
        projected_contradictions = sorted(
            float(projected.contradiction.item()) for projected in result.projected_charts
        )
        if result.spawned_chart_id is not None:
            if projected_contradictions:
                chosen_second_gaps.append(projected_contradictions[0] - contradiction)
        elif len(projected_contradictions) > 1:
            chosen_second_gaps.append(projected_contradictions[1] - contradiction)

        seen_regime = event.regime_name in canonical_chart_by_regime
        canonical_chart = canonical_chart_by_regime.get(event.regime_name)
        spawned = result.spawned_chart_id is not None
        spawned_flags.append(spawned)
        false_spawn_flags.append(bool(spawned and seen_regime))
        warmup_reuse_flags.append(bool((not spawned) and (not result.chosen_port.used_calibration)))
        match_ids = [match.chart_id for match in result.matches]

        is_corrupted = bool(event.metadata.get("corrupted", 0.0) > 0.0)
        if is_corrupted:
            corruption_spawn_flags.append(spawned)
            corrupted_regime_chart_counts.setdefault(event.regime_name, {})
            corrupted_regime_chart_counts[event.regime_name][chart_id] = (
                corrupted_regime_chart_counts[event.regime_name].get(chart_id, 0) + 1
            )
            if seen_regime and canonical_chart is not None:
                corruption_retrieval_stability.append(canonical_chart in match_ids)

        if not seen_regime:
            canonical_chart_by_regime[event.regime_name] = chart_id
        else:
            regime_reuse_flags.append(chart_id == canonical_chart)
            top_k_hits.append(canonical_chart in match_ids)
            nearest_confusions.append(bool(result.matches) and result.matches[0].chart_id != canonical_chart)
            canonical_projected = next(
                (
                    float(projected.contradiction.item())
                    for projected in result.projected_charts
                    if projected.chart.chart_id == canonical_chart
                ),
                None,
            )
            if canonical_projected is not None:
                chosen_canonical_gaps.append(canonical_projected - contradiction)

        delay_gap = float(event.metadata.get("delay_gap", 0.0))
        if delay_gap > 0.0 and seen_regime:
            delayed_reuse = bool(chart_id == canonical_chart)
            delayed_reuse_flags.append(delayed_reuse)
            delayed_false_spawn_flags.append(spawned)
            if delayed_reuse:
                first_reuse_latencies.append(0.0)
            else:
                pending_delay_by_regime[event.regime_name] = 1
        elif event.regime_name in pending_delay_by_regime:
            if canonical_chart is not None and chart_id == canonical_chart:
                first_reuse_latencies.append(float(pending_delay_by_regime[event.regime_name]))
                del pending_delay_by_regime[event.regime_name]
            else:
                pending_delay_by_regime[event.regime_name] += 1

    for latency in pending_delay_by_regime.values():
        first_reuse_latencies.append(float(latency))

    summary = AtlasBenchmarkSummary(
        contradiction=summarize_contradiction(contradictions, support_scores),
        spawn=summarize_spawn(
            spawned_flags=spawned_flags,
            false_spawn_flags=false_spawn_flags,
            regime_reuse_flags=regime_reuse_flags,
            chart_usage_counts=chart_usage_counts,
            chart_count=len(runtime.store),
        ),
        retrieval=summarize_retrieval(top_k_hits=top_k_hits, nearest_confusions=nearest_confusions),
        compactness=summarize_compactness(observations_by_chart),
        purity=summarize_purity(
            regime_chart_counts=regime_chart_counts,
            chart_regime_counts=chart_regime_counts,
            chart_count=len(runtime.store),
        ),
        gaps=summarize_gaps(
            chosen_second_gaps=chosen_second_gaps,
            chosen_canonical_gaps=chosen_canonical_gaps,
        ),
        delay=summarize_delay(
            delayed_reuse_flags=delayed_reuse_flags,
            delayed_false_spawn_flags=delayed_false_spawn_flags,
            first_reuse_latencies=first_reuse_latencies,
        ),
        corruption=summarize_corruption(
            corruption_spawn_flags=corruption_spawn_flags,
            corrupted_regime_chart_counts=corrupted_regime_chart_counts,
            retrieval_stability_flags=corruption_retrieval_stability,
        ),
        calibration=summarize_calibration(
            thresholds=calibration_thresholds,
            sample_counts=calibration_sample_counts,
            calibrated_flags=calibrated_decision_flags,
            warmup_reuse_flags=warmup_reuse_flags,
        ),
        decision=summarize_decision(
            ambiguous_flags=ambiguous_flags,
            trusted_update_flags=trusted_update_flags,
        ),
        geometry=summarize_geometry(
            center_shifts=geometry_center_shifts,
            radius_shifts=geometry_radius_shifts,
        ),
        family=world.family,
        config_name=config.name,
    )
    return BenchmarkResult(family=world.family, config=config, summary=summary)


def run_benchmark_suite(
    configs: List[BenchmarkConfig] | None = None,
    variants: List[BenchmarkWorld] | None = None,
) -> List[BenchmarkResult]:
    configs = configs if configs is not None else default_configs()
    variants = variants if variants is not None else default_variants()
    results: List[BenchmarkResult] = []
    for variant in variants:
        for config in configs:
            results.append(_run_one(variant, config))
    return results
