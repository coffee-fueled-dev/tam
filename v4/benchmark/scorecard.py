"""Scorecard reporting for benchmark results.

This module turns raw benchmark outputs into a small set of judgments:

- theory_pass
- geometry_pass
- transfer_pass
- use_case_pass
- overall

Important current limitation:
- `task.success_rate` is now separated from `claim_success_rate`, but the corridor
  benchmark still uses a simple world objective (`goal_reached` / progress). That
  means use-case scoring is intentionally conservative and remains provisional
  until richer external objectives are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal

from v4.benchmark.runner import BenchmarkResult

Status = Literal["PASS", "WEAK_PASS", "PROVISIONAL", "FAIL"]
OverallStatus = Literal["strongly meaningful", "meaningful", "not yet meaningful"]


@dataclass
class ScorecardRow:
    """One scorecard row for a benchmark family."""

    benchmark_family: str
    best_structural_config: str
    theory_pass: Status
    geometry_pass: Status
    transfer_pass: Status
    use_case_pass: Status
    overall: OverallStatus
    rationale: str


def _group_by_variant(results: Iterable[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
    grouped: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.variant.name, []).append(result)
    return grouped


def _best_structural_result(results: List[BenchmarkResult]) -> BenchmarkResult | None:
    structural = [
        result
        for result in results
        if result.config.name.startswith("shared_") or result.config.name.startswith("dimension_specific_")
    ]
    if not structural:
        return None
    return min(structural, key=lambda result: result.summary.geometry.mean_contradiction)


def _by_name(results: List[BenchmarkResult]) -> Dict[str, BenchmarkResult]:
    return {result.config.name: result for result in results}


def _theory_status(results: List[BenchmarkResult]) -> Status:
    if not results:
        return "FAIL"
    if any(result.summary.geometry.coverage < 0.0 for result in results):
        return "FAIL"
    return "PASS"


def _geometry_status(best_structural: BenchmarkResult | None, baselines: Dict[str, BenchmarkResult]) -> Status:
    if best_structural is None:
        return "FAIL"

    geometry = best_structural.summary.geometry
    point = baselines.get("shared_point")
    diagonal = baselines.get("shared_diagonal")

    stronger_than_point = point is None or geometry.mean_contradiction < point.summary.geometry.mean_contradiction
    stronger_than_diagonal = (
        diagonal is None or geometry.mean_contradiction < diagonal.summary.geometry.mean_contradiction
    )
    reasonable_radius = geometry.mean_radius <= 1.25
    well_calibrated = geometry.confidence_calibration > 0.0

    if stronger_than_point and stronger_than_diagonal and geometry.coverage >= 0.7 and reasonable_radius and well_calibrated:
        return "PASS"
    if geometry.coverage >= 0.6 and stronger_than_point:
        return "WEAK_PASS"
    return "FAIL"


def _transfer_status(results: List[BenchmarkResult], current_variant: str) -> Status:
    by_name = _by_name(results)
    structural = by_name.get("shared_fiber")
    raw = by_name.get("raw_fiber")
    dimension_specific = by_name.get("dimension_specific_fiber")
    if structural is None or raw is None:
        return "FAIL"

    hard_variant = any(token in current_variant for token in ("hidden", "4d", "shuffled"))
    structural_better = structural.summary.geometry.mean_contradiction < raw.summary.geometry.mean_contradiction
    shared_better_than_specific = (
        dimension_specific is None
        or structural.summary.geometry.mean_contradiction < dimension_specific.summary.geometry.mean_contradiction
    )

    if hard_variant and structural_better and shared_better_than_specific:
        return "PASS"
    if structural_better:
        return "WEAK_PASS"
    return "FAIL"


def _use_case_status(best_structural: BenchmarkResult | None, baselines: Dict[str, BenchmarkResult]) -> Status:
    if best_structural is None:
        return "FAIL"

    raw = baselines.get("raw_fiber")
    task = best_structural.summary.task
    current_gap = abs(task.claim_success_rate - task.success_rate)

    if current_gap > 1e-6:
        if raw is not None and task.final_progress > raw.summary.task.final_progress:
            return "PROVISIONAL"
        return "FAIL"

    # If claim and task success are identical, we still treat use-case scoring as provisional
    # because the current benchmark only recently separated them and does not yet encode richer
    # world-level objectives than corridor completion/progress.
    if raw is not None and task.final_progress > raw.summary.task.final_progress:
        return "PROVISIONAL"
    return "FAIL"


def _overall_status(theory_pass: Status, geometry_pass: Status, transfer_pass: Status, use_case_pass: Status) -> OverallStatus:
    if theory_pass == "PASS" and geometry_pass == "PASS" and transfer_pass in ("PASS", "WEAK_PASS"):
        if use_case_pass == "PASS":
            return "strongly meaningful"
        return "meaningful"
    return "not yet meaningful"


def _rationale(
    theory_pass: Status,
    geometry_pass: Status,
    transfer_pass: Status,
    use_case_pass: Status,
    best_structural: BenchmarkResult | None,
) -> str:
    if best_structural is None:
        return "No structural benchmark result was available."

    geometry = best_structural.summary.geometry
    task = best_structural.summary.task
    return (
        f"theory={theory_pass}, geometry={geometry_pass}, transfer={transfer_pass}, "
        f"use_case={use_case_pass}; contradiction={geometry.mean_contradiction:.3f}, "
        f"radius={geometry.mean_radius:.3f}, coverage={geometry.coverage:.3f}, "
        f"task_success={task.success_rate:.3f}, claim_success={task.claim_success_rate:.3f}, "
        f"final_progress={task.final_progress:.3f}"
    )


def score_benchmark_results(results: List[BenchmarkResult]) -> List[ScorecardRow]:
    """Convert raw benchmark results into scorecard rows grouped by variant."""
    rows: List[ScorecardRow] = []
    for variant_name, variant_results in _group_by_variant(results).items():
        by_name = _by_name(variant_results)
        best_structural = _best_structural_result(variant_results)
        theory_pass = _theory_status(variant_results)
        geometry_pass = _geometry_status(best_structural, by_name)
        transfer_pass = _transfer_status(variant_results, variant_name)
        use_case_pass = _use_case_status(best_structural, by_name)
        overall = _overall_status(theory_pass, geometry_pass, transfer_pass, use_case_pass)
        rows.append(
            ScorecardRow(
                benchmark_family=variant_name,
                best_structural_config=best_structural.config.name if best_structural is not None else "none",
                theory_pass=theory_pass,
                geometry_pass=geometry_pass,
                transfer_pass=transfer_pass,
                use_case_pass=use_case_pass,
                overall=overall,
                rationale=_rationale(
                    theory_pass=theory_pass,
                    geometry_pass=geometry_pass,
                    transfer_pass=transfer_pass,
                    use_case_pass=use_case_pass,
                    best_structural=best_structural,
                ),
            )
        )
    return rows
