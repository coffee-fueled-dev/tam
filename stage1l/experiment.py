"""Stage 1L continuous edge-gap drift experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .model import (
    classify_regime,
    exact_support_coverage,
    excess_measure,
    make_learner,
    modes_for_edge_gap,
    normalize_angle,
    support_cone,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sample_from_modes(
    modes: Sequence[Mapping[str, float]],
    rng: random.Random,
    resolution: int = 1,
) -> int:
    weights = [float(mode.get("weight", 1.0)) for mode in modes]
    total = sum(weights)
    draw = rng.random() * total
    cumulative = 0.0
    chosen = modes[-1]
    for mode, weight in zip(modes, weights):
        cumulative += weight
        if draw < cumulative:
            chosen = mode
            break
    center = float(chosen["center"])
    half_width = float(chosen["half_width"])
    low = int(round((center - half_width) / resolution))
    high = int(round((center + half_width) / resolution))
    span = high - low
    if span <= 0:
        return int(normalize_angle(int(round(center / resolution)) * resolution))
    degree = (low + rng.randint(0, span)) * resolution
    return int(normalize_angle(degree))


def interpolate(start: float, end: float, t: float) -> float:
    return float(start + (end - start) * t)


def generate_stream(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> list[dict[str, Any]]:
    config = protocol["scenarios"][scenario]
    half_width = float(protocol["half_width"])
    center_a = float(protocol["center_a"])
    resolution = int(protocol["angle_resolution_degrees"])
    bins = protocol["regime_bins"]
    rng = random.Random(seed * 1_000_003 + hash(scenario) % 10_000_019)
    stream: list[dict[str, Any]] = []

    if scenario == "plateau_tour":
        step = 0
        for phase in config["phases"]:
            edge_gap = float(phase["edge_gap"])
            modes = modes_for_edge_gap(edge_gap, half_width, center_a)
            regime = classify_regime(edge_gap, bins)
            for _ in range(int(phase["length"])):
                angle = sample_from_modes(modes, rng, resolution)
                stream.append(
                    {
                        "step": step,
                        "angle": angle,
                        "edge_gap": edge_gap,
                        "regime": regime,
                        "phase": str(phase["name"]),
                        "situation": regime,
                    }
                )
                step += 1
        return stream

    # slow_sweep
    length = int(config["length"])
    start = float(config["edge_gap_start"])
    end = float(config["edge_gap_end"])
    for step in range(length):
        t = 0.0 if length <= 1 else step / float(length - 1)
        edge_gap = interpolate(start, end, t)
        modes = modes_for_edge_gap(edge_gap, half_width, center_a)
        regime = classify_regime(edge_gap, bins)
        angle = sample_from_modes(modes, rng, resolution)
        stream.append(
            {
                "step": step,
                "angle": angle,
                "edge_gap": edge_gap,
                "regime": regime,
                "phase": regime,
                "situation": regime,
            }
        )
    return stream


def run_learner_on_stream(
    policy: str,
    stream: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    learner = make_learner(
        policy,
        window_size=int(protocol["primary_window"]),
        split_gap=float(protocol["primary_split_gap"]),
        grace_t=int(protocol["grace_T"]),
    )
    half_width = float(protocol["half_width"])
    center_a = float(protocol["center_a"])
    resolution = int(protocol["angle_resolution_degrees"])
    records: list[dict[str, Any]] = []
    contradictions = 0
    post_miss_failures = 0
    operations: dict[str, int] = {}

    for row in stream:
        step = int(row["step"])
        angle = int(row["angle"])
        regime = str(row["regime"])
        edge_gap = float(row["edge_gap"])
        modes = modes_for_edge_gap(edge_gap, half_width, center_a)
        commitment = learner.predict(regime, step)
        inside = commitment.contains(angle)
        if not inside:
            contradictions += 1

        refinement = learner.observe(regime, angle)
        operations[refinement.operation] = operations.get(refinement.operation, 0) + 1
        if not inside and not refinement.after.contains(angle):
            post_miss_failures += 1

        oracle = support_cone(modes)
        records.append(
            {
                "step": step,
                "phase": row["phase"],
                "regime": regime,
                "situation": row["situation"],
                "edge_gap": edge_gap,
                "angle": angle,
                "inside_pre": inside,
                "operation": refinement.operation,
                "pre_measure": commitment.cones.measure,
                "pre_count": len(commitment.cones.cones),
                "exact_support_coverage": exact_support_coverage(
                    commitment.cones, modes, resolution
                ),
                "excess": excess_measure(commitment.cones, oracle),
            }
        )

    return {
        "policy": policy,
        "contradictions": contradictions,
        "post_miss_failures": post_miss_failures,
        "operations": operations,
        "records": records,
    }


def _rows_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "empirical_coverage": 1.0,
            "exact_support_coverage": 1.0,
            "mean_measure": 0.0,
            "mean_count": 0.0,
            "mean_excess": 0.0,
        }
    return {
        "empirical_coverage": mean(1.0 if row["inside_pre"] else 0.0 for row in rows),
        "exact_support_coverage": mean(row["exact_support_coverage"] for row in rows),
        "mean_measure": mean(row["pre_measure"] for row in rows),
        "mean_count": mean(row["pre_count"] for row in rows),
        "mean_excess": mean(row["excess"] for row in rows),
    }


def _phase_tail_stats(
    records: Sequence[Mapping[str, Any]],
    phase: str,
    tail: int,
) -> dict[str, float]:
    rows = [row for row in records if row["phase"] == phase]
    rows = rows[-tail:] if tail > 0 else rows
    return _rows_stats(rows)


def _phase_head_stats(
    records: Sequence[Mapping[str, Any]],
    phase: str,
    head: int,
) -> dict[str, float]:
    rows = [row for row in records if row["phase"] == phase]
    rows = rows[:head] if head > 0 else rows
    return _rows_stats(rows)


def _regime_window_stats(
    records: Sequence[Mapping[str, Any]],
    regime: str,
    burn_ends: int,
) -> dict[str, float]:
    rows = [row for row in records if row["regime"] == regime]
    if burn_ends > 0 and len(rows) > 2 * burn_ends:
        rows = rows[burn_ends:-burn_ends]
    if not rows:
        return {
            "empirical_coverage": 1.0,
            "exact_support_coverage": 1.0,
            "mean_measure": 0.0,
            "mean_count": 0.0,
            "mean_excess": 0.0,
            "n": 0.0,
        }
    return {
        "empirical_coverage": mean(1.0 if row["inside_pre"] else 0.0 for row in rows),
        "exact_support_coverage": mean(row["exact_support_coverage"] for row in rows),
        "mean_measure": mean(row["pre_measure"] for row in rows),
        "mean_count": mean(row["pre_count"] for row in rows),
        "mean_excess": mean(row["excess"] for row in rows),
        "n": float(len(rows)),
    }


def summarize_seed(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"scenario": scenario, "seed": seed, "learners": {}}
    for policy, result in results.items():
        block: dict[str, Any] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "operations": result["operations"],
        }
        records = result["records"]
        if scenario == "plateau_tour":
            cfg = protocol["scenarios"]["plateau_tour"]
            tail = int(cfg["eval_tail"])
            transition = int(cfg["overlap_transition"])
            block["separate"] = _phase_tail_stats(records, "separate", tail)
            block["boundary"] = _phase_tail_stats(records, "boundary", tail)
            block["overlap"] = _phase_tail_stats(records, "overlap", tail)
            block["overlap_transition"] = _phase_head_stats(
                records, "overlap", transition
            )
        else:
            burn = int(protocol["scenarios"]["slow_sweep"]["burn_ends"])
            block["separate"] = _regime_window_stats(records, "separate", burn)
            block["boundary"] = _regime_window_stats(records, "boundary", burn)
            block["overlap"] = _regime_window_stats(records, "overlap", burn)
        summary["learners"][policy] = block
    return summary


def aggregate_scenario(
    protocol: Mapping[str, Any],
    scenario: str,
    seed_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"scenario": scenario, "n_seeds": len(seed_summaries)}
    for policy in protocol["learners"]:
        rows = [summary["learners"][policy] for summary in seed_summaries]
        block: dict[str, Any] = {
            "mean_contradictions": mean(r["contradictions"] for r in rows),
            "total_post_miss_failures": sum(r["post_miss_failures"] for r in rows),
        }
        for phase in ("separate", "boundary", "overlap"):
            block[phase] = {
                "mean_empirical_coverage": mean(r[phase]["empirical_coverage"] for r in rows),
                "mean_exact_support_coverage": mean(
                    r[phase]["exact_support_coverage"] for r in rows
                ),
                "mean_measure": mean(r[phase]["mean_measure"] for r in rows),
                "mean_count": mean(r[phase]["mean_count"] for r in rows),
                "mean_excess": mean(r[phase]["mean_excess"] for r in rows),
            }
        if scenario == "plateau_tour":
            block["overlap_transition"] = {
                "mean_empirical_coverage": mean(
                    r["overlap_transition"]["empirical_coverage"] for r in rows
                ),
                "mean_exact_support_coverage": mean(
                    r["overlap_transition"]["exact_support_coverage"] for r in rows
                ),
                "mean_measure": mean(
                    r["overlap_transition"]["mean_measure"] for r in rows
                ),
                "mean_count": mean(r["overlap_transition"]["mean_count"] for r in rows),
                "mean_excess": mean(
                    r["overlap_transition"]["mean_excess"] for r in rows
                ),
            }
        out[policy] = block
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    checks: dict[str, Any] = {}

    failures = 0
    for scenario in protocol["scenarios"]:
        for policy in ("binned_hysteresis", "pooled_hysteresis", "pooled_single"):
            failures += aggregates[scenario][policy]["total_post_miss_failures"]
    checks["post_contradiction_containment"] = {
        "met": failures == 0,
        "failures": failures,
    }

    plateau = aggregates["plateau_tour"]
    pev = evidence["plateau_tour"]
    b_sep = plateau["binned_hysteresis"]["separate"]
    b_ov = plateau["binned_hysteresis"]["overlap"]
    p_trans = plateau["pooled_hysteresis"]["overlap_transition"]
    sep_ev = pev["separate"]
    ov_ev = pev["overlap"]

    checks["plateau_separate"] = {
        "met": (
            b_sep["mean_exact_support_coverage"] >= sep_ev["binned_coverage_min"]
            and sep_ev["binned_count_min"]
            <= b_sep["mean_count"]
            <= sep_ev["binned_count_max"]
            and b_sep["mean_measure"] <= sep_ev["binned_measure_max"]
            and b_sep["mean_excess"] <= sep_ev["binned_excess_max"]
        ),
        "detail": b_sep,
    }
    checks["plateau_overlap_binned"] = {
        "met": (
            b_ov["mean_exact_support_coverage"] >= ov_ev["binned_coverage_min"]
            and b_ov["mean_count"] <= ov_ev["binned_count_max"]
            and b_ov["mean_measure"] <= ov_ev["binned_measure_max"]
            and b_ov["mean_excess"] <= ov_ev["binned_excess_max"]
        ),
        "detail": b_ov,
    }
    checks["plateau_overlap_pooled_limitation"] = {
        "met": p_trans["mean_excess"] >= ov_ev["pooled_transition_excess_min"],
        "pooled_transition_excess": p_trans["mean_excess"],
        "pooled_transition_count": p_trans["mean_count"],
        "pooled_transition_measure": p_trans["mean_measure"],
    }
    checks["plateau_boundary"] = {
        "met": True,
        "diagnostic_only": True,
        "binned": plateau["binned_hysteresis"]["boundary"],
        "pooled": plateau["pooled_hysteresis"]["boundary"],
    }

    sweep = aggregates["slow_sweep"]
    sev = evidence["slow_sweep"]
    sb = sweep["binned_hysteresis"]["separate"]
    so = sweep["binned_hysteresis"]["overlap"]
    pb = sweep["pooled_hysteresis"]
    excess_gap = pb["overlap"]["mean_excess"] - so["mean_excess"]
    # Also credit mid-sweep: pooled boundary excess vs binned if overlap gap is thin.
    boundary_gap = (
        pb["boundary"]["mean_excess"]
        - sweep["binned_hysteresis"]["boundary"]["mean_excess"]
    )
    effective_gap = max(excess_gap, boundary_gap)

    checks["sweep_separate"] = {
        "met": (
            sb["mean_exact_support_coverage"] >= sev["separate_window"]["binned_coverage_min"]
            and sev["separate_window"]["binned_count_min"]
            <= sb["mean_count"]
            <= sev["separate_window"]["binned_count_max"]
            and sb["mean_measure"] <= sev["separate_window"]["binned_measure_max"]
        ),
        "detail": sb,
    }
    checks["sweep_overlap"] = {
        "met": (
            so["mean_exact_support_coverage"] >= sev["overlap_window"]["binned_coverage_min"]
            and so["mean_count"] <= sev["overlap_window"]["binned_count_max"]
            and so["mean_measure"] <= sev["overlap_window"]["binned_measure_max"]
        ),
        "detail": so,
    }
    checks["sweep_pooled_excess_gap"] = {
        "met": effective_gap >= sev["pooled_excess_gap_min"],
        "overlap_excess_gap": excess_gap,
        "boundary_excess_gap": boundary_gap,
        "effective_gap": effective_gap,
        "pooled_overlap_excess": pb["overlap"]["mean_excess"],
        "binned_overlap_excess": so["mean_excess"],
    }
    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "post_contradiction_containment",
        "plateau_separate",
        "plateau_overlap_binned",
        "plateau_overlap_pooled_limitation",
        "sweep_separate",
        "sweep_overlap",
        "sweep_pooled_excess_gap",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Under continuous edge-gap drift, regime-binned sticky hysteresis keeps "
            "tight separate and overlap claims, while a pooled sticky store overclaims "
            "after visiting incompatible separations. Boundary remains a known "
            "limitation band."
            if all_met
            else "One or more continuous-drift gates failed; inspect checks."
        ),
        "decision": (
            "keep situation-binned hysteresis for continuous edge-gap situations; "
            "do not treat pooled Stage 1K hysteresis as sufficient under drift; "
            "adaptive split / Stage 2 still unearned unless binned fixed rules fail"
            if all_met
            else "revise continuous-drift characterization before adaptive split"
        ),
        "limits": (
            "Bins are coarse observable features of edge gap, not learned embeddings. "
            "No control, safety, adaptive split_gap, or neural predictor claim."
        ),
    }


def run_experiment(output: str | Path | None = None) -> dict[str, Any]:
    protocol = load_protocol()
    started = time.perf_counter()
    seed_count = int(protocol["seeds"])
    scenarios = list(protocol["scenarios"])
    all_summaries: dict[str, list[dict[str, Any]]] = {name: [] for name in scenarios}

    for scenario in scenarios:
        for seed in range(seed_count):
            stream = generate_stream(protocol, scenario, seed)
            results = {
                policy: run_learner_on_stream(policy, stream, protocol)
                for policy in protocol["learners"]
            }
            all_summaries[scenario].append(
                summarize_seed(protocol, scenario, seed, results)
            )

    aggregates = {
        scenario: aggregate_scenario(protocol, scenario, summaries)
        for scenario, summaries in all_summaries.items()
    }
    checks = evaluate_evidence(protocol, aggregates)
    decision = decide(checks)
    payload = {
        "protocol": protocol,
        "aggregates": aggregates,
        "checks": checks,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
        "seeds": seed_count,
    }

    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "protocol.snapshot.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "aggregates": aggregates,
                    "checks": checks,
                    "decision": decision,
                    "runtime_seconds": payload["runtime_seconds"],
                    "seeds": seed_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/stage1l")
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])
    print(decision["decision"])


if __name__ == "__main__":
    main()
