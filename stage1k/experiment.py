"""Stage 1K overlapping / near-merge mode experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .model import (
    DIAGNOSTIC_LEARNERS,
    PRIMARY_LEARNERS,
    SITUATION_LABEL,
    exact_support_coverage,
    excess_measure,
    make_learner,
    normalize_angle,
    scenario_modes,
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


def generate_stream(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> list[dict[str, Any]]:
    modes = scenario_modes(protocol, scenario)
    length = int(protocol["length"])
    resolution = int(protocol["angle_resolution_degrees"])
    rng = random.Random(seed * 1_000_003 + hash(scenario) % 10_000_019)
    stream: list[dict[str, Any]] = []
    for step in range(length):
        angle = sample_from_modes(modes, rng, resolution)
        stream.append(
            {
                "step": step,
                "angle": angle,
                "regime": scenario,
                "phase": scenario,
                "situation": protocol.get("situation_label", SITUATION_LABEL),
            }
        )
    return stream


def run_learner_on_stream(
    policy: str,
    stream: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    window_size: int | None = None,
    split_gap: float | None = None,
    grace_t: int | None = None,
) -> dict[str, Any]:
    learner = make_learner(
        policy,
        window_size=window_size or int(protocol["primary_window"]),
        split_gap=split_gap,
        grace_t=grace_t if grace_t is not None else int(protocol["grace_T"]),
    )
    resolution = int(protocol["angle_resolution_degrees"])
    modes = scenario_modes(protocol, str(stream[0]["regime"])) if stream else []
    records: list[dict[str, Any]] = []
    contradictions = 0
    post_miss_failures = 0
    operations: dict[str, int] = {}

    for row in stream:
        step = int(row["step"])
        angle = int(row["angle"])
        commitment = learner.predict(step)
        inside = commitment.contains(angle)
        if not inside:
            contradictions += 1

        refinement = learner.observe(angle)
        operations[refinement.operation] = operations.get(refinement.operation, 0) + 1
        if not inside and not refinement.after.contains(angle):
            post_miss_failures += 1

        oracle = support_cone(modes)
        records.append(
            {
                "step": step,
                "phase": row["phase"],
                "regime": row["regime"],
                "situation": row["situation"],
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
        "final_measure": learner.cone_set.measure,
        "final_count": len(learner.cone_set.cones),
    }


def summarize_seed(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    burn_in = int(protocol["burn_in"])
    summary: dict[str, Any] = {"scenario": scenario, "seed": seed, "learners": {}}
    for policy, result in results.items():
        records = result["records"]
        after = [row for row in records if row["step"] >= burn_in]
        mean_count = mean(row["pre_count"] for row in after) if after else 0.0
        summary["learners"][policy] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "operations": result["operations"],
            "final_measure": result["final_measure"],
            "final_count": result["final_count"],
            "empirical_coverage": (
                mean(1.0 if row["inside_pre"] else 0.0 for row in after)
                if after
                else 1.0
            ),
            "exact_support_coverage": (
                mean(row["exact_support_coverage"] for row in after) if after else 1.0
            ),
            "mean_measure": mean(row["pre_measure"] for row in after) if after else 0.0,
            "mean_count": mean_count,
            "mean_excess": mean(row["excess"] for row in after) if after else 0.0,
            "fraction_count_1": (
                mean(1.0 if abs(row["pre_count"] - 1.0) < 1e-9 else 0.0 for row in after)
                if after
                else 0.0
            ),
            "fraction_count_2": (
                mean(1.0 if abs(row["pre_count"] - 2.0) < 1e-9 else 0.0 for row in after)
                if after
                else 0.0
            ),
        }
    return summary


def aggregate_scenario(
    protocol: Mapping[str, Any],
    scenario: str,
    seed_summaries: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"scenario": scenario, "n_seeds": len(seed_summaries)}
    for policy in policies:
        rows = [summary["learners"][policy] for summary in seed_summaries]
        out[policy] = {
            "mean_contradictions": mean(r["contradictions"] for r in rows),
            "total_post_miss_failures": sum(r["post_miss_failures"] for r in rows),
            "mean_empirical_coverage": mean(r["empirical_coverage"] for r in rows),
            "mean_exact_support_coverage": mean(
                r["exact_support_coverage"] for r in rows
            ),
            "mean_measure": mean(r["mean_measure"] for r in rows),
            "mean_count": mean(r["mean_count"] for r in rows),
            "mean_excess": mean(r["mean_excess"] for r in rows),
            "mean_fraction_count_1": mean(r["fraction_count_1"] for r in rows),
            "mean_fraction_count_2": mean(r["fraction_count_2"] for r in rows),
        }
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    checks: dict[str, Any] = {}

    failures = 0
    for scenario in ("separate70", "overlap5", "wide_unimodal"):
        for policy in ("hysteresis_gap30", "window_single"):
            failures += aggregates[scenario][policy]["total_post_miss_failures"]
    checks["post_contradiction_containment"] = {
        "met": failures == 0,
        "failures": failures,
    }

    sep = aggregates["separate70"]["hysteresis_gap30"]
    sep_single = aggregates["separate70"]["window_single"]
    sev = evidence["separate70"]
    checks["separate70"] = {
        "met": (
            sep["mean_exact_support_coverage"] >= sev["coverage_min"]
            and sev["count_min"] <= sep["mean_count"] <= sev["count_max"]
            and sep["mean_measure"] <= sev["measure_max"]
            and sep["mean_excess"] <= sev["excess_max"]
            and sep_single["mean_measure"] >= sev["single_measure_min"]
        ),
        "hysteresis_gap30": {
            "coverage": sep["mean_exact_support_coverage"],
            "count": sep["mean_count"],
            "measure": sep["mean_measure"],
            "excess": sep["mean_excess"],
        },
        "window_single_measure": sep_single["mean_measure"],
    }

    ov = aggregates["overlap5"]["hysteresis_gap30"]
    oev = evidence["overlap5"]
    checks["overlap5"] = {
        "met": (
            ov["mean_exact_support_coverage"] >= oev["coverage_min"]
            and ov["mean_count"] <= oev["count_max"]
            and ov["mean_measure"] <= oev["measure_max"]
            and ov["mean_excess"] <= oev["excess_max"]
        ),
        "detail": {
            "coverage": ov["mean_exact_support_coverage"],
            "count": ov["mean_count"],
            "measure": ov["mean_measure"],
            "excess": ov["mean_excess"],
        },
    }

    wide = aggregates["wide_unimodal"]["hysteresis_gap30"]
    wev = evidence["wide_unimodal"]
    checks["wide_unimodal"] = {
        "met": (
            wide["mean_exact_support_coverage"] >= wev["coverage_min"]
            and wide["mean_count"] <= wev["count_max"]
            and wide["mean_measure"] <= wev["measure_max"]
            and wide["mean_excess"] <= wev["excess_max"]
        ),
        "detail": {
            "coverage": wide["mean_exact_support_coverage"],
            "count": wide["mean_count"],
            "measure": wide["mean_measure"],
            "excess": wide["mean_excess"],
        },
    }

    bound = aggregates["boundary30"]["hysteresis_gap30"]
    checks["boundary30"] = {
        "met": True,
        "diagnostic_only": True,
        "mean_count": bound["mean_count"],
        "fraction_count_1": bound["mean_fraction_count_1"],
        "fraction_count_2": bound["mean_fraction_count_2"],
        "mean_measure": bound["mean_measure"],
        "mean_excess": bound["mean_excess"],
        "mean_coverage": bound["mean_exact_support_coverage"],
    }

    sensitivity: dict[str, Any] = {"met": True, "diagnostic_only": True}
    for scenario in ("separate70", "overlap5"):
        sensitivity[scenario] = {
            gap_policy: {
                "count": aggregates[scenario][gap_policy]["mean_count"],
                "measure": aggregates[scenario][gap_policy]["mean_measure"],
                "coverage": aggregates[scenario][gap_policy][
                    "mean_exact_support_coverage"
                ],
                "excess": aggregates[scenario][gap_policy]["mean_excess"],
            }
            for gap_policy in DIAGNOSTIC_LEARNERS
        }
    checks["gap_sensitivity"] = sensitivity
    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "post_contradiction_containment",
        "separate70",
        "overlap5",
        "wide_unimodal",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    boundary = checks["boundary30"]
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Fixed split_gap=30° keeps two cones on well-separated modes, merges "
            "clearly overlapping supports, and does not false-split a wide unimodal "
            f"arc. Boundary edge-gap≈30° remains a known limitation "
            f"(mean_count={boundary['mean_count']:.2f}, "
            f"frac_count_1={boundary['fraction_count_1']:.2f}, "
            f"frac_count_2={boundary['fraction_count_2']:.2f})."
            if all_met
            else "One or more separation-regime gates failed; inspect checks."
        ),
        "decision": (
            "keep fixed split_gap only inside the documented working regime "
            "(well-separated and clearly overlapping/unimodal); treat boundary "
            "instability as a known limitation, not a bug"
            if all_met
            else "revise fixed split_gap characterization before adaptive split"
        ),
        "limits": (
            "Fixed (W,T,split_gap) only. Continuous situations and learned "
            "split_gap remain next if boundary failure matters in target domains."
        ),
    }


def run_experiment(output: str | Path | None = None) -> dict[str, Any]:
    protocol = load_protocol()
    started = time.perf_counter()
    seed_count = int(protocol["seeds"])
    scenarios = list(protocol["scenarios"])
    policies = list(PRIMARY_LEARNERS) + list(DIAGNOSTIC_LEARNERS)
    all_summaries: dict[str, list[dict[str, Any]]] = {name: [] for name in scenarios}

    for scenario in scenarios:
        run_policies = list(PRIMARY_LEARNERS)
        if scenario in ("separate70", "overlap5"):
            run_policies = policies
        for seed in range(seed_count):
            stream = generate_stream(protocol, scenario, seed)
            results = {
                policy: run_learner_on_stream(policy, stream, protocol)
                for policy in run_policies
            }
            # Ensure diagnostic keys exist on non-sensitivity scenarios as None skips.
            for policy in policies:
                if policy not in results and policy in DIAGNOSTIC_LEARNERS:
                    continue
            all_summaries[scenario].append(
                summarize_seed(protocol, scenario, seed, results)
            )

    aggregates = {
        scenario: aggregate_scenario(
            protocol,
            scenario,
            summaries,
            policies=(
                list(PRIMARY_LEARNERS) + list(DIAGNOSTIC_LEARNERS)
                if scenario in ("separate70", "overlap5")
                else list(PRIMARY_LEARNERS)
            ),
        )
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
    parser.add_argument("--output", default="artifacts/stage1k")
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])
    print(decision["decision"])


if __name__ == "__main__":
    main()
