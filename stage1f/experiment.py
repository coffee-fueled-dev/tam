"""Stage 1F geometric cone refinement experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from .model import (
    CIRCLE,
    AngularCone,
    ConeSet,
    GeometricLearner,
    excess_measure,
    mode_represented,
    normalize_angle,
    oracle_support,
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
    weights = [float(mode["weight"]) for mode in modes]
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
    # Inclusive integer-degree support within the continuous sector.
    low = int(round((center - half_width) / resolution))
    high = int(round((center + half_width) / resolution))
    # Sample uniformly among integer degrees on the wrapped arc.
    span = high - low
    if span <= 0:
        degree = int(round(center / resolution)) * resolution
        return int(normalize_angle(degree))
    offset = rng.randint(0, span)
    degree = (low + offset) * resolution
    return int(normalize_angle(degree))


def scenario_modes_at_step(
    protocol: Mapping[str, Any],
    scenario: str,
    step: int,
) -> list[dict[str, float]]:
    config = protocol["scenarios"][scenario]
    if scenario == "emergence":
        if step < int(config["phase1_end"]):
            return [dict(config["mode_a"])]
        return [dict(mode) for mode in config["phase2_modes"]]
    return [dict(mode) for mode in config["modes"]]


def oracle_modes_for_scenario(
    protocol: Mapping[str, Any],
    scenario: str,
) -> list[dict[str, float]]:
    config = protocol["scenarios"][scenario]
    if scenario == "emergence":
        return [dict(mode) for mode in config["phase2_modes"]]
    return [dict(mode) for mode in config["modes"]]


def generate_stream(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> list[int]:
    config = protocol["scenarios"][scenario]
    length = int(config["length"])
    resolution = int(protocol["angle_resolution_degrees"])
    rng = random.Random(seed * 1_000_003 + hash(scenario) % 10_000_019)
    return [
        sample_from_modes(
            scenario_modes_at_step(protocol, scenario, step),
            rng,
            resolution=resolution,
        )
        for step in range(length)
    ]


def run_learner_on_stream(
    policy: str,
    stream: Sequence[int],
    split_gap: float,
) -> dict[str, Any]:
    learner = GeometricLearner(policy, split_gap=split_gap)
    records: list[dict[str, Any]] = []
    contradictions = 0
    post_miss_failures = 0
    operations = {"widen": 0, "add": 0, "noop": 0}
    first_add_step: int | None = None

    for step, angle in enumerate(stream):
        commitment = learner.predict(step)
        inside = commitment.contains(angle)
        if not inside:
            contradictions += 1
        refinement = learner.observe(angle)
        operations[refinement.operation] = operations.get(refinement.operation, 0) + 1
        if refinement.operation == "add" and first_add_step is None and step > 0:
            first_add_step = step
        if not inside and not refinement.after.contains(angle):
            post_miss_failures += 1
        records.append(
            {
                "step": step,
                "angle": angle,
                "inside_pre": inside,
                "operation": refinement.operation,
                "pre": commitment.to_dict(),
                "post": refinement.after.to_dict(),
            }
        )

    return {
        "policy": policy,
        "contradictions": contradictions,
        "post_miss_failures": post_miss_failures,
        "operations": operations,
        "first_add_step": first_add_step,
        "final": learner.cone_set.to_dict(),
        "final_count": len(learner.cone_set.cones),
        "final_measure": learner.cone_set.measure,
        "records": records,
        "learner": learner,
    }


def held_out_coverage(
    records: Sequence[Mapping[str, Any]],
    eval_start: int,
    burn_in: int,
) -> float:
    start = max(eval_start, burn_in)
    relevant = [row for row in records if row["step"] >= start]
    if not relevant:
        return 1.0
    return mean(1.0 if row["inside_pre"] else 0.0 for row in relevant)


def summarize_seed(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    stream: Sequence[int],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    support = oracle_support(oracle_modes_for_scenario(protocol, scenario))
    config = protocol["scenarios"][scenario]
    eval_start = int(config["eval_start"])
    burn_in = int(protocol["burn_in"])
    summary: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "oracle_measure": support.measure,
        "learners": {},
    }

    for policy, result in results.items():
        learner = result["learner"]
        excess = excess_measure(learner.cone_set, support)
        summary["learners"][policy] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "operations": result["operations"],
            "final_count": result["final_count"],
            "final_measure": result["final_measure"],
            "excess_measure": excess,
            "held_out_coverage": held_out_coverage(
                result["records"], eval_start, burn_in
            ),
            "final": result["final"],
        }

        if scenario == "emergence":
            mode_b = protocol["scenarios"]["emergence"]["mode_b"]
            phase1_end = int(config["phase1_end"])
            center = float(mode_b["center"])
            half = float(mode_b["half_width"])
            first_b_step = None
            for step, angle in enumerate(stream):
                if step < phase1_end:
                    continue
                delta = min(
                    abs(normalize_angle(angle) - center),
                    CIRCLE - abs(normalize_angle(angle) - center),
                )
                if delta <= half + 1e-9:
                    first_b_step = step
                    break

            first_b_contradiction = None
            representation_delay = None
            if first_b_step is not None:
                row = result["records"][first_b_step]
                first_b_contradiction = not bool(row["inside_pre"])
                # Immediate representation: the realized new-mode trajectory is
                # inside the post-update cone set at the contradiction step.
                post_cones = tuple(
                    AngularCone(c["center"], c["half_width"])
                    for c in row["post"]["cones"]
                )
                representation_delay = (
                    0 if ConeSet(post_cones).contains(stream[first_b_step]) else None
                )

            summary["learners"][policy]["emergence"] = {
                "first_b_step": first_b_step,
                "first_b_is_contradiction": first_b_contradiction,
                "representation_delay": representation_delay,
                "mode_b_represented_final": mode_represented(
                    learner.cone_set, mode_b
                ),
            }

    return summary


def aggregate_scenario(
    protocol: Mapping[str, Any],
    scenario: str,
    seed_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    learners = protocol["learners"]
    out: dict[str, Any] = {"scenario": scenario, "n_seeds": len(seed_summaries)}
    for policy in learners:
        rows = [row["learners"][policy] for row in seed_summaries]
        block: dict[str, Any] = {
            "mean_held_out_coverage": mean(r["held_out_coverage"] for r in rows),
            "mean_final_measure": mean(r["final_measure"] for r in rows),
            "mean_excess": mean(r["excess_measure"] for r in rows),
            "mean_final_count": mean(r["final_count"] for r in rows),
            "median_final_count": median(r["final_count"] for r in rows),
            "mean_contradictions": mean(r["contradictions"] for r in rows),
            "total_post_miss_failures": sum(r["post_miss_failures"] for r in rows),
            "fraction_exact_cone_count": None,
        }
        if scenario == "unimodal":
            target = 1
            block["fraction_exact_cone_count"] = mean(
                1.0 if r["final_count"] == target else 0.0 for r in rows
            )
        elif scenario in ("bimodal", "emergence"):
            target = 2 if policy == "multi_cone" else None
            if target is not None:
                block["fraction_exact_cone_count"] = mean(
                    1.0 if r["final_count"] == target else 0.0 for r in rows
                )
        if scenario == "emergence":
            emerg = [r["emergence"] for r in rows]
            block["fraction_first_b_contradiction"] = mean(
                1.0 if e["first_b_is_contradiction"] else 0.0 for e in emerg
            )
            delays = [
                e["representation_delay"]
                for e in emerg
                if e["representation_delay"] is not None
            ]
            block["mean_representation_delay"] = mean(delays) if delays else None
            block["max_representation_delay"] = max(delays) if delays else None
            block["fraction_mode_b_represented"] = mean(
                1.0 if e["mode_b_represented_final"] else 0.0 for e in emerg
            )
        out[policy] = block

    # Pairwise union ratio multi vs single
    if "multi_cone" in learners and "single_widen" in learners:
        ratios = []
        for row in seed_summaries:
            single = row["learners"]["single_widen"]["final_measure"]
            multi = row["learners"]["multi_cone"]["final_measure"]
            ratios.append(multi / single if single > 1e-9 else 0.0)
        out["multi_vs_single_union_ratio_mean"] = mean(ratios)
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    checks: dict[str, Any] = {}

    # Global post-containment
    total_failures = sum(
        aggregates[scenario][policy]["total_post_miss_failures"]
        for scenario in aggregates
        for policy in protocol["learners"]
    )
    checks["contradiction_post_containment"] = {
        "met": total_failures == 0,
        "failures": total_failures,
        "threshold": evidence["contradiction_post_containment_min"],
    }

    # Held-out coverage for adaptive learners
    coverage_ok = True
    coverage_detail = {}
    for scenario, agg in aggregates.items():
        for policy in ("single_widen", "multi_cone"):
            value = agg[policy]["mean_held_out_coverage"]
            coverage_detail[f"{scenario}:{policy}"] = value
            if value < evidence["held_out_coverage_min"]:
                coverage_ok = False
    checks["held_out_coverage"] = {
        "met": coverage_ok,
        "detail": coverage_detail,
        "threshold": evidence["held_out_coverage_min"],
    }

    uni = aggregates["unimodal"]
    uni_ev = evidence["unimodal"]
    checks["unimodal_tight"] = {
        "met": (
            uni["single_widen"]["mean_final_count"] <= uni_ev["max_cones_single"]
            and uni["multi_cone"]["mean_final_count"] <= uni_ev["max_cones_multi"]
            and uni["single_widen"]["mean_excess"] <= uni_ev["max_excess_degrees"]
            and uni["multi_cone"]["mean_excess"] <= uni_ev["max_excess_degrees"]
            and uni["full_circle"]["mean_final_measure"]
            >= evidence["full_circle_measure"] - 1e-6
        ),
        "single_count": uni["single_widen"]["mean_final_count"],
        "multi_count": uni["multi_cone"]["mean_final_count"],
        "single_excess": uni["single_widen"]["mean_excess"],
        "multi_excess": uni["multi_cone"]["mean_excess"],
    }

    bi = aggregates["bimodal"]
    bi_ev = evidence["bimodal"]
    checks["bimodal_multi_tighter"] = {
        "met": (
            bi["multi_cone"]["fraction_exact_cone_count"] >= 0.9
            and bi["multi_vs_single_union_ratio_mean"]
            <= bi_ev["union_ratio_max_vs_single"]
            and bi["multi_cone"]["mean_excess"] <= bi_ev["max_excess_degrees_multi"]
        ),
        "fraction_two_cones": bi["multi_cone"]["fraction_exact_cone_count"],
        "union_ratio": bi["multi_vs_single_union_ratio_mean"],
        "multi_excess": bi["multi_cone"]["mean_excess"],
        "single_measure": bi["single_widen"]["mean_final_measure"],
        "multi_measure": bi["multi_cone"]["mean_final_measure"],
    }

    em = aggregates["emergence"]
    em_ev = evidence["emergence"]
    checks["emergence_add"] = {
        "met": (
            em["multi_cone"]["fraction_first_b_contradiction"] >= 0.95
            and em["multi_cone"]["max_representation_delay"] is not None
            and em["multi_cone"]["max_representation_delay"]
            <= em_ev["representation_delay_max"]
            and em["multi_cone"]["fraction_exact_cone_count"] >= 0.9
            and em["multi_vs_single_union_ratio_mean"]
            <= em_ev["union_ratio_max_vs_single"]
        ),
        "fraction_first_contradiction": em["multi_cone"][
            "fraction_first_b_contradiction"
        ],
        "max_representation_delay": em["multi_cone"]["max_representation_delay"],
        "fraction_two_cones": em["multi_cone"]["fraction_exact_cone_count"],
        "union_ratio": em["multi_vs_single_union_ratio_mean"],
    }

    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "contradiction_post_containment",
        "held_out_coverage",
        "unimodal_tight",
        "bimodal_multi_tighter",
        "emergence_add",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Geometric cone proliferation earns tighter reality-containing "
            "commitments on separated directional modes; single connected "
            "widening pays an unsupported bridge cost."
            if all_met
            else "One or more geometric refinement gates failed; inspect checks."
        ),
        "limits": (
            "Result is limited to unit one-step directional trajectories. "
            "Multi-step embeddings, contextual applicability, narrowing, "
            "control, and learned refinement remain untested."
        ),
    }


def run_experiment(output: str | Path | None = None) -> dict[str, Any]:
    protocol = load_protocol()
    started = time.perf_counter()
    seed_count = int(protocol["seeds"])
    split_gap = float(protocol["split_gap_degrees"])
    scenarios = list(protocol["scenarios"])
    learners = list(protocol["learners"])

    all_seed_summaries: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in scenarios
    }
    run_artifacts: dict[str, Any] = {}

    for scenario in scenarios:
        for seed in range(seed_count):
            stream = generate_stream(protocol, scenario, seed)
            results = {}
            for policy in learners:
                results[policy] = run_learner_on_stream(policy, stream, split_gap)
            summary = summarize_seed(protocol, scenario, seed, stream, results)
            all_seed_summaries[scenario].append(summary)
            run_artifacts.setdefault(scenario, {})[str(seed)] = {
                "stream": stream,
                "summary": {
                    key: value
                    for key, value in summary.items()
                    if key != "learners"
                },
                "learners": {
                    policy: {
                        **{
                            k: v
                            for k, v in summary["learners"][policy].items()
                            if k != "final"
                        },
                        "final": summary["learners"][policy]["final"],
                    }
                    for policy in learners
                },
            }

    aggregates = {
        scenario: aggregate_scenario(protocol, scenario, summaries)
        for scenario, summaries in all_seed_summaries.items()
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
        # Compact run artifacts without full per-step records to keep size sane.
        compact_runs = {}
        for scenario, by_seed in run_artifacts.items():
            compact_runs[scenario] = {}
            for seed, data in by_seed.items():
                compact_runs[scenario][seed] = {
                    "stream_hash": hash(tuple(data["stream"])) & 0xFFFFFFFF,
                    "stream_len": len(data["stream"]),
                    "learners": data["learners"],
                }
        (out / "runs.json").write_text(
            json.dumps(compact_runs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_path = out / "summary.json"
        summary_path.write_text(
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
    parser.add_argument(
        "--output",
        default="artifacts/stage1f",
        help="directory for protocol snapshot, runs, and summary",
    )
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])


if __name__ == "__main__":
    main()
