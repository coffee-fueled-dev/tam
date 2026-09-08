"""Stage 1I multimodal cone pruning experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .model import (
    SITUATION_LABEL,
    exact_support_coverage,
    excess_measure,
    make_learner,
    normalize_angle,
    regime_modes,
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


def expand_schedule(schedule: Sequence[Sequence[Any]]) -> list[str]:
    labels: list[str] = []
    for name, length in schedule:
        labels.extend([str(name)] * int(length))
    return labels


def phase_name_at(protocol: Mapping[str, Any], scenario: str, step: int) -> str:
    config = protocol["scenarios"][scenario]
    if "phase_names" not in config:
        return str(config["schedule"][0][0])
    names = config["phase_names"]
    cursor = 0
    for index, (_, length) in enumerate(config["schedule"]):
        if step < cursor + int(length):
            return str(names[index])
        cursor += int(length)
    return str(names[-1])


def generate_stream(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> list[dict[str, Any]]:
    config = protocol["scenarios"][scenario]
    regimes = expand_schedule(config["schedule"])
    resolution = int(protocol["angle_resolution_degrees"])
    rng = random.Random(seed * 1_000_003 + hash(scenario) % 10_000_019)
    stream: list[dict[str, Any]] = []
    for step, regime in enumerate(regimes):
        modes = regime_modes(protocol, regime)
        angle = sample_from_modes(modes, rng, resolution)
        stream.append(
            {
                "step": step,
                "angle": angle,
                "regime": regime,
                "phase": phase_name_at(protocol, scenario, step),
                "situation": protocol.get("situation_label", SITUATION_LABEL),
            }
        )
    return stream


def run_learner_on_stream(
    policy: str,
    stream: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    window_size: int | None = None,
) -> dict[str, Any]:
    learner = make_learner(
        policy,
        window_size=window_size,
        split_gap=float(protocol["split_gap_degrees"]),
    )
    resolution = int(protocol["angle_resolution_degrees"])
    records: list[dict[str, Any]] = []
    contradictions = 0
    post_miss_failures = 0
    operations: dict[str, int] = {}

    for row in stream:
        step = int(row["step"])
        angle = int(row["angle"])
        regime = str(row["regime"])
        modes = regime_modes(protocol, regime)
        commitment = learner.predict(step)
        inside = commitment.contains(angle)
        if not inside:
            contradictions += 1

        refinement = learner.observe(angle)
        operations[refinement.operation] = operations.get(refinement.operation, 0) + 1
        if not inside and not refinement.after.contains(angle):
            post_miss_failures += 1

        exact = exact_support_coverage(commitment.cones, modes, resolution)
        excess = excess_measure(commitment.cones, support_cone(modes))
        records.append(
            {
                "step": step,
                "phase": row["phase"],
                "regime": regime,
                "situation": row["situation"],
                "angle": angle,
                "inside_pre": inside,
                "operation": refinement.operation,
                "pre_measure": commitment.cones.measure,
                "pre_count": len(commitment.cones.cones),
                "post_measure": refinement.after.measure,
                "post_count": len(refinement.after.cones),
                "exact_support_coverage": exact,
                "excess": excess,
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


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        out.append(mean(values[start : index + 1]))
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def summarize_seed(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    burn_in = int(protocol["burn_in"])
    roll_w = int(protocol["rolling_coverage_window"])
    summary: dict[str, Any] = {"scenario": scenario, "seed": seed, "learners": {}}

    for policy, result in results.items():
        records = result["records"]
        block: dict[str, Any] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "operations": result["operations"],
            "final_measure": result["final_measure"],
            "final_count": result["final_count"],
        }

        if scenario == "stationary_AB":
            after = [row for row in records if row["step"] >= burn_in]
            block["empirical_coverage"] = (
                mean(1.0 if row["inside_pre"] else 0.0 for row in after)
                if after
                else 1.0
            )
            block["exact_coverage"] = (
                mean(row["exact_support_coverage"] for row in after) if after else 1.0
            )
            block["mean_measure"] = (
                mean(row["pre_measure"] for row in after) if after else 0.0
            )
            block["mean_count"] = (
                mean(row["pre_count"] for row in after) if after else 0.0
            )

        if scenario.startswith("hidden_AB_to_"):
            ab_rows = [row for row in records if row["phase"] == "AB"]
            post_rows = [row for row in records if row["phase"] == "post"]
            block["AB"] = {
                "mean_measure": mean(row["pre_measure"] for row in ab_rows)
                if ab_rows
                else 0.0,
                "mean_count": mean(row["pre_count"] for row in ab_rows)
                if ab_rows
                else 0.0,
                "empirical_coverage": mean(
                    1.0 if row["inside_pre"] else 0.0 for row in ab_rows
                )
                if ab_rows
                else 1.0,
            }
            block["post"] = {
                "empirical_coverage": mean(
                    1.0 if row["inside_pre"] else 0.0 for row in post_rows
                )
                if post_rows
                else 1.0,
                "exact_coverage": mean(
                    row["exact_support_coverage"] for row in post_rows
                )
                if post_rows
                else 1.0,
                "mean_measure": mean(row["pre_measure"] for row in post_rows)
                if post_rows
                else 0.0,
                "mean_count": mean(row["pre_count"] for row in post_rows)
                if post_rows
                else 0.0,
            }

            exacts = [row["exact_support_coverage"] for row in post_rows]
            measures = [row["pre_measure"] for row in post_rows]
            counts = [row["pre_count"] for row in post_rows]
            roll = rolling_mean(exacts, roll_w)
            cov_min = evidence["prune"]["rolling_exact_coverage_min"]
            measure_max = evidence["prune"]["measure_max"]
            count_target = evidence["prune"]["count_target"]
            prune_delay = None
            for index, value in enumerate(roll):
                if (
                    value >= cov_min
                    and measures[index] <= measure_max
                    and counts[index] <= count_target
                ):
                    prune_delay = index
                    break
            block["prune_delay"] = prune_delay

            last_n = int(evidence["final_post"]["last_n"])
            final_rows = post_rows[-last_n:]
            block["final_post"] = {
                "empirical_coverage": mean(
                    1.0 if row["inside_pre"] else 0.0 for row in final_rows
                )
                if final_rows
                else 1.0,
                "mean_measure": mean(row["pre_measure"] for row in final_rows)
                if final_rows
                else 0.0,
                "mean_count": mean(row["pre_count"] for row in final_rows)
                if final_rows
                else 0.0,
            }

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
        if scenario == "stationary_AB":
            block.update(
                {
                    "mean_empirical_coverage": mean(
                        r["empirical_coverage"] for r in rows
                    ),
                    "mean_exact_coverage": mean(r["exact_coverage"] for r in rows),
                    "mean_measure": mean(r["mean_measure"] for r in rows),
                    "mean_count": mean(r["mean_count"] for r in rows),
                }
            )
        if scenario.startswith("hidden_AB_to_"):
            delays = [r["prune_delay"] for r in rows if r["prune_delay"] is not None]
            block["prune"] = {
                "detected_fraction": len(delays) / len(rows),
                "mean_delay": mean(delays) if delays else None,
                "p90_delay": percentile(delays, 0.9) if delays else None,
            }
            block["final_post"] = {
                "mean_empirical_coverage": mean(
                    r["final_post"]["empirical_coverage"] for r in rows
                ),
                "mean_measure": mean(r["final_post"]["mean_measure"] for r in rows),
                "mean_count": mean(r["final_post"]["mean_count"] for r in rows),
            }
            block["post_mean_measure"] = mean(r["post"]["mean_measure"] for r in rows)
            block["post_mean_count"] = mean(r["post"]["mean_count"] for r in rows)
            block["ab_mean_measure"] = mean(r["AB"]["mean_measure"] for r in rows)
            block["ab_mean_count"] = mean(r["AB"]["mean_count"] for r in rows)
        out[policy] = block
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    checks: dict[str, Any] = {}

    failures = 0
    for scenario in ("stationary_AB", "hidden_AB_to_A"):
        for policy in ("window_multi", "window_single"):
            failures += aggregates[scenario][policy]["total_post_miss_failures"]
    checks["post_contradiction_containment"] = {
        "met": failures == 0,
        "failures": failures,
    }

    sta = aggregates["stationary_AB"]
    wm = sta["window_multi"]
    ws = sta["window_single"]
    sta_ev = evidence["stationary_AB"]
    checks["stationary_AB"] = {
        "met": (
            sta_ev["window_multi_count_min"]
            <= wm["mean_count"]
            <= sta_ev["window_multi_count_max"]
            and wm["mean_measure"] <= sta_ev["window_multi_measure_max"]
            and wm["mean_empirical_coverage"] >= sta_ev["coverage_min"]
            and wm["mean_exact_coverage"] >= sta_ev["exact_coverage_min"]
            and ws["mean_measure"] >= sta_ev["window_single_measure_min"]
        ),
        "window_multi_count": wm["mean_count"],
        "window_multi_measure": wm["mean_measure"],
        "window_multi_coverage": wm["mean_empirical_coverage"],
        "window_single_measure": ws["mean_measure"],
    }

    drop = aggregates["hidden_AB_to_A"]
    prune = drop["window_multi"]["prune"]
    checks["prune_AB_to_A"] = {
        "met": (
            prune["detected_fraction"] >= 0.95
            and prune["p90_delay"] is not None
            and prune["p90_delay"] <= evidence["prune"]["p90_delay_max"]
        ),
        "detected_fraction": prune["detected_fraction"],
        "p90_delay": prune["p90_delay"],
        "mean_delay": prune["mean_delay"],
    }

    final = drop["window_multi"]["final_post"]
    fin_ev = evidence["final_post"]
    checks["final_post"] = {
        "met": (
            final["mean_empirical_coverage"] >= fin_ev["empirical_coverage_min"]
            and final["mean_measure"] <= fin_ev["mean_measure_max"]
            and final["mean_count"] <= fin_ev["mean_count_max"]
        ),
        "empirical_coverage": final["mean_empirical_coverage"],
        "mean_measure": final["mean_measure"],
        "mean_count": final["mean_count"],
    }

    cum = drop["cumulative_multi"]["final_post"]
    cum_ev = evidence["cumulative_limitation"]
    checks["cumulative_limitation"] = {
        "met": (
            cum["mean_measure"] >= cum_ev["final_measure_min"]
            and cum["mean_count"] >= cum_ev["final_count_min"]
        ),
        "mean_measure": cum["mean_measure"],
        "mean_count": cum["mean_count"],
    }

    single_ab = drop["window_single"]["ab_mean_measure"]
    checks["single_limitation"] = {
        "met": single_ab >= evidence["single_limitation"]["ab_phase_measure_min"],
        "ab_mean_measure": single_ab,
        "note": (
            "Window-single bridges under AB; after the window flushes it can "
            "shrink. Permanent dead-mode retention is the cumulative_multi failure."
        ),
    }

    checks["diagnostics"] = {
        "met": True,
        "diagnostic_only": True,
        "hidden_AB_to_B": aggregates.get("hidden_AB_to_B"),
        "sensitivity": aggregates.get("sensitivity"),
    }
    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "post_contradiction_containment",
        "stationary_AB",
        "prune_AB_to_A",
        "final_post",
        "cumulative_limitation",
        "single_limitation",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Window multi-cone pruning drops vanished directional modes; "
            "single-window bridging and cumulative multi-cones retain dead geometry."
            if all_met
            else "One or more multimodal prune gates failed; inspect checks."
        ),
        "decision": (
            "keep window multi-cone pruning"
            if all_met
            else "revise or remove window multi-cone pruning"
        ),
        "limits": (
            "Result limited to separated unimodal modes with fixed split/window. "
            "Mode birth/death churn, overlapping modes, continuous situations, "
            "and control remain untested."
        ),
    }


def run_sensitivity(
    protocol: Mapping[str, Any],
    seed_count: int = 20,
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    roll_w = int(protocol["rolling_coverage_window"])
    out: dict[str, Any] = {}
    for window in protocol["sensitivity_windows"]:
        delays: list[int] = []
        final_measures: list[float] = []
        final_counts: list[float] = []
        for seed in range(seed_count):
            stream = generate_stream(protocol, "hidden_AB_to_A", seed)
            result = run_learner_on_stream(
                "window_multi",
                stream,
                protocol,
                window_size=int(window),
            )
            post = [row for row in result["records"] if row["phase"] == "post"]
            roll = rolling_mean(
                [row["exact_support_coverage"] for row in post],
                roll_w,
            )
            measures = [row["pre_measure"] for row in post]
            counts = [row["pre_count"] for row in post]
            cov_min = evidence["prune"]["rolling_exact_coverage_min"]
            measure_max = evidence["prune"]["measure_max"]
            count_target = evidence["prune"]["count_target"]
            delay = None
            for index, value in enumerate(roll):
                if (
                    value >= cov_min
                    and measures[index] <= measure_max
                    and counts[index] <= count_target
                ):
                    delay = index
                    break
            if delay is not None:
                delays.append(delay)
            last_n = int(evidence["final_post"]["last_n"])
            final_rows = post[-last_n:]
            final_measures.append(mean(row["pre_measure"] for row in final_rows))
            final_counts.append(mean(row["pre_count"] for row in final_rows))
        out[str(window)] = {
            "n_seeds": seed_count,
            "prune_p90": percentile(delays, 0.9) if delays else None,
            "final_measure": mean(final_measures) if final_measures else None,
            "final_count": mean(final_counts) if final_counts else None,
        }
    return out


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

    aggregates: dict[str, Any] = {
        scenario: aggregate_scenario(protocol, scenario, summaries)
        for scenario, summaries in all_summaries.items()
    }
    aggregates["sensitivity"] = run_sensitivity(protocol)
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
    parser.add_argument("--output", default="artifacts/stage1i")
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])


if __name__ == "__main__":
    main()
