"""Stage 1J mode churn / hysteresis experiments."""

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


def build_churn_schedule(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return list of (regime, phase_tag)."""
    events: list[tuple[str, str]] = []
    warmup = int(config["warmup_AB"])
    events.extend([("AB", "warmup")] * warmup)
    off_len = int(config["off_len"])
    on_len = int(config["on_len"])
    for cycle in range(int(config["cycles"])):
        events.extend([("A", f"gap_{cycle}")] * off_len)
        events.extend([("AB", f"on_{cycle}")] * on_len)
    return events


def generate_stream(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> list[dict[str, Any]]:
    config = protocol["scenarios"][scenario]
    resolution = int(protocol["angle_resolution_degrees"])
    rng = random.Random(seed * 1_000_003 + hash(scenario) % 10_000_019)

    if scenario in ("fast_churn", "slow_churn"):
        events = build_churn_schedule(config)
    else:
        regimes = expand_schedule(config["schedule"])
        names = config.get("phase_names")
        events = []
        if names:
            cursor = 0
            for index, (_, length) in enumerate(config["schedule"]):
                for _ in range(int(length)):
                    events.append((regimes[cursor], str(names[index])))
                    cursor += 1
        else:
            events = [(regime, regime) for regime in regimes]

    stream: list[dict[str, Any]] = []
    for step, (regime, phase) in enumerate(events):
        modes = regime_modes(protocol, regime)
        angle = sample_from_modes(modes, rng, resolution)
        stream.append(
            {
                "step": step,
                "angle": angle,
                "regime": regime,
                "phase": phase,
                "situation": protocol.get("situation_label", SITUATION_LABEL),
            }
        )
    return stream


def run_learner_on_stream(
    policy: str,
    stream: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    window_size: int | None = None,
    grace_t: int | None = None,
) -> dict[str, Any]:
    learner = make_learner(
        policy,
        window_size=window_size or int(protocol["primary_window"]),
        split_gap=float(protocol["split_gap_degrees"]),
        grace_t=grace_t if grace_t is not None else int(protocol["grace_T"]),
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
        if len(refinement.before.cones) > len(refinement.after.cones):
            operations["count_drop"] = operations.get("count_drop", 0) + 1
        if not inside and not refinement.after.contains(angle):
            post_miss_failures += 1

        b_cov = exact_support_coverage(
            commitment.cones, [dict(protocol["modes"]["B"])], resolution
        )
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
                "exact_support_coverage": exact_support_coverage(
                    commitment.cones, modes, resolution
                ),
                "b_mode_coverage": b_cov,
                "excess": max(
                    0.0,
                    commitment.cones.measure - support_cone(modes).measure,
                ),
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


def prune_rate_per_1000(operations: Mapping[str, int], n_steps: int) -> float:
    if n_steps <= 0:
        return 0.0
    drops = float(operations.get("count_drop", 0) + operations.get("prune", 0))
    return 1000.0 * drops / float(n_steps)


def summarize_seed(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    burn_in = int(protocol["burn_in"])
    summary: dict[str, Any] = {"scenario": scenario, "seed": seed, "learners": {}}

    for policy, result in results.items():
        records = result["records"]
        block: dict[str, Any] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "operations": result["operations"],
            "prune_rate_per_1000": prune_rate_per_1000(
                result["operations"], len(records)
            ),
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
            block["mean_measure"] = (
                mean(row["pre_measure"] for row in after) if after else 0.0
            )
            block["mean_count"] = (
                mean(row["pre_count"] for row in after) if after else 0.0
            )

        if scenario == "sustained_drop":
            post = [row for row in records if row["phase"] == "post"]
            last_n = int(evidence["sustained_drop"]["last_n"])
            final_rows = post[-last_n:]
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
            measure_max = evidence["sustained_drop"]["final_measure_max"]
            count_max = evidence["sustained_drop"]["final_count_max"]
            cov_min = evidence["sustained_drop"]["empirical_coverage_min"]
            delay = None
            for offset, row in enumerate(post):
                # Require a short stable run of tight geometry.
                window = post[offset : offset + 20]
                if len(window) < 20:
                    break
                if (
                    all(r["pre_measure"] <= measure_max for r in window)
                    and all(r["pre_count"] <= count_max for r in window)
                    and mean(1.0 if r["inside_pre"] else 0.0 for r in window) >= cov_min
                ):
                    delay = offset
                    break
            block["drop_delay"] = delay

        if scenario in ("fast_churn", "slow_churn"):
            gap_end_steps = int(
                protocol["evidence"]["fast_churn"].get("gap_end_steps", 5)
            )
            gap_end_rows: list[Mapping[str, Any]] = []
            # Collect last gap_end_steps of each gap_* phase.
            by_gap: dict[str, list[Mapping[str, Any]]] = {}
            for row in records:
                phase = str(row["phase"])
                if phase.startswith("gap_"):
                    by_gap.setdefault(phase, []).append(row)
            for rows in by_gap.values():
                gap_end_rows.extend(rows[-gap_end_steps:])
            on_rows = [row for row in records if str(row["phase"]).startswith("on_")]
            block["gap_end_mean_count"] = (
                mean(row["pre_count"] for row in gap_end_rows) if gap_end_rows else 0.0
            )
            block["on_b_coverage"] = (
                mean(row["b_mode_coverage"] for row in on_rows) if on_rows else 1.0
            )

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
            "mean_prune_rate_per_1000": mean(r["prune_rate_per_1000"] for r in rows),
        }
        if scenario == "stationary_AB":
            block.update(
                {
                    "mean_empirical_coverage": mean(
                        r["empirical_coverage"] for r in rows
                    ),
                    "mean_measure": mean(r["mean_measure"] for r in rows),
                    "mean_count": mean(r["mean_count"] for r in rows),
                }
            )
        if scenario == "sustained_drop":
            delays = [r["drop_delay"] for r in rows if r["drop_delay"] is not None]
            block["final_post"] = {
                "mean_empirical_coverage": mean(
                    r["final_post"]["empirical_coverage"] for r in rows
                ),
                "mean_measure": mean(r["final_post"]["mean_measure"] for r in rows),
                "mean_count": mean(r["final_post"]["mean_count"] for r in rows),
            }
            block["drop"] = {
                "detected_fraction": len(delays) / len(rows),
                "mean_delay": mean(delays) if delays else None,
                "p90_delay": percentile(delays, 0.9) if delays else None,
            }
        if scenario in ("fast_churn", "slow_churn"):
            block["gap_end_mean_count"] = mean(r["gap_end_mean_count"] for r in rows)
            block["on_b_coverage"] = mean(r["on_b_coverage"] for r in rows)
        out[policy] = block
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    checks: dict[str, Any] = {}

    failures = 0
    for scenario in ("stationary_AB", "sustained_drop", "fast_churn"):
        for policy in ("window_multi", "hysteresis_multi"):
            failures += aggregates[scenario][policy]["total_post_miss_failures"]
    checks["post_contradiction_containment"] = {
        "met": failures == 0,
        "failures": failures,
    }

    sta = aggregates["stationary_AB"]
    sta_ev = evidence["stationary_AB"]
    sta_ok = True
    detail = {}
    for policy in ("window_multi", "hysteresis_multi"):
        block = sta[policy]
        detail[policy] = {
            "coverage": block["mean_empirical_coverage"],
            "count": block["mean_count"],
            "measure": block["mean_measure"],
        }
        if (
            block["mean_empirical_coverage"] < sta_ev["coverage_min"]
            or not (
                sta_ev["count_min"] <= block["mean_count"] <= sta_ev["count_max"]
            )
            or block["mean_measure"] > sta_ev["measure_max"]
        ):
            sta_ok = False
    checks["stationary_AB"] = {"met": sta_ok, "detail": detail}

    fast = aggregates["fast_churn"]
    hyst = fast["hysteresis_multi"]
    win = fast["window_multi"]
    fast_ev = evidence["fast_churn"]
    prune_ratio = (
        hyst["mean_prune_rate_per_1000"] / win["mean_prune_rate_per_1000"]
        if win["mean_prune_rate_per_1000"] > 1e-9
        else (0.0 if hyst["mean_prune_rate_per_1000"] <= 1e-9 else 1.0)
    )
    checks["fast_churn_hysteresis"] = {
        "met": (
            hyst["gap_end_mean_count"] >= fast_ev["hysteresis_gap_end_count_min"]
            and hyst["on_b_coverage"] >= fast_ev["hysteresis_onblock_b_coverage_min"]
            and prune_ratio <= fast_ev["prune_rate_ratio_max"]
        ),
        "gap_end_mean_count": hyst["gap_end_mean_count"],
        "on_b_coverage": hyst["on_b_coverage"],
        "prune_rate_ratio": prune_ratio,
        "hysteresis_prune_rate": hyst["mean_prune_rate_per_1000"],
        "window_prune_rate": win["mean_prune_rate_per_1000"],
    }

    lim = evidence["window_multi_limitation"]
    checks["window_multi_limitation"] = {
        "met": win["gap_end_mean_count"] <= lim["gap_end_count_max"],
        "gap_end_mean_count": win["gap_end_mean_count"],
        "prune_rate": win["mean_prune_rate_per_1000"],
    }

    drop = aggregates["sustained_drop"]
    hdrop = drop["hysteresis_multi"]
    sev = evidence["sustained_drop"]
    checks["sustained_drop"] = {
        "met": (
            hdrop["final_post"]["mean_measure"] <= sev["final_measure_max"]
            and hdrop["final_post"]["mean_count"] <= sev["final_count_max"]
            and hdrop["final_post"]["mean_empirical_coverage"]
            >= sev["empirical_coverage_min"]
            and hdrop["drop"]["detected_fraction"] >= 0.95
            and hdrop["drop"]["p90_delay"] is not None
            and hdrop["drop"]["p90_delay"] <= sev["p90_delay_max"]
        ),
        "final": hdrop["final_post"],
        "drop": hdrop["drop"],
    }

    cum = drop["cumulative_multi"]["final_post"]
    cev = evidence["cumulative_limitation"]
    checks["cumulative_limitation"] = {
        "met": (
            cum["mean_count"] >= cev["final_count_min"]
            and cum["mean_measure"] >= cev["final_measure_min"]
        ),
        "final": cum,
    }

    checks["diagnostics"] = {
        "met": True,
        "diagnostic_only": True,
        "slow_churn": aggregates.get("slow_churn"),
    }
    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "post_contradiction_containment",
        "stationary_AB",
        "fast_churn_hysteresis",
        "window_multi_limitation",
        "sustained_drop",
        "cumulative_limitation",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Sticky hysteresis retains modes across short flicker gaps and cuts "
            "prune churn versus plain window rebuild, while still pruning after "
            "sustained absence."
            if all_met
            else "One or more churn/hysteresis gates failed; inspect checks."
        ),
        "decision": (
            "keep sticky multi-cone hysteresis"
            if all_met
            else "revise or remove sticky hysteresis"
        ),
        "limits": (
            "Fixed (W,T,split_gap) on two separated modes only. Overlapping modes, "
            "continuous situations, and learned grace remain untested."
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
    parser.add_argument("--output", default="artifacts/stage1j")
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])


if __name__ == "__main__":
    main()
