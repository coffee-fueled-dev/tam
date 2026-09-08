"""Stage 1H situational cone narrowing experiments."""

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
    support_cone,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sample_angle(
    support: Mapping[str, float],
    rng: random.Random,
    resolution: int = 1,
) -> int:
    center = float(support["center"])
    half_width = float(support["half_width"])
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
    schedule = config["schedule"]
    if scenario == "hidden_ABA":
        names = config["phase_names"]
        cursor = 0
        for index, (_, length) in enumerate(schedule):
            if step < cursor + int(length):
                return str(names[index])
            cursor += int(length)
        return str(names[-1])
    return str(schedule[0][0])


def generate_stream(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
) -> list[dict[str, Any]]:
    config = protocol["scenarios"][scenario]
    regimes = expand_schedule(config["schedule"])
    resolution = int(protocol["angle_resolution_degrees"])
    supports = protocol["supports"]
    rng = random.Random(seed * 1_000_003 + hash(scenario) % 10_000_019)
    stream: list[dict[str, Any]] = []
    for step, regime in enumerate(regimes):
        angle = sample_angle(supports[regime], rng, resolution)
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
    learner = make_learner(policy, window_size=window_size)
    supports = protocol["supports"]
    resolution = int(protocol["angle_resolution_degrees"])
    records: list[dict[str, Any]] = []
    contradictions = 0
    post_miss_failures = 0
    operations: dict[str, int] = {}
    freeze_step = None
    if policy == "frozen_a1":
        # Freeze at first transition out of A.
        for row in stream:
            if row["regime"] != "A":
                freeze_step = int(row["step"])
                break

    for row in stream:
        step = int(row["step"])
        if freeze_step is not None and step == freeze_step and not learner.frozen:
            learner.freeze()

        angle = int(row["angle"])
        regime = str(row["regime"])
        commitment = learner.predict(step)
        inside = commitment.contains(angle)
        if not inside:
            contradictions += 1

        refinement = learner.observe(angle)
        operations[refinement.operation] = operations.get(refinement.operation, 0) + 1
        if not inside and not refinement.after.contains(angle):
            post_miss_failures += 1

        exact = exact_support_coverage(commitment.cones, supports[regime], resolution)
        excess = excess_measure(commitment.cones, support_cone(supports, regime))
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
                "post_measure": refinement.after.measure,
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
        "learner": learner,
        "final_measure": learner.cone_set.measure,
    }


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        chunk = values[start : index + 1]
        out.append(mean(chunk))
    return out


def summarize_seed(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    burn_in = int(protocol["burn_in"])
    roll_w = int(protocol["rolling_coverage_window"])
    supports = protocol["supports"]
    summary: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "learners": {},
    }

    for policy, result in results.items():
        records = result["records"]
        block: dict[str, Any] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "operations": result["operations"],
            "final_measure": result["final_measure"],
        }

        if scenario in ("stationary_A", "stationary_B"):
            regime = "A" if scenario.endswith("A") else "B"
            after = [row for row in records if row["step"] >= burn_in]
            block["empirical_coverage"] = (
                mean(1.0 if row["inside_pre"] else 0.0 for row in after)
                if after
                else 1.0
            )
            block["exact_coverage"] = (
                mean(row["exact_support_coverage"] for row in after) if after else 1.0
            )
            block["mean_excess"] = mean(row["excess"] for row in after) if after else 0.0
            block["mean_measure"] = (
                mean(row["pre_measure"] for row in after) if after else 0.0
            )
            block["oracle_measure"] = support_cone(supports, regime).measure

        if scenario == "hidden_ABA":
            by_phase: dict[str, list[Mapping[str, Any]]] = {
                "A1": [],
                "B": [],
                "A2": [],
            }
            for row in records:
                by_phase[str(row["phase"])].append(row)

            for phase, rows in by_phase.items():
                block[phase] = {
                    "empirical_coverage": mean(
                        1.0 if row["inside_pre"] else 0.0 for row in rows
                    )
                    if rows
                    else 1.0,
                    "exact_coverage": mean(
                        row["exact_support_coverage"] for row in rows
                    )
                    if rows
                    else 1.0,
                    "mean_measure": mean(row["pre_measure"] for row in rows)
                    if rows
                    else 0.0,
                    "mean_excess": mean(row["excess"] for row in rows) if rows else 0.0,
                }

            # Expansion delay: rolling exact coverage in B
            b_exact = [row["exact_support_coverage"] for row in by_phase["B"]]
            b_roll = rolling_mean(b_exact, roll_w)
            target = evidence["expansion"]["rolling_exact_coverage_min"]
            expansion_delay = next(
                (index for index, value in enumerate(b_roll) if value >= target),
                None,
            )
            block["expansion_delay"] = expansion_delay

            # Contraction delay: rolling exact coverage and measure in A2
            a2_exact = [row["exact_support_coverage"] for row in by_phase["A2"]]
            a2_measure = [row["pre_measure"] for row in by_phase["A2"]]
            a2_roll = rolling_mean(a2_exact, roll_w)
            cov_min = evidence["contraction"]["rolling_exact_coverage_min"]
            measure_max = evidence["contraction"]["measure_max"]
            contraction_delay = None
            for index, value in enumerate(a2_roll):
                if value >= cov_min and a2_measure[index] <= measure_max:
                    contraction_delay = index
                    break
            block["contraction_delay"] = contraction_delay

            last_n = int(evidence["final_a2"]["last_n"])
            final_rows = by_phase["A2"][-last_n:]
            block["final_a2_empirical_coverage"] = (
                mean(1.0 if row["inside_pre"] else 0.0 for row in final_rows)
                if final_rows
                else 1.0
            )
            block["final_a2_mean_measure"] = (
                mean(row["pre_measure"] for row in final_rows) if final_rows else 0.0
            )
            block["final_a2_mean_excess"] = (
                mean(row["excess"] for row in final_rows) if final_rows else 0.0
            )
            # Width churn: absolute measure changes
            measures = [row["pre_measure"] for row in records]
            churn = (
                mean(abs(measures[i] - measures[i - 1]) for i in range(1, len(measures)))
                if len(measures) > 1
                else 0.0
            )
            block["width_churn"] = churn

        summary["learners"][policy] = block
    return summary


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
        if scenario in ("stationary_A", "stationary_B"):
            block.update(
                {
                    "mean_empirical_coverage": mean(r["empirical_coverage"] for r in rows),
                    "mean_exact_coverage": mean(r["exact_coverage"] for r in rows),
                    "mean_excess": mean(r["mean_excess"] for r in rows),
                    "mean_measure": mean(r["mean_measure"] for r in rows),
                }
            )
        if scenario == "hidden_ABA":
            for phase in ("A1", "B", "A2"):
                block[phase] = {
                    "mean_empirical_coverage": mean(
                        r[phase]["empirical_coverage"] for r in rows
                    ),
                    "mean_exact_coverage": mean(
                        r[phase]["exact_coverage"] for r in rows
                    ),
                    "mean_measure": mean(r[phase]["mean_measure"] for r in rows),
                    "mean_excess": mean(r[phase]["mean_excess"] for r in rows),
                }
            expansion = [r["expansion_delay"] for r in rows if r["expansion_delay"] is not None]
            contraction = [
                r["contraction_delay"] for r in rows if r["contraction_delay"] is not None
            ]
            block["expansion"] = {
                "detected_fraction": len(expansion) / len(rows),
                "mean_delay": mean(expansion) if expansion else None,
                "p90_delay": percentile(expansion, 0.9) if expansion else None,
            }
            block["contraction"] = {
                "detected_fraction": len(contraction) / len(rows),
                "mean_delay": mean(contraction) if contraction else None,
                "p90_delay": percentile(contraction, 0.9) if contraction else None,
            }
            block["final_a2"] = {
                "mean_empirical_coverage": mean(
                    r["final_a2_empirical_coverage"] for r in rows
                ),
                "mean_measure": mean(r["final_a2_mean_measure"] for r in rows),
                "mean_excess": mean(r["final_a2_mean_excess"] for r in rows),
            }
            block["mean_width_churn"] = mean(r["width_churn"] for r in rows)
        out[policy] = block
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    checks: dict[str, Any] = {}

    failures = 0
    for scenario in protocol["scenarios"]:
        agg = aggregates[scenario]
        for policy in ("window64", "cumulative"):
            failures += agg[policy]["total_post_miss_failures"]
    checks["post_contradiction_containment"] = {
        "met": failures == 0,
        "failures": failures,
    }

    stationary_ok = True
    stationary_detail = {}
    for scenario in ("stationary_A", "stationary_B"):
        block = aggregates[scenario]["window64"]
        stationary_detail[scenario] = {
            "empirical": block["mean_empirical_coverage"],
            "exact": block["mean_exact_coverage"],
            "excess": block["mean_excess"],
        }
        if (
            block["mean_empirical_coverage"] < evidence["stationary"]["coverage_min"]
            or block["mean_exact_coverage"] < evidence["stationary"]["exact_coverage_min"]
            or block["mean_excess"] > evidence["stationary"]["excess_max"] + 1e-9
        ):
            stationary_ok = False
    checks["stationary"] = {"met": stationary_ok, "detail": stationary_detail}

    aba = aggregates["hidden_ABA"]
    window = aba["window64"]
    expansion = window["expansion"]
    checks["expansion"] = {
        "met": (
            expansion["detected_fraction"] >= 0.95
            and expansion["p90_delay"] is not None
            and expansion["p90_delay"] <= evidence["expansion"]["p90_delay_max"]
        ),
        "detected_fraction": expansion["detected_fraction"],
        "p90_delay": expansion["p90_delay"],
        "mean_delay": expansion["mean_delay"],
    }

    contraction = window["contraction"]
    checks["contraction"] = {
        "met": (
            contraction["detected_fraction"] >= 0.95
            and contraction["p90_delay"] is not None
            and contraction["p90_delay"] <= evidence["contraction"]["p90_delay_max"]
        ),
        "detected_fraction": contraction["detected_fraction"],
        "p90_delay": contraction["p90_delay"],
        "mean_delay": contraction["mean_delay"],
    }

    final = window["final_a2"]
    checks["final_a2"] = {
        "met": (
            final["mean_empirical_coverage"]
            >= evidence["final_a2"]["empirical_coverage_min"]
            and final["mean_measure"] <= evidence["final_a2"]["mean_measure_max"]
        ),
        "empirical_coverage": final["mean_empirical_coverage"],
        "mean_measure": final["mean_measure"],
    }

    cum = aba["cumulative"]["final_a2"]
    checks["cumulative_limitation"] = {
        "met": (
            cum["mean_measure"] >= evidence["cumulative_limitation"]["final_a2_measure_min"]
            and cum["mean_excess"]
            >= evidence["cumulative_limitation"]["final_a2_excess_min"]
        ),
        "mean_measure": cum["mean_measure"],
        "mean_excess": cum["mean_excess"],
    }

    frozen = aba["frozen_a1"]
    checks["frozen_tradeoff"] = {
        "met": (
            frozen["B"]["mean_exact_coverage"]
            <= evidence["frozen_tradeoff"]["b_exact_coverage_max"]
            and frozen["A2"]["mean_exact_coverage"]
            >= evidence["frozen_tradeoff"]["a2_exact_coverage_min"]
        ),
        "b_exact_coverage": frozen["B"]["mean_exact_coverage"],
        "a2_exact_coverage": frozen["A2"]["mean_exact_coverage"],
    }

    checks["diagnostics"] = {
        "met": True,
        "diagnostic_only": True,
        "width_churn_window64": window["mean_width_churn"],
        "sensitivity": aggregates.get("sensitivity"),
    }
    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "post_contradiction_containment",
        "stationary",
        "expansion",
        "contraction",
        "final_a2",
        "cumulative_limitation",
        "frozen_tradeoff",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Rolling-window cones widen under hidden expansion and narrow after "
            "return to tight support at matched coverage; cumulative cones remain wide."
            if all_met
            else "One or more narrowing gates failed; inspect checks."
        ),
        "decision": (
            "keep rolling situational narrowing"
            if all_met
            else "revise or remove rolling-window narrowing"
        ),
        "limits": (
            "Result limited to bounded unimodal one-step directional support. "
            "Multimodal contraction, continuous situations, control, and optimal "
            "window selection remain untested."
        ),
    }


def run_sensitivity(
    protocol: Mapping[str, Any],
    seed_count: int = 20,
) -> dict[str, Any]:
    """Window-size sensitivity on hidden_ABA for a subset of seeds."""
    evidence = protocol["evidence"]
    roll_w = int(protocol["rolling_coverage_window"])
    out: dict[str, Any] = {}
    for window in protocol["sensitivity_windows"]:
        expansion_delays: list[int] = []
        contraction_delays: list[int] = []
        final_measures: list[float] = []
        final_coverages: list[float] = []
        for seed in range(seed_count):
            stream = generate_stream(protocol, "hidden_ABA", seed)
            result = run_learner_on_stream(
                f"window{window}",
                stream,
                protocol,
                window_size=int(window),
            )
            records = result["records"]
            by_phase = {"B": [], "A2": []}
            for row in records:
                if row["phase"] in by_phase:
                    by_phase[row["phase"]].append(row)

            b_roll = rolling_mean(
                [row["exact_support_coverage"] for row in by_phase["B"]],
                roll_w,
            )
            target = evidence["expansion"]["rolling_exact_coverage_min"]
            delay = next(
                (index for index, value in enumerate(b_roll) if value >= target),
                None,
            )
            if delay is not None:
                expansion_delays.append(delay)

            a2_exact = [row["exact_support_coverage"] for row in by_phase["A2"]]
            a2_measure = [row["pre_measure"] for row in by_phase["A2"]]
            a2_roll = rolling_mean(a2_exact, roll_w)
            cov_min = evidence["contraction"]["rolling_exact_coverage_min"]
            measure_max = evidence["contraction"]["measure_max"]
            cdelay = None
            for index, value in enumerate(a2_roll):
                if value >= cov_min and a2_measure[index] <= measure_max:
                    cdelay = index
                    break
            if cdelay is not None:
                contraction_delays.append(cdelay)

            last_n = int(evidence["final_a2"]["last_n"])
            final_rows = by_phase["A2"][-last_n:]
            final_measures.append(mean(row["pre_measure"] for row in final_rows))
            final_coverages.append(
                mean(1.0 if row["inside_pre"] else 0.0 for row in final_rows)
            )

        out[str(window)] = {
            "n_seeds": seed_count,
            "expansion_p90": (
                percentile(expansion_delays, 0.9) if expansion_delays else None
            ),
            "contraction_p90": (
                percentile(contraction_delays, 0.9) if contraction_delays else None
            ),
            "final_a2_measure": mean(final_measures) if final_measures else None,
            "final_a2_coverage": mean(final_coverages) if final_coverages else None,
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

    aggregates = {
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
    parser.add_argument("--output", default="artifacts/stage1h")
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])


if __name__ == "__main__":
    main()
