"""Stage 1G situational cone-volume experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .model import (
    CIRCLE,
    AngularCone,
    ConeSet,
    Refinement,
    SituationalLearner,
    dominant_center,
    normalize_angle,
    situation_excess,
    support_for,
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


def build_schedule(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    schedule_cfg = protocol["schedule"]
    interleave = list(schedule_cfg["interleave"])
    events: list[dict[str, Any]] = []

    for step in range(int(schedule_cfg["train_length"])):
        situation = interleave[step % len(interleave)]
        events.append(
            {
                "phase": "train",
                "situation": situation,
                "support_key": situation,
                "learn": True,
            }
        )

    for step in range(int(schedule_cfg["eval_length"])):
        situation = interleave[step % len(interleave)]
        events.append(
            {
                "phase": "eval",
                "situation": situation,
                "support_key": situation,
                "learn": False,
            }
        )

    diagnostic = protocol["hidden_volume_change"]
    for _ in range(int(schedule_cfg["diagnostic_length"])):
        events.append(
            {
                "phase": "diagnostic",
                "situation": diagnostic["situation_label"],
                "support_key": diagnostic["support_like"],
                "learn": True,
            }
        )
    return events


def generate_stream(
    protocol: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    events = build_schedule(protocol)
    resolution = int(protocol["angle_resolution_degrees"])
    supports = protocol["supports"]
    rng = random.Random(seed * 1_000_003 + 17)
    stream: list[dict[str, Any]] = []
    for step, event in enumerate(events):
        angle = sample_angle(supports[event["support_key"]], rng, resolution)
        stream.append(
            {
                "step": step,
                "angle": angle,
                "phase": event["phase"],
                "situation": event["situation"],
                "support_key": event["support_key"],
                "learn": event["learn"],
            }
        )
    return stream


def _coneset_from_dict(payload: Mapping[str, Any]) -> ConeSet:
    cones = tuple(
        AngularCone(float(cone["center"]), float(cone["half_width"]))
        for cone in payload["cones"]
    )
    return ConeSet(cones)


def run_learner_on_stream(
    policy: str,
    stream: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    learner = SituationalLearner(
        policy,
        situations=protocol["situations"],
        split_gap=float(protocol["split_gap_degrees"]),
    )
    records: list[dict[str, Any]] = []
    contradictions = 0
    post_miss_failures = 0
    freeze_snapshot: dict[str, dict[str, Any]] | None = None
    entered_eval = False

    for row in stream:
        if row["phase"] == "eval" and not entered_eval:
            learner.freeze()
            freeze_snapshot = {
                situation: learner.cone_set(situation).to_dict()
                for situation in protocol["situations"]
            }
            entered_eval = True
        if row["phase"] == "diagnostic" and learner.frozen:
            learner.frozen = False

        situation = row["situation"]
        angle = int(row["angle"])
        commitment = learner.predict(situation, int(row["step"]))
        inside = commitment.contains(angle)
        if not inside:
            contradictions += 1

        if row["learn"]:
            refinement = learner.observe(situation, angle)
            if not inside and not refinement.after.contains(angle):
                post_miss_failures += 1
        else:
            before = learner.cone_set(situation)
            refinement = Refinement("noop", before, before, float(angle))

        records.append(
            {
                "step": row["step"],
                "phase": row["phase"],
                "situation": situation,
                "support_key": row["support_key"],
                "angle": angle,
                "inside_pre": inside,
                "operation": refinement.operation,
                "pre_measure": commitment.cones.measure,
                "post_measure": refinement.after.measure,
                "pre_count": len(commitment.cones.cones),
                "post": refinement.after.to_dict(),
            }
        )

    return {
        "policy": policy,
        "contradictions": contradictions,
        "post_miss_failures": post_miss_failures,
        "records": records,
        "learner": learner,
        "freeze_snapshot": freeze_snapshot or {},
        "final": {
            situation: learner.cone_set(situation).to_dict()
            for situation in protocol["situations"]
        },
    }


def _phase_rows(
    records: Sequence[Mapping[str, Any]],
    phase: str,
    situation: str | None = None,
) -> list[Mapping[str, Any]]:
    rows = [row for row in records if row["phase"] == phase]
    if situation is not None:
        rows = [row for row in rows if row["situation"] == situation]
    return rows


def summarize_seed(
    protocol: Mapping[str, Any],
    seed: int,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    supports = protocol["supports"]
    summary: dict[str, Any] = {"seed": seed, "learners": {}}

    for policy, result in results.items():
        learner = result["learner"]
        per_situation: dict[str, Any] = {}
        for situation in protocol["situations"]:
            eval_rows = _phase_rows(result["records"], "eval", situation)
            coverage = (
                mean(1.0 if row["inside_pre"] else 0.0 for row in eval_rows)
                if eval_rows
                else 1.0
            )
            snapshot = result["freeze_snapshot"].get(situation)
            cone_set = (
                _coneset_from_dict(snapshot)
                if snapshot is not None
                else learner.cone_set(situation)
            )
            excess = situation_excess(cone_set, supports, situation)
            per_situation[situation] = {
                "held_out_coverage": coverage,
                "measure": cone_set.measure,
                "excess": excess,
                "center": dominant_center(cone_set),
                "count": len(cone_set.cones),
                "oracle_measure": support_for(supports, situation).measure,
            }

        diag_rows = _phase_rows(result["records"], "diagnostic")
        diag_measure = learner.cone_set("tight").measure if diag_rows else None
        wide_measure = float(supports["wide"]["half_width"]) * 2
        diag_overwide = (
            None if diag_measure is None else diag_measure >= wide_measure - 5
        )

        summary["learners"][policy] = {
            "contradictions": result["contradictions"],
            "post_miss_failures": result["post_miss_failures"],
            "situations": per_situation,
            "diagnostic_tight_measure": diag_measure,
            "diagnostic_overwide": diag_overwide,
            "freeze_snapshot": result["freeze_snapshot"],
        }

    if (
        "unconditional_multi" in summary["learners"]
        and "situation_multi" in summary["learners"]
    ):
        summary["volume_gap_tight"] = (
            summary["learners"]["unconditional_multi"]["situations"]["tight"][
                "measure"
            ]
            - summary["learners"]["situation_multi"]["situations"]["tight"]["measure"]
        )
    return summary


def aggregate(
    protocol: Mapping[str, Any],
    seed_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(seed_summaries), "learners": {}}
    for policy in protocol["learners"]:
        rows = [summary["learners"][policy] for summary in seed_summaries]
        block: dict[str, Any] = {
            "mean_contradictions": mean(r["contradictions"] for r in rows),
            "total_post_miss_failures": sum(r["post_miss_failures"] for r in rows),
            "situations": {},
            "fraction_diagnostic_overwide": mean(
                1.0 if r["diagnostic_overwide"] else 0.0
                for r in rows
                if r["diagnostic_overwide"] is not None
            ),
            "mean_diagnostic_tight_measure": mean(
                r["diagnostic_tight_measure"]
                for r in rows
                if r["diagnostic_tight_measure"] is not None
            ),
        }
        for situation in protocol["situations"]:
            sit_rows = [r["situations"][situation] for r in rows]
            centers = [s["center"] for s in sit_rows if s["center"] is not None]
            block["situations"][situation] = {
                "mean_held_out_coverage": mean(
                    s["held_out_coverage"] for s in sit_rows
                ),
                "mean_measure": mean(s["measure"] for s in sit_rows),
                "mean_excess": mean(s["excess"] for s in sit_rows),
                "mean_count": mean(s["count"] for s in sit_rows),
                "mean_center": mean(centers) if centers else None,
            }
        out["learners"][policy] = block

    gaps = [s["volume_gap_tight"] for s in seed_summaries if "volume_gap_tight" in s]
    out["mean_volume_gap_tight"] = mean(gaps) if gaps else None
    return out


def evaluate_evidence(
    protocol: Mapping[str, Any],
    aggregates: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = protocol["evidence"]
    learners = aggregates["learners"]
    checks: dict[str, Any] = {}

    total_failures = sum(
        learners[policy]["total_post_miss_failures"] for policy in protocol["learners"]
    )
    checks["contradiction_post_containment"] = {
        "met": total_failures == 0,
        "failures": total_failures,
    }

    sit = learners["situation_multi"]
    coverage_ok = all(
        sit["situations"][situation]["mean_held_out_coverage"]
        >= evidence["situation_multi_coverage_min"]
        for situation in protocol["situations"]
    )
    checks["situation_multi_coverage"] = {
        "met": coverage_ok,
        "detail": {
            situation: sit["situations"][situation]["mean_held_out_coverage"]
            for situation in protocol["situations"]
        },
        "threshold": evidence["situation_multi_coverage_min"],
    }

    tight_ev = evidence["tight"]
    sit_tight = sit["situations"]["tight"]
    unc_tight = learners["unconditional_multi"]["situations"]["tight"]
    gap = aggregates["mean_volume_gap_tight"]
    contamination = unc_tight["mean_measure"] >= tight_ev[
        "unconditional_multi_measure_min"
    ] or (gap is not None and gap >= tight_ev["volume_gap_min"])
    checks["tight_volume"] = {
        "met": (
            sit_tight["mean_measure"] <= tight_ev["situation_multi_measure_max"]
            and sit_tight["mean_excess"] <= tight_ev["situation_multi_excess_max"]
            and contamination
        ),
        "situation_multi_measure": sit_tight["mean_measure"],
        "situation_multi_excess": sit_tight["mean_excess"],
        "unconditional_multi_measure": unc_tight["mean_measure"],
        "volume_gap": gap,
    }

    shifted_ev = evidence["shifted"]
    sit_shifted = sit["situations"]["shifted"]
    center = sit_shifted["mean_center"]
    assert center is not None
    delta = min(abs(center - shifted_ev["center_near"]), CIRCLE - abs(center - shifted_ev["center_near"]))
    checks["shifted_local"] = {
        "met": (
            sit_shifted["mean_measure"] <= shifted_ev["situation_multi_measure_max"]
            and sit_shifted["mean_excess"] <= shifted_ev["situation_multi_excess_max"]
            and delta <= shifted_ev["center_tolerance"]
            and sit_shifted["mean_measure"]
            < sit["situations"]["wide"]["mean_measure"]
        ),
        "measure": sit_shifted["mean_measure"],
        "excess": sit_shifted["mean_excess"],
        "center": center,
        "wide_measure": sit["situations"]["wide"]["mean_measure"],
    }

    full = learners["full_circle"]["situations"]["tight"]["mean_measure"]
    checks["full_circle"] = {
        "met": abs(full - evidence["full_circle_measure"]) < 1e-6,
        "measure": full,
    }

    checks["diagnostic_no_narrowing"] = {
        "met": True,
        "diagnostic_only": True,
        "fraction_situation_multi_overwide": learners["situation_multi"][
            "fraction_diagnostic_overwide"
        ],
        "mean_diagnostic_tight_measure": learners["situation_multi"][
            "mean_diagnostic_tight_measure"
        ],
        "note": (
            "Cumulative situation_multi is expected to remain over-wide after "
            "hidden volume expansion under the tight label; narrowing is out of scope."
        ),
    }
    return checks


def decide(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "contradiction_post_containment",
        "situation_multi_coverage",
        "tight_volume",
        "shifted_local",
        "full_circle",
    )
    met = {name: bool(checks[name]["met"]) for name in required}
    all_met = all(met.values())
    return {
        "verdict": "PASS" if all_met else "FAIL",
        "met": met,
        "interpretation": (
            "Situation-conditioned geometric stores keep tight claims in tight "
            "contexts; unconditional pooling over-claims by absorbing wide support."
            if all_met
            else "One or more situational volume gates failed; inspect checks."
        ),
        "decision": (
            "keep situation-conditioned geometric cone stores"
            if all_met
            else "revise or remove situation conditioning"
        ),
        "limits": (
            "No narrowing/forgetting, control, continuous state, or safety claim. "
            "Hidden volume change under a fixed label remains a documented limitation."
        ),
    }


def run_experiment(output: str | Path | None = None) -> dict[str, Any]:
    protocol = load_protocol()
    started = time.perf_counter()
    seed_count = int(protocol["seeds"])
    seed_summaries: list[dict[str, Any]] = []

    for seed in range(seed_count):
        stream = generate_stream(protocol, seed)
        results = {
            policy: run_learner_on_stream(policy, stream, protocol)
            for policy in protocol["learners"]
        }
        seed_summaries.append(summarize_seed(protocol, seed, results))

    aggregates = aggregate(protocol, seed_summaries)
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
    parser.add_argument("--output", default="artifacts/stage1g")
    args = parser.parse_args(argv)
    result = run_experiment(args.output)
    decision = result["decision"]
    print(json.dumps({"verdict": decision["verdict"], "met": decision["met"]}, indent=2))
    print(decision["interpretation"])


if __name__ == "__main__":
    main()
