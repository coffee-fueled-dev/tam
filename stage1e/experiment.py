"""Stage 1E commitment ledger, stationary audit, and drift detection."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from fractions import Fraction
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from .ledger import (
    LedgerWriter,
    build_seal,
    code_file_hashes,
    digest_payload,
    protocol_hash,
    write_seal,
)
from .model import (
    OUTCOMES,
    PORTS,
    Commitment,
    OutcomeModel,
    brier_score,
    build_commitment,
    make_model,
    negative_log_likelihood,
    oracle_probabilities,
    total_variation,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "protocol.json"
CODE_FILES = (
    "__init__.py",
    "protocol.json",
    "model.py",
    "ledger.py",
    "experiment.py",
    "replay.py",
    "test_stage1e.py",
    "README.md",
)


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def regime_tables(protocol: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        name: {
            port: {str(outcome): float(table[str(outcome)]) for outcome in OUTCOMES}
            for port, table in regime.items()
        }
        for name, regime in protocol["regimes"].items()
    }


def expand_schedule(schedule: list[list[Any]]) -> list[str]:
    labels: list[str] = []
    for name, length in schedule:
        labels.extend([str(name)] * int(length))
    return labels


def forced_port(step: int) -> str:
    return PORTS[step % len(PORTS)]


def sample_outcome(
    regime: Mapping[str, Mapping[str, float]],
    port: str,
    draw: int,
) -> int:
    # draw is an integer in [0, 2^53).
    unit = draw / float(1 << 53)
    cumulative = 0.0
    table = regime[port]
    for outcome in OUTCOMES:
        cumulative += float(table[str(outcome)])
        if unit < cumulative:
            return outcome
    return OUTCOMES[-1]


def oracle_commitment(
    protocol: Mapping[str, Any],
    regime_name: str,
    port: str,
    full_domain: bool = False,
) -> Commitment:
    probs = oracle_probabilities(protocol["regimes"][regime_name], port)
    return build_commitment(probs, full_domain=full_domain)


def _fraction_float(value: Fraction) -> float:
    return float(value)


def run_variant(
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    variant: str,
    ledger_path: Path | None = None,
    protocol_digest: str | None = None,
) -> dict[str, Any]:
    schedule = expand_schedule(protocol["scenarios"][scenario]["schedule"])
    regimes = regime_tables(protocol)
    model = make_model(variant)
    full_domain = variant == "full_domain"
    burn_in = int(protocol["burn_in"])
    rng = random.Random(seed * 100_003 + 17)
    writer = None
    if ledger_path is not None:
        assert protocol_digest is not None
        writer = LedgerWriter(ledger_path, protocol_digest, scenario, seed)

    records: list[dict[str, Any]] = []
    for step, regime_name in enumerate(schedule):
        if variant == "frozen200" and step == burn_in:
            model.freeze()
        port = forced_port(step)
        commitment = model.predict(port, full_domain=full_domain)
        model_digest = digest_payload(model.state_payload())
        draw = rng.getrandbits(53)
        if writer is not None:
            writer.append({
                "kind": "commit",
                "step": step,
                "port": port,
                "model_state_digest": model_digest,
                "probabilities": commitment.probability_pairs(),
                "cone": list(commitment.cone),
                "cone_mass": [
                    commitment.cone_mass.numerator,
                    commitment.cone_mass.denominator,
                ],
            })
        outcome = sample_outcome(regimes[regime_name], port, draw)
        binding = outcome in commitment.cone
        score = brier_score(commitment.probabilities, outcome)
        nll = negative_log_likelihood(commitment.probabilities, outcome)
        model.observe(port, outcome)
        post_digest = digest_payload(model.state_payload())
        if writer is not None:
            writer.append({
                "kind": "resolution",
                "step": step,
                "port": port,
                "draw_bits": draw,
                "outcome": outcome,
                "binding_success": binding,
                "brier": [
                    Fraction(score).limit_denominator(10_000_000).numerator,
                    Fraction(score).limit_denominator(10_000_000).denominator,
                ],
                "nll": nll,
                "post_model_state_digest": post_digest,
                "regime": regime_name,
            })
        records.append({
            "step": step,
            "port": port,
            "regime": regime_name,
            "outcome": outcome,
            "binding_success": binding,
            "cone": list(commitment.cone),
            "cone_mass": _fraction_float(commitment.cone_mass),
            "brier": _fraction_float(score),
            "nll": nll,
            "probabilities": {
                str(outcome_): _fraction_float(commitment.probabilities[outcome_])
                for outcome_ in OUTCOMES
            },
            "oracle_cone": list(
                oracle_commitment(protocol, regime_name, port).cone
            ),
        })

    final_hash = writer.close() if writer is not None else ""
    return {
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "records": records,
        "final_state": model.to_dict(),
        "final_hash": final_hash,
        "event_count": 0 if writer is None else writer.count,
    }


def _held_out_records(
    records: list[dict[str, Any]],
    burn_in: int,
) -> list[dict[str, Any]]:
    return [row for row in records if row["step"] >= burn_in]


def coverage(records: Iterable[dict[str, Any]]) -> float:
    rows = list(records)
    return mean(bool(row["binding_success"]) for row in rows) if rows else 0.0


def mean_cardinality(records: Iterable[dict[str, Any]]) -> float:
    rows = list(records)
    return mean(len(row["cone"]) for row in rows) if rows else 0.0


def set_calibration_gap(records: Iterable[dict[str, Any]]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    return abs(coverage(rows) - mean(float(row["cone_mass"]) for row in rows))


def classwise_ece(records: Iterable[dict[str, Any]], bins: int = 10) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    eces: list[float] = []
    for outcome in OUTCOMES:
        pairs = [
            (
                float(row["probabilities"][str(outcome)]),
                1.0 if row["outcome"] == outcome else 0.0,
            )
            for row in rows
        ]
        edges = [i / bins for i in range(bins + 1)]
        total = 0.0
        for index in range(bins):
            lo, hi = edges[index], edges[index + 1]
            bucket = [
                pair for pair in pairs
                if (pair[0] >= lo and pair[0] < hi)
                or (index == bins - 1 and pair[0] == hi)
            ]
            if not bucket:
                continue
            conf = mean(pair[0] for pair in bucket)
            acc = mean(pair[1] for pair in bucket)
            total += (len(bucket) / len(pairs)) * abs(acc - conf)
        eces.append(total)
    return max(eces)


def mean_brier(records: Iterable[dict[str, Any]]) -> float:
    rows = list(records)
    return mean(float(row["brier"]) for row in rows) if rows else 0.0


def detector_series(
    records: list[dict[str, Any]],
    name: str,
    window: int,
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    if name == "cone_miss":
        recent: deque[int] = deque(maxlen=window)
        for row in records:
            recent.append(0 if row["binding_success"] else 1)
            if len(recent) == window:
                values.append((row["step"], float(sum(recent))))
    elif name == "nll":
        recent_nll: deque[float] = deque(maxlen=window)
        for row in records:
            recent_nll.append(float(row["nll"]))
            if len(recent_nll) == window:
                values.append((row["step"], float(sum(recent_nll))))
    elif name == "raw_two_window":
        recent_outcomes: deque[int] = deque(maxlen=2 * window)
        for row in records:
            recent_outcomes.append(int(row["outcome"]))
            if len(recent_outcomes) == 2 * window:
                previous = list(recent_outcomes)[:window]
                latest = list(recent_outcomes)[window:]
                values.append((row["step"], total_variation(previous, latest)))
    else:
        raise ValueError(name)
    return values


def alarm_steps(
    series: list[tuple[int, float]],
    threshold: float,
    refractory: int,
) -> list[int]:
    alarms: list[int] = []
    next_allowed = -10**9
    for step, value in series:
        if value >= threshold and step >= next_allowed:
            alarms.append(step)
            next_allowed = step + refractory
    return alarms


def false_alarms_per_1000(
    alarms: list[int],
    start: int,
    end: int,
) -> float:
    count = sum(1 for step in alarms if start <= step < end)
    length = max(1, end - start)
    return 1000.0 * count / length


def choose_threshold(
    calibration_runs: list[dict[str, Any]],
    detector: str,
    window: int,
    grid: list[float],
    burn_in: int,
    refractory: int,
    budget: float,
) -> float:
    best = None
    best_key = None
    for threshold in grid:
        rates: list[float] = []
        for run in calibration_runs:
            series = detector_series(run["records"], detector, window)
            alarms = alarm_steps(series, float(threshold), refractory)
            rates.append(
                false_alarms_per_1000(alarms, burn_in, len(run["records"]))
            )
        rate = mean(rates) if rates else 999.0
        if rate <= budget:
            # Most sensitive = lowest threshold that still respects the budget.
            key = (float(threshold), rate)
            if best_key is None or key < best_key:
                best_key = key
                best = float(threshold)
    if best is None:
        best = float(max(grid))
    return best


def detection_delay(
    alarms: list[int],
    boundary: int,
    within: int,
) -> int | None:
    for step in alarms:
        if boundary <= step <= boundary + within:
            return step - boundary
    return None


def recovery_delay(
    records: list[dict[str, Any]],
    boundary: int,
    stable_uses: int,
) -> int | None:
    # After boundary, wait until each port matches oracle cone for stable_uses uses.
    needed = {port: stable_uses for port in PORTS}
    for row in records:
        if row["step"] < boundary:
            continue
        if row["cone"] == row["oracle_cone"]:
            needed[row["port"]] = max(0, needed[row["port"]] - 1)
        else:
            needed[row["port"]] = stable_uses
        if all(value == 0 for value in needed.values()):
            return row["step"] - boundary
    return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def hindsight_disagreement(
    protocol: Mapping[str, Any],
    run: dict[str, Any],
) -> float:
    # Reconstruct B-period binding labels from the final A2 model state.
    final_model = OutcomeModel.from_dict(run["final_state"])
    late_b = [
        row for row in run["records"]
        if row["regime"] == "B" and 900 <= row["step"] < 1000
    ]
    if not late_b:
        return 0.0
    disagreements = 0
    for row in late_b:
        commitment = final_model.predict(row["port"])
        reconstructed = row["outcome"] in commitment.cone
        if reconstructed != row["binding_success"]:
            disagreements += 1
    return disagreements / len(late_b)


def summarize_stationary(
    runs: list[dict[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    burn_in = int(protocol["burn_in"])
    by_port: dict[str, list[dict[str, Any]]] = {port: [] for port in PORTS}
    oracle_rows: list[dict[str, Any]] = []
    for run in runs:
        held = _held_out_records(run["records"], burn_in)
        for row in held:
            by_port[row["port"]].append(row)
        # Oracle Brier on identical outcomes using regime labels.
        for row in held:
            oracle = oracle_commitment(protocol, row["regime"], row["port"])
            oracle_rows.append({
                **row,
                "brier": _fraction_float(
                    brier_score(oracle.probabilities, row["outcome"])
                ),
            })
    summary = {"by_port": {}, "aggregate": {}}
    all_rows: list[dict[str, Any]] = []
    for port in PORTS:
        rows = by_port[port]
        all_rows.extend(rows)
        summary["by_port"][port] = {
            "coverage": coverage(rows),
            "mean_cardinality": mean_cardinality(rows),
            "set_calibration_gap": set_calibration_gap(rows),
            "ece": classwise_ece(rows),
            "brier": mean_brier(rows),
        }
    learner_brier = mean_brier(all_rows)
    oracle_brier = mean_brier(oracle_rows)
    summary["aggregate"] = {
        "coverage": coverage(all_rows),
        "mean_cardinality": mean_cardinality(all_rows),
        "set_calibration_gap": set_calibration_gap(all_rows),
        "ece": classwise_ece(all_rows),
        "brier": learner_brier,
        "oracle_brier": oracle_brier,
        "brier_regret": learner_brier - oracle_brier,
    }
    return summary


def summarize_detectors(
    runs: list[dict[str, Any]],
    stationary_runs: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    thresholds: Mapping[str, float],
    scenario: str,
) -> dict[str, Any]:
    burn_in = int(protocol["burn_in"])
    refractory = int(protocol["refractory_period"])
    detectors = protocol["detectors"]
    boundaries = [500, 1000]
    within_rev = int(protocol["evidence"]["drift"]["reversal_detection_within"])
    within_in = int(protocol["evidence"]["drift"]["inside_cone_detection_within"])
    result: dict[str, Any] = {}
    for name, config in detectors.items():
        window = int(config["window"])
        threshold = float(thresholds[name])
        fa_rates = []
        for run in stationary_runs:
            series = detector_series(run["records"], name, window)
            alarms = alarm_steps(series, threshold, refractory)
            fa_rates.append(false_alarms_per_1000(alarms, burn_in, len(run["records"])))
        detections = {str(boundary): [] for boundary in boundaries}
        delays = {str(boundary): [] for boundary in boundaries}
        within = within_rev if scenario == "hidden_ABA" else within_in
        for run in runs:
            series = detector_series(run["records"], name, window)
            alarms = alarm_steps(series, threshold, refractory)
            for boundary in boundaries:
                delay = detection_delay(alarms, boundary, within)
                detections[str(boundary)].append(delay is not None)
                if delay is not None:
                    delays[str(boundary)].append(delay)
        result[name] = {
            "threshold": threshold,
            "stationary_false_alarms_per_1000": mean(fa_rates) if fa_rates else None,
            "detection_rate": {
                key: mean(values) if values else 0.0
                for key, values in detections.items()
            },
            "mean_delay": {
                key: (mean(values) if values else None)
                for key, values in delays.items()
            },
        }
    return result


def failure_lift(
    runs: list[dict[str, Any]],
    boundary: int,
) -> float:
    lifts: list[float] = []
    for run in runs:
        before = [
            row for row in run["records"]
            if boundary - 100 <= row["step"] < boundary
        ]
        after = [
            row for row in run["records"]
            if boundary <= row["step"] < boundary + 10
        ]
        if not before or not after:
            continue
        before_rate = 1.0 - coverage(before)
        after_rate = 1.0 - coverage(after)
        lifts.append(after_rate - before_rate)
    return mean(lifts) if lifts else 0.0


def summarize_recovery(
    runs: list[dict[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    stable = int(protocol["evidence"]["recovery"]["stable_uses_per_port"])
    delays_b = []
    delays_a = []
    final_metrics = []
    for run in runs:
        delay_b = recovery_delay(run["records"], 500, stable)
        delay_a = recovery_delay(run["records"], 1000, stable)
        if delay_b is not None:
            delays_b.append(delay_b)
        if delay_a is not None:
            delays_a.append(delay_a)
        a1 = [row for row in run["records"] if 300 <= row["step"] < 500]
        a2 = [row for row in run["records"] if 1300 <= row["step"] < 1500]
        final_metrics.append({
            "coverage_diff": abs(coverage(a2) - coverage(a1)),
            "brier_diff": abs(mean_brier(a2) - mean_brier(a1)),
            "cardinality_diff": abs(mean_cardinality(a2) - mean_cardinality(a1)),
        })
    return {
        "b_adaptation": {
            "median": median(delays_b) if delays_b else None,
            "p90": percentile([float(v) for v in delays_b], 0.9) if delays_b else None,
            "n": len(delays_b),
        },
        "a_return": {
            "median": median(delays_a) if delays_a else None,
            "p90": percentile([float(v) for v in delays_a], 0.9) if delays_a else None,
            "n": len(delays_a),
        },
        "a_return_extra_median": (
            None
            if not delays_a or not delays_b
            else median(delays_a) - median(delays_b)
        ),
        "final200": {
            "coverage_diff": mean(item["coverage_diff"] for item in final_metrics),
            "brier_diff": mean(item["brier_diff"] for item in final_metrics),
            "cardinality_diff": mean(
                item["cardinality_diff"] for item in final_metrics
            ),
        },
    }


def evaluate_evidence(
    protocol: Mapping[str, Any],
    stationary: dict[str, Any],
    full_domain_card: float,
    aba_detectors: dict[str, Any],
    ada_detectors: dict[str, Any],
    lift: float,
    recovery: dict[str, Any],
    hindsight: float,
    replay_ok: bool,
) -> dict[str, Any]:
    st = protocol["evidence"]["stationary"]
    dr = protocol["evidence"]["drift"]
    rc = protocol["evidence"]["recovery"]
    ac = protocol["evidence"]["accountability"]
    agg = stationary["aggregate"]

    stationary_ok = (
        agg["coverage"] >= st["coverage_min"]
        and agg["mean_cardinality"] <= st["mean_cardinality_max"]
        and agg["set_calibration_gap"] <= st["set_calibration_gap_max"]
        and agg["ece"] <= st["ece_max"]
        and agg["brier_regret"] <= st["brier_regret_max"]
        and full_domain_card == st["full_domain_cardinality"]
    )
    cone = aba_detectors["cone_miss"]
    reversal_ok = (
        cone["stationary_false_alarms_per_1000"] <= dr["false_alarms_per_1000_max"]
        and cone["detection_rate"]["500"] >= dr["reversal_detection_rate_min"]
        and cone["detection_rate"]["1000"] >= dr["reversal_detection_rate_min"]
        and lift >= dr["first10_failure_lift_min"]
    )
    cone_inside = ada_detectors["cone_miss"]["detection_rate"]
    conventional = max(
        ada_detectors["nll"]["detection_rate"]["500"],
        ada_detectors["nll"]["detection_rate"]["1000"],
        ada_detectors["raw_two_window"]["detection_rate"]["500"],
        ada_detectors["raw_two_window"]["detection_rate"]["1000"],
    )
    # Limitation if cone_miss rarely detects while a conventional detector does.
    inside_limitation = (
        cone_inside["500"] <= dr["cone_miss_inside_detection_max"]
        and cone_inside["1000"] <= dr["cone_miss_inside_detection_max"]
        and conventional >= dr["conventional_inside_detection_min"]
    )
    recovery_ok = (
        recovery["b_adaptation"]["median"] is not None
        and recovery["a_return"]["median"] is not None
        and recovery["b_adaptation"]["median"] <= rc["median_delay_max"]
        and recovery["a_return"]["median"] <= rc["median_delay_max"]
        and recovery["b_adaptation"]["p90"] <= rc["p90_delay_max"]
        and recovery["a_return"]["p90"] <= rc["p90_delay_max"]
        and recovery["a_return_extra_median"] <= rc["a_return_extra_median_max"]
        and recovery["final200"]["coverage_diff"] <= rc["final200_coverage_diff_max"]
        and recovery["final200"]["brier_diff"] <= rc["final200_brier_diff_max"]
        and recovery["final200"]["cardinality_diff"]
        <= rc["final200_cardinality_diff_max"]
    )
    accountability_ok = (
        replay_ok and hindsight >= ac["hindsight_disagreement_min"]
    )
    return {
        "stationary_forecast_quality": {
            "met": stationary_ok,
            "details": agg,
            "full_domain_cardinality": full_domain_card,
        },
        "reversal_drift_signal": {
            "met": reversal_ok,
            "details": {
                "cone_miss": cone,
                "first10_failure_lift": lift,
            },
        },
        "inside_cone_blindness": {
            "met": inside_limitation,
            "limitation": True,
            "details": {
                "cone_miss_detection": cone_inside,
                "conventional_best_detection": conventional,
                "detectors": ada_detectors,
            },
            "criterion": (
                "cone_miss detects <=20% of ADA boundaries within 100 steps while "
                "NLL or raw detector detects >=90%"
            ),
        },
        "adaptation_recovery": {
            "met": recovery_ok,
            "details": recovery,
        },
        "accountability": {
            "met": accountability_ok,
            "replay_ok": replay_ok,
            "hindsight_disagreement": hindsight,
        },
    }


def decide(evidence: Mapping[str, Any]) -> dict[str, Any]:
    keep_summaries = (
        evidence["stationary_forecast_quality"]["met"]
        and evidence["accountability"]["met"]
    )
    keep_drift = evidence["reversal_drift_signal"]["met"]
    scope_inside = evidence["inside_cone_blindness"]["met"]
    if not evidence["accountability"]["replay_ok"]:
        verdict = "reject_accountability_implementation"
    elif keep_summaries and keep_drift and scope_inside:
        verdict = "keep_cones_as_audited_summaries_with_scoped_drift_signal"
    elif keep_summaries and keep_drift:
        verdict = "keep_cones_as_audited_summaries_and_reversal_signal"
    elif keep_summaries:
        verdict = "keep_cones_as_audited_summaries_only"
    else:
        verdict = "do_not_keep_cone_accountability_claims"
    return {
        "verdict": verdict,
        "keep_summaries": keep_summaries,
        "keep_reversal_signal": keep_drift,
        "scope_binding_failures_to_out_of_cone_change": scope_inside,
    }


def run_experiment(output: str | Path) -> dict[str, Any]:
    from . import replay as replay_mod

    protocol = load_protocol()
    digest = protocol_hash(protocol)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "protocol.snapshot.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cal_lo, cal_hi = protocol["seeds"]["calibration"]
    eval_lo, eval_hi = protocol["seeds"]["evaluation"]
    calibration_seeds = list(range(int(cal_lo), int(cal_hi) + 1))
    evaluation_seeds = list(range(int(eval_lo), int(eval_hi) + 1))
    all_seeds = list(range(int(protocol["seeds"]["all"])))

    primary_runs: dict[str, dict[int, dict[str, Any]]] = {
        scenario: {} for scenario in protocol["scenarios"]
    }
    variant_summaries: dict[str, Any] = {}
    final_states: dict[str, Any] = {}
    run_final_hashes: dict[str, str] = {}
    event_counts: dict[str, int] = {}

    # Primary window64 ledgers for all seeds/scenarios.
    for scenario in protocol["scenarios"]:
        for seed in all_seeds:
            ledger_path = output_path / "runs" / scenario / f"{seed}.jsonl"
            result = run_variant(
                protocol,
                scenario,
                seed,
                "window64",
                ledger_path=ledger_path,
                protocol_digest=digest,
            )
            primary_runs[scenario][seed] = result
            key = f"{scenario}:{seed}"
            final_states[key] = result["final_state"]
            run_final_hashes[key] = result["final_hash"]
            event_counts[key] = result["event_count"]

    # Variant traces on evaluation seeds only, summary-level.
    for variant in protocol["variants"]:
        if variant == "window64":
            continue
        variant_summaries[variant] = {}
        for scenario in protocol["scenarios"]:
            runs = [
                run_variant(protocol, scenario, seed, variant)
                for seed in evaluation_seeds
            ]
            variant_summaries[variant][scenario] = summarize_stationary(
                runs, protocol
            )

    stationary_cal = [primary_runs["stationary_A"][seed] for seed in calibration_seeds]
    thresholds = {}
    for name, config in protocol["detectors"].items():
        thresholds[name] = choose_threshold(
            stationary_cal,
            name,
            int(config["window"]),
            list(config["grid"]),
            int(protocol["burn_in"]),
            int(protocol["refractory_period"]),
            float(protocol["detector_false_alarm_budget_per_1000"]),
        )

    stationary_eval = [primary_runs["stationary_A"][seed] for seed in evaluation_seeds]
    aba_eval = [primary_runs["hidden_ABA"][seed] for seed in evaluation_seeds]
    ada_eval = [primary_runs["hidden_ADA"][seed] for seed in evaluation_seeds]
    full_domain_eval = [
        run_variant(protocol, "stationary_A", seed, "full_domain")
        for seed in evaluation_seeds
    ]
    full_domain_card = mean(
        mean_cardinality(_held_out_records(run["records"], int(protocol["burn_in"])))
        for run in full_domain_eval
    )

    stationary_summary = summarize_stationary(stationary_eval, protocol)
    aba_detectors = summarize_detectors(
        aba_eval, stationary_eval, protocol, thresholds, "hidden_ABA"
    )
    ada_detectors = summarize_detectors(
        ada_eval, stationary_eval, protocol, thresholds, "hidden_ADA"
    )
    lift = failure_lift(aba_eval, 500)
    recovery = summarize_recovery(aba_eval, protocol)
    hindsight = mean(hindsight_disagreement(protocol, run) for run in aba_eval)

    replay_ok = True
    for scenario in ("stationary_A", "hidden_ABA"):
        for seed in evaluation_seeds[:5]:
            path = output_path / "runs" / scenario / f"{seed}.jsonl"
            ok, _ = replay_mod.replay_and_verify(
                path, protocol, digest, scenario, seed
            )
            replay_ok = replay_ok and ok

    evidence = evaluate_evidence(
        protocol,
        stationary_summary,
        full_domain_card,
        aba_detectors,
        ada_detectors,
        lift,
        recovery,
        hindsight,
        replay_ok,
    )
    decision = decide(evidence)
    summary = {
        "configuration": {
            "protocol_hash": digest,
            "calibration_seeds": calibration_seeds,
            "evaluation_seeds": evaluation_seeds,
            "thresholds": thresholds,
        },
        "stationary": stationary_summary,
        "variants": variant_summaries,
        "detectors": {
            "hidden_ABA": aba_detectors,
            "hidden_ADA": ada_detectors,
            "first10_failure_lift_at_500": lift,
        },
        "recovery": recovery,
        "hindsight_disagreement": hindsight,
        "evidence_targets": evidence,
        "decision": decision,
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_path / "summary.json").write_text(summary_text, encoding="utf-8")
    (output_path / "final_states.json").write_text(
        json.dumps(final_states, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal = build_seal(
        protocol,
        digest,
        code_file_hashes(PACKAGE_ROOT, CODE_FILES),
        run_final_hashes,
        event_counts,
        {
            key: digest_payload(state)
            for key, state in final_states.items()
        },
        digest_payload(json.loads(summary_text)),
    )
    write_seal(output_path / "seal.json", seal)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/stage1e")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    limitation = {"inside_cone_blindness"}
    for name, result in summary["evidence_targets"].items():
        if name in limitation:
            label = "LIMITATION" if result["met"] else "ABSENT"
        else:
            label = "PASS" if result["met"] else "FAIL"
        print(f"{label}  {name}")
    print(f"DECISION  {summary['decision']['verdict']}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
