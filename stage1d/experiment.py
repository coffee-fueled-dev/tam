"""Stage 1D exact and finite-sample asymmetric-risk evaluation."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

from .model import (
    CATASTROPHE_THRESHOLD,
    CONTROLLERS,
    LOSS_DOMAIN,
    PORTS,
    Commitment,
    LossModel,
    build_commitment,
    exact_probabilities,
    select_port,
)

REGIMES = {
    "below_cutoff": {"p": 0.08, "catastrophe_loss": 18.0},
    "above_cutoff": {"p": 0.12, "catastrophe_loss": 12.0},
    "boundary": {"p": 0.10, "catastrophe_loss": 14.4},
}
PRIMARY_REGIMES = ("below_cutoff", "above_cutoff")
TRAIN_BUDGETS = (100, 500, 2000)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(100))
    evaluation_draws: int = 10_000
    train_budgets: tuple[int, ...] = TRAIN_BUDGETS
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 20260907


def sample_loss(
    port: str,
    catastrophe_probability: float,
    catastrophe_loss: float,
    uniform: float,
) -> float:
    if port == "safe":
        return 2.0
    return catastrophe_loss if uniform < catastrophe_probability else 0.0


def _commitment_payload(commitment: Commitment) -> dict[str, object]:
    return {
        "probabilities": {
            str(loss): commitment.probabilities[loss] for loss in LOSS_DOMAIN
        },
        "cone": list(commitment.cone),
    }


def _empirical_cvar90(losses: list[float]) -> float:
    if not losses:
        return 0.0
    ordered = sorted(losses, reverse=True)
    count = max(1, int(round(0.1 * len(ordered))))
    return mean(ordered[:count])


def _evaluate_controller(
    chosen_port: str,
    commitment: Commitment,
    potential: Mapping[str, list[float]],
) -> dict[str, float]:
    losses = potential[chosen_port]
    binding_failures = [
        loss not in commitment.cone for loss in losses
    ]
    catastrophes = [loss > CATASTROPHE_THRESHOLD for loss in losses]
    failure_and_catastrophe = [
        fail and cat for fail, cat in zip(binding_failures, catastrophes)
    ]
    return {
        "mean_loss": mean(losses),
        "catastrophe_rate": mean(catastrophes),
        "empirical_cvar90": _empirical_cvar90(losses),
        "cone_coverage": 1.0 - mean(binding_failures),
        "cone_cardinality": float(len(commitment.cone)),
        "binding_failure_rate": mean(binding_failures),
        "failure_catastrophe_overlap": mean(failure_and_catastrophe),
        "rare_tail_selection": 1.0 if chosen_port == "rare_tail" else 0.0,
    }


def _potential_outcomes(
    regime: str,
    uniforms: list[float],
) -> dict[str, list[float]]:
    params = REGIMES[regime]
    return {
        port: [
            sample_loss(port, params["p"], params["catastrophe_loss"], uniform)
            for uniform in uniforms
        ]
        for port in PORTS
    }


def _regret(
    metrics: dict[str, float],
    potential: Mapping[str, list[float]],
) -> dict[str, float]:
    mean_by_port = {port: mean(potential[port]) for port in PORTS}
    cat_by_port = {
        port: mean(loss > CATASTROPHE_THRESHOLD for loss in potential[port])
        for port in PORTS
    }
    ev_port = min(PORTS, key=lambda port: (mean_by_port[port], PORTS.index(port)))
    cvar_port = min(
        PORTS,
        key=lambda port: (_empirical_cvar90(potential[port]), PORTS.index(port)),
    )
    return {
        "regret_to_ev_mean": metrics["mean_loss"] - mean_by_port[ev_port],
        "regret_to_cvar_catastrophe": (
            metrics["catastrophe_rate"] - cat_by_port[cvar_port]
        ),
    }


def _exact_port_probabilities(regime: str) -> dict[str, dict[float, float]]:
    params = REGIMES[regime]
    return {
        port: exact_probabilities(port, params["p"], params["catastrophe_loss"])
        for port in PORTS
    }


def _train_model(
    regime: str,
    budget: int,
    seed: int,
) -> LossModel:
    params = REGIMES[regime]
    model = LossModel(history_size=max(budget, 1))
    rng = random.Random(seed * 10_000 + 21 + budget)
    for port in PORTS:
        for _ in range(budget):
            uniform = rng.random()
            loss = sample_loss(port, params["p"], params["catastrophe_loss"], uniform)
            model.observe(port, loss)
    return model


def _learned_probabilities(model: LossModel) -> dict[str, dict[float, float]]:
    return {port: model.probabilities(port) for port in PORTS}


def _run_condition(
    phase: str,
    regime: str,
    seed: int,
    uniforms: list[float],
    port_probabilities: Mapping[str, Mapping[float, float]],
    budget: int | None = None,
    port_support: Mapping[str, tuple[float, ...]] | None = None,
) -> list[dict[str, object]]:
    potential = _potential_outcomes(regime, uniforms)
    records: list[dict[str, object]] = []
    for controller in CONTROLLERS:
        chosen, commitment = select_port(
            port_probabilities,
            controller,
            port_support=port_support,
        )
        metrics = _evaluate_controller(chosen, commitment, potential)
        metrics.update(_regret(metrics, potential))
        record = {
            "phase": phase,
            "regime": regime,
            "seed": seed,
            "controller": controller,
            "chosen_port": chosen,
            "commitment": _commitment_payload(commitment),
            **metrics,
        }
        if budget is not None:
            record["budget"] = budget
        records.append(record)
    return records


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "low": None, "high": None}
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(draw))
    ordered = sorted(means)
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return {
        "mean": mean(values),
        "low": ordered[low_index],
        "high": ordered[high_index],
    }


def _controller_summary(
    rows: list[dict[str, object]],
    config: Config,
) -> dict[str, object]:
    metric_names = (
        "rare_tail_selection",
        "mean_loss",
        "catastrophe_rate",
        "empirical_cvar90",
        "cone_coverage",
        "cone_cardinality",
        "binding_failure_rate",
        "failure_catastrophe_overlap",
        "regret_to_ev_mean",
        "regret_to_cvar_catastrophe",
    )
    summary: dict[str, object] = {}
    for controller in CONTROLLERS:
        subset = [row for row in rows if row["controller"] == controller]
        summary[controller] = {
            name: _bootstrap_mean_ci(
                [float(row[name]) for row in subset],
                config.bootstrap_samples,
                config.bootstrap_seed
                + CONTROLLERS.index(controller) * 17
                + metric_names.index(name),
            )
            for name in metric_names
        }
        summary[controller]["selection_mode"] = (
            "rare_tail"
            if _average(float(row["rare_tail_selection"]) for row in subset) == 1.0
            else (
                "safe"
                if _average(float(row["rare_tail_selection"]) for row in subset) == 0.0
                else "mixed"
            )
        )
    return summary


def _paired_difference(
    rows: list[dict[str, object]],
    left: str,
    right: str,
    metric: str,
    config: Config,
) -> dict[str, float | None]:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["controller"]] = float(row[metric])
    diffs = [
        values[left] - values[right]
        for values in by_seed.values()
        if left in values and right in values
    ]
    return _bootstrap_mean_ci(
        diffs,
        config.bootstrap_samples,
        config.bootstrap_seed + 1000 + CONTROLLERS.index(left) * 31,
    )


def _exact_choice(regime: str, controller: str) -> str:
    chosen, _ = select_port(_exact_port_probabilities(regime), controller)
    return chosen


def _finite_tam_classification_rate(
    rows: list[dict[str, object]],
    regime: str,
    budget: int,
) -> float:
    expected = _exact_choice(regime, "tam_90")
    subset = [
        row for row in rows
        if row["phase"] == "finite_sample"
        and row["regime"] == regime
        and row["budget"] == budget
        and row["controller"] == "tam_90"
    ]
    if not subset:
        return 0.0
    return mean(row["chosen_port"] == expected for row in subset)


def _evidence(rows: list[dict[str, object]], config: Config) -> dict[str, object]:
    exact = [row for row in rows if row["phase"] == "exact"]

    def exact_metric(regime: str, controller: str, metric: str) -> float:
        values = [
            float(row[metric]) for row in exact
            if row["regime"] == regime and row["controller"] == controller
        ]
        return mean(values)

    sanity = {
        "ev_below": _exact_choice("below_cutoff", "expected_value") == "rare_tail",
        "ev_above": _exact_choice("above_cutoff", "expected_value") == "rare_tail",
        "cvar_below": _exact_choice("below_cutoff", "cvar_90") == "safe",
        "cvar_above": _exact_choice("above_cutoff", "cvar_90") == "safe",
        "worst_below": _exact_choice("below_cutoff", "worst_case") == "safe",
        "worst_above": _exact_choice("above_cutoff", "worst_case") == "safe",
        "tam_below": _exact_choice("below_cutoff", "tam_90") == "rare_tail",
        "tam_above": _exact_choice("above_cutoff", "tam_90") == "safe",
    }

    risk_tradeoff = {}
    for regime in PRIMARY_REGIMES:
        cat_reduction = (
            exact_metric(regime, "expected_value", "catastrophe_rate")
            - exact_metric(regime, "cvar_90", "catastrophe_rate")
        )
        mean_cost = (
            exact_metric(regime, "cvar_90", "mean_loss")
            - exact_metric(regime, "expected_value", "mean_loss")
        )
        risk_tradeoff[regime] = {
            "catastrophe_reduction": cat_reduction,
            "mean_loss_cost": mean_cost,
            "met": cat_reduction >= 0.07 and mean_cost <= 0.60,
        }

    above_gap = abs(
        exact_metric("above_cutoff", "tam_90", "catastrophe_rate")
        - exact_metric("above_cutoff", "cvar_90", "catastrophe_rate")
    )
    below_selection = exact_metric("below_cutoff", "tam_90", "rare_tail_selection")
    below_gap = (
        exact_metric("below_cutoff", "tam_90", "catastrophe_rate")
        - exact_metric("below_cutoff", "cvar_90", "catastrophe_rate")
    )

    stability = {}
    for budget in (500, 2000):
        rates = {
            regime: _finite_tam_classification_rate(rows, regime, budget)
            for regime in PRIMARY_REGIMES
        }
        stability[str(budget)] = {
            "rates": rates,
            "met": all(rate >= 0.90 for rate in rates.values()),
        }

    # Pareto dominance of TAM over CVaR on primary exact regimes.
    pareto = False
    for regime in PRIMARY_REGIMES:
        tam_cat = exact_metric(regime, "tam_90", "catastrophe_rate")
        cvar_cat = exact_metric(regime, "cvar_90", "catastrophe_rate")
        tam_mean = exact_metric(regime, "tam_90", "mean_loss")
        cvar_mean = exact_metric(regime, "cvar_90", "mean_loss")
        if (cvar_cat - tam_cat) >= 0.05 and abs(tam_mean - cvar_mean) <= 0.05:
            pareto = True
        if (cvar_mean - tam_mean) >= 0.10 and abs(tam_cat - cvar_cat) <= 0.005:
            pareto = True

    return {
        "exact_selector_sanity": {
            "met": all(sanity.values()),
            "details": sanity,
            "criterion": (
                "EV rare_tail both; CVaR/worst safe both; "
                "TAM rare_tail below and safe above"
            ),
        },
        "conventional_risk_tradeoff": {
            "met": all(item["met"] for item in risk_tradeoff.values()),
            "details": risk_tradeoff,
            "criterion": (
                "CVaR catastrophe rate <= EV - 0.07 and mean-loss cost <= 0.60"
            ),
        },
        "above_cutoff_protection": {
            "met": above_gap <= 0.005,
            "catastrophe_rate_gap": above_gap,
            "criterion": "TAM catastrophe rate within 0.5 pp of CVaR at p=0.12",
        },
        "below_cutoff_tail_blindness": {
            "met": below_selection >= 0.95 and below_gap >= 0.07,
            "rare_tail_selection": below_selection,
            "catastrophe_rate_gap_vs_cvar": below_gap,
            "criterion": (
                "TAM selects rare_tail >=95% and exceeds CVaR catastrophe by >=7 pp; "
                "meeting this is a limitation"
            ),
        },
        "finite_sample_stability": {
            "met": all(item["met"] for item in stability.values()),
            "details": stability,
            "criterion": (
                "At budgets 500 and 2000, >=90% seeds match exact TAM classification"
            ),
        },
        "tam_pareto_over_cvar": {
            "met": pareto,
            "criterion": (
                "TAM has >=5 pp fewer catastrophes at mean loss within 0.05, "
                "or mean loss >=0.10 lower at catastrophe within 0.5 pp"
            ),
        },
        "paired_differences_exact": {
            regime: {
                "tam_minus_cvar_catastrophe": _paired_difference(
                    [row for row in exact if row["regime"] == regime],
                    "tam_90",
                    "cvar_90",
                    "catastrophe_rate",
                    config,
                ),
                "tam_minus_ev_mean_loss": _paired_difference(
                    [row for row in exact if row["regime"] == regime],
                    "tam_90",
                    "expected_value",
                    "mean_loss",
                    config,
                ),
            }
            for regime in PRIMARY_REGIMES
        },
    }


def _decision(evidence: dict[str, object]) -> dict[str, object]:
    blindness = evidence["below_cutoff_tail_blindness"]["met"]
    protection = evidence["above_cutoff_protection"]["met"]
    pareto = evidence["tam_pareto_over_cvar"]["met"]
    if blindness:
        if protection:
            verdict = "cones_as_reporting_only_need_explicit_risk_layer"
        else:
            verdict = "reject_90pct_cone_as_tail_safety_filter"
    elif pareto:
        verdict = "tam_selection_advantage_over_cvar"
    else:
        verdict = "no_tam_selection_advantage"
    return {
        "verdict": verdict,
        "tail_blindness_demonstrated": blindness,
        "above_cutoff_matches_cvar": protection,
        "pareto_over_cvar": pareto,
    }


def run_experiment(output: str | Path, config: Config = Config()) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    models: dict[str, object] = {}
    started = time.perf_counter()

    with (output_path / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for seed in config.seeds:
            rng = random.Random(seed * 10_000 + 7)
            uniforms = [rng.random() for _ in range(config.evaluation_draws)]

            for regime in REGIMES:
                exact_rows = _run_condition(
                    "exact",
                    regime,
                    seed,
                    uniforms,
                    _exact_port_probabilities(regime),
                )
                for row in exact_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows.extend(exact_rows)

                for budget in config.train_budgets:
                    model = _train_model(regime, budget, seed)
                    models[f"{seed}:{regime}:{budget}"] = model.to_dict()
                    support = {port: model.supported(port) for port in PORTS}
                    finite_rows = _run_condition(
                        "finite_sample",
                        regime,
                        seed,
                        uniforms,
                        _learned_probabilities(model),
                        budget=budget,
                        port_support=support,
                    )
                    for row in finite_rows:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.extend(finite_rows)

    aggregate = {
        "exact": {
            regime: _controller_summary(
                [row for row in rows if row["phase"] == "exact" and row["regime"] == regime],
                config,
            )
            for regime in REGIMES
        },
        "finite_sample": {
            str(budget): {
                regime: _controller_summary(
                    [
                        row for row in rows
                        if row["phase"] == "finite_sample"
                        and row["regime"] == regime
                        and row["budget"] == budget
                    ],
                    config,
                )
                for regime in REGIMES
            }
            for budget in config.train_budgets
        },
    }
    evidence = _evidence(rows, config)
    summary = {
        "configuration": {
            **asdict(config),
            "seeds": list(config.seeds),
            "train_budgets": list(config.train_budgets),
            "regimes": REGIMES,
            "ports": list(PORTS),
            "controllers": list(CONTROLLERS),
            "loss_domain": list(LOSS_DOMAIN),
            "catastrophe_threshold": CATASTROPHE_THRESHOLD,
            "commitment_mass": 0.9,
            "cvar_tail": 0.1,
            "purpose": "asymmetric_tail_risk_boundary",
        },
        "aggregate": aggregate,
        "evidence_targets": evidence,
        "decision": _decision(evidence),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "model.json").write_text(
        json.dumps(models, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/stage1d")
    args = parser.parse_args()
    summary = run_experiment(args.output)
    limitation_targets = {"below_cutoff_tail_blindness"}
    for name, result in summary["evidence_targets"].items():
        if "met" not in result:
            continue
        if name in limitation_targets:
            label = "LIMITATION" if result["met"] else "ABSENT"
        else:
            label = "PASS" if result["met"] else "FAIL"
        print(f"{label}  {name}")
    print(f"DECISION  {summary['decision']['verdict']}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
