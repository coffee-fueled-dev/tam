"""Reporting helpers for atlas benchmark runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from atlas.benchmark.runner import BenchmarkResult, run_benchmark_suite


def result_to_jsonable(result: BenchmarkResult) -> dict:
    return asdict(result)


def format_result_line(result: BenchmarkResult) -> str:
    summary = result.summary
    return (
        f"{summary.family:<10} {summary.config_name:<20} "
        f"contr={summary.contradiction.mean_contradiction:.3f} "
        f"support={summary.contradiction.support_rate:.3f} "
        f"spawn={summary.spawn.spawn_rate:.3f} "
        f"false_spawn={summary.spawn.false_spawn_rate:.3f} "
        f"reuse={summary.spawn.reuse_rate:.3f} "
        f"topk={summary.retrieval.top_k_retrieval_hit:.3f} "
        f"purity={summary.purity.chart_purity:.3f} "
        f"frag={summary.purity.regime_fragmentation:.3f} "
        f"overload={summary.purity.chart_overloading:.3f} "
        f"ambig={summary.decision.ambiguous_rate:.3f} "
        f"trust={summary.decision.trusted_calibration_update_rate:.3f} "
        f"calib={summary.calibration.calibrated_decision_rate:.3f} "
        f"samp={summary.calibration.mean_sample_count:.2f} "
        f"gctr={summary.geometry.mean_center_shift:.3f} "
        f"grad={summary.geometry.mean_radius_shift:.3f} "
        f"delay_spawn={summary.delay.delayed_false_spawn_rate:.3f} "
        f"collapse={int(summary.purity.single_chart_collapse)} "
        f"charts={summary.spawn.chart_count}"
    )


def dump_jsonl(output_path: str) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark_suite()
    lines = [json.dumps(result_to_jsonable(result), sort_keys=True) for result in results]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} JSONL rows to {target}")


def print_report() -> None:
    for result in run_benchmark_suite():
        print(format_result_line(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSONL output path.")
    args = parser.parse_args()
    if args.output:
        dump_jsonl(args.output)
    else:
        print_report()


if __name__ == "__main__":
    main()
