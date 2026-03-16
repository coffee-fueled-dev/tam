"""Run the v4 benchmark suite and dump results as JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from v4.benchmark.runner import run_benchmark_suite
from v4.benchmark.scorecard import score_benchmark_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="tmp/v4-benchmark.jsonl",
        help="Path to the JSONL dump file.",
    )
    parser.add_argument(
        "--include-scorecard",
        action="store_true",
        help="Append scorecard rows after raw benchmark results.",
    )
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_benchmark_suite()
    lines = [
        json.dumps(
            {
                "type": "benchmark_result",
                "config_name": result.config.name,
                "variant_name": result.variant.name,
                "result": to_jsonable(result),
            },
            sort_keys=True,
        )
        for result in results
    ]

    if args.include_scorecard:
        for row in score_benchmark_results(results):
            lines.append(
                json.dumps(
                    {
                        "type": "scorecard_row",
                        "benchmark_family": row.benchmark_family,
                        "row": to_jsonable(row),
                    },
                    sort_keys=True,
                )
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} JSONL rows to {output_path}")


if __name__ == "__main__":
    main()
