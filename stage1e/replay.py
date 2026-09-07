"""Independent replay and verification for Stage 1E ledgers."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from .ledger import digest_payload, read_ledger, verify_ledger
from .model import OUTCOMES, OutcomeModel, brier_score, build_commitment


def probabilities_from_pairs(
    pairs: Mapping[str, list[int]],
) -> dict[int, Fraction]:
    return {
        int(outcome): Fraction(int(values[0]), int(values[1]))
        for outcome, values in pairs.items()
    }


def replay_and_verify(
    ledger_path: Path,
    protocol: Mapping[str, Any],
    protocol_digest: str,
    scenario: str,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    ok, message = verify_ledger(ledger_path, protocol_digest, scenario, seed)
    if not ok:
        return False, {"error": message}

    rows = read_ledger(ledger_path)
    model = OutcomeModel(history_size=int(protocol["window_size"]))
    burn_in = int(protocol["burn_in"])
    reconstructed: list[dict[str, Any]] = []

    index = 0
    while index < len(rows):
        commit = rows[index]
        if commit.get("kind") != "commit":
            return False, {"error": f"expected commit at {index}"}
        if index + 1 >= len(rows) or rows[index + 1].get("kind") != "resolution":
            return False, {"error": f"missing resolution after commit {index}"}
        resolution = rows[index + 1]
        if commit["step"] != resolution["step"] or commit["port"] != resolution["port"]:
            return False, {"error": f"commit/resolution mismatch at {index}"}

        if digest_payload(model.state_payload()) != commit["model_state_digest"]:
            return False, {"error": f"pre-state mismatch at step {commit['step']}"}

        commitment = model.predict(commit["port"])
        expected_pairs = commitment.probability_pairs()
        if expected_pairs != commit["probabilities"]:
            return False, {"error": f"probability mismatch at step {commit['step']}"}
        if list(commitment.cone) != commit["cone"]:
            return False, {"error": f"cone mismatch at step {commit['step']}"}

        outcome = int(resolution["outcome"])
        binding = outcome in commitment.cone
        if binding != bool(resolution["binding_success"]):
            return False, {"error": f"binding mismatch at step {commit['step']}"}

        score = brier_score(commitment.probabilities, outcome)
        model.observe(commit["port"], outcome)
        if digest_payload(model.state_payload()) != resolution["post_model_state_digest"]:
            return False, {"error": f"post-state mismatch at step {commit['step']}"}

        reconstructed.append({
            "step": commit["step"],
            "port": commit["port"],
            "outcome": outcome,
            "binding_success": binding,
            "cone": list(commitment.cone),
            "brier": float(score),
            "held_out": commit["step"] >= burn_in,
        })
        index += 2

    return True, {
        "final_state": model.to_dict(),
        "final_hash": message,
        "records": reconstructed,
        "event_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument(
        "--protocol",
        default=str(Path(__file__).with_name("protocol.json")),
    )
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    from .ledger import protocol_hash

    digest = protocol_hash(protocol)
    path = Path(args.ledger)
    # Infer scenario/seed from path .../runs/<scenario>/<seed>.jsonl
    seed = int(path.stem)
    scenario = path.parent.name
    ok, detail = replay_and_verify(path, protocol, digest, scenario, seed)
    if not ok:
        raise SystemExit(f"REPLAY FAIL: {detail}")
    print(f"REPLAY OK  events={detail['event_count']} final_hash={detail['final_hash']}")


if __name__ == "__main__":
    main()
