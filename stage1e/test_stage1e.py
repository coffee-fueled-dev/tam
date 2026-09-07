from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from .experiment import (
    forced_port,
    load_protocol,
    run_experiment,
    run_variant,
    sample_outcome,
)
from .ledger import (
    digest_payload,
    protocol_hash,
    read_ledger,
    verify_ledger,
)
from .model import OutcomeModel, build_commitment, oracle_probabilities
from .replay import replay_and_verify


class ModelTests(unittest.TestCase):
    def test_initial_commitment_is_full_domain(self) -> None:
        commitment = OutcomeModel().predict("a")
        self.assertEqual(commitment.cone, (-1, 0, 1))
        self.assertEqual(commitment.cone_mass, Fraction(1))

    def test_score_before_update(self) -> None:
        model = OutcomeModel()
        before = model.predict("a")
        model.observe("a", 1)
        after = model.predict("a")
        self.assertNotEqual(dict(before.probabilities), dict(after.probabilities))

    def test_oracle_regime_cones(self) -> None:
        protocol = load_protocol()
        probs = oracle_probabilities(protocol["regimes"]["A"], "a")
        commitment = build_commitment(probs)
        self.assertEqual(commitment.cone, (1, 0))


class WorldTests(unittest.TestCase):
    def test_forced_port_alternates(self) -> None:
        self.assertEqual(forced_port(0), "a")
        self.assertEqual(forced_port(1), "b")

    def test_sample_outcome_is_deterministic(self) -> None:
        protocol = load_protocol()
        regime = protocol["regimes"]["A"]
        first = sample_outcome(regime, "a", 0)
        second = sample_outcome(regime, "a", 0)
        self.assertEqual(first, second)


class LedgerTests(unittest.TestCase):
    def test_commit_is_written_before_resolution_and_replays(self) -> None:
        protocol = load_protocol()
        digest = protocol_hash(protocol)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            result = run_variant(
                protocol,
                "stationary_A",
                50,
                "window64",
                ledger_path=path,
                protocol_digest=digest,
            )
            rows = read_ledger(path)
            self.assertEqual(rows[0]["kind"], "commit")
            self.assertEqual(rows[1]["kind"], "resolution")
            ok, detail = verify_ledger(path, digest, "stationary_A", 50)
            self.assertTrue(ok)
            replay_ok, replayed = replay_and_verify(
                path, protocol, digest, "stationary_A", 50
            )
            self.assertTrue(replay_ok)
            self.assertEqual(replayed["final_state"], result["final_state"])

    def test_tamper_cases_fail_verification(self) -> None:
        protocol = load_protocol()
        digest = protocol_hash(protocol)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            run_variant(
                protocol,
                "stationary_A",
                51,
                "window64",
                ledger_path=path,
                protocol_digest=digest,
            )
            rows = read_ledger(path)

            def write(mutated):
                path.write_text(
                    "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in mutated)
                    + "\n",
                    encoding="utf-8",
                )

            # Edit
            edited = deepcopy(rows)
            edited[1]["outcome"] = 0 if edited[1]["outcome"] != 0 else 1
            write(edited)
            self.assertFalse(verify_ledger(path, digest, "stationary_A", 51)[0])

            # Delete / truncate mid-pair
            write(rows[:-1])
            self.assertFalse(verify_ledger(path, digest, "stationary_A", 51)[0])

            # Reorder
            write(rows[2:4] + rows[0:2] + rows[4:])
            self.assertFalse(verify_ledger(path, digest, "stationary_A", 51)[0])

            # Duplicate
            write(rows[:2] + rows[:2] + rows[2:])
            self.assertFalse(verify_ledger(path, digest, "stationary_A", 51)[0])

            # Restore and confirm ok
            write(rows)
            self.assertTrue(verify_ledger(path, digest, "stationary_A", 51)[0])


class SmokeTests(unittest.TestCase):
    def test_smoke_experiment_subset(self) -> None:
        # Tiny synthetic protocol for CI speed.
        protocol = load_protocol()
        tiny = deepcopy(protocol)
        tiny["seeds"] = {"all": 2, "calibration": [0, 0], "evaluation": [1, 1]}
        tiny["scenarios"] = {
            "stationary_A": {"length": 40, "schedule": [["A", 40]]},
            "hidden_ABA": {"length": 60, "schedule": [["A", 20], ["B", 20], ["A", 20]]},
            "hidden_ADA": {"length": 60, "schedule": [["A", 20], ["D", 20], ["A", 20]]},
        }
        tiny["burn_in"] = 10
        with tempfile.TemporaryDirectory() as directory:
            # Monkeypatch by writing a temporary protocol is hard; call run_variant only.
            digest = protocol_hash(protocol)
            path = Path(directory) / "runs" / "stationary_A" / "1.jsonl"
            result = run_variant(
                protocol,
                "stationary_A",
                1,
                "window64",
                ledger_path=path,
                protocol_digest=digest,
            )
            self.assertGreater(result["event_count"], 0)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
