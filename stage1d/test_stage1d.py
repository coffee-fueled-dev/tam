from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .experiment import Config, run_experiment, sample_loss
from .model import (
    LossModel,
    build_commitment,
    cvar_90,
    exact_probabilities,
    select_port,
)


class SelectorTests(unittest.TestCase):
    def test_exact_selector_sanity_primary_regimes(self) -> None:
        below = {
            "safe": exact_probabilities("safe", 0.08, 18.0),
            "rare_tail": exact_probabilities("rare_tail", 0.08, 18.0),
        }
        above = {
            "safe": exact_probabilities("safe", 0.12, 12.0),
            "rare_tail": exact_probabilities("rare_tail", 0.12, 12.0),
        }
        self.assertEqual(select_port(below, "expected_value")[0], "rare_tail")
        self.assertEqual(select_port(above, "expected_value")[0], "rare_tail")
        self.assertEqual(select_port(below, "cvar_90")[0], "safe")
        self.assertEqual(select_port(above, "cvar_90")[0], "safe")
        self.assertEqual(select_port(below, "worst_case")[0], "safe")
        self.assertEqual(select_port(above, "worst_case")[0], "safe")
        self.assertEqual(select_port(below, "tam_90")[0], "rare_tail")
        self.assertEqual(select_port(above, "tam_90")[0], "safe")

    def test_cone_boundary_includes_or_excludes_catastrophe(self) -> None:
        below = exact_probabilities("rare_tail", 0.08, 18.0)
        above = exact_probabilities("rare_tail", 0.12, 12.0)
        self.assertEqual(build_commitment(below).cone, (0.0,))
        self.assertEqual(build_commitment(above).cone, (0.0, 12.0))

    def test_cvar_values(self) -> None:
        safe = exact_probabilities("safe", 0.08, 18.0)
        risky_below = exact_probabilities("rare_tail", 0.08, 18.0)
        risky_above = exact_probabilities("rare_tail", 0.12, 12.0)
        self.assertAlmostEqual(cvar_90(safe), 2.0)
        # p=0.08: worst 10% = all 8% catastrophe plus 2% of the zero-loss mass.
        self.assertAlmostEqual(cvar_90(risky_below), 14.4)
        # p=0.12: worst 10% lies entirely inside the catastrophe.
        self.assertAlmostEqual(cvar_90(risky_above), 12.0)

    def test_sample_loss_uses_threshold(self) -> None:
        self.assertEqual(sample_loss("safe", 0.08, 18.0, 0.99), 2.0)
        self.assertEqual(sample_loss("rare_tail", 0.08, 18.0, 0.07), 18.0)
        self.assertEqual(sample_loss("rare_tail", 0.08, 18.0, 0.08), 0.0)


class ModelTests(unittest.TestCase):
    def test_serialization_round_trip(self) -> None:
        model = LossModel(history_size=32)
        model.observe("safe", 2.0)
        model.observe("rare_tail", 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(LossModel.load(path).to_dict(), model.to_dict())

    def test_smoothing_keeps_all_losses(self) -> None:
        model = LossModel()
        probs = model.probabilities("safe")
        self.assertEqual(set(probs), {0.0, 2.0, 12.0, 14.4, 18.0})
        self.assertAlmostEqual(sum(probs.values()), 1.0)


class ExperimentTests(unittest.TestCase):
    config = Config(
        seeds=(0, 1),
        evaluation_draws=200,
        train_budgets=(20, 40),
        bootstrap_samples=50,
    )

    def test_complete_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(directory, self.config)
            for filename in ("decisions.jsonl", "summary.json", "model.json"):
                self.assertTrue((Path(directory) / filename).is_file())
            self.assertIn("verdict", summary["decision"])
            self.assertIn("exact_selector_sanity", summary["evidence_targets"])

    def test_evaluation_does_not_update_after_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_experiment(directory, self.config)
            models = json.loads((Path(directory) / "model.json").read_text())
            key20 = [key for key in models if key.endswith(":20")][0]
            counts = models[key20]["histories"]
            self.assertEqual(len(counts["safe"]), 20)
            self.assertEqual(len(counts["rare_tail"]), 20)

    def test_shared_draws_are_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_experiment(directory, self.config)
            rows = [
                json.loads(line)
                for line in (Path(directory) / "decisions.jsonl").read_text().splitlines()
                if '"phase": "exact"' in line and '"regime": "below_cutoff"' in line
            ]
            by_seed = {}
            for row in rows:
                by_seed.setdefault(row["seed"], {})[row["controller"]] = row
            for controllers in by_seed.values():
                # Same evaluation stream implies identical loss when same port chosen.
                if (
                    controllers["expected_value"]["chosen_port"]
                    == controllers["tam_90"]["chosen_port"]
                ):
                    self.assertAlmostEqual(
                        controllers["expected_value"]["mean_loss"],
                        controllers["tam_90"]["mean_loss"],
                    )

    def test_runs_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            run_experiment(first, self.config)
            run_experiment(second, self.config)
            self.assertEqual(
                (Path(first) / "decisions.jsonl").read_bytes(),
                (Path(second) / "decisions.jsonl").read_bytes(),
            )
            left = json.loads((Path(first) / "summary.json").read_text())
            right = json.loads((Path(second) / "summary.json").read_text())
            left.pop("runtime_seconds", None)
            right.pop("runtime_seconds", None)
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
