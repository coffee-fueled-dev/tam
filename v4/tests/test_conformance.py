"""Conformance tests for the v4 theory/runtime boundary."""

from __future__ import annotations

import unittest

import torch

from v4.core import ContextAtom, ContextWindow, Situation
from v4.runtime_torch import TorchAgent, initial_history
from v4.training import training_loss
from v4.worlds import PlaneWorld


class V4ConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.world = PlaneWorld(
            drift_by_port={
                "port_0": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "port_1": torch.tensor([0.0, 1.0], dtype=torch.float32),
            },
            noise_scale=0.0,
        )
        self.agent = TorchAgent(observed_dim=2, latent_dim=4, n_ports=2, fiber_dim=1)

    def test_bind_time_context_is_frozen(self) -> None:
        history = initial_history()
        observed_state = self.world.initial_observed_state()
        history.append_observation(self.world.observe(observed_state))
        situation = self.agent.encode_situation(0, observed_state, history)

        situation.prior_context.append(ContextAtom(value=torch.ones(2), kind="extra"))
        binding, _ = self.agent.select_binding(situation, sample=False)

        situation.prior_context.append(ContextAtom(value=torch.zeros(2), kind="late"))

        self.assertEqual(len(binding.claimed_region.bind_context.atoms), 2)
        self.assertEqual(len(situation.prior_context.atoms), 3)

    def test_next_state_is_wrapped_in_situation(self) -> None:
        history = initial_history()
        observed_state = self.world.initial_observed_state()
        outcome, _ = self.agent.run_cycle(
            world=self.world,
            history=history,
            step=0,
            observed_state=observed_state,
            sample=False,
        )

        self.assertIsInstance(outcome.next_situation, Situation)
        self.assertEqual(outcome.next_situation.step, 1)

    def test_world_returns_episode_before_transition_evaluation(self) -> None:
        history = initial_history()
        observed_state = self.world.initial_observed_state()
        outcome, _ = self.agent.run_cycle(
            world=self.world,
            history=history,
            step=0,
            observed_state=observed_state,
            sample=False,
        )

        self.assertGreaterEqual(len(outcome.episode.atoms), 2)
        self.assertEqual(outcome.realized_transition.delta.shape[-1], 4)

    def test_contradiction_is_primary_geometry_signal(self) -> None:
        contradiction = torch.tensor(2.5, requires_grad=True)
        log_probability = torch.tensor(-0.7, requires_grad=True)
        entropy = torch.tensor(0.3, requires_grad=True)

        total_loss, geometry_loss, policy_loss = training_loss(
            contradiction=contradiction,
            log_probability=log_probability,
            entropy=entropy,
        )

        self.assertAlmostEqual(float(geometry_loss.item()), 2.5, places=5)
        self.assertGreater(float(policy_loss.item()), 0.0)
        self.assertGreater(float(total_loss.item()), float(geometry_loss.item()))


if __name__ == "__main__":
    unittest.main()
