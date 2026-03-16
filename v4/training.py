"""Binding-centric training helpers for v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from v4.runtime_torch import TorchAgent, initial_history
from v4.world import World

Tensor = torch.Tensor


@dataclass
class StepStats:
    """Useful values from one training step."""

    contradiction: float
    success: bool
    selected_port: str
    policy_loss: float
    geometry_loss: float
    entropy: float


@dataclass
class TrainingTrace:
    """Collected stats for a short training run."""

    steps: List[StepStats] = field(default_factory=list)

    def contradictions(self) -> List[float]:
        return [step.contradiction for step in self.steps]


def training_loss(
    contradiction: Tensor,
    log_probability: Tensor | None,
    entropy: Tensor,
    policy_weight: float = 0.1,
    entropy_weight: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split learning signal into geometry and selection terms.

    - geometry loss updates the claimed region itself
    - policy loss updates which port gets selected
    """
    geometry_loss = contradiction
    if log_probability is None:
        policy_loss = torch.zeros_like(geometry_loss)
    else:
        policy_loss = contradiction.detach() * (-log_probability)
    total_loss = geometry_loss + policy_weight * policy_loss - entropy_weight * entropy
    return total_loss, geometry_loss, policy_loss


def train_agent(
    agent: TorchAgent,
    world: World,
    optimizer: torch.optim.Optimizer,
    steps: int,
    sample: bool = True,
) -> TrainingTrace:
    """Run a short contradiction-driven training loop."""
    history = initial_history()
    observed_state = world.initial_observed_state()
    trace = TrainingTrace()

    for step in range(steps):
        optimizer.zero_grad()
        outcome, artifacts = agent.run_cycle(
            world=world,
            history=history,
            step=step,
            observed_state=observed_state,
            sample=sample,
        )
        total_loss, geometry_loss, policy_loss = training_loss(
            contradiction=outcome.contradiction,
            log_probability=outcome.binding.selection.log_probability,
            entropy=artifacts.selection_entropy,
        )
        total_loss.backward()
        optimizer.step()

        trace.steps.append(
            StepStats(
                contradiction=float(outcome.contradiction.item()),
                success=outcome.success,
                selected_port=outcome.binding.selection.port_name,
                policy_loss=float(policy_loss.item()),
                geometry_loss=float(geometry_loss.item()),
                entropy=float(artifacts.selection_entropy.item()),
            )
        )
        observed_state = outcome.next_situation.observed_state.detach()

    return trace
