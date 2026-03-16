"""PyTorch runtime adapters for v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from v4.context import LastNContextSelector
from v4.core import (
    BindingOutcome,
    BindingRecord,
    ClaimedRegion,
    ContextWindow,
    PortSelection,
    Situation,
)
from v4.inference import EncoderEpisodeInterpreter, MLPStateEncoder
from v4.ports import FixedFiberPort
from v4.theory import next_situation
from v4.world import World, WorldHistory

Tensor = torch.Tensor


class PortFiberHead(nn.Module):
    """Parameterize a family of fibers from the current latent state."""

    def __init__(self, latent_dim: int, n_ports: int, fiber_dim: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_ports = n_ports
        self.fiber_dim = fiber_dim

        hidden_dim = max(32, latent_dim)
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.anchor_head = nn.Linear(hidden_dim, n_ports * latent_dim)
        self.basis_head = nn.Linear(hidden_dim, n_ports * latent_dim * fiber_dim)
        self.radius_head = nn.Linear(hidden_dim, n_ports * latent_dim)
        self.confidence_head = nn.Linear(hidden_dim, n_ports)
        self.selection_head = nn.Linear(hidden_dim, n_ports)

    def forward(self, latent_state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        shared = self.shared(latent_state)
        anchors = self.anchor_head(shared).view(self.n_ports, self.latent_dim)
        bases = self.basis_head(shared).view(self.n_ports, self.latent_dim, self.fiber_dim)
        radii = torch.nn.functional.softplus(
            self.radius_head(shared).view(self.n_ports, self.latent_dim)
        ) + 0.05
        confidences = torch.sigmoid(self.confidence_head(shared))
        logits = self.selection_head(shared)
        return anchors, bases, radii, confidences, logits


@dataclass
class CycleArtifacts:
    """Extra tensors returned by the torch runtime for training."""

    selection_logits: Tensor
    selection_probabilities: Tensor
    selection_entropy: Tensor


class TorchAgent(nn.Module):
    """A small trainable agent that follows the v4 cycle explicitly."""

    def __init__(
        self,
        observed_dim: int,
        latent_dim: int,
        n_ports: int,
        fiber_dim: int = 1,
        context_size: int = 4,
    ):
        super().__init__()
        self.encoder = MLPStateEncoder(observed_dim, latent_dim)
        self.interpreter = EncoderEpisodeInterpreter(self.encoder)
        self.port_head = PortFiberHead(latent_dim, n_ports, fiber_dim=fiber_dim)
        self.context_selector = LastNContextSelector(context_size)
        self.n_ports = n_ports

    def encode_situation(
        self,
        step: int,
        observed_state: Tensor,
        history: WorldHistory,
    ) -> Situation:
        prior_context = self.context_selector.select(history.atoms)
        latent_state = self.encoder(observed_state)
        return Situation(
            step=step,
            latent_state=latent_state,
            observed_state=observed_state,
            prior_context=prior_context,
        )

    def build_ports(self, situation: Situation) -> tuple[List[FixedFiberPort], Tensor, Tensor]:
        anchors, bases, radii, confidences, logits = self.port_head(situation.latent_state)
        probabilities = torch.softmax(logits, dim=-1)

        ports: List[FixedFiberPort] = []
        for port_idx in range(self.n_ports):
            ports.append(
                FixedFiberPort(
                    name=f"port_{port_idx}",
                    anchor=anchors[port_idx],
                    basis=bases[port_idx],
                    radius=radii[port_idx],
                    confidence=confidences[port_idx],
                )
            )
        return ports, logits, probabilities

    def select_binding(
        self,
        situation: Situation,
        sample: bool = True,
    ) -> tuple[BindingRecord, CycleArtifacts]:
        ports, logits, probabilities = self.build_ports(situation)
        distribution = torch.distributions.Categorical(probabilities)
        if sample:
            port_index = distribution.sample()
        else:
            port_index = torch.argmax(probabilities)

        chosen_index = int(port_index.item())
        chosen_port = ports[chosen_index]
        fiber = chosen_port.make_fiber(situation)
        selection = PortSelection(
            port_name=chosen_port.name,
            port_index=chosen_index,
            logit=logits[chosen_index],
            probability=probabilities[chosen_index],
            log_probability=distribution.log_prob(port_index),
        )
        binding = BindingRecord(
            situation_step=situation.step,
            selection=selection,
            claimed_region=ClaimedRegion(
                port_name=chosen_port.name,
                fiber=fiber,
                bind_context=situation.prior_context.copy(),
            ),
        )
        artifacts = CycleArtifacts(
            selection_logits=logits,
            selection_probabilities=probabilities,
            selection_entropy=distribution.entropy(),
        )
        return binding, artifacts

    def run_cycle(
        self,
        world: World,
        history: WorldHistory,
        step: int,
        observed_state: Tensor,
        sample: bool = True,
    ) -> tuple[BindingOutcome, CycleArtifacts]:
        history.append_observation(world.observe(observed_state))
        situation = self.encode_situation(step, observed_state, history)
        binding, artifacts = self.select_binding(situation, sample=sample)
        episode = world.bind(binding, situation)
        realized_transition = self.interpreter(situation, episode)
        contradiction = binding.claimed_region.contradiction(realized_transition)
        history.append_episode(episode)

        next_observed_state = episode.last()
        next_prior_context = self.context_selector.select(history.atoms)
        next_latent_state = self.encoder(next_observed_state)
        next_state = next_situation(
            previous=situation,
            next_latent_state=next_latent_state,
            next_observed_state=next_observed_state,
            next_prior_context=next_prior_context,
        )
        outcome = BindingOutcome(
            binding=binding,
            episode=episode,
            realized_transition=realized_transition,
            contradiction=contradiction,
            success=binding.claimed_region.contains(realized_transition),
            next_situation=next_state,
        )
        return outcome, artifacts


def initial_history() -> WorldHistory:
    """Create an empty world history for a new run."""
    return WorldHistory()
