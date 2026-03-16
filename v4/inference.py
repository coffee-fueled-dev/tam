"""Inference helpers for v4."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from v4.core import ContextEpisode, LatentTransition, Situation
from v4.theory import transition_from_states

Tensor = torch.Tensor


class SituationEncoder(Protocol):
    """Maps observed state into latent state."""

    def __call__(self, observed_state: Tensor) -> Tensor:
        ...


class EpisodeInterpreter(Protocol):
    """Turns a context episode into a realized latent transition."""

    def __call__(self, situation: Situation, episode: ContextEpisode) -> LatentTransition:
        ...


class MLPStateEncoder(nn.Module):
    """A tiny encoder used by the reference runtime."""

    def __init__(self, observed_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observed_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, observed_state: Tensor) -> Tensor:
        return self.net(observed_state)


class EncoderEpisodeInterpreter(nn.Module):
    """Interpret an episode by encoding the last observed state."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, situation: Situation, episode: ContextEpisode) -> LatentTransition:
        end_latent = self.encoder(episode.last())
        return transition_from_states(situation.latent_state, end_latent)
