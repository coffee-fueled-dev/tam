"""Situation encoder for the atlas runtime."""

from __future__ import annotations

from typing import Optional, Protocol

import torch
import torch.nn as nn

from atlas.core import InferredSituation

Tensor = torch.Tensor


def normalize_observed_context(observed_context: Tensor) -> Tensor:
    """Bound nuisance magnitude without removing informative scale."""
    flat = observed_context.reshape(-1).to(dtype=torch.float32)
    return torch.tanh(flat / 3.0)


class SituationEncoder(Protocol):
    """Maps raw context into retrieval and geometry latents."""

    def infer(self, observed_context: Tensor) -> InferredSituation:
        ...


class ReferenceSituationEncoder(nn.Module):
    """Small shared-trunk encoder with separate retrieval and geometry heads."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        query_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        feature_dropout: float = 0.0,
    ):
        super().__init__()
        query_dim = query_dim if query_dim is not None else latent_dim
        hidden_dim = hidden_dim if hidden_dim is not None else max(32, input_dim, latent_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.query_head = nn.Linear(hidden_dim, query_dim)
        self.novelty_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, observed_context: Tensor) -> InferredSituation:
        flat = normalize_observed_context(observed_context)
        shared = self.shared(self.feature_dropout(flat))
        situation_latent = self.latent_head(shared)
        query_key = self.query_head(shared)
        novelty_hint = torch.sigmoid(self.novelty_head(shared))
        confidence_hint = torch.sigmoid(self.confidence_head(shared))
        return InferredSituation(
            observed_context=flat,
            situation_latent=situation_latent,
            query_key=query_key,
            novelty_hint=novelty_hint,
            confidence_hint=confidence_hint,
        )

    def infer(self, observed_context: Tensor) -> InferredSituation:
        return self.forward(observed_context)


class IdentitySituationEncoder:
    """Deterministic encoder useful for tests and tiny demos."""

    def __init__(self, query_dims: Optional[int] = None):
        self.query_dims = query_dims

    def infer(self, observed_context: Tensor) -> InferredSituation:
        flat = normalize_observed_context(observed_context)
        query_key = flat if self.query_dims is None else flat[: self.query_dims]
        novelty_hint = torch.zeros(1, dtype=torch.float32)
        confidence_hint = torch.ones(1, dtype=torch.float32)
        return InferredSituation(
            observed_context=flat,
            situation_latent=flat,
            query_key=query_key,
            novelty_hint=novelty_hint,
            confidence_hint=confidence_hint,
        )
