"""Benchmark-specific runtime pieces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn

from v4.benchmark.observation import (
    dimension_feature_matrix,
)
from v4.core import ContextAtom
from v4.runtime_torch import PortFiberHead

Tensor = torch.Tensor

EncoderMode = Literal["raw", "shared", "dimension_specific"]
GeometryMode = Literal["fiber", "diagonal", "point"]


@dataclass
class EncodedObservation:
    """The encoded tensor plus optional feature-matrix metadata."""

    encoded: Tensor
    feature_matrix: Tensor | None = None


class RawObservationEncoder(nn.Module):
    """Simple baseline: use the raw flattened observation directly."""

    def __init__(self, observed_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observed_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, atom: ContextAtom) -> EncodedObservation:
        return EncodedObservation(encoded=self.net(atom.value), feature_matrix=None)


class SharedFeatureEncoder(nn.Module):
    """Encode per-dimension feature summaries with shared weights."""

    def __init__(self, observed_dim: int, latent_dim: int, per_dim_feature_dim: int):
        super().__init__()
        self.raw_projection = nn.Linear(observed_dim, latent_dim)
        self.per_dim = nn.Sequential(
            nn.Linear(per_dim_feature_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, atom: ContextAtom) -> EncodedObservation:
        frame = atom.metadata["frame"]
        feature_matrix, _ = dimension_feature_matrix(frame)
        feature_matrix = feature_matrix.to(atom.value.device)

        summaries: List[Tensor] = []
        for dim_idx in range(feature_matrix.shape[0]):
            summaries.append(self.per_dim(feature_matrix[dim_idx]))

        pooled = torch.stack(summaries, dim=0).mean(dim=0) if summaries else 0.0
        encoded = self.raw_projection(atom.value) + pooled
        return EncodedObservation(encoded=encoded, feature_matrix=feature_matrix)


class DimensionSpecificFeatureEncoder(nn.Module):
    """A dimension-specific ablation for the shared feature encoder."""

    def __init__(
        self,
        observed_dim: int,
        latent_dim: int,
        state_dim: int,
        per_dim_feature_dim: int,
    ):
        super().__init__()
        self.raw_projection = nn.Linear(observed_dim, latent_dim)
        self.per_dim = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(per_dim_feature_dim, latent_dim),
                    nn.Tanh(),
                    nn.Linear(latent_dim, latent_dim),
                )
                for _ in range(state_dim)
            ]
        )

    def forward(self, atom: ContextAtom) -> EncodedObservation:
        frame = atom.metadata["frame"]
        feature_matrix, _ = dimension_feature_matrix(frame)
        feature_matrix = feature_matrix.to(atom.value.device)

        summaries: List[Tensor] = []
        for dim_idx in range(feature_matrix.shape[0]):
            summaries.append(self.per_dim[dim_idx](feature_matrix[dim_idx]))

        pooled = torch.stack(summaries, dim=0).mean(dim=0) if summaries else 0.0
        encoded = self.raw_projection(atom.value) + pooled
        return EncodedObservation(encoded=encoded, feature_matrix=feature_matrix)


class BenchmarkGeometryHead(nn.Module):
    """Geometry head with ablations for point / diagonal / fiber claims."""

    def __init__(self, latent_dim: int, n_ports: int, geometry_mode: GeometryMode, fiber_dim: int = 1):
        super().__init__()
        self.geometry_mode = geometry_mode
        actual_fiber_dim = fiber_dim if geometry_mode == "fiber" else 0
        self.fiber_head = PortFiberHead(latent_dim, n_ports, fiber_dim=max(actual_fiber_dim, 1))
        self.latent_dim = latent_dim
        self.n_ports = n_ports
        self.actual_fiber_dim = actual_fiber_dim

    def forward(self, latent_state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        anchors, bases, radii, confidences, logits = self.fiber_head(latent_state)
        if self.geometry_mode == "point":
            bases = torch.zeros(self.n_ports, self.latent_dim, 0, device=latent_state.device)
            radii = torch.full_like(radii, 0.1)
        elif self.geometry_mode == "diagonal":
            bases = torch.zeros(self.n_ports, self.latent_dim, 0, device=latent_state.device)
        else:
            bases = bases[:, :, : self.actual_fiber_dim]
        return anchors, bases, radii, confidences, logits
