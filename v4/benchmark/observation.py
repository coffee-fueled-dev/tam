"""Observation contract for the geometric benchmark world.

This module keeps the observation stream explicit without introducing a
tokenizer-specific representation. The key idea is:

- observations are made of named feature groups
- most groups are per-dimension
- feature count and ordering can vary across tasks
- encoders can consume either the flattened observation or per-dimension feature
  matrices derived from the same structured frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class FeatureGroup:
    """A named feature group inside one observation frame."""

    name: str
    values: Tensor
    per_dimension: bool = True


@dataclass
class ObservationFrame:
    """Structured observation before flattening.

    The order of `feature_groups` is part of the contract and can be shuffled
    deliberately by benchmark variants.
    """

    state: Tensor
    feature_groups: List[FeatureGroup]
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[-1])


def flatten_frame(frame: ObservationFrame) -> Tensor:
    """Flatten an `ObservationFrame` into a tensor for the current v4 runtime."""
    parts: List[Tensor] = [frame.state]
    for group in frame.feature_groups:
        parts.append(group.values.reshape(-1))
    return torch.cat(parts, dim=0).to(dtype=torch.float32)


def feature_order(frame: ObservationFrame) -> List[str]:
    """Return the explicit feature-group order for this frame."""
    return [group.name for group in frame.feature_groups]


def dimension_feature_matrix(frame: ObservationFrame) -> Tuple[Tensor, List[str]]:
    """Build a per-dimension feature matrix.

    Returns:
    - matrix: `(state_dim, feature_dim)`
    - feature_names: names corresponding to the columns of the matrix
    """
    rows: List[List[float]] = [[] for _ in range(frame.state_dim)]
    names: List[str] = []

    for group in frame.feature_groups:
        if group.per_dimension:
            names.append(group.name)
            for dim_idx in range(frame.state_dim):
                rows[dim_idx].append(float(group.values[dim_idx].item()))
        else:
            flat_values = group.values.reshape(-1)
            for value_idx, value in enumerate(flat_values):
                names.append(f"{group.name}_{value_idx}")
                scalar = float(value.item())
                for dim_idx in range(frame.state_dim):
                    rows[dim_idx].append(scalar)

    matrix = torch.tensor(rows, dtype=torch.float32)
    return matrix, names


def feature_order_robustness(ordered_feature_names: List[List[str]]) -> float:
    """A simple summary of how many distinct feature orders appear."""
    if not ordered_feature_names:
        return 0.0
    unique_orders = len(set(tuple(names) for names in ordered_feature_names))
    return 1.0 / max(unique_orders, 1)
