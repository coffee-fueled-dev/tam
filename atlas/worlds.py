"""Tiny synthetic worlds for atlas development."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import torch

Tensor = torch.Tensor


@dataclass
class Regime:
    """One recurring local regime in observation space."""

    name: str
    center: Tensor
    offsets: List[Tensor] = field(default_factory=list)

    def observations(self) -> list[Tensor]:
        base = self.center.to(dtype=torch.float32)
        if not self.offsets:
            return [base]
        return [base + offset.to(dtype=torch.float32) for offset in self.offsets]


class RecurringRegimeWorld:
    """A deterministic stream with recurring local regimes."""

    def __init__(self, regimes: Iterable[Regime]):
        self.regimes = list(regimes)

    def sequence(self) -> list[Tensor]:
        observations: list[Tensor] = []
        for regime in self.regimes:
            observations.extend(regime.observations())
        return observations


def default_world() -> RecurringRegimeWorld:
    """Create a tiny world with two recurring, separated regimes."""
    return RecurringRegimeWorld(
        regimes=[
            Regime(
                name="left_cluster",
                center=torch.tensor([1.0, 1.0]),
                offsets=[
                    torch.tensor([0.0, 0.0]),
                    torch.tensor([0.1, -0.1]),
                    torch.tensor([-0.1, 0.1]),
                ],
            ),
            Regime(
                name="right_cluster",
                center=torch.tensor([4.0, 4.0]),
                offsets=[
                    torch.tensor([0.0, 0.0]),
                    torch.tensor([0.1, 0.0]),
                    torch.tensor([0.0, -0.1]),
                ],
            ),
            Regime(
                name="left_cluster_repeat",
                center=torch.tensor([1.0, 1.0]),
                offsets=[
                    torch.tensor([0.05, 0.0]),
                    torch.tensor([-0.05, 0.05]),
                ],
            ),
        ]
    )
