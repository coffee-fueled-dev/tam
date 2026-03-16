"""Port abstractions for v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from v4.core import Port, PortFiber, Situation

Tensor = torch.Tensor


@dataclass
class FixedFiberPort(Port):
    """A simple port with a fixed name and externally supplied fiber parameters."""

    name: str
    anchor: Tensor
    basis: Tensor
    radius: Tensor
    confidence: Tensor

    def make_fiber(self, situation: Situation) -> PortFiber:
        del situation
        return PortFiber(
            anchor=self.anchor,
            basis=self.basis,
            radius=self.radius,
            confidence=self.confidence,
        )


@dataclass
class SimplePortSet:
    """A concrete list-based port set."""

    ports: List[Port]

    def ports_for(self, situation: Situation) -> List[Port]:
        del situation
        return self.ports
