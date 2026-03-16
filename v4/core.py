"""Core theory objects for TAM v4.

The design in this file follows a few simple rules:

1. A `Situation` is the current indexed latent state plus the prior context.
2. A port does not directly output an action. It produces a `PortFiber`.
3. A `PortFiber` defines a `ClaimedRegion` in latent transition space.
4. Binding freezes that claim in a `BindingRecord`.
5. The world returns a `ContextEpisode`.
6. The episode is interpreted into a `LatentTransition`.
7. Learning is driven by contradiction between the realized transition and the
   frozen claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import torch

Tensor = torch.Tensor


def _as_column_basis(basis: Tensor, latent_dim: int, device: torch.device) -> Tensor:
    """Normalize basis shape to `(latent_dim, fiber_dim)`."""
    if basis.numel() == 0:
        return torch.zeros(latent_dim, 0, dtype=torch.float32, device=device)
    if basis.dim() == 1:
        return basis.unsqueeze(-1)
    return basis


@dataclass
class ContextAtom:
    """One piece of context returned by the world."""

    value: Tensor
    kind: str = "observation"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """The context view available at a given point in the cycle."""

    atoms: List[ContextAtom] = field(default_factory=list)

    def values(self) -> List[Tensor]:
        return [atom.value for atom in self.atoms]

    def append(self, atom: ContextAtom) -> None:
        self.atoms.append(atom)

    def copy(self) -> "ContextWindow":
        return ContextWindow(
            atoms=[
                ContextAtom(
                    value=atom.value.clone(),
                    kind=atom.kind,
                    metadata=dict(atom.metadata),
                )
                for atom in self.atoms
            ]
        )


@dataclass
class ContextEpisode:
    """A world response: an ordered sequence of context atoms."""

    atoms: List[ContextAtom]
    metadata: Dict[str, float] = field(default_factory=dict)

    def first(self) -> Tensor:
        if not self.atoms:
            raise ValueError("ContextEpisode is empty")
        return self.atoms[0].value

    def last(self) -> Tensor:
        if not self.atoms:
            raise ValueError("ContextEpisode is empty")
        return self.atoms[-1].value


@dataclass
class Situation:
    """An indexed latent state plus the prior context used at selection time."""

    step: int
    latent_state: Tensor
    observed_state: Tensor
    prior_context: ContextWindow = field(default_factory=ContextWindow)
    cache: Dict[str, Tensor] = field(default_factory=dict)


@dataclass
class LatentTransition:
    """A realized movement in latent transition space."""

    start: Tensor
    end: Tensor

    @property
    def delta(self) -> Tensor:
        return self.end - self.start


@dataclass
class PortFiber:
    """A geometric claim in latent transition space.

    The claim is a low-dimensional affine fiber:

        anchor + basis @ coordinates

    together with an axis-aligned transverse radius.
    """

    anchor: Tensor
    basis: Tensor
    radius: Tensor
    confidence: Optional[Tensor] = None

    def __post_init__(self) -> None:
        latent_dim = self.anchor.shape[-1]
        self.basis = _as_column_basis(self.basis, latent_dim, self.anchor.device)
        if self.radius.dim() == 0:
            self.radius = self.radius.repeat(latent_dim)
        if self.confidence is None:
            self.confidence = torch.tensor(1.0, device=self.anchor.device)

    def nearest_point(self, point: Tensor) -> Tensor:
        """Project a point onto the affine fiber."""
        if self.basis.numel() == 0 or self.basis.shape[-1] == 0:
            return self.anchor

        centered = point - self.anchor
        coords = torch.linalg.pinv(self.basis) @ centered
        return self.anchor + self.basis @ coords

    def contradiction(self, point: Tensor, eps: float = 1e-6) -> Tensor:
        """Measure how strongly a realized point violates this claim."""
        nearest = self.nearest_point(point)
        residual = point - nearest
        scaled = residual / (self.radius + eps)
        return torch.sqrt(torch.mean(scaled**2))

    def contains(self, point: Tensor, threshold: float = 1.0) -> bool:
        """A simple boolean success test for debugging and examples."""
        return bool(self.contradiction(point).item() <= threshold)


@dataclass
class ClaimedRegion:
    """The bind-time geometric claim made by a port."""

    port_name: str
    fiber: PortFiber
    bind_context: ContextWindow
    threshold: float = 1.0

    def contradiction(self, realized_transition: LatentTransition) -> Tensor:
        return self.fiber.contradiction(realized_transition.delta)

    def contains(self, realized_transition: LatentTransition) -> bool:
        return self.fiber.contains(
            realized_transition.delta,
            threshold=self.threshold,
        )


@dataclass
class PortSelection:
    """The selected port and the score used to choose it."""

    port_name: str
    port_index: int
    logit: Tensor
    probability: Optional[Tensor] = None
    log_probability: Optional[Tensor] = None


@dataclass
class BindingRecord:
    """The frozen claim made when a port was bound."""

    situation_step: int
    selection: PortSelection
    claimed_region: ClaimedRegion


@dataclass
class BindingOutcome:
    """The result of comparing world reality to the frozen claim."""

    binding: BindingRecord
    episode: ContextEpisode
    realized_transition: LatentTransition
    contradiction: Tensor
    success: bool
    next_situation: Situation


class Port(Protocol):
    """A port maps a situation to a state-conditioned fiber."""

    name: str

    def make_fiber(self, situation: Situation) -> PortFiber:
        ...


class PortSet(Protocol):
    """A family of ports that can be queried from one situation."""

    def ports_for(self, situation: Situation) -> Sequence[Port]:
        ...


class World(Protocol):
    """Minimal world interface for the reference runtime."""

    def initial_observed_state(self) -> Tensor:
        ...

    def observe(self, observed_state: Tensor) -> ContextAtom:
        ...

    def bind(self, binding: BindingRecord, situation: Situation) -> ContextEpisode:
        ...


def choose_port(
    situation: Situation,
    ports: Sequence[Port],
    scores: Optional[Sequence[Tensor]] = None,
) -> BindingRecord:
    """Select a port and freeze its bind-time claim.

    This is intentionally simple:
    - if scores are provided, choose the highest score
    - otherwise choose the first port
    """
    if not ports:
        raise ValueError("At least one port is required")

    chosen_index = 0
    if scores is not None:
        chosen_index = max(range(len(ports)), key=lambda i: float(scores[i].item()))

    chosen_port = ports[chosen_index]
    fiber = chosen_port.make_fiber(situation)
    selection = PortSelection(
        port_name=chosen_port.name,
        port_index=chosen_index,
        logit=scores[chosen_index] if scores is not None else torch.tensor(0.0),
    )
    claimed_region = ClaimedRegion(
        port_name=chosen_port.name,
        fiber=fiber,
        bind_context=situation.prior_context.copy(),
    )
    return BindingRecord(
        situation_step=situation.step,
        selection=selection,
        claimed_region=claimed_region,
    )
