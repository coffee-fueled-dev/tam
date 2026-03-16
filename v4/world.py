"""World interfaces for v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol

from v4.core import BindingRecord, ContextAtom, ContextEpisode, ContextWindow, Situation


class World(Protocol):
    """Minimal world protocol for the reference implementation."""

    def initial_observed_state(self):
        ...

    def observe(self, observed_state) -> ContextAtom:
        ...

    def bind(self, binding: BindingRecord, situation: Situation) -> ContextEpisode:
        ...


@dataclass
class WorldHistory:
    """A tiny helper for tracking context history."""

    atoms: List[ContextAtom] = field(default_factory=list)

    def append_episode(self, episode: ContextEpisode) -> None:
        self.atoms.extend(episode.atoms)

    def append_observation(self, atom: ContextAtom) -> None:
        self.atoms.append(atom)

    def as_window(self) -> ContextWindow:
        return ContextWindow(atoms=list(self.atoms))
