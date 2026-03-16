"""Context selection utilities for v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from v4.core import ContextAtom, ContextWindow


@dataclass
class LastNContextSelector:
    """Select the last `n` context atoms.

    This is intentionally deterministic and easy to understand.
    """

    n: int

    def select(self, history: Iterable[ContextAtom]) -> ContextWindow:
        atoms: List[ContextAtom] = list(history)
        if self.n <= 0:
            return ContextWindow([])
        return ContextWindow(atoms=atoms[-self.n :])
