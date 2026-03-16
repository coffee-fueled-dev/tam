"""Small reference worlds for v4."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from v4.core import BindingRecord, ContextAtom, ContextEpisode, Situation

Tensor = torch.Tensor


@dataclass
class PlaneWorld:
    """A tiny 2D world with one movement vector per port.

    The world ignores the port's geometric claim. It only uses the chosen
    port identity and then returns what actually happened. This keeps the
    separation clear:

    - the agent claims a region of admissible futures
    - the world produces reality
    - contradiction is measured afterwards
    """

    drift_by_port: dict[str, Tensor]
    noise_scale: float = 0.05

    def initial_observed_state(self) -> Tensor:
        return torch.zeros(2, dtype=torch.float32)

    def observe(self, observed_state: Tensor) -> ContextAtom:
        return ContextAtom(value=observed_state.clone(), kind="observation")

    def bind(self, binding: BindingRecord, situation: Situation) -> ContextEpisode:
        start = situation.observed_state
        drift = self.drift_by_port[binding.selection.port_name].to(start.device)
        noise = torch.randn_like(start) * self.noise_scale
        end = start + drift + noise
        midpoint = start + 0.5 * (end - start)
        return ContextEpisode(
            atoms=[
                ContextAtom(value=start.clone(), kind="state"),
                ContextAtom(value=midpoint, kind="state"),
                ContextAtom(value=end, kind="state"),
            ]
        )
