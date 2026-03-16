"""Typed implementation rules for TAM v4.

This module is intentionally small. It translates the formulation into the
objects used by the codebase.

Implementation rules
--------------------

1. `Situation(step=n, latent_state=x_n, prior_context=...)` is the typed form of
   the current situation.
2. A port is conditioned on the situation and produces a `PortFiber`.
3. The fiber is frozen into a `ClaimedRegion` at bind time.
4. The world returns a `ContextEpisode`.
5. The episode is interpreted into a `LatentTransition`.
6. Contradiction is computed against the frozen claim, not against a
   recomputed post-hoc region.
7. The next situation is built explicitly from the next latent state and next
   prior context.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from v4.core import ContextWindow, LatentTransition, Situation

Tensor = torch.Tensor


@dataclass(frozen=True)
class TheoryGlossary:
    """A compact glossary for the main v4 objects."""

    situation: str = "An indexed latent state plus prior context."
    latent_transition: str = "A realized movement in latent transition space."
    port_fiber: str = "A state-conditioned geometric claim over latent transitions."
    claimed_region: str = "The bind-time claim frozen from a port fiber."
    contradiction: str = "The mismatch between a frozen claim and a realized transition."


def transition_from_states(start_latent_state: Tensor, end_latent_state: Tensor) -> LatentTransition:
    """Create a typed latent transition from two latent states."""
    return LatentTransition(start=start_latent_state, end=end_latent_state)


def next_situation(
    previous: Situation,
    next_latent_state: Tensor,
    next_observed_state: Tensor,
    next_prior_context: ContextWindow,
) -> Situation:
    """Create the next typed situation.

    This keeps the state/situation distinction explicit:
    the state is the latent vector, and the situation is the indexed wrapper.
    """
    return Situation(
        step=previous.step + 1,
        latent_state=next_latent_state,
        observed_state=next_observed_state,
        prior_context=next_prior_context,
    )
