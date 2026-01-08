"""
TAM Python implementation - Action-Binding with Continuous Ports.
"""

from .actor import Actor
from .networks import ActorNet, SharedPolicy, SharedTube
from .utils import (
    gaussian_nll,
    interp1d_linear,
    kl_diag_gaussian_to_standard,
    sample_truncated_geometric,
    truncated_geometric_weights,
)

__all__ = [
    # Core TAM
    "Actor",
    # Networks
    "ActorNet",
    "SharedPolicy",
    "SharedTube",
    # Utilities
    "gaussian_nll",
    "kl_diag_gaussian_to_standard",
    "truncated_geometric_weights",
    "sample_truncated_geometric",
    "interp1d_linear",
]
