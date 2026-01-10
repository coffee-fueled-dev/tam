"""Environment package for TAM experiments."""

from .hidden_regime_fault import HiddenRegimeFaultEnv, FaultEvent
from .latent_rule_gridworld import LatentRuleGridworld
from .equivalent_envs import (
    RotatedGridworld,
    ScaledGridworld,
    MirroredGridworld,
    ShiftedRulesGridworld,
    ContinuousGridworld,
    make_standard_gridworld,
    make_rotated_gridworld,
    make_scaled_gridworld,
    make_mirrored_gridworld,
    make_shifted_rules_gridworld,
    make_continuous_gridworld,
    EQUIVALENT_ENV_PAIRS,
)

__all__ = [
    "HiddenRegimeFaultEnv",
    "FaultEvent",
    "LatentRuleGridworld",
    "RotatedGridworld",
    "ScaledGridworld",
    "MirroredGridworld",
    "ShiftedRulesGridworld",
    "ContinuousGridworld",
    "make_standard_gridworld",
    "make_rotated_gridworld",
    "make_scaled_gridworld",
    "make_mirrored_gridworld",
    "make_shifted_rules_gridworld",
    "make_continuous_gridworld",
    "EQUIVALENT_ENV_PAIRS",
]
