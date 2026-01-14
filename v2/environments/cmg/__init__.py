"""
CMG (Commitment-Mode-Gating) Environment.
"""

from .env import CMGEnv, CMGConfig
from .episode import generate_episode, rollout_with_forced_mode
from .diagnostics import (
    fork_separability_test,
    commitment_regret_test,
    gating_irreversibility_test,
    run_cmg_topology_suite,
)

__all__ = [
    "CMGEnv",
    "CMGConfig",
    "generate_episode",
    "rollout_with_forced_mode",
    "fork_separability_test",
    "commitment_regret_test",
    "gating_irreversibility_test",
    "run_cmg_topology_suite",
]
