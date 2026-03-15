"""
Abstract Environment interface following TAM formalism.

The World W responds with context episodes when the Actor binds a port.
Context episodes are sequences: e_{n→n+1} = (c_{n,0}, c_{n,1}, ..., c_{n,k}, c_{n+1})
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from v3.environment_recorder import EnvironmentRecorder
    from v3.goal_motif import GoalMotif

import torch


class Environment(ABC):
    """
    Abstract environment interface following TAM formalism.

    The World W responds with context episodes when the Actor binds a port.
    Context episodes are sequences: e_{n→n+1} = (c_{n,0}, c_{n,1}, ..., c_{n,k}, c_{n+1})
    """

    @abstractmethod
    def get_context(
        self, current_state: torch.Tensor, intent: torch.Tensor
    ) -> torch.Tensor:
        """
        Get prior context before binding (for port affordance evaluation).

        This is the "prior context" c_n^prior that informs port affordance.

        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            intent: Intent/target (state_dim,) or (1, state_dim)

        Returns:
            raw_context: Raw context observation c_n ∈ C (raw_ctx_dim,)
        """
        pass

    @abstractmethod
    def bind_port(
        self, tube: torch.Tensor, sigma: torch.Tensor, current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Bind a port (execute tube) and receive context episode.

        This implements: When A binds port p_n, W responds with context episode e_{n→n+1}

        Args:
            tube: Proposed trajectory τ (affordance tube) (T, state_dim) - relative to current_state
            sigma: Precision vector (T, state_dim) or (T, 1)
            current_state: Current state x_n (state_dim,) or (1, state_dim)

        Returns:
            context_episode: Actual path taken e_{n→n+1} = (c_{n,0}, ..., c_{n+1}) (T', state_dim)
                           This is the sequence of states actually realized by the world
        """
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """
        Dimension of state space.

        This is inferred from the environment, not fixed at construction.
        """
        pass

    @property
    @abstractmethod
    def context_dim(self) -> int:
        """
        Dimension of raw context observation.

        Typically: state_dim + max_obstacles * (state_dim + 1)
        """
        pass

    def get_initial_state(self) -> torch.Tensor:
        """
        Get initial state for training.

        Default implementation returns zeros, can be overridden.
        """
        return torch.zeros(self.state_dim)

    @abstractmethod
    def get_intent(
        self, current_state: torch.Tensor, goal_motif: Optional["GoalMotif"] = None
    ) -> torch.Tensor:
        """
        Get intent/target for current state.

        Can use goal_motif for motif-based navigation, or fall back to position-based.

        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            goal_motif: Optional GoalMotif to navigate toward (if None, uses position-based fallback)

        Returns:
            intent: Intent/target direction (state_dim,)
        """
        raise NotImplementedError("Subclasses must implement get_intent()")

    def get_goal_motif(self) -> Optional[List["GoalMotif"]]:
        """
        Get current active goal motifs (if environment supports motif-based goals).

        Returns:
            List of GoalMotif objects, or None if not using motif-based goals
        """
        return None

    def check_goal_completion(
        self, current_state: torch.Tensor, goal_motif: Optional["GoalMotif"] = None
    ) -> bool:
        """
        Check if a goal motif has been reached (if using motif-based goals).

        Args:
            current_state: Current state x_n (state_dim,) or (1, state_dim)
            goal_motif: Optional GoalMotif to check against

        Returns:
            True if goal motif is matched, False otherwise
        """
        # Default implementation: not using motif-based goals
        return False

    @abstractmethod
    def apply(
        self,
        next_state: torch.Tensor,
        env_recorder: Optional["EnvironmentRecorder"] = None,
    ) -> int:
        """
        Apply next state and update environment internal state.

        This is the required way environments apply state changes.
        Called at environment refresh rate (e.g., each step in bind_port).

        Args:
            next_state: Next state to apply (state_dim,)
            env_recorder: Optional environment recorder to call

        Returns:
            goals_reached: Number of goals reached in this step (0 if not tracking goals)
        """
        pass
