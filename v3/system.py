"""
Abstract TAM system interface.

Implements the core TAM cycle:
1. A is in a situation s_n = (n, x_n)
2. A selects a port from those currently afforded
3. A binds the port
4. W responds with a context episode
5. A interprets the episode and evaluates binding success
6. A new situation arises; cycle repeats
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class TAMSystem(ABC):
    """
    Abstract TAM system interface.

    Implements the core TAM cycle:
    1. A is in a situation s_n = (n, x_n)
    2. A selects a port from those currently afforded
    3. A binds the port
    4. W responds with a context episode
    5. A interprets the episode and evaluates binding success
    6. A new situation arises; cycle repeats
    """

    @abstractmethod
    def infer_situation(
        self,
        context: torch.Tensor,
        previous_situation: torch.Tensor,
        remaining_distance_until_penalty: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Infer latent situation from context (InferenceEngine).

        This implements: x_n = Infer(context_n, x_{n-1})

        Args:
            context: Raw context c_n ∈ C (raw_ctx_dim,)
            previous_situation: Previous latent situation x_{n-1} (latent_dim,)
            remaining_distance_until_penalty: Optional remaining distance before penalty kicks in

        Returns:
            situation: Latent situation x_n (latent_dim,)
        """
        pass

    @abstractmethod
    def propose_ports(self, situation: torch.Tensor, intent: torch.Tensor) -> Tuple:
        """
        Propose affordance ports (Actor).

        This implements: Ports(s_n) = { p ∈ P | Φ_p(x_n, c_n^prior) ≠ ∅ }

        Args:
            situation: Latent situation x_n (latent_dim,)
            intent: Intent/target direction (state_dim,)

        Returns:
            ports: Tuple of (logits, tubes, sigmas, ...) where:
                - logits: (n_ports,) port selection logits
                - tubes: (n_ports, T, state_dim) proposed trajectories
                - sigmas: (n_ports, T, state_dim) precision vectors
                - ... (other outputs as needed)
        """
        pass

    @abstractmethod
    def evaluate_binding(
        self,
        proposed_tube: torch.Tensor,
        actual_episode: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate binding success (loss computation).

        This implements: Check if τ̂_n ∈ Φ_p(x_n, c_n^post)

        Args:
            proposed_tube: Proposed trajectory τ (T, state_dim)
            actual_episode: Actual context episode e_{n→n+1} (T', state_dim)
            sigma: Precision vector (T, state_dim)

        Returns:
            binding_loss: Loss measuring deviation from affordance cone
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of latent situation space."""
        pass
