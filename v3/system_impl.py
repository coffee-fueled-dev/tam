"""
Concrete implementation of TAMSystem.

Wraps InferenceEngine + Actor + TknProcessor into a unified system.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from v3.system import TAMSystem
from v3.inference import TransformerInferenceEngine
from v3.actor import Actor
from v3.tokenizer import UnifiedTknProcessor
from v3.goal_motif import GoalMotif


class TAMSystemWrapper(TAMSystem):
    """
    Concrete implementation of TAMSystem.

    Wraps InferenceEngine + Actor + TknProcessor into a unified system.
    """

    def __init__(
        self,
        inference_engine: TransformerInferenceEngine,
        actor: Actor,
        tkn_processor: UnifiedTknProcessor,
        obstacles: Optional[list] = None,
        max_observed_obstacles: Optional[int] = None,  # None = no limit, observe all
        max_observed_goals: Optional[int] = None,  # None = no limit, observe all
    ):
        """
        Initialize TAMSystemWrapper.

        Args:
            inference_engine: TransformerInferenceEngine for situation inference
            actor: Actor for port proposal
            tkn_processor: UnifiedTknProcessor for tokenization
            obstacles: List of obstacles (for tokenization) - can be None if not needed
            max_observed_obstacles: Maximum obstacles in observation (None = no limit, observe all)
            max_observed_goals: Maximum goals in observation (None = no limit, observe all)
        """
        self.inference_engine = inference_engine
        self.actor = actor
        self.tkn_processor = tkn_processor
        self.obstacles = obstacles
        self.max_observed_obstacles = max_observed_obstacles
        self.max_observed_goals = max_observed_goals

        # Internal state management
        self.h_state = None  # Hidden state (situation)
        self.previous_velocity = None  # For G1 continuity (kept for compatibility)
        self.memory_context = None  # Memory context for temporal learning

    def reset(self):
        """Reset system state."""
        latent_dim = self.inference_engine.latent_dim
        self.h_state = torch.zeros(1, latent_dim)
        self.previous_velocity = None
        self.memory_context = None
        if self.tkn_processor:
            self.tkn_processor.reset_episode()

    def infer_situation(
        self,
        context: torch.Tensor,
        previous_situation: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
        remaining_distance_until_penalty: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Infer situation from context.

        This handles tokenization internally as part of the inference pipeline.

        Args:
            context: Raw context c_n ∈ C (raw_ctx_dim,) or (1, raw_ctx_dim)
            previous_situation: Previous latent situation x_{n-1} (latent_dim,) or (1, latent_dim)
            current_state: Current state x_n (state_dim,) or (1, state_dim) - required for tokenization

        Returns:
            situation: Latent situation x_n (latent_dim,) or (1, latent_dim)
        """
        # Ensure context has batch dimension
        if context.dim() == 1:
            context = context.unsqueeze(0)  # (1, raw_ctx_dim)

        # Ensure previous_situation has batch dimension
        if previous_situation.dim() == 1:
            previous_situation = previous_situation.unsqueeze(0)  # (1, latent_dim)

        # Use stored h_state if previous_situation is not provided or is zeros
        if self.h_state is not None:
            h_state = self.h_state
        else:
            h_state = previous_situation

        # Ensure current_state is provided (required for tokenization)
        if current_state is None:
            raise ValueError(
                "current_state is required for tokenization in infer_situation()"
            )

        # Ensure current_state has correct shape
        if current_state.dim() > 1:
            current_state_flat = current_state.squeeze(0)
        else:
            current_state_flat = current_state

        # Process through tkn (tokenization)
        # UnifiedTknProcessor.process_observation expects:
        # - current_pos: (state_dim,) tensor
        # - raw_obs: (raw_ctx_dim,) tensor
        # - obstacles: list of (position, radius) tuples
        tkn_output = self.tkn_processor.process_observation(
            current_state_flat,
            context.squeeze(0),  # Remove batch dim for tkn processor
            self.obstacles if self.obstacles is not None else [],
            max_observed_obstacles=self.max_observed_obstacles,
            max_observed_goals=self.max_observed_goals,
        )

        # Add remaining_distance_until_penalty to TKN output if provided
        if remaining_distance_until_penalty is not None:
            tkn_output["remaining_distance_until_penalty"] = torch.tensor(
                remaining_distance_until_penalty, dtype=torch.float32
            )

        # Extract relative goal
        rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, state_dim)

        # Extract dimension tokens and traits
        dimension_tokens = tkn_output[
            "dimension_tokens"
        ]  # List of variable-length tensors
        dimension_traits = tkn_output["dimension_traits"]  # List of (1, 2) tensors
        dimension_structural_props = tkn_output.get(
            "dimension_structural_props", None
        )  # List of lists of dicts

        # Call transformer inference engine with memory context
        x_n, situation_sequence, new_memory_context = self.inference_engine(
            dimension_tokens,
            dimension_traits,
            rel_goal_tensor,
            h_state,
            memory_context=self.memory_context,
            dimension_structural_props=dimension_structural_props,
        )

        # Update internal state
        # Detach to prevent graph accumulation across iterations
        self.h_state = x_n.detach()
        self.memory_context = (
            new_memory_context.detach() if new_memory_context is not None else None
        )

        # Store situation_sequence for use in propose_ports (detach to prevent graph reuse)
        self._last_situation_sequence = (
            situation_sequence.detach() if situation_sequence is not None else None
        )

        # Store tkn_output for complexity computation (detach to prevent graph accumulation)
        self._last_tkn_output = {
            "dimension_structural_props": dimension_structural_props,
            "dimension_tokens": dimension_tokens,
            "dimension_traits": dimension_traits,
        }

        # Store remaining_distance_until_penalty for actor access
        self._last_remaining_distance = tkn_output.get(
            "remaining_distance_until_penalty", None
        )

        # Store tkn_output for goal matching (includes structural properties)
        self._last_tkn_props = {
            "dimension_structural_props": dimension_structural_props,
            "dimension_tokens": dimension_tokens,
            "dimension_traits": dimension_traits,
        }

        return x_n

    def propose_ports(self, situation: torch.Tensor, intent: torch.Tensor) -> Tuple:
        """
        Propose affordance ports.

        Args:
            situation: Latent situation x_n (latent_dim,) or (1, latent_dim)
            intent: Intent/target direction (state_dim,) or (1, state_dim)

        Returns:
            ports: Tuple of (logits, mu_next, sigma_next)
        """
        # Ensure batch dimension
        if situation.dim() == 1:
            situation = situation.unsqueeze(0)  # (1, latent_dim)
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)  # (1, state_dim)

        # Get situation_sequence from last inference (stored in infer_situation)
        situation_sequence = getattr(self, "_last_situation_sequence", None)
        if situation_sequence is None:
            raise ValueError(
                "situation_sequence not available - infer_situation must be called before propose_ports"
            )

        # Call actor
        try:
            logits, mu_next, sigma_next = self.actor(
                situation,
                intent,
                situation_sequence=situation_sequence,
            )
        except Exception as e:
            print(f"Error in actor.forward(): {e}")
            print(f"  situation shape: {situation.shape}")
            print(f"  intent shape: {intent.shape}")
            print(
                f"  situation_sequence shape: {situation_sequence.shape if situation_sequence is not None else None}"
            )
            import traceback

            traceback.print_exc()
            raise

        return (logits, mu_next, sigma_next)

    def evaluate_binding(
        self,
        proposed_next: torch.Tensor,
        actual_episode: torch.Tensor,
        sigma: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate binding success.

        Args:
            proposed_next: Proposed next step (state_dim,)
            actual_episode: Actual context episode e_{n→n+1} (T', state_dim)
            sigma: Precision vector (state_dim,)
            current_state: Current state x_n (state_dim,) - required for binding loss computation

        Returns:
            binding_loss: Loss measuring deviation from affordance cone
        """
        # Use Actor's compute_binding_loss method
        if current_state is None:
            # Try to infer from actual_episode (use first point)
            if len(actual_episode) > 0:
                current_state = actual_episode[0]
            else:
                raise ValueError("current_state is required for evaluate_binding()")

        # Ensure current_state is correct shape
        if current_state.dim() > 1:
            current_state_flat = current_state.squeeze()
        else:
            current_state_flat = current_state

        # Ensure all tensors are on the same device
        device = proposed_next.device
        if actual_episode.device != device:
            actual_episode = actual_episode.to(device)
        if sigma.device != device:
            sigma = sigma.to(device)
        if current_state_flat.device != device:
            current_state_flat = current_state_flat.to(device)

        # For single-step generation, use the first actual step (or current_state if episode is empty)
        if len(actual_episode) > 0:
            actual_next = actual_episode[0]  # First step in episode
        else:
            actual_next = current_state_flat  # No movement

        # Compute binding loss using Actor's method
        binding_loss, agency_reward = self.actor.compute_binding_loss(
            proposed_next,
            actual_next,
            sigma,
            current_state_flat,
            intent_target=None,  # Not used, kept for interface compatibility
        )

        # Compute situation complexity and scale agency reward
        complexity = self.get_situation_complexity()
        scaled_agency_reward = agency_reward * complexity

        # Cache for potential reuse (to avoid recomputing)
        self._last_agency_reward = scaled_agency_reward
        self._last_complexity = complexity

        # Return just binding_loss (agency_reward can be accessed separately if needed)
        return binding_loss

    def get_last_agency_cost(self) -> Optional[torch.Tensor]:
        """Get the agency_cost from the last evaluate_binding call (deprecated - use get_last_agency_reward)."""
        return getattr(self, "_last_agency_cost", None)

    def get_last_agency_reward(self) -> Optional[torch.Tensor]:
        """Get the complexity-scaled agency_reward from the last evaluate_binding call."""
        return getattr(self, "_last_agency_reward", None)

    def compute_situation_complexity(
        self, tkn_output: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Compute situation complexity from tkn stream using information and graph theoretic measures.

        Args:
            tkn_output: Optional tkn output dict. If None, uses stored _last_tkn_output.

        Returns:
            complexity: Scalar tensor representing situation complexity (normalized to [0.1, 10.0])
        """
        # Use provided tkn_output or fall back to stored one
        if tkn_output is None:
            tkn_output = getattr(self, "_last_tkn_output", None)

        if tkn_output is None:
            # Fallback: return neutral complexity (1.0)
            return torch.tensor(1.0)

        dimension_structural_props = tkn_output.get("dimension_structural_props", None)
        dimension_tokens = tkn_output.get("dimension_tokens", None)

        if dimension_structural_props is None or len(dimension_structural_props) == 0:
            return torch.tensor(1.0)

        # Collect measures from structural properties
        total_surprise = 0.0
        hub_importances = []
        hub_counts = []
        in_degrees = []
        unique_tokens = set()

        for dim_props in dimension_structural_props:
            if dim_props is None or len(dim_props) == 0:
                continue

            for token_props in dim_props:
                if isinstance(token_props, dict):
                    # Extract structural properties
                    hub_importance = token_props.get("hub_importance", 0.0)
                    hub_count = token_props.get("hub_count", 0.0)
                    in_degree = token_props.get("in_degree", 0.0)
                    is_hub = token_props.get("is_hub", 0.0)

                    # Surprise is implicit in novelty (low hub_count = high surprise)
                    # Use inverse hub_count as surprise proxy (normalized)
                    if hub_count > 0:
                        surprise_proxy = 1.0 / (1.0 + hub_count)
                    else:
                        surprise_proxy = 1.0  # Maximum surprise for novel patterns

                    total_surprise += surprise_proxy

                    if is_hub > 0.5:  # Only count hubs
                        hub_importances.append(hub_importance)
                        hub_counts.append(hub_count)
                        in_degrees.append(in_degree)

        # Collect unique tokens for diversity measure
        if dimension_tokens is not None:
            for dim_tokens in dimension_tokens:
                if isinstance(dim_tokens, torch.Tensor):
                    unique_tokens.update(dim_tokens.tolist())
                elif isinstance(dim_tokens, (list, tuple)):
                    unique_tokens.update(dim_tokens)

        # Compute complexity components
        num_dimensions = len(dimension_structural_props)
        if num_dimensions == 0:
            return torch.tensor(1.0)

        # Information content: normalized total surprise
        info_content = total_surprise / max(
            num_dimensions, 1.0
        )  # Normalize by dimensions

        # Graph connectivity: average hub importance and in-degree
        avg_hub_importance = (
            sum(hub_importances) / max(len(hub_importances), 1.0)
            if hub_importances
            else 0.0
        )
        avg_in_degree = (
            sum(in_degrees) / max(len(in_degrees), 1.0) if in_degrees else 0.0
        )

        # Pattern diversity: ratio of unique tokens to total tokens
        total_tokens = (
            sum(
                len(dim_tokens) if isinstance(dim_tokens, (list, torch.Tensor)) else 1
                for dim_tokens in dimension_tokens
            )
            if dimension_tokens
            else 1
        )
        unique_tokens_ratio = len(unique_tokens) / max(total_tokens, 1.0)

        # Graph density: if we have lattice access, compute transition density
        # For now, use hub count as proxy for graph density
        graph_density_proxy = len(hub_importances) / max(num_dimensions, 1.0)

        # Combine measures with weights
        info_content_weight = 0.3
        graph_weight = 0.3
        diversity_weight = 0.2
        connectivity_weight = 0.2

        # Normalize in_degree by max value
        max_in_degree = max(in_degrees) if in_degrees else 1.0
        normalized_in_degree = avg_in_degree / max(max_in_degree, 1.0)

        complexity = (
            info_content_weight * info_content
            + graph_weight * (avg_hub_importance + graph_density_proxy)
            + diversity_weight * unique_tokens_ratio
            + connectivity_weight * normalized_in_degree
        )

        # Normalize to [0.1, 10.0] range using sigmoid-like transformation
        # Map [0, inf] -> [0.1, 10.0]
        complexity_normalized = 0.1 + 9.9 * torch.sigmoid(torch.tensor(complexity))

        return complexity_normalized

    def get_situation_complexity(self) -> torch.Tensor:
        """
        Get situation complexity from last tkn_output.

        Returns:
            complexity: Scalar tensor (fallback to 1.0 if unavailable)
        """
        return self.compute_situation_complexity()

    def match_goal_motifs(
        self, goal_motifs: Optional[List[GoalMotif]] = None
    ) -> Optional[List[torch.Tensor]]:
        """
        Match current state against goal motifs.

        Args:
            goal_motifs: Optional list of GoalMotif objects to match against

        Returns:
            List of match scores (one per goal motif), or None if no motifs provided
        """
        if goal_motifs is None or len(goal_motifs) == 0:
            return None

        # Get current situation
        current_situation = (
            self.h_state
            if self.h_state is not None
            else torch.zeros(1, self.latent_dim)
        )
        if current_situation.dim() == 1:
            current_situation = current_situation.unsqueeze(0)

        # Get current TKN properties
        current_tkn_props = getattr(self, "_last_tkn_props", None)
        if current_tkn_props is None:
            current_tkn_props = {}

        # Match each goal motif
        match_scores = []
        for goal_motif in goal_motifs:
            match_score = self.actor.match_goal_motif(
                current_tkn_props,
                goal_motif,
                current_situation.squeeze(0)
                if current_situation.shape[0] == 1
                else current_situation,
            )
            match_scores.append(match_score)

        return match_scores

    def generate_intent_from_motifs(
        self,
        goal_motifs: Optional[List[GoalMotif]] = None,
        state_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Generate intent vector from goal motifs.

        Args:
            goal_motifs: Optional list of GoalMotif objects (uses first one if multiple)
            state_dim: Optional state dimension (if None, inferred from goal_motif)

        Returns:
            Intent vector (state_dim,), or None if no motifs provided
        """
        if goal_motifs is None or len(goal_motifs) == 0:
            return None

        # Use first goal motif (can be extended to combine multiple)
        goal_motif = goal_motifs[0]

        # Get current situation
        current_situation = (
            self.h_state
            if self.h_state is not None
            else torch.zeros(1, self.latent_dim)
        )
        if current_situation.dim() == 1:
            current_situation = current_situation.unsqueeze(0)

        # Get current TKN properties
        current_tkn_props = getattr(self, "_last_tkn_props", None)
        if current_tkn_props is None:
            current_tkn_props = {}

        # Generate intent
        intent = self.actor.generate_intent_from_motif(
            goal_motif,
            current_situation.squeeze(0)
            if current_situation.shape[0] == 1
            else current_situation,
            current_tkn_props,
            state_dim=state_dim,
        )

        return intent

    @property
    def latent_dim(self) -> int:
        """Dimension of latent situation space."""
        return self.inference_engine.latent_dim
