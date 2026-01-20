"""
Concrete implementation of TAMSystem.

Wraps InferenceEngine + Actor + TknProcessor into a unified system.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from v3.system import TAMSystem
from v3.inference import TransformerInferenceEngine
from v3.actor import Actor
from v3.tokenizer import UnifiedTknProcessor


class TAMSystemWrapper(TAMSystem):
    """
    Concrete implementation of TAMSystem.
    
    Wraps InferenceEngine + Actor + TknProcessor into a unified system.
    """
    
    def __init__(self, inference_engine: TransformerInferenceEngine, 
                 actor: Actor, tkn_processor: UnifiedTknProcessor,
                 obstacles: Optional[list] = None,
                 max_observed_obstacles: int = 10,
                 max_observed_goals: int = 10):
        """
        Initialize TAMSystemWrapper.
        
        Args:
            inference_engine: TransformerInferenceEngine for situation inference
            actor: Actor for port proposal
            tkn_processor: UnifiedTknProcessor for tokenization
            obstacles: List of obstacles (for tokenization) - can be None if not needed
            max_observed_obstacles: Maximum obstacles in observation
            max_observed_goals: Maximum goals in observation
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
    
    def infer_situation(self, context: torch.Tensor, 
                       previous_situation: torch.Tensor,
                       current_state: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            raise ValueError("current_state is required for tokenization in infer_situation()")
        
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
            max_observed_goals=self.max_observed_goals
        )
        
        # Extract relative goal
        rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, state_dim)
        
        # Extract dimension tokens and traits
        dimension_tokens = tkn_output["dimension_tokens"]  # List of variable-length tensors
        dimension_traits = tkn_output["dimension_traits"]  # List of (1, 2) tensors
        dimension_structural_props = tkn_output.get("dimension_structural_props", None)  # List of lists of dicts
        
        # Call transformer inference engine with memory context
        x_n, situation_sequence, new_memory_context = self.inference_engine(
            dimension_tokens, 
            dimension_traits, 
            rel_goal_tensor, 
            h_state, 
            memory_context=self.memory_context,
            dimension_structural_props=dimension_structural_props
        )
        
        # Update internal state
        # Detach to prevent graph accumulation across iterations
        self.h_state = x_n.detach()
        self.memory_context = new_memory_context.detach() if new_memory_context is not None else None
        
        # Store situation_sequence for use in propose_ports (detach to prevent graph reuse)
        self._last_situation_sequence = situation_sequence.detach() if situation_sequence is not None else None
        
        return x_n
    
    def propose_ports(self, situation: torch.Tensor, intent: torch.Tensor) -> Tuple:
        """
        Propose affordance ports.
        
        Args:
            situation: Latent situation x_n (latent_dim,) or (1, latent_dim)
            intent: Intent/target direction (state_dim,) or (1, state_dim)
            
        Returns:
            ports: Tuple of (logits, tubes, sigmas, knot_mask, basis_weights)
        """
        # Ensure batch dimension
        if situation.dim() == 1:
            situation = situation.unsqueeze(0)  # (1, latent_dim)
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)  # (1, state_dim)
        
        # Get situation_sequence from last inference (stored in infer_situation)
        situation_sequence = getattr(self, '_last_situation_sequence', None)
        if situation_sequence is None:
            raise ValueError("situation_sequence not available - infer_situation must be called before propose_ports")
        
        # Get markov_lattice (for structural properties, not spatial queries)
        markov_lattice = self.tkn_processor.lattice if hasattr(self.tkn_processor, 'lattice') else None
        
        # No explicit look-ahead - transformer learns through pattern recognition
        current_pos_np = None
        
        # Call actor
        try:
            logits, mu_t, sigma_t, knot_mask, basis_weights = self.actor(
                situation, 
                intent,
                situation_sequence=situation_sequence,
            )
        except Exception as e:
            print(f"Error in actor.forward(): {e}")
            print(f"  situation shape: {situation.shape}")
            print(f"  intent shape: {intent.shape}")
            print(f"  situation_sequence shape: {situation_sequence.shape if situation_sequence is not None else None}")
            import traceback
            traceback.print_exc()
            raise
        
        return (logits, mu_t, sigma_t, knot_mask, basis_weights)
    
    def evaluate_binding(self, proposed_tube: torch.Tensor, 
                        actual_episode: torch.Tensor, 
                        sigma: torch.Tensor,
                        current_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate binding success.
        
        Args:
            proposed_tube: Proposed trajectory τ (T, state_dim)
            actual_episode: Actual context episode e_{n→n+1} (T', state_dim)
            sigma: Precision vector (T, state_dim)
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
        device = proposed_tube.device
        if actual_episode.device != device:
            actual_episode = actual_episode.to(device)
        if sigma.device != device:
            sigma = sigma.to(device)
        if current_state_flat.device != device:
            current_state_flat = current_state_flat.to(device)
        
        # Compute binding loss using Actor's method
        # Cache agency_cost for potential reuse
        # Note: intent_target not available here, so efficiency reward will be segment-length only
        result = self.actor.compute_binding_loss(
            proposed_tube, 
            actual_episode, 
            sigma, 
            current_state_flat,
            knot_mask=None,  # Can be passed if available
            intent_target=None  # Not available in evaluate_binding interface
        )
        
        # Handle both old (2-tuple) and new (3-tuple) return signatures
        if len(result) == 2:
            binding_loss, agency_cost = result
        elif len(result) == 3:
            binding_loss, agency_cost, _ = result  # Ignore efficiency_reward in evaluate_binding
        else:
            raise ValueError(f"Unexpected return value from compute_binding_loss: {len(result)} values")
        
        # Cache agency_cost for potential reuse (to avoid recomputing)
        self._last_agency_cost = agency_cost
        
        # Return just binding_loss (agency_cost can be accessed separately if needed)
        return binding_loss
    
    def get_last_agency_cost(self) -> Optional[torch.Tensor]:
        """Get the agency_cost from the last evaluate_binding call."""
        return getattr(self, '_last_agency_cost', None)
    
    @property
    def latent_dim(self) -> int:
        """Dimension of latent situation space."""
        return self.inference_engine.latent_dim
