"""
Goal Motif: Generic pattern-based goal representation.

Goals are represented as patterns in TKN space, enabling:
- Spatial environments (patterns in observation space)
- Discrete environments (graph patterns, token sequences)
- Abstract goals (learned embeddings)
- Natural language goals (via learned encodings)
"""

import torch
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class GoalMotif:
    """
    Generic goal representation as TKN pattern.

    Supports multiple goal description forms:
    - Structural properties (hub_importance, in_degree, etc.)
    - Token patterns (sequences of tokens)
    - Learned embeddings (for abstract/natural language goals)
    - Natural language strings (future: encoded to embeddings)
    """

    # Pattern-based description (works for any environment)
    structural_properties: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Target ranges for TKN properties: {"hub_importance": (min, max), "in_degree": (min, max), ...}

    token_patterns: Optional[List[List[int]]] = None
    # Token sequences to match (optional)

    # Learned embedding (for natural language / abstract goals)
    learned_embedding: Optional[torch.Tensor] = None
    # (latent_dim,) or (token_embed_dim,) - learned goal representation

    # Natural language encoding (future extension)
    natural_language: Optional[str] = None
    # "Find a place with high connectivity" - will be encoded to embedding

    # Flexible matching
    match_tolerance: float = 0.1
    # How strict is pattern matching (0.0 = exact, 1.0 = very loose)

    # Optional: Spatial position hint (for backward compatibility / learning)
    position_hint: Optional[torch.Tensor] = None
    # (state_dim,) - optional spatial hint, not required for matching

    def __post_init__(self):
        """Validate and normalize goal motif."""
        # Ensure learned_embedding is a tensor if provided
        if self.learned_embedding is not None and not isinstance(
            self.learned_embedding, torch.Tensor
        ):
            self.learned_embedding = torch.tensor(
                self.learned_embedding, dtype=torch.float32
            )

        # Ensure position_hint is a tensor if provided
        if self.position_hint is not None and not isinstance(
            self.position_hint, torch.Tensor
        ):
            self.position_hint = torch.tensor(self.position_hint, dtype=torch.float32)

    def has_structural_properties(self) -> bool:
        """Check if motif has structural property constraints."""
        return len(self.structural_properties) > 0

    def has_learned_embedding(self) -> bool:
        """Check if motif has learned embedding."""
        return self.learned_embedding is not None

    def has_token_patterns(self) -> bool:
        """Check if motif has token patterns."""
        return self.token_patterns is not None and len(self.token_patterns) > 0

    def get_primary_form(self) -> str:
        """Get primary form of goal description."""
        if self.has_learned_embedding():
            return "embedding"
        elif self.has_structural_properties():
            return "structural"
        elif self.has_token_patterns():
            return "token_patterns"
        elif self.natural_language is not None:
            return "natural_language"
        else:
            return "empty"


def extract_motif_from_tkn_output(tkn_output: Dict[str, Any]) -> GoalMotif:
    """
    Extract goal motif from TKN output (for learning goals from examples).

    Args:
        tkn_output: TKN output dict with dimension_structural_props, dimension_tokens, etc.

    Returns:
        GoalMotif extracted from current TKN patterns
    """
    dimension_structural_props = tkn_output.get("dimension_structural_props", None)
    dimension_tokens = tkn_output.get("dimension_tokens", None)

    if dimension_structural_props is None or len(dimension_structural_props) == 0:
        return GoalMotif()

    # Aggregate structural properties across dimensions
    structural_properties = {}
    hub_importances = []
    hub_counts = []
    in_degrees = []

    for dim_props in dimension_structural_props:
        if dim_props is None or len(dim_props) == 0:
            continue

        for token_props in dim_props:
            if isinstance(token_props, dict):
                hub_importance = token_props.get("hub_importance", 0.0)
                hub_count = token_props.get("hub_count", 0.0)
                in_degree = token_props.get("in_degree", 0.0)

                if hub_importance > 0:
                    hub_importances.append(hub_importance)
                if hub_count > 0:
                    hub_counts.append(hub_count)
                if in_degree > 0:
                    in_degrees.append(in_degree)

    # Compute target ranges (mean ± std for flexibility)
    if hub_importances:
        mean_imp = sum(hub_importances) / len(hub_importances)
        std_imp = (
            sum((x - mean_imp) ** 2 for x in hub_importances) / len(hub_importances)
        ) ** 0.5
        structural_properties["hub_importance"] = (
            max(0.0, mean_imp - std_imp),
            min(1.0, mean_imp + std_imp),
        )

    if in_degrees:
        mean_deg = sum(in_degrees) / len(in_degrees)
        std_deg = (
            sum((x - mean_deg) ** 2 for x in in_degrees) / len(in_degrees)
        ) ** 0.5
        structural_properties["in_degree"] = (
            max(0.0, mean_deg - std_deg),
            mean_deg + std_deg,
        )

    if hub_counts:
        mean_count = sum(hub_counts) / len(hub_counts)
        std_count = (
            sum((x - mean_count) ** 2 for x in hub_counts) / len(hub_counts)
        ) ** 0.5
        structural_properties["hub_count"] = (
            max(0.0, mean_count - std_count),
            mean_count + std_count,
        )

    # Extract token patterns if available
    token_patterns = None
    if dimension_tokens is not None:
        token_patterns = []
        for dim_tokens in dimension_tokens:
            if isinstance(dim_tokens, torch.Tensor):
                token_patterns.append(dim_tokens.tolist())
            elif isinstance(dim_tokens, (list, tuple)):
                token_patterns.append(list(dim_tokens))

    return GoalMotif(
        structural_properties=structural_properties,
        token_patterns=token_patterns if token_patterns else None,
        match_tolerance=0.2,  # Default tolerance
    )


def create_motif_from_position(
    position: torch.Tensor,
    tkn_processor,
    context: torch.Tensor,
    obstacles: Optional[List] = None,
) -> GoalMotif:
    """
    Create goal motif from spatial position (for backward compatibility / learning).

    This extracts the TKN pattern at a given position to create a motif.

    Args:
        position: (state_dim,) goal position
        tkn_processor: UnifiedTknProcessor instance
        context: Raw context observation at that position
        obstacles: Optional obstacles list

    Returns:
        GoalMotif extracted from position's TKN pattern
    """
    # Process observation at goal position
    tkn_output = tkn_processor.process_observation(
        position,
        context,
        obstacles if obstacles is not None else [],
        max_observed_obstacles=None,  # No limit - observe all
        max_observed_goals=None,  # No limit - observe all
    )

    # Extract motif from TKN output
    motif = extract_motif_from_tkn_output(tkn_output)

    # Store position as hint (for learning, but not required for matching)
    motif.position_hint = (
        position.clone()
        if isinstance(position, torch.Tensor)
        else torch.tensor(position)
    )

    return motif
