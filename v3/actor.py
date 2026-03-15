import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from v3.goal_motif import GoalMotif


class Actor(nn.Module):
    """
    Fibration Actor: Next-step Actor with cross-attention.
    Key features:
    - Generates single next step per port
    - Per-dimension precision (sigma vector) for anisotropic affordance
    - Dimension-agnostic: infers state_dim from intent.shape
    - Cross-attention for transformer-based architecture
    """

    def __init__(self, latent_dim, n_ports=4, token_embed_dim=64, n_attention_heads=8):
        """
        Args:
            latent_dim: Dimension of latent situation space
            n_ports: Number of affordance ports to propose
            token_embed_dim: Embedding dimension from transformer
            n_attention_heads: Number of attention heads
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_ports = n_ports
        self.token_embed_dim = token_embed_dim

        # Cross-attention: Actor queries situation transformer keys
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_embed_dim, num_heads=n_attention_heads, batch_first=True
        )

        # Intent projection: maps intent to query space (per-dimension)
        self.intent_proj = nn.Linear(1, token_embed_dim)

        # Situation projection: maps latent situation to key/value space
        self.situation_proj = nn.Linear(latent_dim, token_embed_dim)

        # Output projection: attended features + intent -> port outputs
        # Intent is included so model can learn to use it for port selection
        self.port_head = nn.Sequential(
            nn.Linear(token_embed_dim + 1, 256),  # +1 for intent norm
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

        # Port output head: generates logits, next step, and sigmas
        # Output: logits (1) + next_step (max_supported_dim) + sigmas (max_supported_dim)
        max_supported_dim = 12
        self.port_output_head = nn.Linear(
            256, n_ports * (1 + max_supported_dim + max_supported_dim)
        )

        # Learnable intent bias: learns how much to bias next step toward goal
        # Takes intent distance and outputs a bias factor (0 = no bias, 1 = full bias)
        self.intent_bias_head = nn.Sequential(
            nn.Linear(1, 32),  # Intent distance -> hidden
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),  # -> bias factor
            nn.Sigmoid(),  # Constrain to [0, 1]
        )

        # Learnable weights for port selection
        # These control how much intent alignment and agency influence port selection
        self.intent_bias_weight = nn.Parameter(
            torch.tensor(2.0)
        )  # Weight for intent alignment in port selection
        self.agency_bias_weight = nn.Parameter(
            torch.tensor(1.0)
        )  # Weight for agency score in port selection

        # Universal Goal Matcher: learns to match current state to goal motifs
        # Input: aggregated TKN properties (64) + goal embedding (latent_dim) + current situation (latent_dim)
        # Output: match score (0-1)
        self.goal_matcher = nn.Sequential(
            nn.Linear(
                64 + latent_dim + latent_dim, 256
            ),  # TKN features + goal embedding + current situation
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Match score [0, 1]
        )

        # Universal Intent Generator: learns to generate navigation direction from goal motifs
        # Input: goal motif + current situation + current TKN properties
        # Output: intent vector (state_dim,)
        max_supported_state_dim = 12
        self.intent_generator = nn.Sequential(
            nn.Linear(
                latent_dim + latent_dim + 64, 256
            ),  # goal_embedding + current_situation + tkn_features
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(
                256, max_supported_state_dim
            ),  # Output intent (will be sliced to actual state_dim)
        )

    def forward(self, latent_situation, intent, situation_sequence=None):
        """
        Generate next step for each port with anisotropic sigma.

        Args:
            latent_situation: (B, latent_dim) latent situation from InferenceEngine
            intent: (B, state_dim) intent/target direction tensor - INFERS state_dim from this!
            situation_sequence: (B, state_dim, token_embed_dim) sequence from transformer
                             Required for cross-attention - used to generate attended features

        Returns:
            logits: (B, M) port selection logits
            mu_next: (B, M, state_dim) next step for each port
            sigma_next: (B, M, state_dim) per-dimension precision (sigma vector)
        """
        B = latent_situation.size(0)

        # INFER state_dim from intent shape
        state_dim = intent.shape[-1]

        # Cross-attention: Actor queries situation transformer keys
        # Project intent to queries (one per dimension)
        intent_queries = self.intent_proj(
            intent.unsqueeze(-1)
        )  # (B, state_dim, token_embed_dim)

        # Cross-attend: queries attend to situation sequence
        if situation_sequence is None:
            raise ValueError("situation_sequence is required for Actor cross-attention")

        # Debug: check shapes match
        if intent_queries.shape[1] != situation_sequence.shape[1]:
            raise ValueError(
                f"Shape mismatch: intent_queries has {intent_queries.shape[1]} dims, "
                f"situation_sequence has {situation_sequence.shape[1]} dims"
            )

        attended, attention_weights = self.cross_attention(
            query=intent_queries,  # (B, state_dim, token_embed_dim)
            key=situation_sequence,  # (B, state_dim, token_embed_dim)
            value=situation_sequence,  # (B, state_dim, token_embed_dim)
        )

        # Aggregate attended features (mean or weighted by attention)
        attended_feat = attended.mean(dim=1)  # (B, token_embed_dim)

        # Project latent situation to same space as attended features
        latent_proj = self.situation_proj(latent_situation)  # (B, token_embed_dim)

        # Combine attended features with latent situation projection
        # Use addition to merge information from both sources
        combined_feat = attended_feat + latent_proj  # (B, token_embed_dim)

        # Include intent norm in port selection
        intent_norm = torch.norm(intent, dim=-1, keepdim=True)  # (B, 1)
        port_input = torch.cat(
            [combined_feat, intent_norm], dim=-1
        )  # (B, token_embed_dim + 1)

        # Project through port head
        situation = self.port_head(port_input)  # (B, 256)

        # Generate port outputs
        raw_out = self.port_output_head(situation).view(B, self.n_ports, -1)
        logits = raw_out[:, :, 0]  # (B, M)

        # Extract next step and precision (sigma) vector
        max_supported_dim = 12
        mu_next_raw = raw_out[
            :, :, 1 : 1 + state_dim
        ]  # (B, M, state_dim) - slice to inferred dim
        sigma_raw = raw_out[
            :, :, 1 + max_supported_dim : 1 + max_supported_dim + state_dim
        ]  # (B, M, state_dim)

        # Apply intent bias to next step
        # Bias the proposed steps toward the goal direction
        intent_distance = torch.norm(intent, dim=-1, keepdim=True)  # (B, 1)
        intent_bias_factor = self.intent_bias_head(
            intent_distance
        )  # (B, 1) - learnable bias strength [0, 1]

        # Normalize intent to get direction, then scale by average step size for proportional bias
        intent_normalized = intent / (
            intent_distance + 1e-6
        )  # (B, state_dim) - normalized direction
        avg_step_size = torch.mean(
            torch.norm(mu_next_raw, dim=-1, keepdim=True), dim=1, keepdim=True
        )  # (B, 1, 1) - average step size across ports
        intent_bias = (
            intent_normalized.view(B, 1, state_dim) * avg_step_size
        )  # (B, 1, state_dim) - scaled direction

        # Apply bias: stronger bias when intent_bias_factor is high (closer to goal)
        # Broadcast: (B, 1, state_dim) * (B, 1, 1) -> (B, M, state_dim) via broadcasting
        mu_next = mu_next_raw + intent_bias * intent_bias_factor.unsqueeze(
            -1
        )  # (B, M, state_dim)

        # Per-dimension precision: one sigma per dimension (anisotropic affordance)
        # Use exp for stability (log-space precision)
        sigma_next = (
            torch.exp(sigma_raw) + 0.1
        )  # (B, M, state_dim) - per-dimension precision

        return logits, mu_next, sigma_next

    def compute_agency_reward(self, mu_next, sigma_next, step_size_factor=0.1):
        """
        Compute agency reward based on cone precision and horizon commitment.

        Args:
            mu_next: (state_dim,) relative next step
            sigma_next: (state_dim,) per-dimension precision (cone width)
            step_size_factor: Weight for horizon commitment (default 0.1)

        Returns:
            agency_reward: Scalar tensor - positive reward for agency
                - Cone precision reward: -mean(sigma^2) (narrower cones = higher reward)
                - Horizon commitment reward: +step_size_factor * ||mu_next|| (longer steps = higher reward)
        """
        # Cone precision reward: narrower cones (lower sigma) = higher reward
        cone_precision_reward = -torch.mean(sigma_next**2)

        # Horizon commitment reward: longer steps = higher reward
        step_size = torch.norm(mu_next)
        horizon_reward = step_size_factor * step_size

        # Combined agency reward
        agency_reward = cone_precision_reward + horizon_reward

        return agency_reward

    def compute_binding_loss(
        self, proposed_next, actual_next, sigma_next, current_pos, intent_target=None
    ):
        """
        Compute principled TAM loss based on binding failure for single next step.

        This is the core TAM loss: binding succeeds when actual_next stays within
        the affordance cone (sigma-weighted), fails otherwise.

        Args:
            proposed_next: (state_dim,) relative next step
            actual_next: (state_dim,) actual next position taken from world (in global coordinates)
            sigma_next: (state_dim,) per-dimension precision (cone width)
            current_pos: (state_dim,) or (1, state_dim) current position
            intent_target: Optional (state_dim,) target intent/goal position (kept for interface compatibility, not used)

        Returns:
            binding_loss: Scalar tensor - weighted deviation from affordance cone
            agency_reward: Scalar tensor - agency reward (precision + horizon commitment)
        """
        # Ensure current_pos is (state_dim,)
        if current_pos.dim() > 1:
            current_pos = current_pos.squeeze()

        device = proposed_next.device

        # Expected next position: proposed step in global coordinates
        expected_next = proposed_next + current_pos  # (state_dim,)

        # Per-dimension deviation from expected next step
        deviation_per_dim = actual_next - expected_next  # (state_dim,)

        # Weight by per-dimension precision (sigma)
        # Penalty is inversely proportional to sigma^2 (narrowness squared)
        # Higher sigma = wider cone = more tolerance = exponentially lower penalty
        # Lower sigma = narrower cone = less tolerance = exponentially higher penalty
        # This ensures that narrow commitments (small sigma) result in much higher
        # penalties for deviations, preventing agents from using narrow tubes to
        # offset binding failures through agency reward
        weighted_deviation = (deviation_per_dim**2) / (
            sigma_next**2 + 1e-6
        )  # (state_dim,) - penalty ∝ 1/sigma^2

        # Binding loss: sum of weighted deviations (binding failure measure)
        binding_loss = torch.sum(weighted_deviation)

        # Agency reward: computed using dedicated method
        agency_reward = self.compute_agency_reward(
            proposed_next, sigma_next, step_size_factor=0.1
        )

        return binding_loss, agency_reward

    def match_goal_motif(
        self,
        current_tkn_props: Dict[str, Any],
        goal_motif: GoalMotif,
        current_situation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Universal goal matcher: learns to detect goal-like states.

        Works with any GoalMotif form (structural properties, embeddings, etc.).

        Args:
            current_tkn_props: Dict with current TKN structural properties
                - Can include: hub_importance, in_degree, hub_count, etc.
                - Or 'dimension_structural_props' list format
            goal_motif: GoalMotif to match against
            current_situation: Optional (latent_dim,) or (1, latent_dim) current situation embedding

        Returns:
            match_score: Scalar tensor [0, 1] indicating how well current state matches goal
        """
        device = next(self.parameters()).device

        # Aggregate current TKN properties into feature vector
        tkn_features = []

        # Extract structural properties
        if isinstance(current_tkn_props, dict):
            # Handle dimension_structural_props format
            dim_props = current_tkn_props.get("dimension_structural_props", [])
            hub_importances = []
            in_degrees = []
            hub_counts = []

            if dim_props:
                for dim_prop_list in dim_props:
                    if isinstance(dim_prop_list, list):
                        for token_props in dim_prop_list:
                            if isinstance(token_props, dict):
                                hub_importances.append(
                                    token_props.get("hub_importance", 0.0)
                                )
                                in_degrees.append(token_props.get("in_degree", 0.0))
                                hub_counts.append(token_props.get("hub_count", 0.0))

            # Compute aggregate statistics
            avg_hub_importance = (
                sum(hub_importances) / max(len(hub_importances), 1.0)
                if hub_importances
                else 0.0
            )
            avg_in_degree = (
                sum(in_degrees) / max(len(in_degrees), 1.0) if in_degrees else 0.0
            )
            avg_hub_count = (
                sum(hub_counts) / max(len(hub_counts), 1.0) if hub_counts else 0.0
            )

            tkn_features = [avg_hub_importance, avg_in_degree, avg_hub_count]
        else:
            # Fallback: use zeros
            tkn_features = [0.0, 0.0, 0.0]

        # Pad/truncate to fixed size (64 features)
        while len(tkn_features) < 64:
            tkn_features.append(0.0)
        tkn_features = tkn_features[:64]
        tkn_feat_tensor = torch.tensor(
            tkn_features, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 64)

        # Encode goal motif to unified representation
        if goal_motif.has_learned_embedding():
            # Use learned embedding directly
            goal_embedding = goal_motif.learned_embedding.to(device)
            if goal_embedding.dim() == 1:
                goal_embedding = goal_embedding.unsqueeze(
                    0
                )  # (1, latent_dim or token_embed_dim)
            # Project to match input size if needed
            if goal_embedding.shape[1] != self.latent_dim:
                # Use a simple projection if dimensions don't match
                if not hasattr(self, "_goal_embedding_proj"):
                    self._goal_embedding_proj = nn.Linear(
                        goal_embedding.shape[1], self.latent_dim
                    ).to(device)
                goal_embedding = self._goal_embedding_proj(goal_embedding)
        elif goal_motif.has_structural_properties():
            # Encode structural properties to embedding
            struct_features = []
            for prop_name in ["hub_importance", "in_degree", "hub_count"]:
                if prop_name in goal_motif.structural_properties:
                    min_val, max_val = goal_motif.structural_properties[prop_name]
                    struct_features.extend(
                        [min_val, max_val, (min_val + max_val) / 2.0]
                    )
                else:
                    struct_features.extend([0.0, 1.0, 0.5])

            # Pad to latent_dim
            while len(struct_features) < self.latent_dim:
                struct_features.append(0.0)
            struct_features = struct_features[: self.latent_dim]
            goal_embedding = torch.tensor(
                struct_features, dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, latent_dim)
        else:
            # Fallback: use zeros
            goal_embedding = torch.zeros(1, self.latent_dim, device=device)

        # Prepare current situation
        if current_situation is not None:
            if current_situation.dim() == 1:
                current_situation = current_situation.unsqueeze(0)  # (1, latent_dim)
        else:
            current_situation = torch.zeros(1, self.latent_dim, device=device)

        # Combine inputs: tkn_features + goal_embedding + current_situation
        matcher_input = torch.cat(
            [tkn_feat_tensor, goal_embedding, current_situation], dim=-1
        )  # (1, 64 + latent_dim + latent_dim)

        # Compute match score
        match_score = self.goal_matcher(matcher_input)  # (1, 1)

        return match_score.squeeze(-1).squeeze(-1)  # Scalar

    def generate_intent_from_motif(
        self,
        goal_motif: GoalMotif,
        current_situation: torch.Tensor,
        current_tkn_props: Optional[Dict[str, Any]] = None,
        state_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Universal intent generator: learns to produce navigation direction from goal motifs.

        Works with any GoalMotif form and generates intent vector guiding toward goal-like states.

        Args:
            goal_motif: GoalMotif (any form) to navigate toward
            current_situation: (latent_dim,) or (1, latent_dim) current situation embedding
            current_tkn_props: Optional dict with current TKN properties
            state_dim: Optional state dimension (if None, inferred from goal_motif or defaults)

        Returns:
            intent: (state_dim,) intent vector guiding toward goal pattern
        """
        device = next(self.parameters()).device

        # Ensure current_situation has batch dimension
        if current_situation.dim() == 1:
            current_situation = current_situation.unsqueeze(0)  # (1, latent_dim)

        # Infer state_dim if not provided
        if state_dim is None:
            if goal_motif.position_hint is not None:
                state_dim = goal_motif.position_hint.shape[0]
            else:
                state_dim = 2  # Default fallback

        # Encode goal motif to unified representation
        if goal_motif.has_learned_embedding():
            goal_embedding = goal_motif.learned_embedding.to(device)
            if goal_embedding.dim() == 1:
                goal_embedding = goal_embedding.unsqueeze(
                    0
                )  # (1, latent_dim or token_embed_dim)
            # Project to latent_dim if needed
            if goal_embedding.shape[1] != self.latent_dim:
                if not hasattr(self, "_goal_embedding_proj_intent"):
                    self._goal_embedding_proj_intent = nn.Linear(
                        goal_embedding.shape[1], self.latent_dim
                    ).to(device)
                goal_embedding = self._goal_embedding_proj_intent(goal_embedding)
        elif goal_motif.has_structural_properties():
            # Encode structural properties to embedding
            struct_features = []
            for prop_name in ["hub_importance", "in_degree", "hub_count"]:
                if prop_name in goal_motif.structural_properties:
                    min_val, max_val = goal_motif.structural_properties[prop_name]
                    struct_features.extend(
                        [min_val, max_val, (min_val + max_val) / 2.0]
                    )
                else:
                    struct_features.extend([0.0, 1.0, 0.5])

            # Pad to latent_dim
            while len(struct_features) < self.latent_dim:
                struct_features.append(0.0)
            struct_features = struct_features[: self.latent_dim]
            goal_embedding = torch.tensor(
                struct_features, dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, latent_dim)
        else:
            # Fallback: use zeros
            goal_embedding = torch.zeros(1, self.latent_dim, device=device)

        # Aggregate current TKN properties
        tkn_features = []
        if current_tkn_props is not None and isinstance(current_tkn_props, dict):
            dim_props = current_tkn_props.get("dimension_structural_props", [])
            hub_importances = []
            in_degrees = []
            hub_counts = []

            if dim_props:
                for dim_prop_list in dim_props:
                    if isinstance(dim_prop_list, list):
                        for token_props in dim_prop_list:
                            if isinstance(token_props, dict):
                                hub_importances.append(
                                    token_props.get("hub_importance", 0.0)
                                )
                                in_degrees.append(token_props.get("in_degree", 0.0))
                                hub_counts.append(token_props.get("hub_count", 0.0))

            avg_hub_importance = (
                sum(hub_importances) / max(len(hub_importances), 1.0)
                if hub_importances
                else 0.0
            )
            avg_in_degree = (
                sum(in_degrees) / max(len(in_degrees), 1.0) if in_degrees else 0.0
            )
            avg_hub_count = (
                sum(hub_counts) / max(len(hub_counts), 1.0) if hub_counts else 0.0
            )

            tkn_features = [avg_hub_importance, avg_in_degree, avg_hub_count]
        else:
            tkn_features = [0.0, 0.0, 0.0]

        # Pad to 64 features
        while len(tkn_features) < 64:
            tkn_features.append(0.0)
        tkn_features = tkn_features[:64]
        tkn_feat_tensor = torch.tensor(
            tkn_features, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 64)

        # Combine inputs: goal_embedding + current_situation + tkn_features
        intent_input = torch.cat(
            [goal_embedding, current_situation, tkn_feat_tensor], dim=-1
        )  # (1, latent_dim + latent_dim + 64)

        # Generate intent
        intent_raw = self.intent_generator(intent_input)  # (1, max_supported_state_dim)

        # Slice to actual state_dim
        intent = intent_raw[:, :state_dim]  # (1, state_dim)

        return intent.squeeze(0)  # (state_dim,)

    def select_port(self, logits, mu_next, intent_target):
        """
        Select port based on closest next step to target, weighted by logits.

        Args:
            logits: (B, M) port logits
            mu_next: (B, M, state_dim) next step for each port
            intent_target: (B, state_dim) or (state_dim,) target intent vector

        Returns:
            selected_indices: (B,) indices of selected ports
        """
        if intent_target.dim() == 1:
            intent_target = intent_target.unsqueeze(0)  # (1, state_dim)
        dist = torch.norm(mu_next - intent_target.unsqueeze(1), dim=-1)  # (B, M)
        score = F.log_softmax(logits, dim=-1) - dist
        return torch.argmax(score, dim=-1)
