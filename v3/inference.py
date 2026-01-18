import torch
import torch.nn as nn

class TransformerInferenceEngine(nn.Module):
    """
    Transformer-based inference engine that processes dimensions as sequences.
    
    Processes dimensions as variable-length sequences with attention, eliminating
    the need for padding. Each dimension is treated as a "geometric motif" in
    a sequence that the transformer attends to dynamically.
    
    Input: Sequence of dimension-tokens (variable length based on state_dim)
    Output: Latent situation (fixed size) + dimension sequence (for Actor attention)
    """
    def __init__(self, vocab_size=65536, token_embed_dim=64, 
                 latent_dim=128, n_layers=3, n_heads=8, 
                 max_dimension_embed=32, dropout=0.1, memory_window=10):
        """
        Args:
            vocab_size: Token vocabulary size
            token_embed_dim: Embedding dimension per token
            latent_dim: Dimension of latent situation space
            n_layers: Number of transformer encoder layers
            n_heads: Number of attention heads
            max_dimension_embed: Maximum dimension index for positional embedding
            dropout: Dropout rate for transformer layers
            memory_window: Number of recent situations to remember for temporal context
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.memory_window = memory_window
        
        # Token embedding: pattern IDs -> dense vectors
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)
        
        # Dimension positional embedding: which dimension is this?
        # Allows model to distinguish delta_0 from delta_1, etc.
        self.dim_embedding = nn.Embedding(max_dimension_embed, token_embed_dim)
        
        # Trait embedding: hub-ness + surprise -> features
        self.trait_proj = nn.Linear(2, token_embed_dim)
        
        # Intent embedding: rel_goal components -> features (per-dimension)
        self.intent_proj = nn.Linear(1, token_embed_dim)
        
        # Transformer encoder: processes sequence of dimension-tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_embed_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection: sequence -> latent situation
        self.output_proj = nn.Sequential(
            nn.Linear(token_embed_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # CLS token for aggregation (learned)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_embed_dim))
    
    def forward(self, dimension_tokens, dimension_traits, rel_goal, h_prev=None, memory_context=None):
        """
        Process dimensions as a sequence.
        
        Args:
            dimension_tokens: List of token IDs per dimension
                           Each element: (B,) tensor of token IDs for that dimension
                           OR (B, num_heads_per_dim) if multiple tokens per dimension
            dimension_traits: List of traits per dimension
                            Each element: (B, 2) tensor of [hub_count, surprise]
            rel_goal: (B, state_dim) relative goal vector
            h_prev: Optional, kept for interface compatibility (not used in transformer)
            memory_context: Optional (B, memory_window, token_embed_dim) sliding window of recent situations
            
        Returns:
            situation: (B, latent_dim) latent situation
            situation_sequence: (B, state_dim, token_embed_dim) sequence for Actor attention
            new_memory_context: (B, memory_window, token_embed_dim) updated memory context
        """
        B = rel_goal.shape[0]
        state_dim = rel_goal.shape[-1]
        device = rel_goal.device
        
        # Build sequence: one token per dimension
        # Each dimension gets: [token_embed + dim_embed + trait_embed + intent_embed]
        sequence_tokens = []
        
        for dim_idx in range(state_dim):
            # Get token IDs for this dimension
            if isinstance(dimension_tokens, list):
                # UnifiedTknProcessor returns list of (1,) tensors
                dim_token_tensor = dimension_tokens[dim_idx]  # (1,) from UnifiedTknProcessor
                # Ensure it's 1D, then expand to batch size
                if dim_token_tensor.dim() == 0:
                    dim_token_tensor = dim_token_tensor.unsqueeze(0)  # (1,)
                # Expand to batch size: (1,) -> (B,)
                if dim_token_tensor.shape[0] == 1:
                    dim_token_ids = dim_token_tensor.expand(B)  # (B,)
                else:
                    dim_token_ids = dim_token_tensor  # Already (B,)
                # Add dimension for embedding: (B,) -> (B, 1)
                dim_token_ids = dim_token_ids.unsqueeze(-1)  # (B, 1)
            else:
                # Fallback: if passed as tensor, assume it's per-dimension
                dim_token_ids = dimension_tokens[:, dim_idx] if dimension_tokens.dim() > 1 else dimension_tokens
                if dim_token_ids.dim() == 1:
                    dim_token_ids = dim_token_ids.unsqueeze(-1)  # (B, 1)
            
            # Token embedding (from tkn processor)
            # dim_token_ids should be (B, 1) now
            token_embeds = self.token_embedding(dim_token_ids)  # (B, 1, token_embed_dim)
            token_embed = token_embeds.squeeze(1)  # (B, token_embed_dim)
            
            # Dimension positional embedding
            dim_pos = self.dim_embedding(torch.tensor(dim_idx, device=device))  # (token_embed_dim,)
            dim_pos = dim_pos.unsqueeze(0).expand(B, -1)  # (B, token_embed_dim)
            
            # Trait embedding (hub-ness, surprise)
            if isinstance(dimension_traits, list):
                # UnifiedTknProcessor returns list of (1, 2) tensors
                dim_trait_tensor = dimension_traits[dim_idx]  # (1, 2) from UnifiedTknProcessor
                # Expand to batch size if needed
                if dim_trait_tensor.shape[0] == 1 and B > 1:
                    dim_traits = dim_trait_tensor.expand(B, -1)  # (B, 2)
                else:
                    dim_traits = dim_trait_tensor  # (B, 2)
            else:
                # Fallback: if passed as tensor, extract per-dimension
                dim_traits = dimension_traits[:, dim_idx, :] if dimension_traits.dim() == 3 else dimension_traits
            
            trait_embed = self.trait_proj(dim_traits)  # (B, token_embed_dim)
            
            # Intent embedding (rel_goal component)
            intent_component = rel_goal[:, dim_idx:dim_idx+1]  # (B, 1)
            intent_embed = self.intent_proj(intent_component)  # (B, token_embed_dim)
            
            # Combine all embeddings for this dimension
            dim_token = token_embed + dim_pos + trait_embed + intent_embed  # Should be (B, token_embed_dim)
            
            # Ensure dim_token is exactly 2D (B, token_embed_dim)
            if dim_token.dim() == 1:
                # Single dimension: expand to batch if needed
                if B == 1:
                    dim_token = dim_token.unsqueeze(0)  # (1, token_embed_dim)
                else:
                    dim_token = dim_token.unsqueeze(0).expand(B, -1)  # (B, token_embed_dim)
            elif dim_token.dim() > 2:
                # Too many dimensions: flatten
                dim_token = dim_token.view(B, -1)
                # If flattened to wrong size, take first token_embed_dim elements
                if dim_token.shape[1] != self.token_embed_dim:
                    dim_token = dim_token[:, :self.token_embed_dim]
            
            # Final check: ensure shape is exactly (B, token_embed_dim)
            assert dim_token.shape == (B, self.token_embed_dim), \
                f"dim_token shape mismatch at dim {dim_idx}: got {dim_token.shape}, expected {(B, self.token_embed_dim)}"
            
            sequence_tokens.append(dim_token)
        
        # Stack into sequence: (B, state_dim, token_embed_dim)
        # All tokens should now be exactly (B, token_embed_dim)
        sequence = torch.stack(sequence_tokens, dim=1)  # (B, state_dim, token_embed_dim)
        
        # Verify sequence shape before concatenating with CLS
        assert sequence.dim() == 3, f"Expected sequence to be 3D (B, state_dim, token_embed_dim), got {sequence.shape}"
        assert sequence.shape == (B, state_dim, self.token_embed_dim), \
            f"Sequence shape mismatch: got {sequence.shape}, expected {(B, state_dim, self.token_embed_dim)}"
        
        # Add CLS token at beginning
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, token_embed_dim)
        sequence_with_cls = torch.cat([cls, sequence], dim=1)  # (B, state_dim+1, token_embed_dim)
        
        # Add memory context if provided (for temporal attention)
        if memory_context is not None:
            # memory_context: (B, memory_window, token_embed_dim)
            # Concatenate to sequence for temporal attention
            sequence_with_memory = torch.cat([sequence_with_cls, memory_context], dim=1)  # (B, state_dim+1+memory_window, token_embed_dim)
            encoded = self.transformer(sequence_with_memory)  # (B, state_dim+1+memory_window, token_embed_dim)
        else:
            # No memory context: standard processing
            encoded = self.transformer(sequence_with_cls)  # (B, state_dim+1, token_embed_dim)
        
        # Extract CLS token as situation representation
        cls_output = encoded[:, 0, :]  # (B, token_embed_dim)
        
        # Project to latent space
        situation = self.output_proj(cls_output)  # (B, latent_dim)
        
        # Extract dimension sequence (without CLS) for Actor attention
        situation_sequence = encoded[:, 1:1+state_dim, :]  # (B, state_dim, token_embed_dim)
        
        # Update memory context: sliding window of recent situation sequences
        # Use the current situation_sequence as the new memory entry
        if memory_context is not None:
            # Append current situation_sequence and keep only last memory_window entries
            # situation_sequence: (B, state_dim, token_embed_dim)
            # We'll use mean pooling across dimensions to get a single vector per situation
            current_situation_embed = situation_sequence.mean(dim=1, keepdim=True)  # (B, 1, token_embed_dim)
            new_memory_context = torch.cat([memory_context, current_situation_embed], dim=1)  # (B, memory_window+1, token_embed_dim)
            # Keep only last memory_window entries
            new_memory_context = new_memory_context[:, -self.memory_window:, :]  # (B, memory_window, token_embed_dim)
        else:
            # Initialize memory context with current situation
            current_situation_embed = situation_sequence.mean(dim=1, keepdim=True)  # (B, 1, token_embed_dim)
            # Pad to memory_window size
            if self.memory_window > 1:
                # Repeat current situation to fill memory window
                new_memory_context = current_situation_embed.repeat(1, self.memory_window, 1)  # (B, memory_window, token_embed_dim)
            else:
                new_memory_context = current_situation_embed  # (B, 1, token_embed_dim)
        
        return situation, situation_sequence, new_memory_context
