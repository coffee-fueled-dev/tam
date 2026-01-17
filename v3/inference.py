import torch
import torch.nn as nn

class HybridInferenceEngine(nn.Module):
    """
    Hybrid Inference Engine that processes:
    1. Discrete tokens (what) - geometric pattern IDs
    2. Lattice traits (epistemic status) - hub-ness, surprise, buffer state
    3. High-fidelity intent (where) - raw rel_goal vector
    
    This creates a rich "Full Situation" representation that combines:
    - Semantic meaning (token embeddings)
    - Epistemic confidence (lattice topology)
    - Spatial precision (direct goal vector)
    """
    def __init__(self, num_heads, vocab_size=65536, token_embed_dim=16, latent_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.latent_dim = latent_dim
        
        # Token Stream: Discrete pattern IDs -> Dense embeddings
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)
        
        # Lattice Traits Stream: Hub-ness + Surprise -> Features
        # (num_heads, 2) -> (num_heads * 2) -> 32
        self.trait_fc = nn.Sequential(
            nn.Linear(num_heads * 2, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Intent Stream: High-fidelity rel_goal (3,) -> Features
        self.intent_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Combined GRU input: (Token Embeds) + Traits + Intent
        # (num_heads * token_embed_dim) + 32 + 32
        total_input_dim = (num_heads * token_embed_dim) + 32 + 32
        self.gru = nn.GRUCell(total_input_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        
    def forward(self, lattice_tokens, lattice_traits, rel_goal, h_prev):
        """
        Convert geometric tokens + lattice traits + intent to latent situation.
        
        Args:
            lattice_tokens: (B, num_heads) tensor of token IDs
            lattice_traits: (B, num_heads, 2) tensor of [hub_count, surprise] per head
            rel_goal: (B, 3) tensor of high-fidelity relative goal vector
            h_prev: (B, latent_dim) previous hidden state (situation from last step)
        
        Returns:
            h_next: (B, latent_dim) new hidden state (current situation x_n)
        """
        # 1. Token Stream: Embed discrete pattern IDs
        token_embeds = self.token_embedding(lattice_tokens)  # (B, num_heads, token_embed_dim)
        token_flat = token_embeds.view(token_embeds.size(0), -1)  # (B, num_heads * token_embed_dim)
        
        # 2. Lattice Traits Stream: Process epistemic status
        trait_flat = lattice_traits.view(lattice_traits.size(0), -1)  # (B, num_heads * 2)
        trait_feat = self.trait_fc(trait_flat)  # (B, 32)
        
        # 3. Intent Stream: Process high-fidelity goal vector
        intent_feat = self.intent_fc(rel_goal)  # (B, 32)
        
        # 4. Concatenate all channels (The 'Full Situation')
        combined = torch.cat([token_flat, trait_feat, intent_feat], dim=-1)  # (B, total_input_dim)
        
        # 5. Update latent situation x_n
        h_next = self.gru(combined, h_prev)
        return self.ln(h_next)
