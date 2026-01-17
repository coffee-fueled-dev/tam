"""
Main training script for TAM v3.

This script initializes the models and starts training.
"""

from v3.actor import Actor
from v3.inference import HybridInferenceEngine
from v3.simulation import train_actor

if __name__ == "__main__":
    # System configuration
    STATE_DIM = 3  # Dimension of state space (change this to switch between 3D, 6D, etc.)
    MAX_OBSERVED_OBSTACLES = 10  # Maximum obstacles in observation (can add/remove obstacles freely!)
    LATENT_DIM = 128  # Dimension of latent situation space
    
    # Calculate number of heads dynamically based on state_dim
    # Heads: state_dim delta heads + 1 proximity head + state_dim goal_dir heads
    NUM_HEADS = STATE_DIM + 1 + STATE_DIM  # = 2 * STATE_DIM + 1
    
    VOCAB_SIZE = 65536  # Token vocabulary size
    TOKEN_EMBED_DIM = 16  # Embedding dimension per token (reduced since we have traits + intent)
    
    inference_engine = HybridInferenceEngine(
        num_heads=NUM_HEADS,
        intent_dim=STATE_DIM,  # Intent dimension equals state dimension
        vocab_size=VOCAB_SIZE,
        token_embed_dim=TOKEN_EMBED_DIM,
        latent_dim=LATENT_DIM
    )
    print(f"Using Dimension-Agnostic Hybrid System:")
    print(f"  - State dimension: {STATE_DIM}")
    print(f"  - Number of heads: {NUM_HEADS} ({STATE_DIM} delta + 1 proximity + {STATE_DIM} goal_dir)")
    print(f"  - Vocab size: {VOCAB_SIZE}")
    print(f"  - Token embeddings: {TOKEN_EMBED_DIM} dim")
    print(f"  - Lattice traits: hub-ness + surprise")
    print(f"  - High-fidelity intent: raw rel_goal vector ({STATE_DIM}D)")
    
    # Initialize Actor (operates on latent situations)
    # New: Basis-Projection Actor with basis functions
    actor = Actor(latent_dim=LATENT_DIM, state_dim=STATE_DIM, n_knots=6, n_basis=8, interp_res=40)
    
    # Train both models together with live plotting
    train_actor(
        inference_engine=inference_engine,
        actor=actor,
        total_moves=500,  # Total number of moves (not episodes)
        plot_live=True,
        max_observed_obstacles=MAX_OBSERVED_OBSTACLES,
        latent_dim=LATENT_DIM,
        state_dim=STATE_DIM
    )
