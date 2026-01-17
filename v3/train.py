"""
Main training script for TAM v3.

This script initializes the models and starts training.
"""

from v3.actor import Actor
from v3.inference import HybridInferenceEngine
from v3.simulation import train_actor

if __name__ == "__main__":
    # System configuration
    MAX_OBSERVED_OBSTACLES = 10  # Maximum obstacles in observation (can add/remove obstacles freely!)
    LATENT_DIM = 128  # Dimension of latent situation space
    
    # Hybrid system: Geometric tokens + Lattice traits + High-fidelity intent
    NUM_HEADS = 7  # delta_x, delta_y, delta_z, proximity, goal_dir_x, goal_dir_y, goal_dir_z
    VOCAB_SIZE = 65536  # Token vocabulary size
    TOKEN_EMBED_DIM = 16  # Embedding dimension per token (reduced since we have traits + intent)
    
    inference_engine = HybridInferenceEngine(
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        token_embed_dim=TOKEN_EMBED_DIM,
        latent_dim=LATENT_DIM
    )
    print(f"Using Hybrid System: {NUM_HEADS} heads, vocab_size={VOCAB_SIZE}")
    print(f"  - Token embeddings: {TOKEN_EMBED_DIM} dim")
    print(f"  - Lattice traits: hub-ness + surprise")
    print(f"  - High-fidelity intent: raw rel_goal vector")
    
    # Initialize Actor (operates on latent situations)
    actor = Actor(latent_dim=LATENT_DIM, intent_dim=3, n_knots=6, interp_res=40)
    
    # Train both models together with live plotting
    train_actor(
        inference_engine=inference_engine,
        actor=actor,
        total_moves=500,  # Total number of moves (not episodes)
        plot_live=True,
        max_observed_obstacles=MAX_OBSERVED_OBSTACLES,
        latent_dim=LATENT_DIM
    )
