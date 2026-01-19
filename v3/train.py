"""
Main training script for TAM v3.

This script initializes the models and starts training.
All environment and training configuration options are defined here.
"""

from v3.actor import Actor
from v3.inference import TransformerInferenceEngine
from v3.simulation import train_actor

if __name__ == "__main__":
    # ============================================================================
    # MODEL ARCHITECTURE CONFIGURATION
    # ============================================================================
    STATE_DIM = 3  # Dimension of state space (change this to switch between 3D, 6D, etc.)
    LATENT_DIM = 128  # Dimension of latent situation space
    VOCAB_SIZE = 65536  # Token vocabulary size
    TOKEN_EMBED_DIM = 64  # Embedding dimension per token
    
    # Transformer Inference Engine Configuration
    TRANSFORMER_CONFIG = {
        "vocab_size": VOCAB_SIZE,
        "token_embed_dim": TOKEN_EMBED_DIM,
        "latent_dim": LATENT_DIM,
        "n_layers": 3,
        "n_heads": 8,
        "max_dimension_embed": 32,
        "dropout": 0.1,
        "memory_window": 10  # Number of recent situations to remember for temporal context
    }
    
    # Actor Configuration
    ACTOR_CONFIG = {
        "latent_dim": LATENT_DIM,
        "n_ports": 4,  # Number of affordance ports to propose
        "n_knots": 6,  # Number of knots per tube
        "n_basis": 8,  # Number of basis functions for knot generation
        "interp_res": 40,  # Resolution for spline interpolation
        "token_embed_dim": TOKEN_EMBED_DIM,
        "n_attention_heads": 8
    }
    
    # ============================================================================
    # ENVIRONMENT CONFIGURATION
    # ============================================================================
    ENV_CONFIG = {
        # Environment boundaries
        "bounds": {
            "min": [-2.0] * STATE_DIM,
            "max": [12.0] * STATE_DIM
        },
        
        # Obstacle generation parameters
        "obstacles": {
            "num_obstacles": 40,  # More obstacles = harder
            "min_radius": .5,
            "max_radius": 2,
            "packing_threshold": 0.1,  # 10% gap between obstacles (ensures spacing)
            "min_open_path_width": 1,  # Minimum corridor width (higher = easier navigation)
            "seed": 60  # Deterministic obstacle layout (change for different layouts)
        },
        
        # Initial target position
        "initial_target": [10.0] * STATE_DIM,
        
        # Goal generation (when goal is reached) - Curriculum Learning
        "goal_generation": {
            "margin": 0.5,  # Margin from boundaries when generating new goals
            "min_dist_from_current": 1.0,  # Minimum distance from current position (start close)
            "min_dist_from_obstacles": 1.0,  # Minimum distance from obstacles (added to radius)
            "max_attempts": 100,  # Maximum attempts to find valid goal position (increased for distance constraint)
            "initial_max_goal_distance": 2.0,  # Start with goals within 2.0 units (easy)
            "distance_increase_per_goal": None  # Auto-calculated: reaches max diagonal in ~50 goals
        },
        
        # Goal reaching threshold
        "goal_reached_threshold": 0.5,  # Consider goal "reached" if within this distance
        
        # Multi-goal environment configuration
        "initial_goal_count": 20,  # Number of goals at session start
        "max_observed_goals": 15,  # Maximum goals in observation (can add/remove goals freely!)
        
        # Observation configuration
        "max_observed_obstacles": 10,  # Maximum obstacles in observation (can add/remove obstacles freely!)
        
        # Energy/resource mechanics
        "energy": {
            "max_energy": 100.0,  # Maximum energy capacity
            "initial_energy": 100.0,  # Starting energy (default: max_energy)
            "energy_per_unit_distance": 2.5,  # Energy cost per unit distance traveled (increased for faster depletion)
            "energy_replenish_amount": 50.0  # Energy restored when goal reached
        }
    }
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    TRAINING_CONFIG = {
        "total_moves": 1500,  # Total number of moves
        "learning_rate": 1e-3,
        
        # Loss weights (principled TAM loss components)
        "loss_weights": {
            "binding_loss": 1.0,  # Contradiction: produce cones which are not contradicted by the environment
            "agency_cost": 0.1,  # Agency: produce the tightest cones possible subject to binding failure
        },
        
        # Surprise modulation (for novel patterns)
        "surprise_factor": 0.3,  # Multiplier for agency cost when novel patterns detected (1.0 to 1.3)
    }
    
    # ============================================================================
    # TOKENIZER CONFIGURATION
    # ============================================================================
    TOKENIZER_CONFIG = {
        "quantization_bins": 11,  # Number of quantization bins per head
        "quant_range": (-2.0, 2.0),  # Default quantization range (min, max)
        "proximity_quant_range": (-5.0, 10.0),  # Quantization range for proximity head
        "vocab_size": VOCAB_SIZE,
        "hub_threshold": 3  # Minimum hub_count to be considered a hub (for MarkovLattice)
    }
    
    # ============================================================================
    # HUB GRAPH CONFIGURATION
    # ============================================================================
    HUB_GRAPH_CONFIG = {
        "hub_threshold": 3,  # Minimum hub_count to be considered a hub
        "enable_look_ahead": True  # Enable trajectory queries for proactive planning
    }
    
    # ============================================================================
    # VISUALIZATION CONFIGURATION
    # ============================================================================
    VISUALIZATION_CONFIG = {
        "max_visualization_history": 5,  # Maximum number of moves to display in visualization (for future use)
        "plot_update_frequency": 1,  # Update plot every N moves (1 = every move) (for future use)
    }
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    LOGGING_CONFIG = {
        "artifacts_dir": "/Users/zach/Documents/dev/cfd/tam/artifacts",
        "tkn_log_batch_size": 20,  # Write tkn logs every N moves (reduces I/O overhead)
        "save_training_session": True,  # Save full training session data
        "save_goal_stats": True,  # Save goal statistics
        "save_tkn_stats": True,  # Save tokenizer statistics
        "generate_progress_plots": True  # Generate training progress plots at end
    }
    
    # ============================================================================
    # INITIALIZE MODELS
    # ============================================================================
    print("=" * 80)
    print("TAM v3 - Trajectory-Affordance Model Training")
    print("=" * 80)
    
    # Transformer-based inference engine (native dimension-agnostic)
    inference_engine = TransformerInferenceEngine(**TRANSFORMER_CONFIG)
    print(f"\nTransformer-Based Dimension-Agnostic System:")
    print(f"  - State dimension: {STATE_DIM} (inferred from environment)")
    print(f"  - Vocab size: {TRANSFORMER_CONFIG['vocab_size']}")
    print(f"  - Token embeddings: {TRANSFORMER_CONFIG['token_embed_dim']} dim")
    print(f"  - Transformer layers: {TRANSFORMER_CONFIG['n_layers']}")
    print(f"  - Attention heads: {TRANSFORMER_CONFIG['n_heads']}")
    
    actor = Actor(**ACTOR_CONFIG)
    print(f"\nActor Configuration:")
    print(f"  - Latent dimension: {ACTOR_CONFIG['latent_dim']}")
    print(f"  - Ports: {ACTOR_CONFIG['n_ports']}")
    print(f"  - Knots per tube: {ACTOR_CONFIG['n_knots']}")
    print(f"  - Basis functions: {ACTOR_CONFIG['n_basis']}")
    
    print(f"\nEnvironment Configuration:")
    print(f"  - State dimension: {STATE_DIM}")
    print(f"  - Bounds: {ENV_CONFIG['bounds']['min']} to {ENV_CONFIG['bounds']['max']}")
    print(f"  - Obstacles: {ENV_CONFIG['obstacles']['num_obstacles']} (seed={ENV_CONFIG['obstacles']['seed']}, deterministic)")
    print(f"  - Obstacle radius range: {ENV_CONFIG['obstacles']['min_radius']} to {ENV_CONFIG['obstacles']['max_radius']}")
    print(f"  - Packing threshold: {ENV_CONFIG['obstacles']['packing_threshold']}")
    print(f"  - Min open path width: {ENV_CONFIG['obstacles']['min_open_path_width']}")
    print(f"  - Max observed obstacles: {ENV_CONFIG['max_observed_obstacles']}")
    print(f"  - Goal reached threshold: {ENV_CONFIG['goal_reached_threshold']}")
    
    print(f"\nTraining Configuration:")
    print(f"  - Total moves: {TRAINING_CONFIG['total_moves']}")
    print(f"  - Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  - Loss weights: binding={TRAINING_CONFIG['loss_weights']['binding_loss']}, "
          f"agency={TRAINING_CONFIG['loss_weights']['agency_cost']}")
    
    print("=" * 80)
    
    # ============================================================================
    # START TRAINING
    # ============================================================================
    train_actor(
        inference_engine=inference_engine,
        actor=actor,
        config={
            "state_dim": STATE_DIM,
            "latent_dim": LATENT_DIM,
            "env": ENV_CONFIG,
            "training": TRAINING_CONFIG,
            "tokenizer": TOKENIZER_CONFIG,
            "visualization": VISUALIZATION_CONFIG,
            "logging": LOGGING_CONFIG,
            "hub_graph": HUB_GRAPH_CONFIG
        }
    )
