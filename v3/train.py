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
    TOKEN_EMBED_DIM = 64  # Embedding dimension per token (transformer uses larger embeddings)
    
    # Transformer Inference Engine Configuration
    TRANSFORMER_CONFIG = {
        "vocab_size": VOCAB_SIZE,
        "token_embed_dim": TOKEN_EMBED_DIM,
        "latent_dim": LATENT_DIM,
        "n_layers": 3,
        "n_heads": 8,
        "max_dimension_embed": 32,
        "dropout": 0.1
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
            "num_obstacles": 60,  # More obstacles = harder
            "min_radius": .5,
            "max_radius": 2,
            "packing_threshold": 0.1,  # 10% gap between obstacles (ensures spacing)
            "min_open_path_width": 1,  # Minimum corridor width (higher = easier navigation)
            "seed": 60  # Deterministic obstacle layout (change for different layouts)
        },
        
        # Initial target position
        "initial_target": [10.0] * STATE_DIM,
        
        # Goal generation (when goal is reached)
        "goal_generation": {
            "margin": 0.5,  # Margin from boundaries when generating new goals
            "min_dist_from_current": 3.0,  # Minimum distance from current position
            "min_dist_from_obstacles": 1.0,  # Minimum distance from obstacles (added to radius)
            "max_attempts": 50  # Maximum attempts to find valid goal position
        },
        
        # Goal reaching threshold
        "goal_reached_threshold": 0.5,  # Consider goal "reached" if within this distance
        
        # Observation configuration
        "max_observed_obstacles": 10  # Maximum obstacles in observation (can add/remove obstacles freely!)
    }
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    TRAINING_CONFIG = {
        "total_moves": 10_000,  # Total number of moves (not episodes)
        "learning_rate": 1e-3,
        
        # Loss weights (principled TAM loss components)
        "loss_weights": {
            "binding_loss": 1.0,  # Contradiction: binding failure = wasted energy
            "agency_cost": 0.1,  # Agency: narrow cones = high agency = efficient
            "geometry_cost": 0.05,  # Geometry: fewer knots + smoother paths = simpler
            "intent_loss": 0.3  # Intent: goal-directed (supervision, not TAM-principled)
        },
        
        # Surprise modulation (for novel patterns)
        "surprise_factor": 0.3,  # Multiplier for agency cost when novel patterns detected (1.0 to 1.3)
        
        # Intent bias (adaptive goal-reaching)
        "intent_bias": {
            "close_threshold": 1.0,  # Distance threshold for "close" to goal
            "close_factor": 1.0,  # Push factor when close (100% toward goal)
            "far_factor": 0.5  # Push factor when far (50% toward goal)
        }
    }
    
    # ============================================================================
    # TOKENIZER CONFIGURATION
    # ============================================================================
    TOKENIZER_CONFIG = {
        "quantization_bins": 11,  # Number of quantization bins per head
        "quant_range": (-2.0, 2.0),  # Default quantization range (min, max)
        "proximity_quant_range": (-5.0, 10.0),  # Quantization range for proximity head
        "vocab_size": VOCAB_SIZE
    }
    
    # ============================================================================
    # VISUALIZATION CONFIGURATION
    # ============================================================================
    VISUALIZATION_CONFIG = {
        "plot_live": False,  # Whether to show live visualization
        "max_visualization_history": 5,  # Maximum number of moves to display in visualization
        "plot_update_frequency": 1,  # Update plot every N moves (1 = every move)
        "replay_frame_delay": 0.01  # Delay between frames in replay (seconds). Lower = faster replay
        # Examples: 0.05 = 20 FPS (fast), 0.1 = 10 FPS (normal), 0.2 = 5 FPS (slow), 0.01 = 100 FPS (very fast)
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
    # SYSTEM CONFIGURATION
    # ============================================================================
    SYSTEM_CONFIG = {
        # Always uses TransformerInferenceEngine (HybridInferenceEngine removed)
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
    print(f"  - Native sequence processing: no padding needed")
    
    # Initialize Actor with cross-attention (for transformer architecture)
    actor = Actor(**ACTOR_CONFIG)
    print(f"\nActor Configuration:")
    print(f"  - Latent dimension: {ACTOR_CONFIG['latent_dim']}")
    print(f"  - Ports: {ACTOR_CONFIG['n_ports']}")
    print(f"  - Knots per tube: {ACTOR_CONFIG['n_knots']}")
    print(f"  - Basis functions: {ACTOR_CONFIG['n_basis']}")
    print(f"  - Cross-attention: Enabled (transformer architecture)")
    
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
          f"agency={TRAINING_CONFIG['loss_weights']['agency_cost']}, "
          f"geometry={TRAINING_CONFIG['loss_weights']['geometry_cost']}, "
          f"intent={TRAINING_CONFIG['loss_weights']['intent_loss']}")
    
    print(f"\nVisualization Configuration:")
    print(f"  - Live plotting: {VISUALIZATION_CONFIG['plot_live']}")
    print(f"  - Max history: {VISUALIZATION_CONFIG['max_visualization_history']} moves")
    print(f"  - Update frequency: every {VISUALIZATION_CONFIG['plot_update_frequency']} moves")
    print(f"  - Replay speed: {1.0/VISUALIZATION_CONFIG['replay_frame_delay']:.1f} FPS (frame_delay={VISUALIZATION_CONFIG['replay_frame_delay']}s)")
    
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
            "system": SYSTEM_CONFIG
        }
    )
