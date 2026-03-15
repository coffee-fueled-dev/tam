"""
Training script for Simple2DEnvironment using the generic train_tam_system().

This demonstrates how to use the abstract Environment and TAMSystem interfaces
to create a complete training setup.
"""

import torch
import random
from v3.system_impl import TAMSystemWrapper
from v3.inference import TransformerInferenceEngine
from v3.actor import Actor
from v3.tokenizer import UnifiedTknProcessor
from v3.train_tam import train_tam_system
from v3.stats_recorder import JSONLStatsRecorder
from v3.environment_recorder import JSONLEnvironmentRecorder
from v3.d2.environment import Simple2DEnvironment
from v3.goal_motif import GoalMotif, create_motif_from_position
import os
from datetime import datetime

# For running directly from this directory
if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path if needed
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_obstacles_2d(num_obstacles: int = 5, bounds: dict = None, seed: int = 42):
    """Generate simple 2D obstacles."""
    if bounds is None:
        bounds = {"min": [-5.0, -5.0], "max": [5.0, 5.0]}

    random.seed(seed)
    obstacles = []
    min_radius = 0.3
    max_radius = 0.8

    for _ in range(num_obstacles):
        pos = [
            random.uniform(bounds["min"][0] + 1.0, bounds["max"][0] - 1.0),
            random.uniform(bounds["min"][1] + 1.0, bounds["max"][1] - 1.0),
        ]
        radius = random.uniform(min_radius, max_radius)
        obstacles.append((pos, radius))

    return obstacles


def generate_goals_2d(num_goals: int = 3, bounds: dict = None, seed: int = 42):
    """Generate simple 2D goals."""
    if bounds is None:
        bounds = {"min": [-5.0, -5.0], "max": [5.0, 5.0]}

    random.seed(seed + 100)  # Different seed from obstacles
    goals = []

    for _ in range(num_goals):
        pos = torch.tensor(
            [
                random.uniform(bounds["min"][0] + 1.0, bounds["max"][0] - 1.0),
                random.uniform(bounds["min"][1] + 1.0, bounds["max"][1] - 1.0),
            ],
            dtype=torch.float32,
        )
        goals.append(pos)

    return goals


if __name__ == "__main__":
    print("=" * 80)
    print("TAM v3 - Simple 2D Environment Training")
    print("=" * 80)

    # Configuration
    STATE_DIM = 2
    LATENT_DIM = 64
    VOCAB_SIZE = 65536
    TOKEN_EMBED_DIM = 32

    # Model configuration
    TRANSFORMER_CONFIG = {
        "vocab_size": VOCAB_SIZE,
        "token_embed_dim": TOKEN_EMBED_DIM,
        "latent_dim": LATENT_DIM,
        "n_layers": 2,
        "n_heads": 4,
        "max_dimension_embed": 16,
        "dropout": 0.1,
        "memory_window": 5,
    }

    ACTOR_CONFIG = {
        "latent_dim": LATENT_DIM,
        "n_ports": 4,
        "token_embed_dim": TOKEN_EMBED_DIM,
        "n_attention_heads": 4,
    }

    # Environment configuration
    bounds = {"min": [-5.0, -5.0], "max": [5.0, 5.0]}

    obstacles = generate_obstacles_2d(num_obstacles=20, bounds=bounds, seed=60)
    goals = generate_goals_2d(num_goals=40, bounds=bounds, seed=42)

    # Initialize models
    print("\nInitializing models...")
    inference_engine = TransformerInferenceEngine(**TRANSFORMER_CONFIG)
    actor = Actor(**ACTOR_CONFIG)

    # Initialize tokenizer
    tkn_processor = UnifiedTknProcessor(
        quantization_bins=11,
        quant_range=(-2.0, 2.0),
        vocab_size=VOCAB_SIZE,
        hub_threshold=3,
    )

    # Create TAMSystemWrapper
    system = TAMSystemWrapper(
        inference_engine=inference_engine,
        actor=actor,
        tkn_processor=tkn_processor,
        obstacles=obstacles,
        max_observed_obstacles=None,  # No limit - observe all obstacles
        max_observed_goals=None,  # No limit - observe all goals
    )

    # Create environment
    environment = Simple2DEnvironment(
        state_dim=STATE_DIM,
        bounds=bounds,
        obstacles=obstacles,
        goals=goals,
        max_observed_obstacles=None,  # No limit - observe all obstacles
        max_observed_goals=None,  # No limit - observe all goals
    )

    # Convert goal positions to goal motifs (for motif-based navigation)
    # Extract TKN patterns from goal positions to create motifs
    goal_motifs = []
    if len(goals) > 0:
        zero_intent = torch.zeros(STATE_DIM)
        for goal_pos in goals:
            # Get context at goal position to extract TKN pattern
            context = environment.get_context(goal_pos, zero_intent)
            # Create motif with extracted TKN pattern (structural properties, token patterns)
            goal_motif = create_motif_from_position(
                goal_pos, tkn_processor, context, obstacles
            )
            goal_motifs.append(goal_motif)

        # Set goal motifs in environment (enables motif-based navigation)
        environment.set_goal_motifs(goal_motifs)
        # Set TKN processor so environment can extract patterns from new goals
        environment.set_tkn_processor(tkn_processor)
        print(
            f"Initialized {len(goal_motifs)} goal motifs with TKN patterns extracted from positions"
        )

    # Training configuration
    total_moves = 1000
    learning_rate = 1e-3
    loss_weights = {
        "binding_loss": 1.0,
        "agency_reward": 1.0,  # Weight for agency reward (positive reward reduces loss)
        "goal_reward": 1.0,  # Weight for goal reward (positive reward reduces loss)
        "goal_penalty": 1.0,  # Weight for distance-based penalty (positive penalty increases loss)
    }

    # Goal reward configuration
    training_config = {
        "goal_reward_base": 10.0,  # Base reward when goal reached (constant, no decay)
        "goal_distance_threshold": 10.0,  # Maximum distance allowed before penalty kicks in
        "goal_distance_penalty": 0.1,  # Penalty per unit distance over threshold
    }

    print(f"\nEnvironment Configuration:")
    print(f"  - State dimension: {STATE_DIM}")
    print(f"  - Bounds: {bounds['min']} to {bounds['max']}")
    print(f"  - Obstacles: {len(obstacles)}")
    print(f"  - Goals: {len(goals)}")
    print(f"  - Context dimension: {environment.context_dim}")

    print(f"\nTraining Configuration:")
    print(f"  - Total moves: {total_moves}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Loss weights: {loss_weights}")

    print("=" * 80)
    print("\nStarting training...\n")

    # Create stats recorder (optional)
    artifacts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts"
    )
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, f"d2_run_{session_timestamp}")

    # Convert obstacles to serializable format: list of [x, y, radius] for 2D
    obstacles_config = []
    for pos, radius in obstacles:
        if len(pos) == 2:
            obstacles_config.append(
                [pos[0], pos[1], 0.0, radius]
            )  # [x, y, z=0, radius]
        else:
            obstacles_config.append(
                [pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0, radius]
            )

    # Convert goals to serializable format
    goals_config = []
    for goal in goals:
        if isinstance(goal, torch.Tensor):
            goal_list = goal.tolist()
        else:
            goal_list = list(goal)
        # Pad to 3D if needed
        while len(goal_list) < 3:
            goal_list.append(0.0)
        goals_config.append(goal_list)

    # Get initial position from environment
    initial_position = environment.get_initial_state()
    if isinstance(initial_position, torch.Tensor):
        initial_position_list = initial_position.tolist()
    else:
        initial_position_list = list(initial_position)
    # Pad to 3D if needed
    while len(initial_position_list) < 3:
        initial_position_list.append(0.0)

    stats_recorder = JSONLStatsRecorder(
        output_dir=run_dir,
        flush_interval=20,  # Flush every 20 moves
        system_config={
            "latent_dim": LATENT_DIM,
            "vocab_size": VOCAB_SIZE,
            "token_embed_dim": TOKEN_EMBED_DIM,
            "transformer_config": TRANSFORMER_CONFIG,
            "actor_config": ACTOR_CONFIG,
        },
        environment_config={
            "state_dim": STATE_DIM,
            "bounds": bounds,
            "initial_position": initial_position_list,  # Include initial position
            "obstacles": obstacles_config,  # Include full obstacle data
            "num_obstacles": len(obstacles),
            "goals": goals_config,  # Include initial goals
            "num_goals": len(goals),
        },
    )

    # Create environment recorder (optional, for visualization)
    environment_recorder = JSONLEnvironmentRecorder(
        output_dir=run_dir,
        flush_interval=50,  # Flush every 50 steps (environment refresh rate)
        state_dim=STATE_DIM,
    )

    # Train using generic training loop
    train_tam_system(
        system=system,
        environment=environment,
        total_moves=total_moves,
        learning_rate=learning_rate,
        loss_weights=loss_weights,
        goal_reached_threshold=0.5,
        energy_replenish_amount=20.0,
        config=training_config,
        stats_recorder=stats_recorder,
        environment_recorder=environment_recorder,
    )

    print("\n" + "=" * 80)
    print("Training complete!")
