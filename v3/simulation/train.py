import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import json
import os
from datetime import datetime

from v3.simulation.visualization_recorder import VisualizationRecorder
from v3.simulation.wrapper import SimulationWrapper
from v3.tokenizer import UnifiedTknProcessor
from v3.simulation.environment import generate_obstacles

def train_actor(inference_engine, actor, config=None):
    """
    Train the InferenceEngine and Actor together in the environment.
    
    This implements dual-optimization: both models learn simultaneously.
    - InferenceEngine learns to represent context such that Actor can successfully bind
    - Actor learns to use that representation to propose tubes
    
    Training runs for a fixed number of moves, continuously generating new goals when reached.
    Visualization shows only the last N moves (configurable via max_visualization_history).
    
    Args:
        inference_engine: TransformerInferenceEngine model
        actor: Actor model (operates on latent situations)
        config: Configuration dictionary with all training parameters
    """
    # Extract configuration with defaults
    if config is None:
        raise ValueError("config dictionary is required")
    
    state_dim = config.get("state_dim", 3)
    latent_dim = config.get("latent_dim", 128)
    env_config = config.get("env", {})
    training_config = config.get("training", {})
    tokenizer_config = config.get("tokenizer", {})
    visualization_config = config.get("visualization", {})
    logging_config = config.get("logging", {})
    
    # Extract environment configuration
    bounds = env_config.get("bounds", {"min": [-2.0] * state_dim, "max": [12.0] * state_dim})
    obstacles_config = env_config.get("obstacles", {})
    goal_gen_config = env_config.get("goal_generation", {})
    max_observed_obstacles = env_config.get("max_observed_obstacles", 10)
    max_observed_goals = env_config.get("max_observed_goals", 10)
    goal_reached_threshold = env_config.get("goal_reached_threshold", 0.5)
    initial_goal_count = env_config.get("initial_goal_count", 5)  # Number of goals at session start
    
    # Extract energy configuration
    energy_config = env_config.get("energy", {})
    max_energy = energy_config.get("max_energy", 100.0)
    initial_energy = energy_config.get("initial_energy", max_energy)
    energy_per_unit_distance = energy_config.get("energy_per_unit_distance", 1.0)
    energy_replenish_amount = energy_config.get("energy_replenish_amount", 20.0)
    
    # Extract training configuration
    total_moves = training_config.get("total_moves", 500)
    learning_rate = training_config.get("learning_rate", 1e-3)
    loss_weights = training_config.get("loss_weights", {
        "binding_loss": 1.0,
        "agency_cost": 0.1,
        "geometry_cost": 0.05,
    })
    surprise_factor = training_config.get("surprise_factor", 0.3)
    
    
    # Extract tokenizer configuration
    quantization_bins = tokenizer_config.get("quantization_bins", 11)
    quant_range = tokenizer_config.get("quant_range", (-2.0, 2.0))
    vocab_size = tokenizer_config.get("vocab_size", 65536)
    
    # Extract visualization configuration
    max_visualization_history = visualization_config.get("max_visualization_history", 5)
    plot_update_frequency = visualization_config.get("plot_update_frequency", 1)
    compression = visualization_config.get("compression", "none")  # "none", "gzip", "delta"
    
    # Extract logging configuration
    artifacts_dir = logging_config.get("artifacts_dir", "/Users/zach/Documents/dev/cfd/tam/artifacts")
    tkn_log_batch_size = logging_config.get("tkn_log_batch_size", 20)
    
    # Combine parameters for joint optimization
    optimizer = optim.Adam(
        list(inference_engine.parameters()) + list(actor.parameters()), 
        lr=learning_rate
    )
    
    # Generate deterministic obstacle layout
    obstacle_seed = obstacles_config.get("seed", 42)
    num_obstacles_requested = obstacles_config.get("num_obstacles", 12)
    obstacles = generate_obstacles(
        state_dim=state_dim,
        bounds=bounds,
        num_obstacles=num_obstacles_requested,
        min_radius=obstacles_config.get("min_radius", 0.8),
        max_radius=obstacles_config.get("max_radius", 2.2),
        packing_threshold=obstacles_config.get("packing_threshold", 0.5),
        min_open_path_width=obstacles_config.get("min_open_path_width", 2.5),
        seed=obstacle_seed
    )
    
    num_obstacles_generated = len(obstacles)
    if num_obstacles_generated < num_obstacles_requested:
        print(f"⚠️  Warning: Only generated {num_obstacles_generated} obstacles out of {num_obstacles_requested} requested")
        print(f"   This may be due to strict packing constraints (packing_threshold={obstacles_config.get('packing_threshold', 0.5)}, "
              f"min_open_path_width={obstacles_config.get('min_open_path_width', 2.5)})")
        print(f"   Consider increasing packing_threshold or decreasing num_obstacles")
    else:
        print(f"✓ Generated {num_obstacles_generated} obstacles (seed={obstacle_seed}, deterministic)")
    
    # Initialize simulation with energy config
    sim = SimulationWrapper(
        obstacles, 
        state_dim=state_dim, 
        bounds=bounds,
        energy_config={
            "max_energy": max_energy,
            "initial_energy": initial_energy,
            "energy_per_unit_distance": energy_per_unit_distance
        }
    )
    
    # Initialize active goals: generate N goals at session start
    def generate_random_goal():
        """Generate a random goal position within bounds, avoiding obstacles."""
        margin = goal_gen_config.get("margin", 0.5)
        min_bounds = [b + margin for b in bounds['min']]
        max_bounds = [b - margin for b in bounds['max']]
        
        min_dist_from_obstacles = goal_gen_config.get("min_dist_from_obstacles", 1.0)
        max_attempts = goal_gen_config.get("max_attempts", 50)
        
        for attempt in range(max_attempts):
            goal_pos = torch.tensor([
                random.uniform(min_bounds[i], max_bounds[i]) for i in range(state_dim)
            ])
            
            # Check obstacle collision
            valid = True
            for obs_p, obs_r in obstacles:
                obs_pos = torch.tensor(obs_p)
                if torch.norm(goal_pos - obs_pos) < obs_r + min_dist_from_obstacles:
                    valid = False
                    break
            
            if valid:
                return goal_pos
        
        # Fallback: return a goal at the center if all attempts fail
        center = [(bounds['min'][i] + bounds['max'][i]) / 2 for i in range(state_dim)]
        return torch.tensor(center)
    
    # Generate initial goals
    active_goals = []
    for _ in range(initial_goal_count):
        goal = generate_random_goal()
        active_goals.append(goal)
    
    goals_replenished = False  # Flag to track if we've started replenishing
    
    # Calculate raw context dimension: 
    # - max_observed_obstacles * (state_dim + 2) (rel_obs_pos + obs_radius + color)
    # - max_observed_goals * (state_dim + 2) (rel_goal_pos + goal_radius + color)
    # - 2 * state_dim (boundary distances: min and max per dimension)
    # - 2 (energy_value, energy_normalized)
    # This is FIXED regardless of actual obstacle/goal count (uses padding/truncation)
    raw_ctx_dim = max_observed_obstacles * (state_dim + 2) + max_observed_goals * (state_dim + 2) + 2 * state_dim + 2
    
    # Create session directory and file
    os.makedirs(artifacts_dir, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory for all dump files
    run_dir = os.path.join(artifacts_dir, f"run_{session_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    goal_stats_file = os.path.join(run_dir, f"goal_stats_{session_timestamp}.jsonl")
    
    # Initialize tkn processor (UnifiedTknProcessor for transformer architecture)
    vocab_size_from_engine = inference_engine.vocab_size if hasattr(inference_engine, 'vocab_size') else vocab_size
    hub_threshold = tokenizer_config.get("hub_threshold", 3)
    tkn_processor = UnifiedTknProcessor(
        quantization_bins=quantization_bins,
        quant_range=quant_range,
        vocab_size=vocab_size_from_engine,
        hub_threshold=hub_threshold
    )
    tkn_log_file = os.path.join(run_dir, f"tkn_patterns_{session_timestamp}.jsonl")
    print(f"TKN enabled: Pattern discovery logging to {tkn_log_file}")
    print(f"  - Using UnifiedTknProcessor: shared head across dimensions")
    print(f"  - Vocab size: {vocab_size_from_engine}")
    
    # Verify models are compatible
    if hasattr(inference_engine, 'latent_dim') and inference_engine.latent_dim != latent_dim:
        raise ValueError(f"InferenceEngine latent_dim mismatch: expected {latent_dim}, "
                        f"but model has {inference_engine.latent_dim}")
    
    # Actor validation (dimension-agnostic, so no state_dim check needed)
    # Verify latent_dim matches transformer attention setup
    if hasattr(actor, 'situation_proj'):
        if actor.situation_proj.in_features != latent_dim:
            raise ValueError(f"Actor latent_dim mismatch: expected {latent_dim}, "
                           f"but actor.situation_proj expects {actor.situation_proj.in_features}")
    
    # Initialize VisualizationRecorder for data recording (generates visualization_data.jsonl and visualization_metadata.json)
    plotter = VisualizationRecorder(
        obstacles, active_goals=active_goals, bounds=bounds,
        max_history=max_visualization_history,
        artifacts_dir=run_dir,
        config=config,
        world_seed=obstacle_seed
    )
    
    print(f"Training for {total_moves} moves... (Session: {session_timestamp})")
    print(f"Run directory: {run_dir}")
    print(f"Raw context dimension: {raw_ctx_dim} ({max_observed_obstacles} max obstacles * {state_dim + 2} + {max_observed_goals} max goals * {state_dim + 2} + {2 * state_dim} boundaries + 2 energy)")
    print(f"Latent dimension: {latent_dim}")
    print(f"State dimension: {state_dim}")
    print(f"Obstacles: {len(obstacles)} (seed={obstacle_seed}, deterministic)")
    print(f"Initial goals: {len(active_goals)}")
    print(f"Bounds: {bounds['min']} to {bounds['max']}")

    # Initialize training state
    current_pos = torch.zeros((1, state_dim))
    previous_velocity = None  # Track previous path direction for G1 continuity
    move_count = 0  # Total moves executed
    
    # Reset InferenceEngine hidden state at start
    h_state = torch.zeros(1, latent_dim)  # (1, latent_dim) - initial situation
    
    # Initialize memory context for temporal learning
    memory_context = None  # Will be initialized on first forward pass
    
    # Reset tkn processor at start
    tkn_processor.reset_episode()
    
    # Goal tracking for logging (use numpy arrays for better performance)
    goal_number = 0  # Track total goals reached
    goal_start_pos = current_pos.clone()  # Position when current goal was set
    moves_to_current_goal = 0  # Moves taken toward current goal
    segment_lengths = []  # Track segment lengths for current goal (keep as list for extend)
    agency_values = []  # Track sigma (agency) values for current goal (keep as list for extend)
    
    # Performance optimizations:
    # 1. Batch I/O: Buffer tkn logs and write in batches (reduces file I/O overhead)
    # 2. Optimized tensor operations: Avoid unnecessary copies, use efficient numpy conversions
    # 3. Optimized statistics: Use float32 arrays, compute stats efficiently
    tkn_log_buffer = []  # Buffer tkn logs to write in batches
    
    # Track loss history for analysis
    loss_history = []  # Track loss values over time
    
    # Main training loop: run for total_moves
    while move_count < total_moves:
        # 1. INFER: Convert raw world data to latent situation
        raw_obs = sim.get_raw_observation(
            current_pos, 
            active_goals=active_goals, 
            max_obstacles=max_observed_obstacles,
            max_observed_goals=max_observed_goals
        )  # (raw_ctx_dim,)
        raw_obs = raw_obs.unsqueeze(0)  # (1, raw_ctx_dim) for batch dimension
        
        # Process through tkn (always enabled)
        # Pass max_observed_obstacles and max_observed_goals so tokenizer can extract obstacle/goal directions from raw_obs
        tkn_output = tkn_processor.process_observation(
            current_pos, raw_obs.squeeze(0), obstacles, 
            max_observed_obstacles=max_observed_obstacles,
            max_observed_goals=max_observed_goals
        )
        
        # Extract relative goal
        rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, state_dim) high-fidelity intent
        
        # Transformer architecture: process dimensions as sequence
        # UnifiedTknProcessor returns list of (1,) tensors for tokens and (1, 2) for traits
        dimension_tokens = tkn_output["dimension_tokens"]  # List of (1,) tensors
        dimension_traits = tkn_output["dimension_traits"]  # List of (1, 2) tensors
        
        # Call transformer inference engine with memory context
        # Keep gradients flowing for temporal learning within the memory window
        x_n, situation_sequence, new_memory_context = inference_engine(
            dimension_tokens, dimension_traits, rel_goal_tensor, h_state, memory_context=memory_context
        )  # x_n: (1, latent_dim), situation_sequence: (1, state_dim, token_embed_dim), new_memory_context: (1, memory_window, token_embed_dim)
        
        # For logging (legacy format compatibility)
        lattice_tokens = tkn_output.get("lattice_tokens", torch.zeros(len(dimension_tokens), dtype=torch.long))
        surprise_mask = tkn_output.get("surprise_mask", torch.zeros(len(dimension_tokens), dtype=torch.bool))
        lattice_traits = tkn_output.get("lattice_traits", torch.zeros(len(dimension_tokens), 2))
        
        # Keep h_state connected for gradient flow (will be detached after backward pass)
        h_state = x_n
        
        # Query hub graph for structural analogies if surprise detected
        if surprise_mask is not None and surprise_mask.any():
            markov_lattice = tkn_processor.lattice
            if markov_lattice and hasattr(markov_lattice, 'query_structural_analogies'):
                # Query for structural analogies to inform transformer
                query_embedding = situation_sequence.mean(dim=1)  # (1, token_embed_dim)
                # Use first dimension's tokens for query (or first token of first dimension if variable-length)
                query_token = dimension_tokens[0] if dimension_tokens and len(dimension_tokens) > 0 else None
                if query_token is not None and isinstance(query_token, torch.Tensor) and query_token.numel() > 0:
                    # Extract first token from variable-length sequence
                    query_token_id = query_token.flatten()[0].item()
                    analogies = markov_lattice.query_structural_analogies(
                        query_token_id, query_embedding, k=5
                    )
                else:
                    analogies = []
                # Note: Analogies can be used to modulate attention or add context
                # For now, we just track them (can be enhanced later)
        
        # Buffer tkn logs for batch writing (much faster than writing every move)
        log_entry = {
            "move": move_count,
            "lattice_tokens": [t.tolist() for t in dimension_tokens],
            "surprise_mask": surprise_mask.tolist() if isinstance(surprise_mask, torch.Tensor) else [0] * len(dimension_tokens),
            "lattice_traits": [t.squeeze(0).tolist() for t in dimension_traits],
            "buffer_hashes": tkn_output.get("buffer_hashes", torch.zeros(len(dimension_tokens), dtype=torch.long)).tolist(),
            **tkn_output["metadata"]
        }
        tkn_log_buffer.append(log_entry)
        
        # Write in batches to reduce I/O overhead
        if len(tkn_log_buffer) >= tkn_log_batch_size:
            with open(tkn_log_file, 'a') as f:
                for entry in tkn_log_buffer:
                    json.dump(entry, f)
                    f.write('\n')
            tkn_log_buffer.clear()
        
        # Extract rel_goal from tkn_output (extracted from nearest goal in observation)
        # Actor will eventually learn to infer this from observation, but for now we use extracted rel_goal
        rel_goal = tkn_output["rel_goal"]  # (state_dim,) - Already extracted from nearest goal in observation
        rel_goal_batched = rel_goal.unsqueeze(0)  # (1, state_dim) - Add batch dimension for Actor
        
        # 2. ACT: Propose affordances based on latent situation x_n
        # Actor operates on latent representation, not raw spatial features
        # Actor uses cross-attention to situation_sequence from transformer
        # Pass markov_lattice for look-ahead queries
        markov_lattice = tkn_processor.lattice if hasattr(tkn_processor, 'lattice') else None
        current_pos_np = current_pos.squeeze().cpu().numpy() if isinstance(current_pos, torch.Tensor) else current_pos
        
        logits, mu_t, sigma_t, knot_mask, basis_weights = actor(
            x_n, rel_goal_batched, previous_velocity=previous_velocity, situation_sequence=situation_sequence,
            markov_lattice=markov_lattice, current_pos=current_pos_np
        )
        
        # Selection (Categorical sampling for exploration)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        idx = m.sample()
        
        # Extract selected tube - ensure we get (T, state_dim) shape
        idx_int = idx.item() if isinstance(idx, torch.Tensor) else idx
        selected_tube = mu_t[0, idx_int].squeeze(0)  # (T, state_dim)
        selected_sigma = sigma_t[0, idx_int].squeeze(0)  # (T, state_dim) - per-dimension precision
        
        # Ensure shapes are correct
        if selected_tube.dim() > 2:
            selected_tube = selected_tube.view(-1, state_dim)
        if selected_sigma.dim() > 2:
            selected_sigma = selected_sigma.view(-1, state_dim)
            
        # Track agency (sigma) values for current goal - use mean across dimensions
        agency_values.append(float(selected_sigma.mean().detach().item()))
        
        # Track segment lengths (distances between consecutive points in the tube)
        # Only calculate if we have multiple points (optimization)
        if len(selected_tube) > 1:
            # Optimized: compute norm in one go, convert to list efficiently
            tube_segments = selected_tube[1:] - selected_tube[:-1]
            segment_lengths_for_move = torch.norm(tube_segments, dim=-1).detach().cpu()
            segment_lengths.extend(segment_lengths_for_move.tolist())
        
        # Execute the tube
        # selected_sigma is now (T, state_dim) - need to convert to scalar for execute()
        # Use mean sigma across dimensions for execution (or max for conservative)
        selected_sigma_scalar = selected_sigma.mean(dim=-1, keepdim=True)  # (T, 1)
        actual_p = sim.execute(selected_tube, selected_sigma_scalar, current_pos)
        
        # 3. PRINCIPLED TAM LOSS: Computed by Actor based on binding outcomes
        # The Actor computes its own loss from binding failure and agency
        # This is dimension-agnostic and principled based on the TAM framework
        # Binding failure naturally captures all contradictions, including energy depletion:
        # - If energy runs out, actual_path is shorter than proposed_tube
        # - The binding loss automatically penalizes the unexecuted portion
        # - No special "death" concept needed - binding failure is universal
        
        # Get knot mask for selected port (if available)
        selected_knot_mask = knot_mask[0, idx_int] if knot_mask is not None else None
        
        # Compute binding loss, agency cost within Actor
        binding_loss, agency_cost = actor.compute_binding_loss(
            selected_tube, actual_p, selected_sigma, current_pos.squeeze(), 
            knot_mask=selected_knot_mask
        )
        
        # MODULATE AGENCY WITH SURPRISE: Novel patterns reduce agency cost penalty, allowing wider cones (lower precision) until learned
        # This is principled: novel situations require wider cones until learned
        if surprise_mask is not None:
            surprise_multiplier = 1.0 + surprise_factor * surprise_mask.float().mean().item()  # 1.0 to 1.0+surprise_factor multiplier
            agency_cost = agency_cost / surprise_multiplier
        
        # Principled TAM Loss: Based on binding failure and agency
        # Energy minimization emerges naturally:
        # - Binding failures = wasted energy (binding_loss)
        # - Wide cones = low agency = wasted energy (agency_cost)
        # - Goal proximity reward = positive signal for moving toward goals
        # - Execution reward = positive signal for completing moves
        # 
        # The environment teaches everything else through binding failures:
        # - Knot count: Too many/few knots → binding failures → actor learns optimal count
        # - Tube start: Doesn't start at current position → binding failure → actor learns to start at origin
        # - Path smoothness: Sharp turns cause collisions → binding failures → actor learns smooth paths
        # - Complex paths that cause binding failures will be avoided through learning
        # - The Actor learns to select simpler ports naturally via binding feedback
        # - Fewer moves emerge from selecting ports that reach goals without binding failures
        loss = (loss_weights["binding_loss"] * binding_loss
               + loss_weights["agency_cost"] * agency_cost
               )
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for temporal learning (prevents gradient explosion)
        torch.nn.utils.clip_grad_norm_(inference_engine.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track loss for analysis
        loss_history.append(float(loss.item()))
        
        # Detach memory_context and h_state after backward pass to prevent graph accumulation
        # This allows temporal learning within each iteration's computation graph,
        # but prevents unbounded graph growth across iterations
        memory_context = new_memory_context.detach()
        h_state = h_state.detach()
        
        # Handle energy depletion: if energy depleted, reset position and PARTIAL energy (not full)
        # This makes energy depletion costly, not rewarding
        # Binding failure already captured the contradiction - no special "death" concept needed
        # Goal stays in place until reached - actor must learn to reach it to survive
        if sim.is_dead or sim.current_energy <= 0:
            print(f"Move {move_count}/{total_moves} | Energy depleted! Resetting position and PARTIAL energy, keeping goal.")
            # Reset energy to PARTIAL amount (not full) - makes depletion costly
            # This prevents the agent from learning that depletion is rewarding
            sim.current_energy = sim.initial_energy * 0.3  # Only 30% energy on reset (not 100%)
            sim.is_dead = False
            # Reset position to origin
            current_pos = torch.zeros((1, state_dim))
            # Keep the same goal - don't generate a new one
            # This forces the actor to learn to reach the goal to avoid repeated energy depletion
            # Reset goal tracking counters (but keep target_pos unchanged)
            goal_start_pos = current_pos.clone()
            goal_start_move = move_count
            moves_to_current_goal = 0
            segment_lengths = []
            agency_values = []
            # Reset tkn processor
            tkn_processor.reset_episode()
            # Continue training (don't break - just reset and continue)
        
        # Update position
        current_pos = actual_p[-1].view(1, state_dim).detach()
        move_count += 1
        moves_to_current_goal += 1
        
        # Note: previous_velocity tracking removed - G1 continuity alignment removed
        # The model should learn path smoothness through binding failures if needed
        previous_velocity = None  # Keep for interface compatibility but not used
            
        # Record data for visualization (no rendering during training - zero overhead)
        if plotter is not None and move_count % plot_update_frequency == 0:
            plotter.update(selected_tube, selected_sigma, actual_p, current_pos, 
                         episode=0, step=move_count, active_goals=active_goals,
                         energy=sim.current_energy, max_energy=sim.max_energy,
                         loss=loss.item())
        
        # Check if any goal is reached - check all active goals
        reached_goal = None
        reached_goal_idx = None
        for idx, goal_pos in enumerate(active_goals):
            goal_distance = torch.norm(current_pos.squeeze() - goal_pos).item()
            if goal_distance < goal_reached_threshold:
                reached_goal = goal_pos
                reached_goal_idx = idx
                break
        
        if reached_goal is not None:
            # Remove reached goal from active goals
            active_goals.pop(reached_goal_idx)
            
            # Replenish energy when goal is reached
            sim.replenish_energy(energy_replenish_amount)
            
            # Mark goal as reached in plotter (increments goal counter)
            if plotter is not None:
                plotter.mark_goal_reached(reached_goal)
            
            # Calculate statistics for this goal (optimized: compute all stats in one pass)
            if len(segment_lengths) > 0:
                seg_array = np.asarray(segment_lengths, dtype=np.float32)  # Use float32 for speed
                seg_mean = float(seg_array.mean())
                seg_max = float(seg_array.max())
                seg_min = float(seg_array.min())
                seg_std = float(seg_array.std())
            else:
                seg_mean = seg_max = seg_min = seg_std = 0.0
            
            if len(agency_values) > 0:
                agency_array = np.asarray(agency_values, dtype=np.float32)  # Use float32 for speed
                agency_mean = float(agency_array.mean())
                agency_max = float(agency_array.max())
                agency_min = float(agency_array.min())
                agency_std = float(agency_array.std())
            else:
                agency_mean = agency_max = agency_min = agency_std = 0.0
            
            # Increment goal counter
            goal_number += 1
            
            # Log goal statistics
            goal_stats = {
                "goal_number": goal_number,
                "start_position": goal_start_pos.squeeze().tolist(),
                "goal_position": reached_goal.tolist(),
                "moves_taken": moves_to_current_goal,
                "segment_lengths": {
                    "mean": seg_mean,
                    "max": seg_max,
                    "min": seg_min,
                    "std": seg_std,
                    "num_segments": len(segment_lengths)
                },
                "agency": {
                    "mean": agency_mean,
                    "max": agency_max,
                    "min": agency_min,
                    "std": agency_std,
                    "num_samples": len(agency_values)
                }
            }
            
            # Write goal statistics to JSONL file
            with open(goal_stats_file, 'a') as f:
                json.dump(goal_stats, f)
                f.write('\n')
            
            # Replenishment logic: only start replenishing after all original goals are gone
            # Then replenish decreasing number of goals until reaching 1 new goal per completed goal
            if len(active_goals) == 0:
                # All original goals are gone - start replenishment
                if not goals_replenished:
                    goals_replenished = True
                    print(f"Move {move_count}/{total_moves} | All original goals reached! Starting replenishment.")
                
                # Calculate how many goals to replenish (decreasing schedule)
                # Start with initial_goal_count/2, then decrease by half each time until reaching 1
                replenish_count = max(2, initial_goal_count // (2 ** goal_number))
                
                # Generate new goals
                for _ in range(replenish_count):
                    new_goal = generate_random_goal()
                    active_goals.append(new_goal)
                
                print(f"Move {move_count}/{total_moves} | Replenished {replenish_count} goal(s). Active goals: {len(active_goals)}")
            
            # Reset goal tracking for next goal
            goal_start_pos = current_pos.clone()
            moves_to_current_goal = 0
            segment_lengths = []
            agency_values = []
            
            # Print goal reached summary
            print(f"Move {move_count}/{total_moves} | Goal reached! | Moves: {goal_stats['moves_taken']} | "
                  f"Active goals: {len(active_goals)} | "
                  f"Segments: {seg_mean:.3f}±{seg_std:.3f} | Agency: {agency_mean:.3f}±{agency_std:.3f}")
    
    # Flush any remaining buffered tkn logs
    if len(tkn_log_buffer) > 0:
        with open(tkn_log_file, 'a') as f:
            for entry in tkn_log_buffer:
                json.dump(entry, f)
                f.write('\n')
        tkn_log_buffer.clear()
    
    print(f"\nTraining complete. Total moves: {move_count}")
    print(f"Goal statistics saved to: {goal_stats_file}")
    print(f"TKN patterns saved to: {tkn_log_file}")
    
    # Finalize visualization data files
    if plotter is not None:
        try:
            plotter.finalize(output_dir=run_dir, compression=compression)
        except Exception as e:
            print(f"\n⚠️  Warning: Error finalizing visualization data: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll run files saved to: {run_dir}")
    print(f"  - Goal statistics: {goal_stats_file}")
    print(f"  - TKN patterns: {tkn_log_file}")
    if plotter is not None and plotter.jsonl_path:
        print(f"  - Visualization data: {plotter.jsonl_path}")
        print(f"  - Visualization metadata: {plotter.metadata_path}")
