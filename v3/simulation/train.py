import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
from datetime import datetime

from v3.simulation.plotter import LivePlotter
from v3.simulation.wrapper import SimulationWrapper
from v3.tokenizer import UnifiedTknProcessor
from v3.simulation.analysis import plot_training_progress, generate_training_summary
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
    system_config = config.get("system", {})
    
    # Extract environment configuration
    bounds = env_config.get("bounds", {"min": [-2.0] * state_dim, "max": [12.0] * state_dim})
    obstacles_config = env_config.get("obstacles", {})
    goal_gen_config = env_config.get("goal_generation", {})
    max_observed_obstacles = env_config.get("max_observed_obstacles", 10)
    goal_reached_threshold = env_config.get("goal_reached_threshold", 0.5)
    initial_target = env_config.get("initial_target", [10.0] * state_dim)
    
    # Extract training configuration
    total_moves = training_config.get("total_moves", 500)
    learning_rate = training_config.get("learning_rate", 1e-3)
    loss_weights = training_config.get("loss_weights", {
        "binding_loss": 1.0,
        "agency_cost": 0.1,
        "geometry_cost": 0.05,
        "intent_loss": 0.3
    })
    surprise_factor = training_config.get("surprise_factor", 0.3)
    intent_bias_config = training_config.get("intent_bias", {
        "close_threshold": 1.0,
        "close_factor": 1.0,
        "far_factor": 0.5
    })
    
    # Extract tokenizer configuration
    quantization_bins = tokenizer_config.get("quantization_bins", 11)
    quant_range = tokenizer_config.get("quant_range", (-2.0, 2.0))
    vocab_size = tokenizer_config.get("vocab_size", 65536)
    
    # Extract visualization configuration
    plot_live = visualization_config.get("plot_live", True)
    max_visualization_history = visualization_config.get("max_visualization_history", 5)
    plot_update_frequency = visualization_config.get("plot_update_frequency", 1)
    replay_frame_delay = visualization_config.get("replay_frame_delay", 0.1)
    
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
    
    sim = SimulationWrapper(obstacles, state_dim=state_dim, bounds=bounds)
    # Initialize target position
    target_pos = torch.tensor(initial_target)
    
    # Calculate raw context dimension: 
    # - state_dim (rel_goal) 
    # + max_observed_obstacles * (state_dim + 1) (rel_obs_pos + obs_radius)
    # + 2 * state_dim (boundary distances: min and max per dimension)
    # This is FIXED regardless of actual obstacle count (uses padding/truncation)
    raw_ctx_dim = state_dim + max_observed_obstacles * (state_dim + 1) + 2 * state_dim
    
    # Create session directory and file
    os.makedirs(artifacts_dir, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory for all dump files
    run_dir = os.path.join(artifacts_dir, f"run_{session_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    session_file = os.path.join(run_dir, f"training_session_{session_timestamp}.json")
    goal_stats_file = os.path.join(run_dir, f"goal_stats_{session_timestamp}.jsonl")
    
    # Initialize tkn processor (UnifiedTknProcessor for transformer architecture)
    vocab_size_from_engine = inference_engine.vocab_size if hasattr(inference_engine, 'vocab_size') else vocab_size
    tkn_processor = UnifiedTknProcessor(
        quantization_bins=quantization_bins,
        quant_range=quant_range,
        vocab_size=vocab_size_from_engine
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
    
    # Training data storage
    training_data = {
        "session_timestamp": session_timestamp,
        "total_moves": total_moves,
        "obstacles": obstacles,
        "obstacle_seed": obstacle_seed,  # Save seed for reproducibility
        "bounds": bounds,
        "raw_ctx_dim": raw_ctx_dim,
        "latent_dim": latent_dim,
        "state_dim": state_dim,
        "max_observed_obstacles": max_observed_obstacles,
        "initial_target": target_pos.tolist(),
        "use_tokenized": True,  # Always using tokenized system
        "config": config  # Save full config for reproducibility
    }
    
    # Initialize plotter in record mode (collects data for replay after training)
    plotter = None
    if plot_live:
        plotter = LivePlotter(obstacles, target_pos.numpy(), bounds=bounds, 
                             max_history=max_visualization_history, record_mode=True)
    
    print(f"Training for {total_moves} moves... (Session: {session_timestamp})")
    print(f"Run directory: {run_dir}")
    print(f"Raw context dimension: {raw_ctx_dim} ({state_dim} goal + {max_observed_obstacles} max obstacles * {state_dim + 1})")
    print(f"Latent dimension: {latent_dim}")
    print(f"State dimension: {state_dim}")
    print(f"Obstacles: {len(obstacles)} (seed={obstacle_seed}, deterministic)")
    print(f"Bounds: {bounds['min']} to {bounds['max']}")
    print(f"Visualization will show last {max_visualization_history} moves only")

    # Initialize training state
    current_pos = torch.zeros((1, state_dim))
    previous_velocity = None  # Track previous path direction for G1 continuity
    move_count = 0  # Total moves executed
    
    # Reset InferenceEngine hidden state at start
    h_state = torch.zeros(1, latent_dim)  # (1, latent_dim) - initial situation
    
    # Reset tkn processor at start
    tkn_processor.reset_episode()
    
    # Goal tracking for logging (use numpy arrays for better performance)
    goal_start_pos = current_pos.clone()  # Position when current goal was set
    goal_start_move = 0  # Move number when current goal was set
    moves_to_current_goal = 0  # Moves taken toward current goal
    segment_lengths = []  # Track segment lengths for current goal (keep as list for extend)
    agency_values = []  # Track sigma (agency) values for current goal (keep as list for extend)
    
    # Performance optimizations:
    # 1. Batch I/O: Buffer tkn logs and write in batches (reduces file I/O overhead)
    # 2. Reduced plot frequency: Update visualization every N moves (reduces matplotlib overhead)
    # 3. Disabled expensive sphere visualization (was creating many 3D surfaces)
    # 4. Optimized tensor operations: Avoid unnecessary copies, use efficient numpy conversions
    # 5. Optimized statistics: Use float32 arrays, compute stats efficiently
    tkn_log_buffer = []  # Buffer tkn logs to write in batches
    
    # Track loss history for analysis
    loss_history = []  # Track loss values over time
    
    # Main training loop: run for total_moves
    while move_count < total_moves:
        # 1. INFER: Convert raw world data to latent situation
        raw_obs = sim.get_raw_observation(current_pos, target_pos, max_obstacles=max_observed_obstacles)  # (raw_ctx_dim,)
        raw_obs = raw_obs.unsqueeze(0)  # (1, raw_ctx_dim) for batch dimension
        
        # Process through tkn (always enabled)
        # Pass max_observed_obstacles so tokenizer can extract obstacle directions from raw_obs
        tkn_output = tkn_processor.process_observation(
            current_pos, raw_obs.squeeze(0), obstacles, max_observed_obstacles=max_observed_obstacles
        )
        
        # Extract relative goal
        rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, state_dim) high-fidelity intent
        
        # Transformer architecture: process dimensions as sequence
        # UnifiedTknProcessor returns list of (1,) tensors for tokens and (1, 2) for traits
        dimension_tokens = tkn_output["dimension_tokens"]  # List of (1,) tensors
        dimension_traits = tkn_output["dimension_traits"]  # List of (1, 2) tensors
        
        # Call transformer inference engine
        x_n, situation_sequence = inference_engine(
            dimension_tokens, dimension_traits, rel_goal_tensor, h_state
        )  # x_n: (1, latent_dim), situation_sequence: (1, state_dim, token_embed_dim)
        
        # For logging (legacy format compatibility)
        lattice_tokens = tkn_output.get("lattice_tokens", torch.zeros(len(dimension_tokens), dtype=torch.long))
        surprise_mask = tkn_output.get("surprise_mask", torch.zeros(len(dimension_tokens), dtype=torch.bool))
        lattice_traits = tkn_output.get("lattice_traits", torch.zeros(len(dimension_tokens), 2))
        
        h_state = x_n.detach()  # Pass memory forward (detach to prevent backprop through time)
        
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
        
        rel_goal = target_pos - current_pos 
        
        # 2. ACT: Propose affordances based on latent situation x_n
        # Actor operates on latent representation, not raw spatial features
        # Actor uses cross-attention to situation_sequence from transformer
        logits, mu_t, sigma_t, knot_mask, basis_weights = actor(
            x_n, rel_goal, previous_velocity=previous_velocity, situation_sequence=situation_sequence
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
        # The Actor computes its own loss from binding failure, agency, and geometry
        # This is dimension-agnostic and principled based on the TAM framework
        
        # Get knot mask for selected port (if available)
        selected_knot_mask = knot_mask[0, idx_int] if knot_mask is not None else None
        
        # Compute binding loss, agency cost, and geometry cost within Actor
        binding_loss, agency_cost, geometry_cost = actor.compute_binding_loss(
            selected_tube, actual_p, selected_sigma, current_pos.squeeze(), 
            knot_mask=selected_knot_mask
        )
        
        # MODULATE AGENCY WITH SURPRISE: Novel patterns require wider cones (lower agency)
        # This is principled: novel situations require wider cones until learned
        if surprise_mask is not None:
            surprise_multiplier = 1.0 + surprise_factor * surprise_mask.float().mean().item()  # 1.0 to 1.0+surprise_factor multiplier
            agency_cost = agency_cost * surprise_multiplier
        
        # INTENT LOSS: Direct supervision to move toward goal
        # This is not strictly TAM-principled (it's external supervision), but necessary
        # for goal-directed behavior. The Actor learns to select ports that move toward
        # the goal to minimize binding failures over the long term.
        tube_endpoint_global = selected_tube[-1] + current_pos.squeeze()
        goal_distance = torch.norm(tube_endpoint_global - target_pos)
        
        # Principled intent loss: penalize distance, but don't penalize overshooting
        # If tube reaches or overshoots goal (distance < threshold), reward it
        # This encourages the Actor to make tubes that actually reach the goal
        # Dimension-agnostic: uses norm which works in any dimension
        if goal_distance < goal_reached_threshold:
            # Goal reached: minimal penalty (or even reward for efficiency)
            # The closer to goal, the better (but we don't want to penalize overshooting)
            intent_loss = goal_distance**2  # Quadratic penalty, but small when close
        else:
            # Goal not reached: standard MSE loss
            intent_loss = F.mse_loss(tube_endpoint_global, target_pos)
        
        # Principled TAM Loss: Based on binding failure and agency
        # Energy minimization emerges naturally:
        # - Binding failures = wasted energy (binding_loss)
        # - Wide cones = low agency = wasted energy (agency_cost)
        # - Intent loss = direct supervision to move toward goal
        # 
        # The environment teaches everything else through binding failures:
        # - Knot count: Too many/few knots → binding failures → actor learns optimal count
        # - Tube start: Doesn't start at current position → binding failure → actor learns to start at origin
        # - Tube end: Doesn't reach goal → intent_loss + inefficient paths → actor learns to aim for goal
        # - Path smoothness: Sharp turns cause collisions → binding failures → actor learns smooth paths
        # - Complex paths that cause binding failures will be avoided through learning
        # - The Actor learns to select simpler ports naturally via binding feedback
        # - Fewer moves emerge from selecting ports that reach goals without binding failures
        loss = (loss_weights["binding_loss"] * binding_loss
               + loss_weights["agency_cost"] * agency_cost
               + loss_weights["geometry_cost"] * geometry_cost  # Will be 0.0, kept for interface compatibility
               + loss_weights["intent_loss"] * intent_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss for analysis
        loss_history.append(float(loss.item()))
        
        # Update position
        current_pos = actual_p[-1].view(1, state_dim).detach()
        move_count += 1
        moves_to_current_goal += 1
        
        # Note: previous_velocity tracking removed - G1 continuity alignment removed
        # The model should learn path smoothness through binding failures if needed
        previous_velocity = None  # Keep for interface compatibility but not used
            
        # Record data for replay (no rendering during training - zero overhead)
        if plotter is not None and move_count % plot_update_frequency == 0:
            # Record mode: just collects data, no rendering
            plotter.update(selected_tube, selected_sigma, actual_p, current_pos, 
                         episode=0, step=move_count, goal_pos=target_pos)
        
        # Check if goal is reached - if so, log goal statistics and generate new goal
        # Optimized: compute distance once and reuse
        goal_distance = torch.norm(current_pos.squeeze() - target_pos).item()
        if goal_distance < 1.0:
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
            
            # Log goal statistics
            goal_stats = {
                "goal_number": len(training_data.get("goals_data", [])) + 1,
                "start_position": goal_start_pos.squeeze().tolist(),
                "goal_position": target_pos.tolist(),
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
            
            # Store in training data
            if "goals_data" not in training_data:
                training_data["goals_data"] = []
            training_data["goals_data"].append(goal_stats)
            
            # Record goal reached (will be visualized during replay)
            # Note: mark_goal_reached is for live mode, we'll handle this in replay
            
            # Generate a new random goal (avoiding obstacles, current position, and respecting boundaries)
            margin = goal_gen_config.get("margin", 0.5)
            min_bounds = [b + margin for b in bounds['min']]
            max_bounds = [b - margin for b in bounds['max']]
            
            new_goal = torch.tensor([
                random.uniform(min_bounds[i], max_bounds[i]) for i in range(state_dim)
            ])
            
            # Ensure new goal is not too close to current position or obstacles
            min_dist_from_current = goal_gen_config.get("min_dist_from_current", 3.0)
            min_dist_from_obstacles = goal_gen_config.get("min_dist_from_obstacles", 1.0)
            max_attempts = goal_gen_config.get("max_attempts", 50)
            attempts = 0
            while (torch.norm(new_goal - current_pos.squeeze()) < min_dist_from_current or
                   any(torch.norm(new_goal - torch.tensor(obs[0])) < obs[1] + min_dist_from_obstacles 
                       for obs in obstacles)) and attempts < max_attempts:
                new_goal = torch.tensor([
                    random.uniform(min_bounds[i], max_bounds[i]) for i in range(state_dim)
                ])
                attempts += 1
            
            # Reset goal tracking for new goal
            target_pos = new_goal
            goal_start_pos = current_pos.clone()
            goal_start_move = move_count
            moves_to_current_goal = 0
            segment_lengths = []
            agency_values = []
            
            # Goal position updates are recorded in plotter.update() calls
            # No need to update visualization during training (record mode)
            
            # Print goal reached summary
            print(f"Move {move_count}/{total_moves} | Goal reached! | Moves: {goal_stats['moves_taken']} | "
                  f"Segments: {seg_mean:.3f}±{seg_std:.3f} | Agency: {agency_mean:.3f}±{agency_std:.3f}")
    
    # Flush any remaining buffered tkn logs
    if len(tkn_log_buffer) > 0:
        with open(tkn_log_file, 'a') as f:
            for entry in tkn_log_buffer:
                json.dump(entry, f)
                f.write('\n')
        tkn_log_buffer.clear()
    
    # Save training data to file
    with open(session_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\nTraining complete. Total moves: {move_count}")
    print(f"Session data saved to: {session_file}")
    print(f"Goal statistics saved to: {goal_stats_file}")
    
    # Save tkn statistics
    tkn_stats_file = os.path.join(run_dir, f"tkn_stats_{session_timestamp}.json")
    tkn_stats = {
        "total_novel_patterns": len(tkn_processor.lattice.novel_patterns),
        "head_names": tkn_processor.head_names,
        "vocab_size": tkn_processor.lattice.vocab_size,
        "novel_patterns_sample": list(tkn_processor.lattice.novel_patterns)[:20]  # Sample of novel patterns
    }
    with open(tkn_stats_file, 'w') as f:
        json.dump(tkn_stats, f, indent=2)
    print(f"TKN statistics saved to: {tkn_stats_file}")
    print(f"Total novel patterns discovered: {tkn_stats['total_novel_patterns']}")
    
    # Generate training progress plots
    progress_plot_file = os.path.join(run_dir, f"training_progress_{session_timestamp}.png")
    summary_file = os.path.join(run_dir, f"training_summary_{session_timestamp}.json")
    
    try:
        plot_training_progress(goal_stats_file, loss_history=loss_history, output_path=progress_plot_file)
        summary = generate_training_summary(goal_stats_file, output_path=summary_file)
        
        # Print summary statistics
        if summary:
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print(f"{'='*60}")
            print(f"Total Goals Reached: {summary['total_goals']}")
            print(f"Total Moves: {summary['total_moves']}")
            print(f"\nMoves per Goal:")
            print(f"  Overall: {summary['moves_per_goal']['overall']['mean']:.2f} ± {summary['moves_per_goal']['overall']['std']:.2f}")
            if summary['moves_per_goal']['early_phase']['mean']:
                print(f"  Early:   {summary['moves_per_goal']['early_phase']['mean']:.2f} ± {summary['moves_per_goal']['early_phase']['std']:.2f}")
            if summary['moves_per_goal']['late_phase']['mean']:
                print(f"  Late:    {summary['moves_per_goal']['late_phase']['mean']:.2f} ± {summary['moves_per_goal']['late_phase']['std']:.2f}")
            if summary['improvement']['moves_reduction']:
                print(f"  Improvement: {summary['improvement']['moves_reduction']:.2f} moves reduction")
            print(f"\nAgency (σ):")
            print(f"  Overall: {summary['agency']['overall_mean']:.3f} ± {summary['agency']['overall_std']:.3f}")
            if summary['agency']['early_mean'] and summary['agency']['late_mean']:
                print(f"  Early:   {summary['agency']['early_mean']:.3f}")
                print(f"  Late:   {summary['agency']['late_mean']:.3f}")
                if summary['improvement']['agency_change']:
                    print(f"  Change: {summary['improvement']['agency_change']:.3f} ({'↓' if summary['improvement']['agency_change'] > 0 else '↑'} more confident)")
            print(f"{'='*60}")
    except Exception as e:
        print(f"Warning: Could not generate training plots: {e}")
    
    print(f"\nAll run files saved to: {run_dir}")
    print(f"  - Training session: {session_file}")
    print(f"  - Goal statistics: {goal_stats_file}")
    print(f"  - TKN patterns: {tkn_log_file}")
    print(f"  - TKN statistics: {tkn_stats_file}")
    if 'progress_plot_file' in locals():
        print(f"  - Training progress plot: {progress_plot_file}")
    if 'summary_file' in locals():
        print(f"  - Training summary: {summary_file}")
    
    # Replay visualization after training completes
    if plotter is not None and plot_live:
        print("\n" + "="*60)
        print("Starting visualization replay...")
        print(f"Replay speed: {1.0/replay_frame_delay:.1f} FPS (frame_delay={replay_frame_delay}s)")
        print("="*60)
        plotter.replay(frame_delay=replay_frame_delay)
