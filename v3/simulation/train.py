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
from v3.tokenizer import TknProcessor
from v3.simulation.analysis import plot_training_progress, generate_training_summary
from v3.simulation.environment import generate_obstacles

def train_actor(inference_engine, actor, total_moves=500, plot_live=True, max_observed_obstacles=10, 
                latent_dim=64, state_dim=3, max_visualization_history=5):
    """
    Train the InferenceEngine and Actor together in the environment.
    
    This implements dual-optimization: both models learn simultaneously.
    - InferenceEngine learns to represent context such that Actor can successfully bind
    - Actor learns to use that representation to propose tubes
    
    Training runs for a fixed number of moves, continuously generating new goals when reached.
    Visualization shows only the last N moves (configurable via max_visualization_history).
    
    Args:
        inference_engine: HybridInferenceEngine model (GRU-based World Model with tokenized inputs)
        actor: Actor model (operates on latent situations)
        total_moves: Total number of moves to execute (not episodes)
        plot_live: Whether to show live visualization
        max_observed_obstacles: Maximum number of obstacles in observation (fixed size)
        latent_dim: Dimension of latent situation space
        max_visualization_history: Maximum number of moves to display in visualization (default: 5)
    """
    # Combine parameters for joint optimization
    optimizer = optim.Adam(
        list(inference_engine.parameters()) + list(actor.parameters()), 
        lr=1e-3
    )
    # Define environment boundaries
    bounds = {
        'min': [-2.0] * state_dim,
        'max': [12.0] * state_dim
    }
    
    # Generate deterministic obstacle layout
    # Parameters can be tuned for difficulty:
    # - num_obstacles: More = harder
    # - packing_threshold: Higher = more spread out (easier navigation)
    # - min_open_path_width: Higher = wider corridors (easier navigation)
    # - seed: Change for different layouts (but same seed = same layout)
    obstacle_seed = 42
    obstacles = generate_obstacles(
        state_dim=state_dim,
        bounds=bounds,
        num_obstacles=12,  # More obstacles for challenge
        min_radius=0.8,
        max_radius=2.2,
        packing_threshold=0.5,  # 50% gap between obstacles (ensures spacing)
        min_open_path_width=2.5,  # Minimum 2.5 unit wide corridors
        seed=obstacle_seed
    )
    
    print(f"Generated {len(obstacles)} obstacles (seed={obstacle_seed}, deterministic)")
    
    sim = SimulationWrapper(obstacles, state_dim=state_dim, bounds=bounds)
    # Initialize target position with state_dim dimensions
    target_pos = torch.tensor([10.0] * state_dim)
    
    # Calculate raw context dimension: state_dim (rel_goal) + max_observed_obstacles * (state_dim + 1)
    # This is FIXED regardless of actual obstacle count (uses padding/truncation)
    raw_ctx_dim = state_dim + max_observed_obstacles * (state_dim + 1)
    
    # Create session directory and file
    artifacts_dir = "/Users/zach/Documents/dev/cfd/tam/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory for all dump files
    run_dir = os.path.join(artifacts_dir, f"run_{session_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    session_file = os.path.join(run_dir, f"training_session_{session_timestamp}.json")
    goal_stats_file = os.path.join(run_dir, f"goal_stats_{session_timestamp}.jsonl")
    
    # Initialize tkn processor (required for HybridInferenceEngine)
    vocab_size = inference_engine.vocab_size
    tkn_processor = TknProcessor(state_dim=state_dim, quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=vocab_size)
    tkn_log_file = os.path.join(run_dir, f"tkn_patterns_{session_timestamp}.jsonl")
    print(f"TKN enabled: Pattern discovery logging to {tkn_log_file}")
    print(f"  - {len(tkn_processor.head_names)} heads: {', '.join(tkn_processor.head_names)}")
    print(f"  - Vocab size: {tkn_processor.lattice.vocab_size}")
    print(f"  - Using HybridInferenceEngine: tokens + traits + intent")
    
    # Verify models are compatible
    expected_heads = len(tkn_processor.head_names)
    if inference_engine.num_heads != expected_heads:
        raise ValueError(f"InferenceEngine num_heads mismatch: expected {expected_heads}, "
                       f"but model has {inference_engine.num_heads}")
    
    if inference_engine.latent_dim != latent_dim:
        raise ValueError(f"InferenceEngine latent_dim mismatch: expected {latent_dim}, "
                        f"but model has {inference_engine.latent_dim}")
    
    if hasattr(actor, 'encoder'):
        # Check the first layer input size (should be latent_dim + intent_dim)
        first_layer = list(actor.encoder.children())[0]
        if isinstance(first_layer, nn.Linear):
            expected_input = first_layer.in_features - state_dim  # Subtract intent_dim (which equals state_dim)
            if expected_input != latent_dim:
                raise ValueError(f"Actor latent_dim mismatch: expected {latent_dim}, "
                               f"but actor expects {expected_input}. "
                               f"Initialize Actor with latent_dim={latent_dim}")
    
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
    }
    
    # Initialize live plotter
    plotter = None
    if plot_live:
        plotter = LivePlotter(obstacles, target_pos.numpy(), bounds=bounds, max_history=max_visualization_history)
    
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
    tkn_log_batch_size = 20  # Write tkn logs every N moves
    plot_update_frequency = 1  # Update plot every N moves (1 = every move for better visualization)
    
    # Track loss history for analysis
    loss_history = []  # Track loss values over time
    
    # Main training loop: run for total_moves
    while move_count < total_moves:
        # 1. INFER: Convert raw world data to latent situation
        raw_obs = sim.get_raw_observation(current_pos, target_pos, max_obstacles=max_observed_obstacles)  # (raw_ctx_dim,)
        raw_obs = raw_obs.unsqueeze(0)  # (1, raw_ctx_dim) for batch dimension
        
        # Process through tkn (always enabled)
        tkn_output = tkn_processor.process_observation(
            current_pos, raw_obs.squeeze(0), obstacles
        )
        lattice_tokens = tkn_output["lattice_tokens"].unsqueeze(0)  # (1, num_heads)
        surprise_mask = tkn_output["surprise_mask"]  # (num_heads,)
        lattice_traits = tkn_output["lattice_traits"].unsqueeze(0)  # (1, num_heads, 2)
        rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, state_dim) high-fidelity intent
        
        # Buffer tkn logs for batch writing (much faster than writing every move)
        log_entry = {
            "move": move_count,
            "lattice_tokens": lattice_tokens.squeeze(0).tolist(),
            "surprise_mask": surprise_mask.tolist(),
            "lattice_traits": lattice_traits.squeeze(0).tolist(),
            "buffer_hashes": tkn_output["buffer_hashes"].tolist(),
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
        
        # Use HybridInferenceEngine: tokens + traits + intent
        x_n = inference_engine(lattice_tokens, lattice_traits, rel_goal_tensor, h_state)  # (1, latent_dim)
        
        h_state = x_n.detach()  # Pass memory forward (detach to prevent backprop through time)
        
        rel_goal = target_pos - current_pos 
        
        # 2. ACT: Propose affordances based on latent situation x_n
        # Actor operates on latent representation, not raw spatial features
        # New: Actor returns basis_weights instead of knot_steps, and per-dimension sigma
        logits, mu_t, sigma_t, knot_mask, basis_weights = actor(x_n, rel_goal, previous_velocity=previous_velocity)
        
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
        
        # 3. KNOT-THEORETIC BINDING LOSS: Topological binding with per-dimension precision
        # This implements the "fibration binding" concept: verify the section stays within the fiber
        
        # A) CONTRADICTION: Did we stay in the tube? (Topological Binding)
        # Use per-dimension sigmas for anisotropic affordance tubes
        min_len = min(len(actual_p), len(selected_tube))
        if min_len > 0:
            expected_p_slice = selected_tube[:min_len] + current_pos.squeeze()  # (T, state_dim)
            actual_p_slice = actual_p[:min_len]  # (T, state_dim)
            
            # Per-dimension deviation
            deviation_per_dim = (actual_p_slice - expected_p_slice)  # (T, state_dim)
            
            # Weight by per-dimension precision (sigma)
            # Higher sigma = more tolerance in that dimension
            # Binding loss: weighted squared deviation per dimension
            sigma_slice = selected_sigma[:min_len]  # (T, state_dim)
            weighted_deviation = (deviation_per_dim**2) / (sigma_slice + 1e-6)  # (T, state_dim)
            
            # Sum across dimensions, then average over time
            binding_loss = torch.mean(weighted_deviation.sum(dim=-1))
        else:
            binding_loss = torch.tensor(0.0, device=selected_tube.device)
            
        # B) AGENCY COST: How 'expensive' was this path?
        # Penalize basis function weights (complexity of the section)
        selected_basis_weights = basis_weights[0, idx_int]  # (n_basis,)
        path_cost = torch.mean(selected_basis_weights**2)  # Penalize large basis weights
        
        # Penalize uncertainty (narrower tubes are better)
        # MODULATE WITH SURPRISE: Novel patterns increase sigma (expand tube, move cautiously)
        # Per-dimension uncertainty cost
        base_uncertainty_cost = torch.mean(selected_sigma**2)  # Mean across all dimensions
        
        if surprise_mask is not None:
            # Surprise mechanism: Novel patterns -> higher sigma -> more cautious
            surprise_factor = 1.0 + 0.3 * surprise_mask.float().mean().item()  # 1.0 to 1.3 multiplier
            # Note: We don't directly modify sigma here, but we could scale the cost
            uncertainty_cost = base_uncertainty_cost * surprise_factor
        else:
            uncertainty_cost = base_uncertainty_cost
        
        # C) INTENT LOSS: Direct supervision to move toward goal
        # Optimized: avoid unnecessary unsqueeze operations
        tube_endpoint_global = selected_tube[-1] + current_pos.squeeze()
        intent_loss = F.mse_loss(tube_endpoint_global, target_pos)
        
        # Simplified Principled Loss
        loss = (1.0 * binding_loss          # Contradiction: stay in the tube
               + 0.1 * path_cost            # Agency: minimize displacement (residual offsets)
               + 0.05 * uncertainty_cost    # Agency: minimize uncertainty (narrower tubes)
               + 0.5 * intent_loss)         # Intent: move toward goal
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss for analysis
        loss_history.append(float(loss.item()))
        
        # Update position and track velocity for G1 continuity
        current_pos = actual_p[-1].view(1, state_dim).detach()
        move_count += 1
        moves_to_current_goal += 1
        
        # Track velocity: direction of actual path taken (for G1 continuity)
        # Use the last segment of the actual path to capture real movement direction
        if len(actual_p) > 1:
            # Use last segment of actual path (what really happened)
            previous_velocity = (actual_p[-1] - actual_p[-2]).detach()  # (state_dim,)
        elif len(actual_p) == 1 and previous_velocity is None:
            # First step: no previous velocity, use a small default direction
            # This will be updated on next step
            previous_velocity = torch.zeros(state_dim, device=current_pos.device)
        # If path is only one point and we have previous velocity, keep it
            
        # Update live plot (only show last N moves)
        # Reduce update frequency for performance (update every N moves)
        if plotter is not None and move_count % plot_update_frequency == 0:
            # Use a fixed episode number (0) so we don't clear on every move
            # The max_history logic in LivePlotter will keep only the last N moves
            plotter.update(selected_tube, selected_sigma, actual_p, current_pos, 
                         episode=0, step=move_count)
        
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
            
            # Mark the reached goal in visualization
            if plotter is not None:
                plotter.mark_goal_reached(target_pos)
            
            # Generate a new random goal (avoiding obstacles, current position, and respecting boundaries)
            margin = 0.5
            min_bounds = [b + margin for b in bounds['min']]
            max_bounds = [b - margin for b in bounds['max']]
            
            new_goal = torch.tensor([
                random.uniform(min_bounds[i], max_bounds[i]) for i in range(state_dim)
            ])
            
            # Ensure new goal is not too close to current position or obstacles
            min_dist = 3.0
            max_attempts = 50
            attempts = 0
            while (torch.norm(new_goal - current_pos.squeeze()) < min_dist or
                   any(torch.norm(new_goal - torch.tensor(obs[0])) < obs[1] + 1.0 
                       for obs in obstacles)) and attempts < max_attempts:
                new_goal = torch.tensor([
                    random.uniform(min_bounds[0], max_bounds[0]),
                    random.uniform(min_bounds[1], max_bounds[1]),
                    random.uniform(min_bounds[2], max_bounds[2])
                ])
                attempts += 1
            
            # Reset goal tracking for new goal
            target_pos = new_goal
            goal_start_pos = current_pos.clone()
            goal_start_move = move_count
            moves_to_current_goal = 0
            segment_lengths = []
            agency_values = []
            
            # Update plotter with new goal
            if plotter is not None:
                plotter.target_pos = target_pos.detach().numpy() if isinstance(target_pos, torch.Tensor) else target_pos
                if plotter.target_marker is not None:
                    plotter.target_marker.remove()
                plotter.target_marker = plotter.ax.scatter(
                    target_pos[0].item(), target_pos[1].item(), target_pos[2].item(),
                    color=plotter.COLORS['goal'], s=300, marker='*', 
                    edgecolors='#D97706', linewidths=1.5, label='Target', zorder=10
                )
                plotter.ax.legend()
                plt.draw()
            
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
    
    if plotter is not None:
        print("Close the plot window to exit.")
        plt.ioff()
    plt.show()
