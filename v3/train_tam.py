"""
Generic TAM training loop that works with any System + Environment.

This implements the core TAM cycle abstractly, allowing any environment
that can respond to port binding with context episodes.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any
from v3.system import TAMSystem
from v3.environment import Environment
from v3.stats_recorder import StatsRecorder
from v3.environment_recorder import EnvironmentRecorder


def train_tam_system(
    system: TAMSystem,
    environment: Environment,
    total_moves: int = 500,
    learning_rate: float = 1e-3,
    loss_weights: Optional[Dict[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    goal_reached_threshold: float = 0.5,
    energy_replenish_amount: float = 20.0,
    config: Optional[Dict[str, Any]] = None,
    stats_recorder: Optional[StatsRecorder] = None,
    environment_recorder: Optional[EnvironmentRecorder] = None
):
    """
    Generic TAM training loop that works with any System + Environment.
    
    This implements the core TAM cycle:
    1. Get context (prior context for port affordance)
    2. Infer situation
    3. Propose ports
    4. Select and bind port
    5. World responds with context episode
    6. Evaluate binding and update
    7. Update state (next situation)
    
    Args:
        system: TAMSystem instance (implements infer_situation, propose_ports, evaluate_binding)
        environment: Environment instance (implements get_context, bind_port, get_intent)
        total_moves: Total number of moves to train
        learning_rate: Learning rate for optimizer
        loss_weights: Dict with loss component weights (binding_loss, agency_cost, etc.)
        optimizer: Optional optimizer (if None, creates Adam optimizer)
        goal_reached_threshold: Distance threshold for considering goal reached
        energy_replenish_amount: Energy to replenish when goal reached (if environment supports it)
        config: Optional configuration dict (for compatibility with existing code)
        stats_recorder: Optional StatsRecorder instance for recording training statistics
        environment_recorder: Optional EnvironmentRecorder instance for recording environment state
    """
    if loss_weights is None:
        loss_weights = {
            "binding_loss": 1.0,
            "agency_cost": 0.1,
            "efficiency_reward": 0.05,  # Small reward for longer segments and goal efficiency (scaled down)
            "collision_penalty": 1.0,  # Penalty for collisions to balance efficiency
        }
    
    # Create optimizer if not provided
    if optimizer is None:
        # Get trainable parameters from system
        # Note: This assumes system components are PyTorch modules
        params = []
        if hasattr(system, 'inference_engine'):
            params.extend(system.inference_engine.parameters())
        if hasattr(system, 'actor'):
            params.extend(system.actor.parameters())
        if len(params) == 0:
            raise ValueError("System must have trainable parameters (inference_engine and/or actor)")
        
        optimizer = optim.Adam(params, lr=learning_rate)
    
    # Initialize system state
    system.reset()
    current_state = environment.get_initial_state()
    
    # Ensure current_state has correct shape
    if current_state.dim() == 1:
        current_state = current_state.unsqueeze(0)  # (1, state_dim)
    
    # Apply initial state to environment (for environment recorder)
    if environment_recorder is not None:
        environment.apply(current_state.squeeze(0), env_recorder=environment_recorder)
    
    # Initialize previous situation (zeros for first step)
    previous_situation = torch.zeros(1, system.latent_dim)
    
    # Track active goals (if environment supports them)
    active_goals = []
    if hasattr(environment, 'active_goals'):
        active_goals = environment.active_goals
        environment.active_goals = active_goals
    
    # Check if infer_situation accepts current_state parameter (once, not every iteration)
    import inspect
    infer_situation_accepts_current_state = False
    if hasattr(system, 'infer_situation'):
        try:
            sig = inspect.signature(system.infer_situation)
            infer_situation_accepts_current_state = 'current_state' in sig.parameters
        except Exception as e:
            print(f"Warning: Could not inspect infer_situation signature: {e}")
            # Default to trying with current_state
            infer_situation_accepts_current_state = True
    
    # Main training loop
    move_count = 0
    
    print("Entering training loop...")
    
    while move_count < total_moves:
        try:
            # 1. Get context (prior context for port affordance evaluation)
            # CRITICAL: Always use environment.active_goals to ensure we see the latest state
            # (goals may have been removed in previous move's apply() calls)
            if hasattr(environment, 'active_goals'):
                current_active_goals = environment.active_goals
            else:
                current_active_goals = active_goals
            
            intent = environment.get_intent(current_state.squeeze(0), active_goals=current_active_goals)
            # get_context() uses environment.active_goals directly, so it will see updated goals
            context = environment.get_context(current_state.squeeze(0), intent)
            
            # 2. Infer situation (through tkn + inference engine)
            # Note: infer_situation may need current_state for tokenization
            # Ensure previous_situation is detached to prevent graph reuse
            previous_situation_detached = previous_situation.detach() if hasattr(previous_situation, 'detach') else previous_situation
            
            if infer_situation_accepts_current_state:
                situation = system.infer_situation(
                    context, 
                    previous_situation_detached, 
                    current_state=current_state.squeeze(0)
                )
            else:
                situation = system.infer_situation(context, previous_situation_detached)
            
            # Ensure situation has batch dimension
            if situation.dim() == 1:
                situation = situation.unsqueeze(0)  # (1, latent_dim)
            
            # 3. Propose ports
            try:
                logits, tubes, sigmas, knot_mask, basis_weights = system.propose_ports(
                    situation.squeeze(0), 
                    intent
                )
            except Exception as e:
                print(f"Error in propose_ports at move {move_count}: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Ensure batch dimensions
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # (1, n_ports)
            if tubes.dim() == 3:
                tubes = tubes.unsqueeze(0)  # (1, n_ports, T, state_dim)
            if sigmas.dim() == 3:
                sigmas = sigmas.unsqueeze(0)  # (1, n_ports, T, state_dim)
            
            # Ensure intent has batch dimension
            if intent.dim() == 1:
                intent = intent.unsqueeze(0)  # (1, state_dim)
            
            # 4. Compute intent-aligned port selection scores
            # Extract tube endpoints (final positions, relative to current_pos)
            tube_endpoints = tubes[:, :, -1, :]  # (B, n_ports, state_dim)
            
            # Compute intent alignment using cosine similarity
            intent_norm = torch.norm(intent, dim=-1, keepdim=True)  # (B, 1)
            device = intent.device
            B = intent.shape[0]
            n_ports = tube_endpoints.shape[1]
            
            # Initialize intent alignment (will be computed for non-zero intents)
            intent_alignment = torch.zeros(B, n_ports, device=device)
            
            # Check which samples have non-zero intent (handle per-sample)
            non_zero_intent_mask = intent_norm.squeeze(-1) > 1e-6  # (B,)
            
            if non_zero_intent_mask.any():
                # Normalize intent direction for non-zero intents
                intent_normalized = intent / (intent_norm + 1e-6)  # (B, state_dim)
                
                # Normalize tube endpoint directions
                tube_endpoint_norms = torch.norm(tube_endpoints, dim=-1, keepdim=True)  # (B, n_ports, 1)
                tube_endpoints_normalized = tube_endpoints / (tube_endpoint_norms + 1e-6)  # (B, n_ports, state_dim)
                
                # Handle zero-length tubes (no movement)
                zero_tube_mask = tube_endpoint_norms.squeeze(-1) < 1e-6  # (B, n_ports)
                
                # Compute cosine similarity: dot product of normalized vectors
                intent_expanded = intent_normalized.unsqueeze(1)  # (B, 1, state_dim)
                cosine_similarity = torch.sum(tube_endpoints_normalized * intent_expanded, dim=-1)  # (B, n_ports)
                
                # Set alignment to 0 for zero-length tubes
                cosine_similarity[zero_tube_mask] = 0.0
                
                # Only set alignment for samples with non-zero intent
                intent_alignment[non_zero_intent_mask] = cosine_similarity[non_zero_intent_mask]
            
            # Compute agency score (negative mean sigma² - tighter cones = higher agency)
            agency_score = -torch.mean(sigmas**2, dim=[-2, -1])  # (B, n_ports)
            # Normalize to similar scale as logits (typical sigma ~0.5-2.0, so sigma² ~0.25-4.0)
            # Divide by 10 to bring to similar scale as logits
            agency_score = agency_score / 10.0
            
            # Get learnable weights from actor
            if hasattr(system, 'actor') and hasattr(system.actor, 'intent_bias_weight'):
                intent_bias_weight = system.actor.intent_bias_weight
                agency_bias_weight = system.actor.agency_bias_weight
            else:
                # Fallback to default values if actor doesn't have learnable weights
                intent_bias_weight = torch.tensor(2.0, device=device)
                agency_bias_weight = torch.tensor(1.0, device=device)
            
            # Combine scores: logits + intent alignment + agency
            combined_score = logits + intent_bias_weight * intent_alignment + agency_bias_weight * agency_score
            
            # 5. Select and bind port using combined score
            # Selection (Categorical sampling for exploration)
            probs = F.softmax(combined_score, dim=-1)
            m = torch.distributions.Categorical(probs)
            selected_port_idx = m.sample()
            
            # Extract selected tube
            selected_port_idx_int = selected_port_idx.item() if isinstance(selected_port_idx, torch.Tensor) else selected_port_idx
            selected_tube = tubes[0, selected_port_idx_int]  # (T, state_dim)
            selected_sigma = sigmas[0, selected_port_idx_int]  # (T, state_dim)
            
            # Extract selected knot_mask and basis_weights for stats recording
            selected_knot_mask = None
            selected_basis_weights = None
            if knot_mask is not None:
                selected_knot_mask = knot_mask[0, selected_port_idx_int]  # (K,)
            if basis_weights is not None:
                selected_basis_weights = basis_weights[0, selected_port_idx_int]  # (n_basis,)
            
            # 5. World responds with context episode
            # bind_port computes physics and returns path, but doesn't update environment
            context_episode = environment.bind_port(
                selected_tube, 
                selected_sigma, 
                current_state.squeeze(0)
            )
            
            # Calculate obstacle metrics
            # Get obstacles from environment
            obstacles = getattr(environment, 'obstacles', [])
            
            # Calculate minimum distance to nearest obstacle at start of move
            obstacle_proximity = float('inf')
            if len(obstacles) > 0:
                current_state_np = current_state.squeeze(0).detach().cpu().numpy()
                if isinstance(current_state_np, torch.Tensor):
                    current_state_np = current_state_np.numpy()
                for obs_pos, obs_r in obstacles:
                    obs_pos_np = np.array(obs_pos)
                    dist = np.linalg.norm(current_state_np - obs_pos_np) - obs_r  # Distance to obstacle surface
                    obstacle_proximity = min(obstacle_proximity, dist)
            
            # Detect collisions in actual path
            obstacle_collision = False
            if len(obstacles) > 0 and len(context_episode) > 0:
                for point in context_episode:
                    point_np = point.detach().cpu().numpy() if isinstance(point, torch.Tensor) else np.array(point)
                    for obs_pos, obs_r in obstacles:
                        obs_pos_np = np.array(obs_pos)
                        dist = np.linalg.norm(point_np - obs_pos_np)
                        if dist < obs_r + 0.1:  # Collision threshold
                            obstacle_collision = True
                            break
                    if obstacle_collision:
                        break
            
            
            # Apply each state in the path to environment (for environment recorder)
            # This allows the recorder to track at environment refresh rate
            # Track total goals reached across all steps in this move
            goals_reached_this_move = 0
            if environment_recorder is not None:
                # Set move number for this move's steps
                environment_recorder.set_move_number(move_count + 1)
                for step_idx, state in enumerate(context_episode):
                    goals_reached_this_step = environment.apply(state, env_recorder=environment_recorder)
                    goals_reached_this_move += goals_reached_this_step
            else:
                # Still track goals even if no environment recorder
                for step_idx, state in enumerate(context_episode):
                    goals_reached_this_step = environment.apply(state, env_recorder=None)
                    goals_reached_this_move += goals_reached_this_step
            
            # 6. Evaluate binding and update
            # Compute binding loss
            try:
                binding_loss = system.evaluate_binding(
                    selected_tube,
                    context_episode,
                    selected_sigma,
                    current_state=current_state.squeeze(0)
                )
            except Exception as e:
                print(f"Error in evaluate_binding: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Compute agency cost and efficiency reward
            # Get intent target (goal position) for efficiency reward
            intent_target = None
            if hasattr(environment, 'active_goals') and len(environment.active_goals) > 0:
                # Use nearest goal as intent target
                current_state_np = current_state.squeeze(0).detach().cpu().numpy()
                if isinstance(current_state_np, torch.Tensor):
                    current_state_np = current_state_np.numpy()
                
                # Find nearest goal
                min_dist = float('inf')
                nearest_goal = None
                for goal_pos in environment.active_goals:
                    goal_pos_np = np.array(goal_pos)
                    dist = np.linalg.norm(current_state_np - goal_pos_np)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_goal = goal_pos_np
                
                if nearest_goal is not None:
                    intent_target = torch.tensor(nearest_goal, dtype=torch.float32, device=binding_loss.device)
            
            # Compute agency cost and efficiency reward
            agency_cost = torch.tensor(0.0, device=binding_loss.device)
            efficiency_reward = torch.tensor(0.0, device=binding_loss.device)
            
            if hasattr(system, 'actor') and hasattr(system.actor, 'compute_binding_loss'):
                try:
                    _, agency_cost, efficiency_reward = system.actor.compute_binding_loss(
                        selected_tube,
                        context_episode,
                        selected_sigma,
                        current_state.squeeze(0),
                        knot_mask=selected_knot_mask,
                        intent_target=intent_target
                    )
                except Exception as e:
                    # If compute_binding_loss doesn't return efficiency_reward (old signature), handle gracefully
                    try:
                        result = system.actor.compute_binding_loss(
                            selected_tube,
                            context_episode,
                            selected_sigma,
                            current_state.squeeze(0),
                            knot_mask=selected_knot_mask,
                            intent_target=intent_target
                        )
                        if len(result) == 2:
                            # Old signature: only binding_loss and agency_cost
                            _, agency_cost = result
                            efficiency_reward = torch.tensor(0.0, device=binding_loss.device)
                        elif len(result) == 3:
                            # New signature: binding_loss, agency_cost, efficiency_reward
                            _, agency_cost, efficiency_reward = result
                    except Exception as e2:
                        # Continue with zero costs if computation fails
                        agency_cost = torch.tensor(0.0, device=binding_loss.device)
                        efficiency_reward = torch.tensor(0.0, device=binding_loss.device)
            
            # Calculate collision penalty (if obstacle collision detected)
            collision_penalty = torch.tensor(0.0, device=binding_loss.device)
            if obstacle_collision:
                # Penalize collisions significantly to balance efficiency reward
                collision_penalty = torch.tensor(10.0, device=binding_loss.device)
            
            # Total loss: binding_loss + agency_cost - efficiency_reward + collision_penalty
            # efficiency_reward is positive (reward), so we subtract it from loss
            # collision_penalty is positive (cost), so we add it to loss
            loss = (loss_weights.get("binding_loss", 1.0) * binding_loss +
                    loss_weights.get("agency_cost", 0.1) * agency_cost -
                    loss_weights.get("efficiency_reward", 0.2) * efficiency_reward +  # Increased from 0.05
                    loss_weights.get("collision_penalty", 1.0) * collision_penalty)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if hasattr(system, 'inference_engine'):
                torch.nn.utils.clip_grad_norm_(system.inference_engine.parameters(), max_norm=1.0)
            if hasattr(system, 'actor'):
                torch.nn.utils.clip_grad_norm_(system.actor.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Detach and clone for next iteration (prevent graph accumulation)
            situation = situation.detach()
            previous_situation = situation.clone()  # Clone to ensure complete independence
            
            # 7. Update state (next situation)
            # Update current_state to last position in context_episode
            # IMPORTANT: Detach to prevent graph accumulation across iterations
            if len(context_episode) > 0:
                current_state = context_episode[-1].detach().view(1, -1)  # (1, state_dim)
            else:
                # Fallback: use current_state (shouldn't happen, but handle gracefully)
                # Still detach to be safe
                if hasattr(current_state, 'detach'):
                    current_state = current_state.detach()
            
            move_count += 1
            
            # Goals are already tracked during apply() calls above
            # Replenish energy if goals were reached (if environment supports it)
            if goals_reached_this_move > 0 and hasattr(environment, 'replenish_energy'):
                environment.replenish_energy(energy_replenish_amount * goals_reached_this_move)
            
            # CRITICAL: Sync active_goals with environment after apply() calls
            # This ensures get_context() and get_intent() see the updated goals on the next move
            if hasattr(environment, 'active_goals'):
                active_goals = environment.active_goals
            
            # Record statistics if stats_recorder is provided
            if stats_recorder is not None:
                # Extract knot components
                knot_components = {}
                knot_components["selected_port"] = selected_port_idx_int
                
                if selected_knot_mask is not None:
                    # Count active knots using epsilon threshold (where mask > epsilon)
                    # This counts knots with meaningful contribution, not just binary threshold
                    epsilon = 1e-3  # Small threshold for "active" knots
                    num_active_knots = (selected_knot_mask > epsilon).sum().item()
                    knot_components["num_active_knots"] = num_active_knots
                    knot_components["knot_mask"] = selected_knot_mask.detach().cpu().tolist()
                    # Also track mean mask value to see overall knot usage
                    knot_components["knot_mask_mean"] = float(selected_knot_mask.mean().item())
                
                if selected_basis_weights is not None:
                    knot_components["basis_weights"] = selected_basis_weights.detach().cpu().tolist()
                
                # Calculate segment lengths between consecutive points in the tube trajectory
                # Segment length = distance between consecutive knots/points in the trajectory
                if len(selected_tube) > 1:
                    tube_segments = selected_tube[1:] - selected_tube[:-1]  # (T-1, state_dim)
                    segment_lengths = torch.norm(tube_segments, dim=-1).detach().cpu()  # (T-1,)
                    
                    # Calculate statistics for segment lengths
                    if len(segment_lengths) > 0:
                        knot_components["segment_length_min"] = float(segment_lengths.min().item())
                        knot_components["segment_length_max"] = float(segment_lengths.max().item())
                        knot_components["segment_length_mean"] = float(segment_lengths.mean().item())
                        knot_components["segment_length_std"] = float(segment_lengths.std().item())
                
                # Add obstacle metrics to knot_components
                knot_components["obstacle_proximity"] = float(obstacle_proximity) if obstacle_proximity != float('inf') else 100.0
                knot_components["obstacle_collision"] = 1 if obstacle_collision else 0
                
                # Bin proximity for conditional analysis
                proximity = knot_components["obstacle_proximity"]
                if proximity < 0.5:
                    knot_components["proximity_bin"] = "0-0.5"
                elif proximity < 1.0:
                    knot_components["proximity_bin"] = "0.5-1.0"
                elif proximity < 2.0:
                    knot_components["proximity_bin"] = "1.0-2.0"
                else:
                    knot_components["proximity_bin"] = "2.0+"
                
                # Record move statistics
                stats_recorder.record_move(
                    move=move_count,  # For internal tracking only
                    binding_loss=binding_loss,
                    agency_cost=agency_cost,
                    knot_components=knot_components if knot_components else None,
                    mu_t=selected_tube,  # Store selected tube trajectory
                    sigma_t=selected_sigma,  # Store selected sigma
                    goals_reached=goals_reached_this_move  # Track goals reached
                )
            
            # Handle energy depletion (if environment supports it)
            if hasattr(environment, 'is_dead') and environment.is_dead:
                # Reset position and partial energy
                if hasattr(environment, 'current_energy') and hasattr(environment, 'initial_energy'):
                    environment.current_energy = environment.initial_energy * 0.3
                    environment.is_dead = False
                current_state = environment.get_initial_state()
                if current_state.dim() == 1:
                    current_state = current_state.unsqueeze(0)
                
                # Reset system state
                if hasattr(system, 'reset'):
                    system.reset()
                previous_situation = torch.zeros(1, system.latent_dim)
            
            # Print progress periodically
            if move_count % 100 == 0:
                print(f"\nMove {move_count}/{total_moves} | Loss: {loss.item():.4f} | "
                      f"Binding: {binding_loss.item():.4f} | Agency: {agency_cost.item():.4f}")
            elif move_count % 10 == 0:
                # More frequent progress updates
                print(f"Move {move_count}/{total_moves}...", end='\r', flush=True)
        
        except Exception as e:
            print(f"\nError at move {move_count}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Finalize stats recorder
    if stats_recorder is not None:
        stats_recorder.finalize()
    
    # Finalize environment recorder
    if environment_recorder is not None:
        environment_recorder.finalize()
    
    print(f"\nTraining complete. Total moves: {move_count}")
