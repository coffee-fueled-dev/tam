"""
Generic TAM training loop that works with any System + Environment.

This implements the core TAM cycle abstractly, allowing any environment
that can respond to port binding with context episodes.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Any, List
from v3.system import TAMSystem
from v3.environment import Environment
from v3.stats_recorder import StatsRecorder
from v3.environment_recorder import EnvironmentRecorder
from v3.goal_motif import GoalMotif, extract_motif_from_tkn_output


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
    environment_recorder: Optional[EnvironmentRecorder] = None,
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
            "agency_reward": 1.0,  # Weight for agency reward (positive reward reduces loss)
            "goal_reward": 1.0,  # Weight for goal reward (negative reward reduces loss)
        }

    # Goal reward configuration
    goal_reward_base = (
        config.get("goal_reward_base", 10.0) if config else 10.0
    )  # Base reward when goal reached
    
    # Distance-based penalty configuration
    goal_distance_threshold = (
        config.get("goal_distance_threshold", 10.0) if config else 10.0
    )  # Maximum distance allowed before penalty kicks in
    goal_distance_penalty = (
        config.get("goal_distance_penalty", 0.1) if config else 0.1
    )  # Penalty per unit distance over threshold

    # Create optimizer if not provided
    if optimizer is None:
        # Get trainable parameters from system
        # Note: This assumes system components are PyTorch modules
        params = []
        if hasattr(system, "inference_engine"):
            params.extend(system.inference_engine.parameters())
        if hasattr(system, "actor"):
            params.extend(system.actor.parameters())
        if len(params) == 0:
            raise ValueError(
                "System must have trainable parameters (inference_engine and/or actor)"
            )

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
    if hasattr(environment, "active_goals"):
        active_goals = environment.active_goals
        environment.active_goals = active_goals

    # Check if infer_situation accepts current_state parameter (once, not every iteration)
    import inspect

    infer_situation_accepts_current_state = False
    if hasattr(system, "infer_situation"):
        try:
            sig = inspect.signature(system.infer_situation)
            infer_situation_accepts_current_state = "current_state" in sig.parameters
        except Exception as e:
            print(f"Warning: Could not inspect infer_situation signature: {e}")
            # Default to trying with current_state
            infer_situation_accepts_current_state = True

    # Main training loop
    move_count = 0
    distance_since_last_goal = 0.0  # Track distance travelled since last goal

    print("Entering training loop...")

    while move_count < total_moves:
        try:
            # 1. Get context (prior context for port affordance evaluation)
            # Check for goal motifs first (motif-based navigation)
            goal_motifs = None
            if hasattr(environment, "get_goal_motif"):
                goal_motifs = environment.get_goal_motif()

            # Get initial intent: use motif-based if available, else fallback to position-based
            if goal_motifs is not None and len(goal_motifs) > 0:
                # Use zero intent initially - will generate from motifs after situation inference
                intent = torch.zeros(
                    environment.state_dim,
                    dtype=torch.float32,
                    device=current_state.device,
                )
            else:
                # Fallback to position-based intent
                if hasattr(environment, "active_goals"):
                    current_active_goals = environment.active_goals
                else:
                    current_active_goals = active_goals
                intent = environment.get_intent(
                    current_state.squeeze(0),
                    goal_motif=None,
                    active_goals=current_active_goals,
                )

            # get_context() uses environment.active_goals directly, so it will see updated goals
            context = environment.get_context(current_state.squeeze(0), intent)

            # 2. Infer situation (through tkn + inference engine)
            # Note: infer_situation may need current_state for tokenization
            # Ensure previous_situation is detached to prevent graph reuse
            previous_situation_detached = (
                previous_situation.detach()
                if hasattr(previous_situation, "detach")
                else previous_situation
            )

            # Compute remaining distance until penalty
            remaining_distance_until_penalty = max(
                0.0, goal_distance_threshold - distance_since_last_goal
            )

            if infer_situation_accepts_current_state:
                situation = system.infer_situation(
                    context,
                    previous_situation_detached,
                    current_state=current_state.squeeze(0),
                    remaining_distance_until_penalty=remaining_distance_until_penalty,
                )
            else:
                situation = system.infer_situation(
                    context,
                    previous_situation_detached,
                    remaining_distance_until_penalty=remaining_distance_until_penalty,
                )

            # Ensure situation has batch dimension
            if situation.dim() == 1:
                situation = situation.unsqueeze(0)  # (1, latent_dim)

            # Generate intent from motifs if using motif-based navigation
            if goal_motifs is not None and len(goal_motifs) > 0:
                if hasattr(system, "generate_intent_from_motifs"):
                    generated_intent = system.generate_intent_from_motifs(
                        goal_motifs, state_dim=environment.state_dim
                    )
                    if generated_intent is not None:
                        intent = generated_intent.to(current_state.device)

            # 3. Propose ports
            # propose_ports handles batch dimensions internally, so pass tensors as-is
            # It expects (latent_dim,) or (1, latent_dim) for situation, (state_dim,) or (1, state_dim) for intent
            try:
                # Remove batch dimension if present (propose_ports will add it back)
                situation_input = (
                    situation.squeeze(0) if situation.dim() > 1 else situation
                )
                intent_input = intent.squeeze(0) if intent.dim() > 1 else intent
                logits, mu_next, sigma_next = system.propose_ports(
                    situation_input, intent_input
                )
            except Exception as e:
                print(f"Error in propose_ports at move {move_count}: {e}")
                import traceback

                traceback.print_exc()
                raise

            # Ensure batch dimensions
            # Actor returns (B, M, state_dim) for mu_next and sigma_next, (B, M) for logits
            # Normalize to (B, n_ports, state_dim) format

            # Handle logits: should be (B, n_ports)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0).unsqueeze(0)  # (1, 1)
            elif logits.dim() == 1:
                logits = logits.unsqueeze(0)  # (1, n_ports)
            # If already 2D, assume it's correct (B, n_ports)

            # Handle mu_next: should be (B, n_ports, state_dim)
            if mu_next.dim() == 4:
                # (1, 1, n_ports, state_dim) -> (1, n_ports, state_dim)
                mu_next = mu_next.squeeze(1)
            elif mu_next.dim() == 3:
                # Already 3D, check if first dim is 1 (batch)
                if mu_next.shape[0] == 1:
                    pass  # Already correct (1, n_ports, state_dim)
                else:
                    mu_next = mu_next.unsqueeze(
                        0
                    )  # (n_ports, state_dim) -> (1, n_ports, state_dim)
            elif mu_next.dim() == 2:
                # (n_ports, state_dim) -> (1, n_ports, state_dim)
                mu_next = mu_next.unsqueeze(0)
            elif mu_next.dim() == 1:
                # (state_dim,) -> (1, 1, state_dim) - single port
                mu_next = mu_next.unsqueeze(0).unsqueeze(0)

            # Handle sigma_next: should be (B, n_ports, state_dim)
            if sigma_next.dim() == 4:
                # (1, 1, n_ports, state_dim) -> (1, n_ports, state_dim)
                sigma_next = sigma_next.squeeze(1)
            elif sigma_next.dim() == 3:
                # Already 3D, check if first dim is 1 (batch)
                if sigma_next.shape[0] == 1:
                    pass  # Already correct (1, n_ports, state_dim)
                else:
                    sigma_next = sigma_next.unsqueeze(
                        0
                    )  # (n_ports, state_dim) -> (1, n_ports, state_dim)
            elif sigma_next.dim() == 2:
                # (n_ports, state_dim) -> (1, n_ports, state_dim)
                sigma_next = sigma_next.unsqueeze(0)
            elif sigma_next.dim() == 1:
                # (state_dim,) -> (1, 1, state_dim) - single port
                sigma_next = sigma_next.unsqueeze(0).unsqueeze(0)

            # Verify shapes are correct
            if mu_next.dim() != 3 or mu_next.shape[1] != logits.shape[1]:
                raise ValueError(
                    f"Shape mismatch: mu_next.shape={mu_next.shape}, logits.shape={logits.shape}, "
                    f"expected mu_next to be (B, n_ports, state_dim) and logits to be (B, n_ports)"
                )

            # Ensure intent has batch dimension
            if intent.dim() == 1:
                intent = intent.unsqueeze(0)  # (1, state_dim)

            # 4. Compute intent-aligned port selection scores
            # mu_next is already the next step (relative to current_pos)
            next_steps = mu_next  # (B, n_ports, state_dim)

            # Compute intent alignment using cosine similarity
            intent_norm = torch.norm(intent, dim=-1, keepdim=True)  # (B, 1)
            device = intent.device
            B = intent.shape[0]
            n_ports = next_steps.shape[1]

            # Check which samples have non-zero intent (handle per-sample)
            non_zero_intent_mask = intent_norm.squeeze(-1) > 1e-6  # (B,)

            # Initialize intent alignment (will be computed for non-zero intents)
            intent_alignment = torch.zeros(B, n_ports, device=device)

            if non_zero_intent_mask.any():
                # Normalize intent direction for non-zero intents
                intent_normalized = intent / (intent_norm + 1e-6)  # (B, state_dim)

                # Normalize next step directions
                next_step_norms = torch.norm(
                    next_steps, dim=-1, keepdim=True
                )  # (B, n_ports, 1)
                next_steps_normalized = next_steps / (
                    next_step_norms + 1e-6
                )  # (B, n_ports, state_dim)

                # Handle zero-length steps (no movement)
                zero_step_mask = next_step_norms.squeeze(-1) < 1e-6  # (B, n_ports)

                # Compute cosine similarity: dot product of normalized vectors
                intent_expanded = intent_normalized.unsqueeze(1)  # (B, 1, state_dim)
                cosine_similarity = torch.sum(
                    next_steps_normalized * intent_expanded, dim=-1
                )  # (B, n_ports)

                # Set alignment to 0 for zero-length steps
                cosine_similarity[zero_step_mask] = 0.0

                # Only set alignment for samples with non-zero intent
                # Multiply by mask to zero out samples with zero intent
                intent_alignment = (
                    cosine_similarity * non_zero_intent_mask.unsqueeze(-1).float()
                )

            # Compute agency score (negative mean sigma² - tighter cones = higher agency)
            agency_score = -torch.mean(sigma_next**2, dim=-1)  # (B, n_ports)
            # Normalize to similar scale as logits (typical sigma ~0.5-2.0, so sigma² ~0.25-4.0)
            # Divide by 10 to bring to similar scale as logits
            agency_score = agency_score / 10.0

            # Get learnable weights from actor for port selection
            if hasattr(system, "actor") and hasattr(system.actor, "intent_bias_weight"):
                intent_bias_weight = (
                    system.actor.intent_bias_weight
                )  # Learnable weight for intent alignment
                agency_bias_weight = (
                    system.actor.agency_bias_weight
                )  # Learnable weight for agency score
            else:
                # Fallback to default values if actor doesn't have learnable weights
                intent_bias_weight = torch.tensor(2.0, device=device)
                agency_bias_weight = torch.tensor(1.0, device=device)

            # Combine scores: logits + intent alignment + agency
            combined_score = (
                logits
                + intent_bias_weight * intent_alignment
                + agency_bias_weight * agency_score
            )

            # 5. Select and bind port using combined score
            # Selection (Categorical sampling for exploration)
            probs = F.softmax(combined_score, dim=-1)
            m = torch.distributions.Categorical(probs)
            selected_port_idx = m.sample()

            # Extract selected next step
            selected_port_idx_int = (
                selected_port_idx.item()
                if isinstance(selected_port_idx, torch.Tensor)
                else selected_port_idx
            )
            selected_next = mu_next[0, selected_port_idx_int]  # (state_dim,)
            selected_sigma = sigma_next[0, selected_port_idx_int]  # (state_dim,)

            # 5. World responds with context episode
            # bind_port computes physics and returns path, but doesn't update environment
            # For single-step generation, we pass the next step as a single-point "tube"
            context_episode = environment.bind_port(
                selected_next.unsqueeze(0),  # (1, state_dim) - single step as "tube"
                selected_sigma.unsqueeze(0),  # (1, state_dim) - single step sigma
                current_state.squeeze(0),
            )

            # Apply each state in the path to environment (for environment recorder)
            # This allows the recorder to track at environment refresh rate
            # Track total goals reached across all steps in this move
            goals_reached_this_move = 0
            if environment_recorder is not None:
                # Set move number for this move's steps
                environment_recorder.set_move_number(move_count + 1)
                for step_idx, state in enumerate(context_episode):
                    goals_reached_this_step = environment.apply(
                        state, env_recorder=environment_recorder
                    )
                    goals_reached_this_move += goals_reached_this_step
            else:
                # Still track goals even if no environment recorder
                for step_idx, state in enumerate(context_episode):
                    goals_reached_this_step = environment.apply(
                        state, env_recorder=None
                    )
                    goals_reached_this_move += goals_reached_this_step

            # Track distance travelled this move (from context episode)
            move_distance = 0.0
            if len(context_episode) > 1:
                move_distance = sum(
                    torch.norm(context_episode[i] - context_episode[i - 1]).item()
                    for i in range(1, len(context_episode))
                )
            distance_since_last_goal += move_distance

            # 6. Evaluate binding and update
            # Compute binding loss
            try:
                binding_loss = system.evaluate_binding(
                    selected_next,
                    context_episode,
                    selected_sigma,
                    current_state=current_state.squeeze(0),
                )
            except Exception as e:
                print(f"Error in evaluate_binding: {e}")
                import traceback

                traceback.print_exc()
                raise

            # Compute agency reward (complexity-scaled, from system)
            agency_reward = torch.tensor(0.0, device=binding_loss.device)

            if hasattr(system, "get_last_agency_reward"):
                try:
                    # Get complexity-scaled agency reward from system
                    agency_reward = system.get_last_agency_reward()
                    if agency_reward is None:
                        agency_reward = torch.tensor(0.0, device=binding_loss.device)
                except Exception as e:
                    # Handle gracefully if computation fails
                    agency_reward = torch.tensor(0.0, device=binding_loss.device)

            # Compute goal reward (constant, no decay)
            goal_reward = torch.tensor(0.0, device=binding_loss.device)
            if goals_reached_this_move > 0:
                # Constant reward per goal reached
                goal_reward = torch.tensor(
                    goal_reward_base * goals_reached_this_move,
                    device=binding_loss.device,
                )
                # Reset distance counter (goals were reached)
                distance_since_last_goal = 0.0

            # Compute distance-based penalty
            goal_penalty = torch.tensor(0.0, device=binding_loss.device)
            if goals_reached_this_move == 0 and distance_since_last_goal > goal_distance_threshold:
                excess_distance = distance_since_last_goal - goal_distance_threshold
                goal_penalty = torch.tensor(
                    goal_distance_penalty * excess_distance,
                    device=binding_loss.device,
                )

            # Total loss: binding_loss - agency_reward - goal_reward + goal_penalty
            loss = (
                loss_weights.get("binding_loss", 1.0) * binding_loss
                - loss_weights.get("agency_reward", 1.0) * agency_reward
                - loss_weights.get("goal_reward", 1.0) * goal_reward
                + loss_weights.get("goal_penalty", 1.0) * goal_penalty
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if hasattr(system, "inference_engine"):
                torch.nn.utils.clip_grad_norm_(
                    system.inference_engine.parameters(), max_norm=1.0
                )
            if hasattr(system, "actor"):
                torch.nn.utils.clip_grad_norm_(system.actor.parameters(), max_norm=1.0)

            optimizer.step()

            # Detach and clone for next iteration (prevent graph accumulation)
            situation = situation.detach()
            previous_situation = (
                situation.clone()
            )  # Clone to ensure complete independence

            # 7. Update state (next situation)
            # Update current_state to last position in context_episode
            # IMPORTANT: Detach to prevent graph accumulation across iterations
            if len(context_episode) > 0:
                current_state = (
                    context_episode[-1].detach().view(1, -1)
                )  # (1, state_dim)
            else:
                # Fallback: use current_state (shouldn't happen, but handle gracefully)
                # Still detach to be safe
                if hasattr(current_state, "detach"):
                    current_state = current_state.detach()

            move_count += 1

            # Goals are already tracked during apply() calls above
            # Replenish energy if goals were reached (if environment supports it)
            if goals_reached_this_move > 0 and hasattr(environment, "replenish_energy"):
                environment.replenish_energy(
                    energy_replenish_amount * goals_reached_this_move
                )

            # CRITICAL: Sync active_goals with environment after apply() calls
            # This ensures get_context() and get_intent() see the updated goals on the next move
            if hasattr(environment, "active_goals"):
                active_goals = environment.active_goals

            # Update goal motifs if using motif-based navigation
            # (motifs may have been updated during goal completion)
            if hasattr(environment, "get_goal_motif"):
                goal_motifs = environment.get_goal_motif()

            # Record statistics if stats_recorder is provided
            if stats_recorder is not None:
                # Extract step components
                step_components = {}
                step_components["selected_port"] = selected_port_idx_int

                # For single-step generation, track step length
                step_length = torch.norm(selected_next).item()
                step_components["step_length"] = float(step_length)

                # Record move statistics
                # For compatibility, store selected_next as a single-point "tube"
                selected_tube_for_stats = selected_next.unsqueeze(0)  # (1, state_dim)
                selected_sigma_for_stats = selected_sigma.unsqueeze(0)  # (1, state_dim)

                # Get situation complexity for stats if available
                situation_complexity = None
                if hasattr(system, "get_situation_complexity"):
                    try:
                        complexity = system.get_situation_complexity()
                        if isinstance(complexity, torch.Tensor):
                            situation_complexity = complexity.item()
                        else:
                            situation_complexity = float(complexity)
                    except:
                        pass

                stats_recorder.record_move(
                    move=move_count,  # For internal tracking only
                    binding_loss=binding_loss,
                    agency_reward=agency_reward,
                    knot_components=step_components if step_components else None,
                    mu_t=selected_tube_for_stats,  # Store selected next step as single-point trajectory
                    sigma_t=selected_sigma_for_stats,  # Store selected sigma
                    goals_reached=goals_reached_this_move,  # Track goals reached
                    situation_complexity=situation_complexity,  # Track situation complexity
                    total_loss=loss,  # Total loss value
                    goal_reward=goal_reward,  # Goal reward value
                    goal_penalty=goal_penalty,  # Goal penalty value
                    loss_weights=loss_weights,  # Loss weights for computing weighted components
                )

            # Handle energy depletion (if environment supports it)
            if hasattr(environment, "is_dead") and environment.is_dead:
                # Reset position and partial energy
                if hasattr(environment, "current_energy") and hasattr(
                    environment, "initial_energy"
                ):
                    environment.current_energy = environment.initial_energy * 0.3
                    environment.is_dead = False
                current_state = environment.get_initial_state()
                if current_state.dim() == 1:
                    current_state = current_state.unsqueeze(0)

                # Reset system state
                if hasattr(system, "reset"):
                    system.reset()
                previous_situation = torch.zeros(1, system.latent_dim)

                # Reset distance counter on episode reset
                distance_since_last_goal = 0.0

            # Print progress periodically
            if move_count % 100 == 0:
                goal_reward_str = (
                    f" | Goal Reward: {goal_reward.item():.4f}"
                    if goals_reached_this_move > 0
                    else ""
                )
                complexity_str = ""
                if hasattr(system, "get_situation_complexity"):
                    try:
                        complexity = system.get_situation_complexity()
                        if isinstance(complexity, torch.Tensor):
                            complexity_str = f" | Complexity: {complexity.item():.4f}"
                    except:
                        pass
                print(
                    f"\nMove {move_count}/{total_moves} | Loss: {loss.item():.4f} | "
                    f"Binding: {binding_loss.item():.4f} | Agency Reward: {agency_reward.item():.4f}{complexity_str}{goal_reward_str}"
                )
            elif move_count % 10 == 0:
                # More frequent progress updates
                print(f"Move {move_count}/{total_moves}...", end="\r", flush=True)

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
