#!/usr/bin/env python3
"""
Test: Binding-Failure Learning for 3D Obstacle Navigation

The actor learns to navigate to a goal while avoiding unknown obstacles.
Obstacles create binding failures - when reality diverges from the predicted tube.

Key insight: The actor doesn't know where obstacles are, but learns which
commitments (z values) lead to binding failures for each situation (s0).
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import json

from v2.actors.actor_2 import Actor
from v2.environments.obstacle_nav import ObstacleNavEnv, ObstacleNavConfig


def create_output_dir():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"artifacts/obstacle_nav/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def train_actor(
    n_episodes: int = 500,
    warmup_episodes: int = 50,
    eval_every: int = 50,
    seed: int = 42,
):
    """
    Train actor to navigate through obstacles using binding failure.
    """
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment with FIXED obstacles
    # Simpler setup: one central obstacle, fixed start/goal to learn one path first
    from v2.environments.obstacle_nav import Obstacle
    
    fixed_obstacles = [
        # One central obstacle that blocks direct path
        Obstacle(center=np.array([0.5, 0.5, 0.0]), radius=0.4),
    ]
    
    env_config = ObstacleNavConfig(
        n_obstacles=len(fixed_obstacles),
        T=20,
        max_steps=60,          # More steps to complete path
        max_speed=0.12,        # Slower but consistent
        goal_radius=0.25,
        randomize_obstacles=False,
        fixed_obstacles=fixed_obstacles,
        randomize_start=False,
        randomize_goal=False,
        fixed_start=np.array([0.0, 0.0, 0.0]),
        fixed_goal=np.array([1.0, 1.0, 0.0]),
    )
    env = ObstacleNavEnv(env_config)
    
    # Create actor
    actor = Actor(
        obs_dim=env.obs_dim,  # 6: [pos, goal]
        z_dim=6,              # Latent commitment dimension (larger for spatial complexity)
        pred_dim=env.pred_dim,  # 3: trajectory in 3D
        T=env_config.T,
        n_knots=7,            # More knots for curved paths around obstacles
        M=12,                 # More heads for diverse strategies (left/right around obstacles)
        alpha_vol=0.05,       # Slightly less volume pressure (allow wider tubes initially)
        alpha_leak=2.0,       # Binding failure weight
        k_sigma=2.5,          # Wider initial tubes
        repel_min=0.1,        # Stronger minimum repulsion (encourage diversity)
        L_refine=3,           # Refine top-3 heads with CEM
    )
    
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    
    # Training metrics
    history = {
        'episode': [],
        'loss': [],                 # Total training loss
        'best_total': [],          # Best head's total loss (binding + terminal + metabolic)
        'best_terminal_nll': [],   # Best head's terminal NLL (goal commitment)
        'best_metabolic': [],      # Best head's metabolic loss (log-length)
        'best_binding_nll': [],   # Best head's binding NLL
        'goal_dist_sq': [],        # Squared distance to goal (from predicted endpoint)
        'goal_dist': [],           # Distance to goal (from predicted endpoint)
        'weighted_binding': [],    # alpha_bind * binding_nll (diagnostic)
        'weighted_terminal': [],   # alpha_terminal * terminal_nll (diagnostic)
        'weighted_metabolic': [],  # alpha_metabolic * metabolic (diagnostic)
        'sigma_min': [],           # Minimum sigma across trajectory (diagnostic)
        'sigma_mean': [],          # Mean sigma across trajectory (diagnostic)
        'sigma_end_mean': [],      # Mean sigma at endpoint (diagnostic)
        'endpoint_mismatch': [],   # ||mu_end - actual_end||^2 (diagnostic: check if they coincide)
        'final_distance': [],      # Final distance to goal (environment metric)
        'n_collisions': [],
        'best_head': [],
        'regret_proxy': [],        # Head diversity metric
        'repel_gate': [],          # Repulsion gating (necessity indicator)
        'repel': [],               # Repulsion loss value
        'usage_entropy': [],       # Head usage entropy (diversity)
    }
    
    print("\n" + "="*60)
    print("TRAINING: Causal Knot-Space Binding-Failure Learning")
    print("="*60)
    print(f"Episodes: {n_episodes}, Warmup: {warmup_episodes}")
    print(f"Obstacles: {env_config.n_obstacles}, T: {env_config.T}")
    print(f"Goal radius: {env_config.goal_radius}")
    print()
    
    # Training loop
    for episode in range(n_episodes):
        device = next(actor.parameters()).device
        
        # Reset environment
        obs = env.reset(seed=seed + episode)
        s0 = torch.tensor(obs, dtype=torch.float32, device=device)
        
        # Starting position and goal
        x0 = obs[:3].copy()  # Position is first 3 elements of rich obs
        goal = env.goal
        
        # Execute with online tube projection and per-step training
        actual_traj_full = []  # Accumulate full trajectory (delta from start) for visualization
        n_collisions = 0
        step_metrics = []  # Accumulate metrics from each step
        last_mu_np = None  # Store last projected tube for visualization
        # Store tubes and positions at each timestep for overlay visualization
        timestep_tubes = []  # List of (position, tube) tuples, where tube is in absolute coordinates
        timestep_positions = []  # List of positions at each timestep
        
        curr_obs = obs
        curr_pos = x0.copy()
        
        # Commitment: keep the same z* unless the world contradicts the prediction
        # "Contradiction" = binding failure = actual step outside predicted tube uncertainty
        # This is energy-efficient: no need to re-think or re-learn when world agrees
        z_star = None  # Will be selected on first step
        
        for t in range(env_config.T):
            # Current situation
            s_t = torch.tensor(curr_obs, dtype=torch.float32, device=device)
            curr_pos_tensor = torch.tensor(curr_pos, dtype=torch.float32, device=device)
            goal_tensor = torch.tensor(goal, dtype=torch.float32, device=device)
            
            # Intent: goal relative to current position (in delta coordinates)
            intent_delta = goal_tensor - curr_pos_tensor
            
            # Project tube from current situation
            # Tube is in cumulative delta coordinates from current position
            # mu[0] = 0 (at current position), mu[i] = cumulative delta to step i
            with torch.no_grad():
                # Select z* only on first step or after world contradicted the prediction
                if z_star is None:
                    z_star = actor.select_z_geometric_multimodal(s_t, intent_delta=intent_delta)
                
                mu_full, sigma_full = actor.get_tube(s_t, z_star)
                mu_np = mu_full.squeeze(0).cpu().numpy()  # (T, 3) cumulative deltas
                sigma_np = sigma_full.squeeze(0).cpu().numpy()  # (T, 3) uncertainties
                
                # Get the predicted first step and its uncertainty
                predicted_step = mu_np[1] if env_config.T > 1 else mu_np[0]
                predicted_sigma = sigma_np[1] if env_config.T > 1 else sigma_np[0]
                
                # Visualization
                curr_pos_delta_from_start = curr_pos - x0
                mu_np_from_start = mu_np + curr_pos_delta_from_start
                last_mu_np = mu_np_from_start
                mu_absolute = curr_pos + mu_np
                timestep_tubes.append((curr_pos.copy(), mu_absolute.copy()))
                timestep_positions.append(curr_pos.copy())
            
            # Execute first step of projected tube
            action = predicted_step
            
            # Step environment
            curr_obs, reward, done, info = env.step(action)
            new_pos = info['x']
            
            # Actual step taken (delta from current position, matching tube coordinate frame)
            actual_step = new_pos - curr_pos
            
            # Check binding failure: did world contradict the prediction?
            # Normalized by sigma - this is the principled measure
            step_error = actual_step - predicted_step
            normalized_error = np.abs(step_error) / (predicted_sigma + 1e-6)
            max_normalized_error = np.max(normalized_error)  # Worst dimension
            
            # World contradicted if error exceeds ~2 sigma in any dimension
            world_contradicted = max_normalized_error > 2.0
            
            # Always train (need baseline learning for goal direction, sigma calibration, etc.)
            # But modulate: contradiction = stronger binding signal
            actual_step_t = torch.tensor(actual_step, dtype=torch.float32, device=device)
            actual_traj_from_curr = torch.stack([
                torch.zeros(3, device=device),  # At current position
                actual_step_t,                   # Where we moved to
            ], dim=0)  # Shape: (2, 3)
            
            # Binding weight scales with contradiction magnitude
            # No contradiction (normalized_error < 1) → reduced binding weight
            # Strong contradiction → full binding weight
            bind_scale = min(1.0, max_normalized_error / 2.0)  # 0 to 1
            effective_alpha_bind = 0.5 + 1.5 * bind_scale  # Range: 0.5 to 2.0
            
            step_metric = actor.train_step_trajectory_jacobian(
                s0=s_t,
                actual_traj=actual_traj_from_curr,
                intent_delta=intent_delta,
                optimizer=optimizer,
                alpha_bind=effective_alpha_bind,
                alpha_terminal=5.0,  # Goal learning always important
                alpha_metabolic=0.1,
                z_executed=z_star,
            )
            step_metrics.append(step_metric)
            
            # Reconsider z* only on strong contradiction
            if world_contradicted:
                z_star = None  # Will reselect on next iteration
            # else: keep same z* (energy-efficient - don't replan if world agrees)
            
            # Update position and accumulate trajectory (for visualization/logging)
            curr_pos = new_pos.copy()
            actual_traj_full.append(new_pos - x0)  # Delta from start (for visualization)
            
            if info['collision']:
                n_collisions += 1
            
            if done:
                break
        
        # Use metrics from last step for logging
        metrics = step_metrics[-1] if step_metrics else {}
        final_distance = info['dist_to_goal']
        
        # For visualization: pad actual trajectory to full T and use last projected tube
        actual_traj = actual_traj_full
        while len(actual_traj) < env_config.T:
            actual_traj.append(actual_traj[-1] if actual_traj else np.zeros(3))
        mu_np = last_mu_np if last_mu_np is not None else np.zeros((env_config.T, 3))
        
        # Add final position to timestep_positions if episode ended early
        if len(timestep_positions) < len(actual_traj_full):
            timestep_positions.append(curr_pos.copy())
        
        # Log metrics (training-relevant losses and diagnostics)
        # Note: metrics may be empty if no contradictions occurred (no training this episode)
        history['episode'].append(episode)
        history['loss'].append(metrics.get('loss', 0.0))
        history['best_total'].append(metrics.get('best_total', 0.0))
        history['best_terminal_nll'].append(metrics.get('best_terminal_nll', 0.0))
        history['best_metabolic'].append(metrics.get('best_metabolic', 0.0))
        history['best_binding_nll'].append(metrics.get('best_binding_nll', 0.0))
        history['goal_dist_sq'].append(metrics.get('goal_dist_sq', 0.0))
        history['goal_dist'].append(metrics.get('goal_dist', 0.0))
        history['weighted_binding'].append(metrics.get('weighted_binding', 0.0))
        history['weighted_terminal'].append(metrics.get('weighted_terminal', 0.0))
        history['weighted_metabolic'].append(metrics.get('weighted_metabolic', 0.0))
        history['sigma_min'].append(metrics.get('sigma_min', 0.0))
        history['sigma_mean'].append(metrics.get('sigma_mean', 0.0))
        history['sigma_end_mean'].append(metrics.get('sigma_end_mean', 0.0))
        if 'endpoint_mismatch' in metrics:
            history['endpoint_mismatch'].append(metrics['endpoint_mismatch'])
        history['final_distance'].append(final_distance)
        history['n_collisions'].append(n_collisions)
        history['best_head'].append(metrics.get('best_head', 0))
        history['regret_proxy'].append(metrics.get('regret_proxy', 0.0))
        history['repel_gate'].append(metrics.get('repel_gate', 0.0))
        history['repel'].append(metrics.get('repel', 0.0))
        history['usage_entropy'].append(metrics.get('usage_entropy', 0.0))
        
        # Print progress with diagnostic breakdown
        if episode % 50 == 0 or episode == n_episodes - 1:
            recent = 50
            recent_total = np.mean(history['best_total'][-recent:])
            recent_terminal = np.mean(history['best_terminal_nll'][-recent:])
            recent_metabolic = np.mean(history['best_metabolic'][-recent:])
            recent_binding = np.mean(history['best_binding_nll'][-recent:])
            recent_goal_dist = np.mean(history['goal_dist'][-recent:])
            recent_final_dist = np.mean(history['final_distance'][-recent:])
            recent_collisions = np.mean(history['n_collisions'][-recent:])
            # Weighted contributions (diagnostics)
            recent_w_bind = np.mean(history['weighted_binding'][-recent:])
            recent_w_term = np.mean(history['weighted_terminal'][-recent:])
            recent_w_metab = np.mean(history['weighted_metabolic'][-recent:])
            # Sigma diagnostics
            recent_sigma_min = np.mean(history['sigma_min'][-recent:])
            recent_sigma_mean = np.mean(history['sigma_mean'][-recent:])
            
            print(f"Episode {episode:4d} | "
                  f"Total: {recent_total:.3f} | "
                  f"GoalNLL: {recent_terminal:.3f} | "
                  f"Metabolic: {recent_metabolic:.3f} | "
                  f"Binding: {recent_binding:.3f} | "
                  f"GoalDist: {recent_goal_dist:.2f} | "
                  f"FinalDist: {recent_final_dist:.2f} | "
                  f"Collisions: {recent_collisions:.1f}")
            
            # Detailed breakdown every 200 episodes
            if episode % 200 == 0:
                print(f"  Weighted: Bind={recent_w_bind:.2f}, Goal={recent_w_term:.2f}, Metab={recent_w_metab:.3f}")
                print(f"  Sigma: min={recent_sigma_min:.4f}, mean={recent_sigma_mean:.4f}")
                # Note: mu_np is now from last step's projection (online updates)
                # Coordinate check is less meaningful now since we project at each step
        
        # Evaluation visualization
        if episode % eval_every == 0 or episode == n_episodes - 1:
            plot_episode(
                env, actor, output_dir, episode, mu_np, actual_traj, x0, info,
                timestep_tubes=timestep_tubes, timestep_positions=timestep_positions
            )
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    final_total = np.mean(history['best_total'][-100:])
    final_terminal = np.mean(history['best_terminal_nll'][-100:])
    final_metabolic = np.mean(history['best_metabolic'][-100:])
    final_binding = np.mean(history['best_binding_nll'][-100:])
    final_goal_dist = np.mean(history['goal_dist'][-100:])
    final_dist = np.mean(history['final_distance'][-100:])
    final_collisions = np.mean(history['n_collisions'][-100:])
    final_entropy = np.mean(history['usage_entropy'][-100:])
    # Weighted contributions
    final_w_bind = np.mean(history['weighted_binding'][-100:])
    final_w_term = np.mean(history['weighted_terminal'][-100:])
    final_w_metab = np.mean(history['weighted_metabolic'][-100:])
    # Sigma diagnostics
    final_sigma_min = np.mean(history['sigma_min'][-100:])
    final_sigma_mean = np.mean(history['sigma_mean'][-100:])
    
    print(f"Final 100 episodes:")
    print(f"  Total loss: {final_total:.4f}")
    print(f"    Binding: {final_binding:.4f} (weighted: {final_w_bind:.4f})")
    print(f"    Goal NLL: {final_terminal:.4f} (weighted: {final_w_term:.4f})")
    print(f"    Metabolic: {final_metabolic:.4f} (weighted: {final_w_metab:.4f})")
    print(f"  Goal dist (predicted endpoint): {final_goal_dist:.3f} | Final dist (actual): {final_dist:.3f}")
    print(f"  Collisions: {final_collisions:.2f} | Head entropy: {final_entropy:.3f}")
    print(f"  Sigma: min={final_sigma_min:.4f}, mean={final_sigma_mean:.4f}")
    
    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot learning curves
    plot_learning_curves(history, output_dir, env_config)
    
    # Save model
    torch.save(actor.state_dict(), output_dir / "actor.pt")
    
    print(f"\nResults saved to: {output_dir}")
    
    return actor, history, output_dir


def plot_episode(env, actor, output_dir, episode, mu_np, actual_traj, x0, info,
                 timestep_tubes=None, timestep_positions=None):
    """Plot a single episode in 3D with overlay of all timestep tubes."""
    fig = plt.figure(figsize=(16, 8))
    
    # 3D trajectory view
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot obstacles
    for obs in env.obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
        y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
        z = obs.center[2] + obs.radius * np.cos(v)
        ax1.plot_surface(x, y, z, alpha=0.3, color='red')
    
    # Plot all timestep tubes (overlay)
    if timestep_tubes is not None and len(timestep_tubes) > 0:
        # Use a colormap to show progression through time
        colors = plt.cm.viridis(np.linspace(0, 1, len(timestep_tubes)))
        for i, (pos, tube) in enumerate(timestep_tubes):
            alpha = 0.3 + 0.4 * (i / max(1, len(timestep_tubes) - 1))  # Fade in over time
            ax1.plot(tube[:, 0], tube[:, 1], tube[:, 2],
                    'b-', linewidth=1.5, alpha=alpha, color=colors[i])
            # Mark the actor's position at this timestep
            ax1.scatter(*pos, color=colors[i], s=30, alpha=0.8, marker='o')
    
    # Plot actual trajectory (thicker, more prominent)
    actual_np = np.array(actual_traj)
    actual_abs = x0 + actual_np
    ax1.plot(actual_abs[:, 0], actual_abs[:, 1], actual_abs[:, 2],
             'g-', linewidth=3, label='Actual trajectory', zorder=10)
    
    # Plot start and goal
    ax1.scatter(*x0, color='blue', s=150, marker='o', label='Start', zorder=11, edgecolors='black', linewidths=2)
    ax1.scatter(*env.goal, color='gold', s=200, marker='*', label='Goal', zorder=11, edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Episode {episode}\nFinal dist: {info["dist_to_goal"]:.2f}\n{len(timestep_tubes) if timestep_tubes else 0} timesteps')
    ax1.legend(fontsize=8)
    
    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(122)
    
    # Plot obstacles (circles)
    for obs in env.obstacles:
        circle = plt.Circle((obs.center[0], obs.center[1]), obs.radius,
                            color='red', alpha=0.3)
        ax2.add_patch(circle)
    
    # Plot all timestep tubes (overlay) in top-down view
    if timestep_tubes is not None and len(timestep_tubes) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(timestep_tubes)))
        for i, (pos, tube) in enumerate(timestep_tubes):
            alpha = 0.3 + 0.4 * (i / max(1, len(timestep_tubes) - 1))
            ax2.plot(tube[:, 0], tube[:, 1], 'b-', linewidth=1.5, alpha=alpha, color=colors[i])
            # Mark the actor's position at this timestep
            ax2.scatter(*pos[:2], color=colors[i], s=30, alpha=0.8, marker='o')
    
    # Plot actual trajectory (thicker, more prominent)
    ax2.plot(actual_abs[:, 0], actual_abs[:, 1], 'g-', linewidth=3, label='Actual trajectory', zorder=10)
    
    # Start and goal
    ax2.scatter(x0[0], x0[1], color='blue', s=150, marker='o', label='Start', zorder=11, edgecolors='black', linewidths=2)
    ax2.scatter(env.goal[0], env.goal[1], color='gold', s=200, marker='*', label='Goal', zorder=11, edgecolors='black', linewidths=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-down view (X-Y)')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"episode_{episode:04d}.png", dpi=100)
    plt.close()


def plot_learning_curves(history, output_dir, env_config):
    """Plot learning curves with training-relevant metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    window = min(50, len(history['loss']) // 4)
    
    # Goal NLL + Goal distance (training-relevant)
    ax = axes[0, 0]
    goal_nll = np.array(history['best_terminal_nll'])  # Now goal_nll
    goal_dist_pred = np.array(history['goal_dist'])
    final_dist = np.array(history['final_distance'])
    if window > 0:
        goal_nll_smooth = np.convolve(goal_nll, np.ones(window)/window, mode='valid')
        goal_dist_smooth = np.convolve(goal_dist_pred, np.ones(window)/window, mode='valid')
        final_smooth = np.convolve(final_dist, np.ones(window)/window, mode='valid')
        ax.plot(goal_nll_smooth, label='Goal NLL', linewidth=2, color='red')
        ax2 = ax.twinx()
        ax2.plot(goal_dist_smooth, label='Predicted endpoint dist', linewidth=2, color='blue')
        ax2.plot(final_smooth, label='Actual final dist', linewidth=2, linestyle='--', color='orange')
        ax2.set_ylabel('Distance to Goal', color='blue')
        ax2.legend(loc='upper right')
    ax.axhline(env_config.goal_radius, color='green', linestyle=':', label='Goal radius', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Goal NLL')
    ax.set_title('Goal Commitment & Distance (smoothed)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Weighted contributions (diagnostic: check scale balance)
    ax = axes[0, 1]
    w_bind = np.array(history['weighted_binding'])
    w_term = np.array(history['weighted_terminal'])
    w_metab = np.array(history['weighted_metabolic'])
    if window > 0:
        w_bind_smooth = np.convolve(w_bind, np.ones(window)/window, mode='valid')
        w_term_smooth = np.convolve(w_term, np.ones(window)/window, mode='valid')
        w_metab_smooth = np.convolve(w_metab, np.ones(window)/window, mode='valid')
        ax.plot(w_bind_smooth, label='α_bind × Binding', linewidth=2)
        ax.plot(w_term_smooth, label='α_term × Terminal', linewidth=2)
        ax.plot(w_metab_smooth, label='α_metab × Metabolic', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Weighted Loss Contribution')
    ax.set_title('Weighted Loss Components (scale check)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Collisions
    ax = axes[0, 2]
    collisions = np.array(history['n_collisions'])
    if window > 0:
        collisions_smooth = np.convolve(collisions, np.ones(window)/window, mode='valid')
        ax.plot(collisions_smooth, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collisions per episode')
    ax.set_title('Collision Rate (smoothed)')
    ax.grid(True, alpha=0.3)
    
    # Metabolic loss (path complexity)
    ax = axes[1, 0]
    metabolic = np.array(history['best_metabolic'])
    if window > 0:
        metabolic_smooth = np.convolve(metabolic, np.ones(window)/window, mode='valid')
        ax.plot(metabolic_smooth, linewidth=2, color='teal')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Metabolic Loss (log-length)')
    ax.set_title('Path Complexity (smoothed)')
    ax.grid(True, alpha=0.3)
    
    # Endpoint mismatch (diagnostic: check if mu_end = actual_end)
    ax = axes[1, 1]
    if 'endpoint_mismatch' in history and len(history['endpoint_mismatch']) > 0:
        endpoint_mismatch = np.array(history['endpoint_mismatch'])
        if window > 0 and len(endpoint_mismatch) >= window:
            mismatch_smooth = np.convolve(endpoint_mismatch, np.ones(window)/window, mode='valid')
            ax.plot(mismatch_smooth, linewidth=2, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Endpoint Mismatch')
        ax.set_title('Predicted vs Actual Endpoint (smoothed)')
        ax.grid(True, alpha=0.3)
    else:
        # Fallback: show regret proxy if endpoint_mismatch not available
        regret = np.array(history['regret_proxy'])
        if window > 0:
            regret_smooth = np.convolve(regret, np.ones(window)/window, mode='valid')
            ax.plot(regret_smooth, linewidth=2, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Regret Proxy')
        ax.set_title('Head Diversity (Regret)')
        ax.grid(True, alpha=0.3)
    
    # Head usage
    ax = axes[1, 2]
    heads = np.array(history['best_head'])
    ax.hist(heads, bins=np.arange(-0.5, max(heads) + 1.5, 1), edgecolor='black')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Count')
    ax.set_title('Head Usage Distribution')
    ax.grid(True, alpha=0.3)
    
    # Binding NLL (diagnostic: check if going negative)
    ax = axes[2, 0]
    binding_nll = np.array(history['best_binding_nll'])
    if window > 0:
        binding_smooth = np.convolve(binding_nll, np.ones(window)/window, mode='valid')
        ax.plot(binding_smooth, linewidth=2, color='purple')
        ax.axhline(0, color='red', linestyle=':', alpha=0.5, label='Zero')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Binding NLL')
    ax.set_title('Binding NLL (check for negative)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sigma statistics (diagnostic: check for collapse)
    ax = axes[2, 1]
    sigma_min = np.array(history['sigma_min'])
    sigma_mean = np.array(history['sigma_mean'])
    if window > 0:
        min_smooth = np.convolve(sigma_min, np.ones(window)/window, mode='valid')
        mean_smooth = np.convolve(sigma_mean, np.ones(window)/window, mode='valid')
        ax.plot(min_smooth, label='Min σ', linewidth=2, color='red')
        ax.plot(mean_smooth, label='Mean σ', linewidth=2, color='blue')
        # Show sigma_min threshold
        ax.axhline(0.05, color='green', linestyle=':', alpha=0.5, label='σ_min=0.05')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sigma')
    ax.set_title('Sigma Statistics (collapse check)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Repulsion metrics (moved to 2,2)
    ax = axes[2, 2]
    repel = np.array(history['repel'])
    repel_gate = np.array(history['repel_gate'])
    if window > 0:
        repel_smooth = np.convolve(repel, np.ones(window)/window, mode='valid')
        gate_smooth = np.convolve(repel_gate, np.ones(window)/window, mode='valid')
        ax.plot(repel_smooth, label='Repulsion', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(gate_smooth, label='Gate', linewidth=2, color='orange', linestyle='--')
        ax2.set_ylabel('Gate (0/1)', color='orange')
        ax2.legend(loc='upper right')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Repulsion Loss')
    ax.set_title('Head Repulsion (smoothed)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Causal Knot-Space Binding-Failure Learning', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close()
    print(f"Saved learning curves to {output_dir / 'learning_curves.png'}")


def eval_actor(actor, n_episodes: int = 20, seed: int = 1000):
    """Evaluate trained actor."""
    from v2.environments.obstacle_nav import Obstacle
    
    # Same fixed obstacles as training
    fixed_obstacles = [
        Obstacle(center=np.array([0.5, 0.5, 0.0]), radius=0.4),
    ]
    
    env_config = ObstacleNavConfig(
        n_obstacles=len(fixed_obstacles),
        T=20,
        max_steps=40,
        max_speed=0.15,
        goal_radius=0.25,
        randomize_obstacles=False,
        fixed_obstacles=fixed_obstacles,
        randomize_start=False,
        randomize_goal=False,
        fixed_start=np.array([0.0, 0.0, 0.0]),
        fixed_goal=np.array([1.0, 1.0, 0.0]),
    )
    env = ObstacleNavEnv(env_config)
    device = next(actor.parameters()).device
    
    successes = 0
    total_collisions = 0
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        s0 = torch.tensor(obs, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            z_star = actor.select_z_geometric_multimodal(s0)
            mu, sigma = actor.get_tube(s0, z_star)
            mu_np = mu.squeeze(0).cpu().numpy()
        
        x0 = obs[:3].copy()  # Position is first 3 elements
        
        for t in range(env_config.T):
            target_delta = mu_np[t]
            target_pos = x0 + target_delta
            current_pos = env.pos
            action = target_pos - current_pos
            
            _, _, done, info = env.step(action)
            
            if info['collision']:
                total_collisions += 1
            
            if done:
                break
        
        if info['reached_goal']:
            successes += 1
    
    print(f"\nEvaluation ({n_episodes} episodes):")
    print(f"  Success rate: {successes / n_episodes * 100:.1f}%")
    print(f"  Avg collisions: {total_collisions / n_episodes:.2f}")
    
    return successes / n_episodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train obstacle navigation with binding failure")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-only", type=str, default=None, help="Path to model for eval only")
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Load and evaluate
        actor = Actor(obs_dim=6, z_dim=4, pred_dim=3, T=20, n_knots=5, M=6)
        actor.load_state_dict(torch.load(args.eval_only))
        eval_actor(actor, n_episodes=50)
    else:
        # Train
        actor, history, output_dir = train_actor(
            n_episodes=args.episodes,
            warmup_episodes=args.warmup,
            seed=args.seed,
        )
        
        # Final evaluation
        eval_actor(actor, n_episodes=50)
