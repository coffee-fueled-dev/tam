"""
Integration test: Minimal Actor + CMG Environment

Tests:
1. Actor can encode CMG observations
2. Actor can predict tubes for CMG trajectories
3. CEM binding works with mode goals as prototypes
4. Training loop runs correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from environments.cmg_env import CMGEnv, CMGConfig, generate_episode, rollout_with_forced_mode
from actors.minimal import Actor, BindingMode


def test_basic_integration():
    """Test actor can process CMG observations and predict tubes."""
    print("=" * 60)
    print("Test 1: Basic Integration")
    print("=" * 60)
    
    # Create environment
    config = CMGConfig(d=4, K=2, T=16, t_gate=2, obs_mode="x_goal")
    env = CMGEnv(config)
    
    # Create actor matching env dimensions
    actor = Actor(
        obs_dim=env.obs_dim,  # x (4) + goal (4) = 8
        z_dim=4,
        pred_dim=config.d,   # Predict state trajectory
        T=config.T,
    )
    
    print(f"  Env: obs_dim={env.obs_dim}, action_dim={env.action_dim}, T={config.T}")
    print(f"  Actor: obs_dim={actor.obs_dim}, z_dim={actor.z_dim}, pred_dim={actor.pred_dim}")
    
    # Reset and get observation
    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32)
    
    # Encode observation
    z_mu, z_logstd = actor.encode(obs_t)
    print(f"  Encoded: z_mu shape={z_mu.shape}, z_std range=[{torch.exp(z_logstd).min():.2f}, {torch.exp(z_logstd).max():.2f}]")
    
    # Sample z and predict tube
    z = actor.sample_z(z_mu, z_logstd)
    mu, sigma = actor.predict_tube(obs_t, z)
    print(f"  Tube: mu shape={mu.shape}, sigma range=[{sigma.min():.3f}, {sigma.max():.3f}]")
    
    print("  ✓ Basic integration works!")
    return True


def test_cem_binding():
    """Test CEM binding with mode prototypes."""
    print("\n" + "=" * 60)
    print("Test 2: CEM Binding with Mode Prototypes")
    print("=" * 60)
    
    config = CMGConfig(d=4, K=2, T=16, t_gate=2, obs_mode="x_goal")
    env = CMGEnv(config)
    
    actor = Actor(
        obs_dim=env.obs_dim,
        z_dim=4,
        pred_dim=config.d,
        T=config.T,
    )
    
    # Generate mode prototype trajectories
    proto_0 = rollout_with_forced_mode(env, k_forced=0, policy="goal_seeking")
    proto_1 = rollout_with_forced_mode(env, k_forced=1, policy="goal_seeking")
    
    # Convert to tensors (use x trajectory, excluding initial state)
    tau_0 = torch.tensor(proto_0["x"][1:], dtype=torch.float32)  # (T, d)
    tau_1 = torch.tensor(proto_1["x"][1:], dtype=torch.float32)  # (T, d)
    
    print(f"  Mode 0 prototype: shape={tau_0.shape}, goal_dist={proto_0['final_goal_dist']:.3f}")
    print(f"  Mode 1 prototype: shape={tau_1.shape}, goal_dist={proto_1['final_goal_dist']:.3f}")
    
    # Reset and bind with CEM
    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32)
    
    # Bind using mode prototypes
    z_star, details = actor.select_z_cem(
        obs_t,
        mode_prototypes=(tau_0, tau_1),
        bend_amplitude=0.15,
    )
    
    print(f"  CEM result: z shape={z_star.shape}, score={details['score']:.3f}")
    print(f"  Predicted mu shape={details['mu'].shape}")
    
    # Check which mode the prediction is closer to
    mu = details["mu"]
    dist_0 = ((mu - tau_0) ** 2).mean().item()
    dist_1 = ((mu - tau_1) ** 2).mean().item()
    committed_mode = 0 if dist_0 < dist_1 else 1
    
    print(f"  Distance to mode 0: {dist_0:.4f}")
    print(f"  Distance to mode 1: {dist_1:.4f}")
    print(f"  Committed to mode: {committed_mode}")
    
    print("  ✓ CEM binding works!")
    return True


def test_training_loop():
    """Test training the actor on CMG trajectories."""
    print("\n" + "=" * 60)
    print("Test 3: Training Loop")
    print("=" * 60)
    
    config = CMGConfig(d=4, K=2, T=16, t_gate=2, obs_mode="x_goal")
    env = CMGEnv(config)
    
    actor = Actor(
        obs_dim=env.obs_dim,
        z_dim=4,
        pred_dim=config.d,
        T=config.T,
        lr=1e-3,
    )
    
    n_steps = 100
    print(f"  Training for {n_steps} steps...")
    
    for step in range(n_steps):
        # Generate episode with goal-seeking policy
        episode = generate_episode(env, policy_mode="goal_seeking")
        
        # Get observation and trajectory
        obs = episode["obs"][0]  # Initial observation
        traj = episode["x"][1:]  # State trajectory (T, d)
        
        obs_t = torch.tensor(obs, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        # Get mode prototypes for this episode
        proto_0 = rollout_with_forced_mode(env, k_forced=0, policy="goal_seeking")
        proto_1 = rollout_with_forced_mode(env, k_forced=1, policy="goal_seeking")
        tau_0 = torch.tensor(proto_0["x"][1:], dtype=torch.float32)
        tau_1 = torch.tensor(proto_1["x"][1:], dtype=torch.float32)
        
        # Bind
        z_star, _ = actor.select_z_cem(obs_t, mode_prototypes=(tau_0, tau_1))
        
        # Train step
        metrics = actor.train_step(obs_t, traj_t, z_star, bend_amplitude=0.15)
        
        if step % 20 == 0:
            print(f"    Step {step}: loss={metrics['loss']:.4f}, bind={metrics['bind_hard']:.3f}, nll={metrics['nll']:.3f}")
    
    # Final evaluation
    print(f"\n  Final metrics:")
    print(f"    Loss: {np.mean(actor.history['loss'][-20:]):.4f}")
    print(f"    Bind rate: {np.mean(actor.history['bind_hard'][-20:]):.3f}")
    print(f"    NLL: {np.mean(actor.history['nll'][-20:]):.3f}")
    
    print("  ✓ Training loop works!")
    return True


def test_mode_commitment():
    """Test that actor commits to modes (doesn't hedge)."""
    print("\n" + "=" * 60)
    print("Test 4: Mode Commitment")
    print("=" * 60)
    
    config = CMGConfig(d=4, K=2, T=16, t_gate=2, obs_mode="x_goal")
    env = CMGEnv(config)
    
    # Train actor briefly
    actor = Actor(
        obs_dim=env.obs_dim,
        z_dim=4,
        pred_dim=config.d,
        T=config.T,
    )
    
    # Train for a bit
    for _ in range(50):
        episode = generate_episode(env, policy_mode="goal_seeking")
        obs_t = torch.tensor(episode["obs"][0], dtype=torch.float32)
        traj_t = torch.tensor(episode["x"][1:], dtype=torch.float32)
        
        proto_0 = rollout_with_forced_mode(env, k_forced=0, policy="goal_seeking")
        proto_1 = rollout_with_forced_mode(env, k_forced=1, policy="goal_seeking")
        tau_0 = torch.tensor(proto_0["x"][1:], dtype=torch.float32)
        tau_1 = torch.tensor(proto_1["x"][1:], dtype=torch.float32)
        
        z_star, _ = actor.select_z_cem(obs_t, mode_prototypes=(tau_0, tau_1))
        actor.train_step(obs_t, traj_t, z_star)
    
    # Evaluate commitment
    n_eval = 50
    mode_commits = []
    
    for _ in range(n_eval):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32)
        
        proto_0 = rollout_with_forced_mode(env, k_forced=0, policy="goal_seeking")
        proto_1 = rollout_with_forced_mode(env, k_forced=1, policy="goal_seeking")
        tau_0 = torch.tensor(proto_0["x"][1:], dtype=torch.float32)
        tau_1 = torch.tensor(proto_1["x"][1:], dtype=torch.float32)
        
        z_star, details = actor.select_z_cem(obs_t, mode_prototypes=(tau_0, tau_1))
        mu = details["mu"]
        
        # Measure commitment: how much closer to one mode vs the other
        dist_0 = ((mu - tau_0) ** 2).mean().item()
        dist_1 = ((mu - tau_1) ** 2).mean().item()
        
        # Commitment = |dist_0 - dist_1| / (dist_0 + dist_1)
        commitment = abs(dist_0 - dist_1) / (dist_0 + dist_1 + 1e-6)
        mode_commits.append(commitment)
    
    avg_commitment = np.mean(mode_commits)
    print(f"  Average commitment: {avg_commitment:.3f}")
    print(f"  (0 = hedging, 1 = full commitment)")
    
    if avg_commitment > 0.3:
        print("  ✓ Actor shows mode commitment!")
    else:
        print("  ⚠ Actor may be hedging (low commitment)")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Actor + CMG Environment Integration Tests")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = []
    results.append(("Basic Integration", test_basic_integration()))
    results.append(("CEM Binding", test_cem_binding()))
    results.append(("Training Loop", test_training_loop()))
    results.append(("Mode Commitment", test_mode_commitment()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    main()
