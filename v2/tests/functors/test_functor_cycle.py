"""
Test 2: Cycle Consistency

Tests the functor law: G(F(z_A)) ≈ z_A (up to basin equivalence)

This is a strong indicator of structure preservation.
If the maps truly capture topology, round-tripping should return
to the same basin.

Run with: pytest test_functor_cycle.py -v
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

from .utils_energy import (
    sample_z_sphere,
    sample_z_angles,
    compute_energy,
    get_basin_centers,
)
from .utils_basins import (
    cluster_by_energy,
    assign_to_nearest,
)
from .zmap import ZMap, FunctorTrainer, cycle_consistency_loss
from .world_variants import create_paired_envs


if HAS_PYTEST:
    @pytest.fixture
    def device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @pytest.fixture
    def z_dim():
        return 2


    @pytest.fixture
    def trained_cycle_maps(device, z_dim):
        """Train functor pair F and G with cycle consistency."""
        from v2.actors.actor import Actor
        
        # Create environments
        env_A, env_B = create_paired_envs(variant="noise", seed=42)
        
        # Create actors
        actor_A = Actor(
            obs_dim=env_A.obs_dim,
            z_dim=z_dim,
            pred_dim=env_A.state_dim,
            T=env_A.T,
        ).to(device)
        
        actor_B = Actor(
            obs_dim=env_B.obs_dim,
            z_dim=z_dim,
            pred_dim=env_B.state_dim,
            T=env_B.T,
        ).to(device)
        
        # Brief actor training
        optimizer_A = torch.optim.Adam(actor_A.parameters(), lr=1e-3)
        optimizer_B = torch.optim.Adam(actor_B.parameters(), lr=1e-3)
        
        for epoch in range(200):
            for actor, env, opt in [(actor_A, env_A, optimizer_A), (actor_B, env_B, optimizer_B)]:
                obs = env.reset()
                s0 = torch.tensor(obs, dtype=torch.float32, device=device)
                traj = []
                curr_obs = obs
                for t in range(env.T):
                    action = np.random.randn(env.state_dim).astype(np.float32) * 0.1
                    curr_obs, _, _, _ = env.step(action)
                    traj.append(curr_obs[:env.state_dim])
                traj_delta = torch.tensor(traj, dtype=torch.float32, device=device) - s0[:env.state_dim]
                actor.train_step_wta(s0, traj_delta, opt)
        
        # Train functor with heavy cycle weight
        trainer = FunctorTrainer(
            actor_A=actor_A,
            actor_B=actor_B,
            z_dim=z_dim,
            device=device,
            cycle_weight=1.0,  # Heavy cycle consistency
            rank_weight=0.5,
        )
        
        trainer.train(env_A, env_B, n_epochs=400, print_every=100)
        
        return trainer, actor_A, actor_B, env_A, env_B


if HAS_PYTEST:
    class TestCycleConsistency:
        """Test G(F(z)) ≈ z cycle consistency."""
        
        def test_pointwise_cycle(self, trained_cycle_maps, device, z_dim):
            """
            Check that G(F(z_A)) ≈ z_A pointwise.
            """
            trainer, actor_A, actor_B, env_A, env_B = trained_cycle_maps
            F = trainer.F
            G = trainer.G
            
            n_samples = 200
            z_A = sample_z_sphere(n_samples, z_dim, device)
            
            # Forward: A -> B
            z_AB = F(z_A)
            
            # Backward: B -> A
            z_ABA = G(z_AB)
            
            # Compute cosine similarity
            z_A_norm = torch.nn.functional.normalize(z_A, p=2, dim=-1)
            z_ABA_norm = torch.nn.functional.normalize(z_ABA, p=2, dim=-1)
            
            sim = (z_A_norm * z_ABA_norm).sum(dim=-1)
            
            mean_sim = sim.mean().item()
            std_sim = sim.std().item()
            min_sim = sim.min().item()
            
            print(f"Cycle cosine similarity: {mean_sim:.3f} ± {std_sim:.3f} (min: {min_sim:.3f})")
            
            # Should have high similarity
            assert mean_sim > 0.8, f"Cycle consistency too low: {mean_sim}"
    
    def test_basin_cycle_consistency(self, trained_cycle_maps, device, z_dim):
        """
        Check that G(F(z)) lands in the same basin as z.
        
        Even if pointwise recovery isn't perfect, basin identity should be preserved.
        """
        trainer, actor_A, actor_B, env_A, env_B = trained_cycle_maps
        F = trainer.F
        G = trainer.G
        
        obs_A = env_A.reset()
        s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
        
        n_samples = 200
        n_basins = 4
        
        z_A = sample_z_sphere(n_samples, z_dim, device)
        energy_A = compute_energy(actor_A, s0_A, z_A)
        
        # Cluster
        clusters = cluster_by_energy(z_A, energy_A, n_basins)
        labels_A = clusters['labels']
        centers_A = clusters['centers']
        
        # Cycle
        z_AB = F(z_A)
        z_ABA = G(z_AB)
        
        # Assign cycled points to basins
        labels_cycled = assign_to_nearest(z_ABA, centers_A)
        
        # Compute basin preservation rate
        same_basin = (labels_A == labels_cycled).float().mean().item()
        
        print(f"Basin preservation rate: {same_basin:.2%}")
        
        # Should preserve basin identity most of the time
        assert same_basin > 0.6, f"Basin preservation too low: {same_basin}"
    
    def test_angular_error(self, trained_cycle_maps, device, z_dim):
        """
        Measure angular error in cycle (for z_dim=2).
        """
        if z_dim != 2:
            pytest.skip("Angular error test only for z_dim=2")
        
        trainer, actor_A, actor_B, env_A, env_B = trained_cycle_maps
        F = trainer.F
        G = trainer.G
        
        n_samples = 200
        z_A = sample_z_angles(n_samples, device)
        
        # Cycle
        z_ABA = G(F(z_A))
        
        # Angular error
        angles_original = torch.atan2(z_A[:, 1], z_A[:, 0])
        angles_cycled = torch.atan2(z_ABA[:, 1], z_ABA[:, 0])
        
        # Handle wrap-around
        diff = angles_original - angles_cycled
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))  # Wrap to [-π, π]
        
        angular_error_deg = (diff.abs() * 180 / np.pi)
        
        mean_error = angular_error_deg.mean().item()
        max_error = angular_error_deg.max().item()
        
        print(f"Angular error: {mean_error:.1f}° mean, {max_error:.1f}° max")
        
        # Should have low angular error
        assert mean_error < 45, f"Angular error too high: {mean_error}°"


    class TestBidirectionalConsistency:
        """Test both directions of cycle: A->B->A and B->A->B."""
        
        def test_both_cycles(self, trained_cycle_maps, device, z_dim):
            """
            Both cycles should have similar quality.
            """
            trainer, actor_A, actor_B, env_A, env_B = trained_cycle_maps
            F = trainer.F
            G = trainer.G
            
            n_samples = 200
            z_A = sample_z_sphere(n_samples, z_dim, device)
            z_B = sample_z_sphere(n_samples, z_dim, device)
            
            # Cycle A: A -> B -> A
            z_ABA = G(F(z_A))
            sim_A = (torch.nn.functional.normalize(z_A, dim=-1) * 
                    torch.nn.functional.normalize(z_ABA, dim=-1)).sum(dim=-1).mean().item()
            
            # Cycle B: B -> A -> B
            z_BAB = F(G(z_B))
            sim_B = (torch.nn.functional.normalize(z_B, dim=-1) * 
                    torch.nn.functional.normalize(z_BAB, dim=-1)).sum(dim=-1).mean().item()
            
            print(f"Cycle A similarity: {sim_A:.3f}")
            print(f"Cycle B similarity: {sim_B:.3f}")
            
            # Both should be high and similar
            assert sim_A > 0.7 and sim_B > 0.7, f"Cycles not consistent: A={sim_A}, B={sim_B}"
            assert abs(sim_A - sim_B) < 0.2, f"Cycles too asymmetric: A={sim_A}, B={sim_B}"


def run_cycle_test(output_dir: Optional[Path] = None, n_epochs: int = 500):
    """
    Run full cycle consistency test with visualizations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 2
    
    from v2.actors.actor import Actor
    
    print("Creating environments...")
    env_A, env_B = create_paired_envs(variant="noise", seed=42)
    
    print("Creating actors...")
    actor_A = Actor(
        obs_dim=env_A.obs_dim,
        z_dim=z_dim,
        pred_dim=env_A.state_dim,
        T=env_A.T,
    ).to(device)
    
    actor_B = Actor(
        obs_dim=env_B.obs_dim,
        z_dim=z_dim,
        pred_dim=env_B.state_dim,
        T=env_B.T,
    ).to(device)
    
    print(f"Training actors ({n_epochs} epochs each)...")
    optimizer_A = torch.optim.Adam(actor_A.parameters(), lr=1e-3)
    optimizer_B = torch.optim.Adam(actor_B.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        for actor, env, opt in [(actor_A, env_A, optimizer_A), (actor_B, env_B, optimizer_B)]:
            obs = env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32, device=device)
            traj = []
            curr_obs = obs
            for t in range(env.T):
                action = np.random.randn(env.state_dim).astype(np.float32) * 0.1
                curr_obs, _, _, _ = env.step(action)
                traj.append(curr_obs[:env.state_dim])
            traj_array = np.array(traj)
            traj_delta = torch.tensor(traj_array, dtype=torch.float32, device=device) - s0[:env.state_dim]
            actor.train_step_wta(s0, traj_delta, opt)
    
    print("Training functor with cycle consistency...")
    trainer = FunctorTrainer(
        actor_A=actor_A,
        actor_B=actor_B,
        z_dim=z_dim,
        device=device,
        cycle_weight=1.0,
        rank_weight=0.5,
    )
    trainer.train(env_A, env_B, n_epochs=600, print_every=100)
    
    print("\nEvaluating cycle consistency...")
    
    # Sample and cycle
    n_samples = 500
    z_A = sample_z_angles(n_samples, device)
    
    z_ABA = trainer.G(trainer.F(z_A))
    
    # Compute similarity
    sim = (torch.nn.functional.normalize(z_A, dim=-1) * 
           torch.nn.functional.normalize(z_ABA, dim=-1)).sum(dim=-1)
    
    mean_sim = sim.mean().item()
    print(f"Mean cycle similarity: {mean_sim:.3f}")
    
    # Plot if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original vs cycled
        z_A_np = z_A.detach().cpu().numpy()
        z_ABA_np = z_ABA.detach().cpu().numpy()
        
        axes[0].scatter(z_A_np[:, 0], z_A_np[:, 1], c='blue', alpha=0.5, s=20, label='Original')
        axes[0].scatter(z_ABA_np[:, 0], z_ABA_np[:, 1], c='red', alpha=0.5, s=20, label='Cycled')
        axes[0].add_patch(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        axes[0].set_xlim(-1.3, 1.3)
        axes[0].set_ylim(-1.3, 1.3)
        axes[0].set_aspect('equal')
        axes[0].legend()
        axes[0].set_title('Cycle A → B → A')
        
        # Similarity histogram
        axes[1].hist(sim.detach().cpu().numpy(), bins=30, edgecolor='black')
        axes[1].axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Cycle Consistency Distribution')
        axes[1].legend()
        
        # Training curves
        cycle_losses = [h['cycle_A'] for h in trainer.history]
        axes[2].plot(cycle_losses, label='Cycle Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Cycle Consistency Training')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'functor_cycle_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_dir / 'functor_cycle_test.png'}")
    
    return {
        'mean_similarity': mean_sim,
        'trainer': trainer,
    }


if __name__ == "__main__":
    output_dir = Path("artifacts/functor_tests")
    results = run_cycle_test(output_dir, n_epochs=500)
    print(f"\nFinal cycle similarity: {results['mean_similarity']:.3f}")
