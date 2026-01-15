"""
Test 1: Rank-Preserving Functor + Warm-Start Transfer

Tests:
1. Energy rank preservation: If E_A(z_i) < E_A(z_j), then E_B(F(z_i)) < E_B(F(z_j))
2. Zero-shot warm-start: Mapped z_A* performs nearly as well as B-CEM
3. Basin confusion matrix approaches permutation

Run with: pytest test_functor_rank.py -v
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Import functor utilities
from .utils_energy import (
    sample_z_sphere,
    sample_z_angles,
    compute_energy,
    compute_energy_landscape,
    get_basin_centers,
)
from .utils_basins import (
    cluster_by_energy,
    match_basins_hungarian,
    compute_basin_confusion_matrix,
    is_permutation_matrix,
)
from .zmap import ZMap, FunctorTrainer, rank_preserving_loss
from .world_variants import (
    create_world_A,
    create_world_B,
    create_paired_envs,
    CMGWorldVariant,
)


if HAS_PYTEST:
    @pytest.fixture
    def device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @pytest.fixture
    def z_dim():
        return 2


    @pytest.fixture
    def trained_actors(device, z_dim):
        """Train actors A and B independently."""
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
        
        # Train actors (brief training for tests)
        optimizer_A = torch.optim.Adam(actor_A.parameters(), lr=1e-3)
        optimizer_B = torch.optim.Adam(actor_B.parameters(), lr=1e-3)
        
        n_epochs = 200
        for epoch in range(n_epochs):
            # Train A
            obs_A = env_A.reset()
            s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
            traj_A = []
            curr_obs = obs_A
            for t in range(env_A.T):
                action = np.random.randn(env_A.state_dim).astype(np.float32) * 0.1
                curr_obs, _, _, _ = env_A.step(action)
                traj_A.append(curr_obs[:env_A.state_dim])
            traj_delta_A = torch.tensor(traj_A, dtype=torch.float32, device=device) - s0_A[:env_A.state_dim]
            actor_A.train_step_wta(s0_A, traj_delta_A, optimizer_A)
            
            # Train B
            obs_B = env_B.reset()
            s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device)
            traj_B = []
            curr_obs = obs_B
            for t in range(env_B.T):
                action = np.random.randn(env_B.state_dim).astype(np.float32) * 0.1
                curr_obs, _, _, _ = env_B.step(action)
                traj_B.append(curr_obs[:env_B.state_dim])
            traj_delta_B = torch.tensor(traj_B, dtype=torch.float32, device=device) - s0_B[:env_B.state_dim]
            actor_B.train_step_wta(s0_B, traj_delta_B, optimizer_B)
        
        return actor_A, actor_B, env_A, env_B


    @pytest.fixture
    def trained_functor(trained_actors, device, z_dim):
        """Train functor F: Z_A -> Z_B."""
        actor_A, actor_B, env_A, env_B = trained_actors
        
        trainer = FunctorTrainer(
            actor_A=actor_A,
            actor_B=actor_B,
            z_dim=z_dim,
            device=device,
        )
        
        trainer.train(env_A, env_B, n_epochs=300, print_every=100)
        
        return trainer


if HAS_PYTEST:
    class TestRankPreservation:
        """Test that the functor preserves energy ranking."""
        
        def test_rank_correlation(self, trained_functor, trained_actors, device, z_dim):
            """
            Check that energy rankings are preserved under mapping.
            
            F should map low-energy z in A to low-energy z in B.
            """
            actor_A, actor_B, env_A, env_B = trained_actors
            F = trained_functor.F
            
            # Sample situation
            obs_A = env_A.reset()
            obs_B = env_B.reset()
            s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
            s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device)
            
            # Sample z
            n_samples = 200
            z_A = sample_z_sphere(n_samples, z_dim, device)
            
            # Compute energies
            energy_A = compute_energy(actor_A, s0_A, z_A)
            
            # Map
            z_mapped = F(z_A)
            
            # Compute energy of mapped
            energy_B_mapped = compute_energy(actor_B, s0_B, z_mapped)
            
            # Compute rank correlation
            from scipy.stats import spearmanr
            
            rank_A = torch.argsort(torch.argsort(energy_A)).cpu().numpy()
            rank_B_mapped = torch.argsort(torch.argsort(energy_B_mapped)).cpu().numpy()
            
            correlation, pvalue = spearmanr(rank_A, rank_B_mapped)
            
            print(f"Rank correlation: {correlation:.3f} (p={pvalue:.3e})")
            
            # Assert positive correlation (structure preservation)
            assert correlation > 0.3, f"Rank correlation too low: {correlation}"
    
    def test_energy_order_pairs(self, trained_functor, trained_actors, device, z_dim):
        """
        Check that pairwise energy ordering is preserved.
        
        If E_A(z_i) < E_A(z_j), then usually E_B(F(z_i)) < E_B(F(z_j)).
        """
        actor_A, actor_B, env_A, env_B = trained_actors
        F = trained_functor.F
        
        obs_A = env_A.reset()
        obs_B = env_B.reset()
        s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
        s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device)
        
        n_samples = 100
        z_A = sample_z_sphere(n_samples, z_dim, device)
        
        energy_A = compute_energy(actor_A, s0_A, z_A)
        z_mapped = F(z_A)
        energy_B_mapped = compute_energy(actor_B, s0_B, z_mapped)
        
        # Check pairwise ordering
        n_pairs = 500
        preserved = 0
        total = 0
        
        for _ in range(n_pairs):
            i, j = np.random.choice(n_samples, 2, replace=False)
            if energy_A[i] < energy_A[j]:
                if energy_B_mapped[i] < energy_B_mapped[j]:
                    preserved += 1
                total += 1
            elif energy_A[i] > energy_A[j]:
                if energy_B_mapped[i] > energy_B_mapped[j]:
                    preserved += 1
                total += 1
        
        preservation_rate = preserved / total if total > 0 else 0
        print(f"Order preservation rate: {preservation_rate:.2%}")
        
        # Should preserve more than random (50%)
        assert preservation_rate > 0.55, f"Order preservation too low: {preservation_rate}"


    class TestWarmStartTransfer:
        """Test zero-shot warm-start transfer from A to B."""
        
        def test_warm_start_performance(self, trained_functor, trained_actors, device, z_dim):
            """
            Test that mapped z_A* performs well in B.
            
            Steps:
            1. Find z_A* (best z in A)
            2. Map: z_hat_B = F(z_A*)
            3. Compare z_hat_B to B-CEM and random
            """
            actor_A, actor_B, env_A, env_B = trained_actors
            F = trained_functor.F
            
            n_trials = 10
            transfer_energies = []
            cem_energies = []
            random_energies = []
            
            for trial in range(n_trials):
                obs_A = env_A.reset()
                obs_B = env_B.reset()
                s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
                s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device)
                
                # Find best z in A (mini CEM)
                n_cem = 50
                z_candidates_A = sample_z_sphere(n_cem, z_dim, device)
                energy_candidates_A = compute_energy(actor_A, s0_A, z_candidates_A)
                best_idx = energy_candidates_A.argmin()
                z_A_star = z_candidates_A[best_idx]
                
                # Map to B
                z_hat_B = F(z_A_star)
                
                # Evaluate in B
                energy_transfer = compute_energy(actor_B, s0_B, z_hat_B).item()
                transfer_energies.append(energy_transfer)
                
                # Compare to B-CEM
                z_candidates_B = sample_z_sphere(n_cem, z_dim, device)
                energy_candidates_B = compute_energy(actor_B, s0_B, z_candidates_B)
                energy_cem = energy_candidates_B.min().item()
                cem_energies.append(energy_cem)
                
                # Compare to random
                z_random = sample_z_sphere(1, z_dim, device)
                energy_random = compute_energy(actor_B, s0_B, z_random).item()
                random_energies.append(energy_random)
            
            # Statistics
            transfer_mean = np.mean(transfer_energies)
            cem_mean = np.mean(cem_energies)
            random_mean = np.mean(random_energies)
            
            print(f"Transfer energy: {transfer_mean:.3f}")
            print(f"B-CEM energy: {cem_mean:.3f}")
            print(f"Random energy: {random_mean:.3f}")
            
            # Transfer should be closer to CEM than to random
            transfer_gap = transfer_mean - cem_mean
            random_gap = random_mean - cem_mean
            
            relative_performance = 1 - (transfer_gap / random_gap) if random_gap > 0 else 0
            print(f"Relative performance: {relative_performance:.2%}")
            
            # Should be significantly better than random
            assert transfer_mean < random_mean, "Transfer should be better than random"


    class TestBasinConfusion:
        """Test that basin structure is preserved (confusion matrix is near-permutation)."""
        
        def test_basin_confusion_matrix(self, trained_functor, trained_actors, device, z_dim):
            """
            Build confusion matrix and check it's close to permutation.
            """
            actor_A, actor_B, env_A, env_B = trained_actors
            F = trained_functor.F
            
            obs_A = env_A.reset()
            obs_B = env_B.reset()
            s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
            s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device)
            
            n_samples = 300
            n_basins = 4  # Same as K
            
            # Get samples and cluster
            z_A = sample_z_sphere(n_samples, z_dim, device)
            energy_A = compute_energy(actor_A, s0_A, z_A)
            
            clusters_A = cluster_by_energy(z_A, energy_A, n_basins)
            labels_A = clusters_A['labels']
            
            # Get B basin centers
            z_B = sample_z_sphere(n_samples, z_dim, device)
            energy_B = compute_energy(actor_B, s0_B, z_B)
            clusters_B = cluster_by_energy(z_B, energy_B, n_basins)
            centers_B = clusters_B['centers']
            
            # Map A samples
            z_mapped = F(z_A)
            
            # Build confusion matrix
            confusion = compute_basin_confusion_matrix(
                z_A, labels_A, z_mapped, centers_B,
                n_basins, n_basins
            )
            
            # Normalize
            row_sums = confusion.sum(dim=1, keepdim=True).clamp(min=1)
            confusion_norm = confusion / row_sums
            
            print("Confusion matrix:")
            print(confusion_norm.cpu().numpy())
            
            # Check if close to permutation
            is_perm = is_permutation_matrix(confusion, threshold=0.5)
            print(f"Is permutation-like: {is_perm}")
            
            # Compute diagonal mass after optimal matching
            from scipy.optimize import linear_sum_assignment
            cost = (1 - confusion_norm).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_mass = sum(confusion_norm[i, j].item() for i, j in zip(row_ind, col_ind))
            matched_fraction = matched_mass / len(row_ind)
            
            print(f"Matched fraction: {matched_fraction:.2%}")
            
            # Should have decent matching
            assert matched_fraction > 0.4, f"Poor basin matching: {matched_fraction}"


def run_full_rank_test(output_dir: Optional[Path] = None, n_epochs: int = 500):
    """
    Run full rank preservation test with visualizations.
    
    Args:
        output_dir: Directory to save plots
        n_epochs: Training epochs
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
        # Train both actors
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
        
        if (epoch + 1) % 100 == 0:
            print(f"  Actor training: {epoch + 1}/{n_epochs}")
    
    print("Training functor...")
    trainer = FunctorTrainer(
        actor_A=actor_A,
        actor_B=actor_B,
        z_dim=z_dim,
        device=device,
    )
    trainer.train(env_A, env_B, n_epochs=500, print_every=100)
    
    print("\nEvaluating...")
    
    # Get a situation for visualization
    obs_A = env_A.reset()
    obs_B = env_B.reset()
    s0_A = torch.tensor(obs_A, dtype=torch.float32, device=device)
    s0_B = torch.tensor(obs_B, dtype=torch.float32, device=device)
    
    # Sample and evaluate
    n_samples = 500
    z_A = sample_z_angles(n_samples, device) if z_dim == 2 else sample_z_sphere(n_samples, z_dim, device)
    
    energy_A = compute_energy(actor_A, s0_A, z_A)
    z_mapped = trainer.F(z_A)
    energy_B_mapped = compute_energy(actor_B, s0_B, z_mapped)
    
    # Compute correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(energy_A.cpu().numpy(), energy_B_mapped.cpu().numpy())
    print(f"Rank correlation: {corr:.3f}")
    
    # Plot if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Energy landscapes
        if z_dim == 2:
            angles = torch.atan2(z_A[:, 1], z_A[:, 0]).detach().cpu().numpy()
            
            axes[0, 0].scatter(angles, energy_A.detach().cpu().numpy(), c='blue', alpha=0.5, s=10)
            axes[0, 0].set_xlabel('Angle')
            axes[0, 0].set_ylabel('Energy')
            axes[0, 0].set_title('Energy Landscape A')
            
            angles_mapped = torch.atan2(z_mapped[:, 1], z_mapped[:, 0]).detach().cpu().numpy()
            axes[0, 1].scatter(angles_mapped, energy_B_mapped.detach().cpu().numpy(), c='red', alpha=0.5, s=10)
            axes[0, 1].set_xlabel('Angle')
            axes[0, 1].set_ylabel('Energy')
            axes[0, 1].set_title('Mapped Energy in B')
        
        # Rank scatter
        axes[1, 0].scatter(energy_A.detach().cpu().numpy(), energy_B_mapped.detach().cpu().numpy(), alpha=0.5, s=10)
        axes[1, 0].set_xlabel('Energy in A')
        axes[1, 0].set_ylabel('Mapped Energy in B')
        axes[1, 0].set_title(f'Energy Correlation (ρ={corr:.3f})')
        
        # Training loss
        losses = [h['loss'] for h in trainer.history]
        axes[1, 1].plot(losses)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Functor Training Loss')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'functor_rank_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_dir / 'functor_rank_test.png'}")
    
    return {
        'rank_correlation': corr,
        'actor_A': actor_A,
        'actor_B': actor_B,
        'F': trainer.F,
        'G': trainer.G,
    }


if __name__ == "__main__":
    # Run standalone test
    output_dir = Path("artifacts/functor_tests")
    results = run_full_rank_test(output_dir, n_epochs=500)
    print(f"\nFinal rank correlation: {results['rank_correlation']:.3f}")
