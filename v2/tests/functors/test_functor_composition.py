"""
Test 3: Functor Composition Law

Tests the category-theoretic composition: F_BC(F_AB(z)) ≈ F_AC(z)

If composition holds, we have genuine structure preservation (not memorization).

Run with: pytest test_functor_composition.py -v
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

from .utils_energy import sample_z_sphere, sample_z_angles, compute_energy
from .utils_basins import cluster_by_energy, assign_to_nearest
from .zmap import ZMap, FunctorTrainer, CompositionChecker
from .world_variants import (
    create_world_A, create_world_B, create_world_C,
    CMGWorldVariant,
)


if HAS_PYTEST:
    @pytest.fixture
    def device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @pytest.fixture
    def z_dim():
        return 2


def train_actor(env, device, z_dim, n_epochs=200):
    """Train a single actor."""
    from v2.actors.actor import Actor
    
    actor = Actor(
        obs_dim=env.obs_dim,
        z_dim=z_dim,
        pred_dim=env.state_dim,
        T=env.T,
    ).to(device)
    
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    
    for _ in range(n_epochs):
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
        actor.train_step_wta(s0, traj_delta, optimizer)
    
    return actor


def train_functor(actor_src, actor_tgt, env_src, env_tgt, device, z_dim, n_epochs=300):
    """Train functor between two actors."""
    trainer = FunctorTrainer(
        actor_A=actor_src,
        actor_B=actor_tgt,
        z_dim=z_dim,
        device=device,
    )
    trainer.train(env_src, env_tgt, n_epochs=n_epochs, print_every=100)
    return trainer.F


if HAS_PYTEST:
    @pytest.fixture
    def three_world_setup(device, z_dim):
        """Set up three worlds A, B, C with trained actors and functors."""
        # Create environments
        env_A = CMGWorldVariant(create_world_A())
        env_B = CMGWorldVariant(create_world_B())
        env_C = CMGWorldVariant(create_world_C())
        
        # Train actors
        print("Training actors...")
        actor_A = train_actor(env_A, device, z_dim, n_epochs=150)
        actor_B = train_actor(env_B, device, z_dim, n_epochs=150)
        actor_C = train_actor(env_C, device, z_dim, n_epochs=150)
        
        # Train functors
        print("Training functors...")
        F_AB = train_functor(actor_A, actor_B, env_A, env_B, device, z_dim, n_epochs=200)
        F_BC = train_functor(actor_B, actor_C, env_B, env_C, device, z_dim, n_epochs=200)
        F_AC = train_functor(actor_A, actor_C, env_A, env_C, device, z_dim, n_epochs=200)
        
        return {
            'actors': (actor_A, actor_B, actor_C),
            'envs': (env_A, env_B, env_C),
            'functors': (F_AB, F_BC, F_AC),
        }


class TestComposition:
    """Test F_BC ∘ F_AB ≈ F_AC."""
    
    def test_composition_similarity(self, three_world_setup, device, z_dim):
        """Check that composed and direct maps give similar results."""
        F_AB, F_BC, F_AC = three_world_setup['functors']
        
        n_samples = 200
        z_A = sample_z_sphere(n_samples, z_dim, device)
        
        # Direct: A -> C
        z_AC = F_AC(z_A)
        
        # Composed: A -> B -> C
        z_AB = F_AB(z_A)
        z_ABC = F_BC(z_AB)
        
        # Compare
        z_AC_norm = torch.nn.functional.normalize(z_AC, dim=-1)
        z_ABC_norm = torch.nn.functional.normalize(z_ABC, dim=-1)
        
        sim = (z_AC_norm * z_ABC_norm).sum(dim=-1)
        mean_sim = sim.mean().item()
        
        print(f"Composition similarity: {mean_sim:.3f}")
        
        # Should be better than random (0.0)
        assert mean_sim > 0.3, f"Composition too weak: {mean_sim}"
    
    def test_angular_composition_error(self, three_world_setup, device, z_dim):
        """Measure angular error in composition (z_dim=2)."""
        if z_dim != 2:
            pytest.skip("Angular test only for z_dim=2")
        
        F_AB, F_BC, F_AC = three_world_setup['functors']
        
        z_A = sample_z_angles(200, device)
        z_AC = F_AC(z_A)
        z_ABC = F_BC(F_AB(z_A))
        
        angles_direct = torch.atan2(z_AC[:, 1], z_AC[:, 0])
        angles_composed = torch.atan2(z_ABC[:, 1], z_ABC[:, 0])
        
        diff = angles_direct - angles_composed
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        mean_error = (diff.abs() * 180 / np.pi).mean().item()
        print(f"Angular composition error: {mean_error:.1f}°")
        
        assert mean_error < 60, f"Angular error too high: {mean_error}°"


    class TestCompositionChecker:
        """Use the CompositionChecker utility."""
        
        def test_checker(self, three_world_setup, device, z_dim):
            F_AB, F_BC, F_AC = three_world_setup['functors']
            
            checker = CompositionChecker(F_AB, F_BC, F_AC, device)
            results = checker.check(n_samples=300, z_dim=z_dim)
            
            print(f"Composition check results:")
            for k, v in results.items():
                print(f"  {k}: {v}")
            
            assert results['mean_cosine_sim'] > 0.3


def run_composition_test(output_dir: Optional[Path] = None, n_epochs: int = 300):
    """Run full composition test with visualizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 2
    
    # Create environments
    env_A = CMGWorldVariant(create_world_A())
    env_B = CMGWorldVariant(create_world_B())
    env_C = CMGWorldVariant(create_world_C())
    
    print("Training actors...")
    actor_A = train_actor(env_A, device, z_dim, n_epochs)
    actor_B = train_actor(env_B, device, z_dim, n_epochs)
    actor_C = train_actor(env_C, device, z_dim, n_epochs)
    
    print("Training functors...")
    F_AB = train_functor(actor_A, actor_B, env_A, env_B, device, z_dim, n_epochs)
    F_BC = train_functor(actor_B, actor_C, env_B, env_C, device, z_dim, n_epochs)
    F_AC = train_functor(actor_A, actor_C, env_A, env_C, device, z_dim, n_epochs)
    
    print("Checking composition...")
    checker = CompositionChecker(F_AB, F_BC, F_AC, device)
    results = checker.check(n_samples=500, z_dim=z_dim)
    
    print(f"\nResults: {results}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize
        z_A = sample_z_angles(200, device)
        z_AC = F_AC(z_A)
        z_ABC = F_BC(F_AB(z_A))
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        z_AC_np = z_AC.detach().cpu().numpy()
        z_ABC_np = z_ABC.detach().cpu().numpy()
        
        ax.scatter(z_AC_np[:, 0], z_AC_np[:, 1], c='blue', alpha=0.5, s=30, label='Direct (F_AC)')
        ax.scatter(z_ABC_np[:, 0], z_ABC_np[:, 1], c='red', alpha=0.5, s=30, label='Composed (F_BC∘F_AB)')
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f"Functor Composition (sim={results['mean_cosine_sim']:.3f})")
        
        plt.savefig(output_dir / 'functor_composition_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_dir / 'functor_composition_test.png'}")
    
    return results


if __name__ == "__main__":
    output_dir = Path("artifacts/functor_tests")
    results = run_composition_test(output_dir, n_epochs=300)
    print(f"\nComposition holds: {results.get('composition_holds', False)}")
    