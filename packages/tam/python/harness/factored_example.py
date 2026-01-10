"""
Example script for factored actor training and intent functor learning.

V2: Uses ordering-based loss and tests on multiple environment pairs.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import torch

from factored_actor import FactoredActor, create_factored_actor
from harness.intent_functor import (
    IntentFunctorConfig,
    run_intent_functor_experiment,
)

try:
    from envs.equivalent_envs import (
        make_standard_gridworld,
        make_mirrored_gridworld,
        make_rotated_gridworld,
        make_scaled_gridworld,
        make_shifted_rules_gridworld,
    )
except ImportError:
    print("Could not import environments")
    make_standard_gridworld = None


def train_factored_actor(
    env_factory,
    env_kwargs,
    actor_kwargs,
    train_steps: int = 3000,
    seed: int = 0,
    name: str = "actor",
) -> FactoredActor:
    """Train a FactoredActor on a given environment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_factory(env_kwargs)
    actor = create_factored_actor(**actor_kwargs)
    
    print(f"\nTraining FactoredActor '{name}'...")
    print(f"  z_intent_dim: {actor.z_intent_dim}, z_real_dim: {actor.z_real_dim}")
    
    for step in range(train_steps):
        obs0 = env.reset()
        s0_t = torch.tensor(obs0, dtype=torch.float32, device=actor.device).unsqueeze(0)
        
        sample = actor.sample_z_factored(s0_t)
        z = sample.z
        
        _, _, stop_logit = actor._tube_init(z, s0_t)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        from utils import truncated_geometric_weights
        _, E_T = truncated_geometric_weights(p_stop, actor.maxH)
        
        def policy_fn(obs_np):
            ot = torch.tensor(obs_np, dtype=torch.float32, device=actor.device).unsqueeze(0)
            action = actor._policy_action(z, ot).detach().cpu().numpy().squeeze()
            if actor.action_dim == 1:
                return float(np.clip(float(action), -2.0, 2.0))
            else:
                return int(np.argmax(action))
        
        obs_seq, state_seq, actions, info = env.rollout(policy_fn=policy_fn, horizon=actor.maxH)
        
        actor.train_on_episode(
            step=step,
            regime=info.get("rule", 0),
            s0=obs0,
            states=state_seq,
            actions=actions,
            z=z,
            z_mu=sample.z_mu,
            z_logstd=sample.z_logstd,
            E_T_imagine=float(E_T.item()),
            z_intent=sample.z_intent,
            z_real=sample.z_real,
            z_intent_mu=sample.z_intent_mu,
            z_intent_logstd=sample.z_intent_logstd,
            z_real_mu=sample.z_real_mu,
            z_real_logstd=sample.z_real_logstd,
        )
        
        if step % 500 == 0:
            recent_bind = np.mean(actor.history["bind_success"][-50:]) if actor.history["bind_success"] else 0
            recent_ET = np.mean(actor.history["E_T_train"][-50:]) if actor.history["E_T_train"] else 0
            print(f"  Step {step}: bind={recent_bind:.3f}, E[T]={recent_ET:.1f}")
    
    print(f"  Training complete.")
    return actor


def main():
    """Single environment pair test."""
    print("=" * 60)
    print("Factored Actor + Intent Functor (Ordering Loss)")
    print("=" * 60)
    
    actor_kwargs = {
        "obs_dim": 9,
        "pred_dim": 2,
        "action_dim": 4,
        "z_intent_dim": 4,
        "z_real_dim": 4,
        "maxH": 64,
        "minT": 2,
        "M": 16,
        "k_sigma": 2.0,
        "bind_success_frac": 0.85,
        "lambda_h": 0.002,
        "beta_kl": 3e-4,
        "halt_bias": -1.0,
        "reasoning_mode": "fixed",
        "intent_only_sigma": True,
        "intent_only_stop": True,
    }
    
    env_kwargs_A = {"seed": 0}
    env_kwargs_B = {"seed": 0, "mirror_x": True}
    
    # Train actors
    actor_A = train_factored_actor(
        env_factory=make_standard_gridworld,
        env_kwargs=env_kwargs_A,
        actor_kwargs=actor_kwargs,
        train_steps=2500,
        seed=0,
        name="standard",
    )
    
    actor_B = train_factored_actor(
        env_factory=make_mirrored_gridworld,
        env_kwargs=env_kwargs_B,
        actor_kwargs=actor_kwargs,
        train_steps=2500,
        seed=42,
        name="mirrored",
    )
    
    # Learn functor with ORDERING loss
    functor_config = IntentFunctorConfig(
        z_intent_dim=4,
        functor_type="affine",
        lr=1e-3,
        n_epochs=40,
        loss_type="ordering",  # Preserve relative ordering
        margin=0.1,
        intent_noise_std=0.2,  # More diversity
        k_sigma=2.0,
        eval_horizon=16,
    )
    
    trainer = run_intent_functor_experiment(
        actor_A=actor_A,
        actor_B=actor_B,
        env_factory_A=make_standard_gridworld,
        env_factory_B=make_mirrored_gridworld,
        env_kwargs_A=env_kwargs_A,
        env_kwargs_B=env_kwargs_B,
        config=functor_config,
        n_source_samples=150,
    )
    
    print(f"\nResults saved to: {trainer.run_dir}")


def run_all_pairs():
    """Test on all environment pairs: mirrored, rotated, scaled, shifted."""
    print("=" * 60)
    print("Multi-Environment Functor Evaluation")
    print("=" * 60)
    
    actor_kwargs = {
        "obs_dim": 9,
        "pred_dim": 2,
        "action_dim": 4,
        "z_intent_dim": 4,
        "z_real_dim": 4,
        "maxH": 64,
        "minT": 2,
        "M": 16,
        "k_sigma": 2.0,
        "bind_success_frac": 0.85,
        "lambda_h": 0.002,
        "beta_kl": 3e-4,
        "halt_bias": -1.0,
        "reasoning_mode": "fixed",
        "intent_only_sigma": True,
        "intent_only_stop": True,
    }
    
    # Train source actor (standard gridworld)
    actor_A = train_factored_actor(
        env_factory=make_standard_gridworld,
        env_kwargs={"seed": 0},
        actor_kwargs=actor_kwargs,
        train_steps=2500,
        seed=0,
        name="source_standard",
    )
    
    # Target environments (from easier to harder)
    targets = [
        ("mirrored", make_mirrored_gridworld, {"seed": 0, "mirror_x": True}),
        ("rotated_45", make_rotated_gridworld, {"seed": 0, "rotation_deg": 45.0}),
        ("scaled_1.5x", make_scaled_gridworld, {"seed": 0, "velocity_scale": 1.5}),
        ("shifted_rules", make_shifted_rules_gridworld, {"seed": 0, "rule_shift": 1}),
    ]
    
    results_summary = {}
    
    for name, env_factory_B, env_kwargs_B in targets:
        print(f"\n{'='*60}")
        print(f"Target: {name}")
        print("=" * 60)
        
        # Train target actor
        actor_B = train_factored_actor(
            env_factory=env_factory_B,
            env_kwargs=env_kwargs_B,
            actor_kwargs=actor_kwargs,
            train_steps=2000,
            seed=42,
            name=f"target_{name}",
        )
        
        # Learn functor
        functor_config = IntentFunctorConfig(
            z_intent_dim=4,
            functor_type="affine",
            lr=1e-3,
            n_epochs=30,
            loss_type="ordering",
            margin=0.1,
            intent_noise_std=0.2,
            output_dir=Path("runs") / f"functor_{name}",
        )
        
        trainer = run_intent_functor_experiment(
            actor_A=actor_A,
            actor_B=actor_B,
            env_factory_A=make_standard_gridworld,
            env_factory_B=env_factory_B,
            env_kwargs_A={"seed": 0},
            env_kwargs_B=env_kwargs_B,
            config=functor_config,
            n_source_samples=100,
        )
        
        if trainer.eval_results:
            res = trainer.eval_results[-1]
            results_summary[name] = {
                "E_T_spearman_trans": res["E_T"]["transported_spearman"],
                "E_T_spearman_id": res["E_T"]["identity_spearman"],
                "E_T_improvement": res["improvement"]["E_T_spearman"],
                "cone_improvement": res["improvement"]["cone_spearman"],
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Rank Preservation Across Environment Pairs")
    print("=" * 60)
    print(f"\n{'Environment':<20} {'E[T] ρ (trans)':<15} {'E[T] ρ (id)':<15} {'Δρ':<10}")
    print("-" * 60)
    
    for name, res in results_summary.items():
        print(f"{name:<20} {res['E_T_spearman_trans']:>12.3f} {res['E_T_spearman_id']:>14.3f} {res['E_T_improvement']:>+9.3f}")
    
    print("\nInterpretation:")
    print("  - Δρ > 0: Functor improves rank preservation over identity")
    print("  - Δρ ≈ 0: Functor doesn't help (or harm)")
    print("  - High ρ: Good rank ordering preservation")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_pairs()
    else:
        main()
