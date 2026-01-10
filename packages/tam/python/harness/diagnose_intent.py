"""
Diagnostic script to check if z_intent actually controls tube geometry.

The functor can only work if:
1. z_intent varies meaningfully across episodes
2. E[T] and cone_vol respond to z_intent changes
3. There's an ordering to preserve
"""

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
import torch

from factored_actor import FactoredActor, create_factored_actor
from harness.intent_functor import get_tube_signature

try:
    from envs.equivalent_envs import make_standard_gridworld
except ImportError:
    make_standard_gridworld = None


def diagnose_intent_variation(actor: FactoredActor, env, n_samples: int = 100):
    """
    Check if z_intent actually controls tube geometry.
    """
    device = actor.device
    
    # Collect samples
    z_intents = []
    E_Ts = []
    cone_vols = []
    
    for i in range(n_samples):
        obs = env.reset()
        s0_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Sample z_intent from posterior
        sample = actor.sample_z_factored(s0_t)
        z_intent = sample.z_intent.detach().cpu().numpy().squeeze()
        
        # Get tube signature
        sig = get_tube_signature(actor, sample.z_intent, s0_t, eval_horizon=16)
        
        z_intents.append(z_intent)
        E_Ts.append(sig.E_T)
        cone_vols.append(sig.log_cone_vol)
    
    z_intents = np.array(z_intents)
    E_Ts = np.array(E_Ts)
    cone_vols = np.array(cone_vols)
    
    print("\n" + "=" * 60)
    print("INTENT VARIATION DIAGNOSTICS")
    print("=" * 60)
    
    print(f"\n1. E[T] statistics:")
    print(f"   Mean: {E_Ts.mean():.4f}")
    print(f"   Std:  {E_Ts.std():.4f}")
    print(f"   Min:  {E_Ts.min():.4f}")
    print(f"   Max:  {E_Ts.max():.4f}")
    print(f"   Range: {E_Ts.max() - E_Ts.min():.4f}")
    
    if E_Ts.std() < 0.1:
        print("   ⚠️  WARNING: E[T] has very low variance!")
        print("   → z_intent is NOT controlling horizon effectively")
    
    print(f"\n2. Log cone volume statistics:")
    print(f"   Mean: {cone_vols.mean():.4f}")
    print(f"   Std:  {cone_vols.std():.4f}")
    print(f"   Range: {cone_vols.max() - cone_vols.min():.4f}")
    
    if cone_vols.std() < 0.5:
        print("   ⚠️  WARNING: Cone volume has very low variance!")
    
    print(f"\n3. z_intent statistics (per dimension):")
    for d in range(z_intents.shape[1]):
        print(f"   dim {d}: mean={z_intents[:, d].mean():.3f}, std={z_intents[:, d].std():.3f}")
    
    # Check correlation between z_intent and tube outputs
    print(f"\n4. Correlation between z_intent dims and E[T]:")
    for d in range(z_intents.shape[1]):
        corr = np.corrcoef(z_intents[:, d], E_Ts)[0, 1]
        marker = "✓" if abs(corr) > 0.3 else "✗"
        print(f"   dim {d} ↔ E[T]: r = {corr:.3f} {marker}")
    
    # Test: Does perturbing z_intent change outputs?
    print(f"\n5. Sensitivity test: perturbing z_intent")
    obs = env.reset()
    s0_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Base z_intent
    z_base = actor.get_intent_embedding(s0_t)
    sig_base = get_tube_signature(actor, z_base, s0_t, eval_horizon=16)
    
    # Perturbed z_intent (add +1 to each dim)
    for d in range(actor.z_intent_dim):
        z_perturbed = z_base.clone()
        z_perturbed[0, d] += 1.0
        sig_perturbed = get_tube_signature(actor, z_perturbed, s0_t, eval_horizon=16)
        
        delta_ET = sig_perturbed.E_T - sig_base.E_T
        delta_cone = sig_perturbed.log_cone_vol - sig_base.log_cone_vol
        
        sensitive = abs(delta_ET) > 0.1 or abs(delta_cone) > 0.1
        marker = "✓" if sensitive else "✗"
        print(f"   Δz[{d}] = +1.0 → ΔE[T] = {delta_ET:+.3f}, Δcone = {delta_cone:+.3f} {marker}")
    
    return z_intents, E_Ts, cone_vols


def plot_diagnostics(z_intents, E_Ts, cone_vols, save_path: Path = None):
    """Plot diagnostic visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. E[T] histogram
    ax = axes[0, 0]
    ax.hist(E_Ts, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(E_Ts.mean(), color='red', linestyle='--', label=f'Mean={E_Ts.mean():.2f}')
    ax.set_xlabel("E[T]")
    ax.set_ylabel("Count")
    ax.set_title(f"E[T] Distribution (std={E_Ts.std():.4f})")
    ax.legend()
    
    # 2. Cone vol histogram
    ax = axes[0, 1]
    ax.hist(cone_vols, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(cone_vols.mean(), color='red', linestyle='--', label=f'Mean={cone_vols.mean():.2f}')
    ax.set_xlabel("Log Cone Volume")
    ax.set_ylabel("Count")
    ax.set_title(f"Cone Vol Distribution (std={cone_vols.std():.4f})")
    ax.legend()
    
    # 3. E[T] vs cone vol scatter
    ax = axes[1, 0]
    ax.scatter(E_Ts, cone_vols, alpha=0.5)
    ax.set_xlabel("E[T]")
    ax.set_ylabel("Log Cone Volume")
    corr = np.corrcoef(E_Ts, cone_vols)[0, 1]
    ax.set_title(f"E[T] vs Cone Vol (r={corr:.3f})")
    
    # 4. z_intent PCA
    ax = axes[1, 1]
    if z_intents.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_intents)
    else:
        z_2d = z_intents[:, :2]
    
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=E_Ts, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='E[T]')
    ax.set_xlabel("z_intent PC1")
    ax.set_ylabel("z_intent PC2")
    ax.set_title("z_intent colored by E[T]")
    
    plt.suptitle("Intent Variation Diagnostics")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.close(fig)


def main():
    from utils import truncated_geometric_weights
    
    print("Training actor and running diagnostics...")
    
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
        "reasoning_mode": "fixed",
        "intent_only_sigma": True,
        "intent_only_stop": True,
    }
    
    env = make_standard_gridworld({"seed": 0})
    actor = create_factored_actor(**actor_kwargs)
    
    # Quick training
    print("\nTraining for 2000 steps...")
    for step in range(2000):
        obs0 = env.reset()
        s0_t = torch.tensor(obs0, dtype=torch.float32, device=actor.device).unsqueeze(0)
        
        sample = actor.sample_z_factored(s0_t)
        z = sample.z
        
        _, _, stop_logit = actor._tube_init(z, s0_t)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        _, E_T = truncated_geometric_weights(p_stop, actor.maxH)
        
        def policy_fn(obs_np):
            ot = torch.tensor(obs_np, dtype=torch.float32, device=actor.device).unsqueeze(0)
            action = actor._policy_action(z, ot).detach().cpu().numpy().squeeze()
            return int(np.argmax(action))
        
        _, state_seq, actions, info = env.rollout(policy_fn=policy_fn, horizon=actor.maxH)
        
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
            print(f"  Step {step}")
    
    # Run diagnostics
    z_intents, E_Ts, cone_vols = diagnose_intent_variation(actor, env, n_samples=100)
    
    # Plot
    Path("runs").mkdir(exist_ok=True)
    plot_diagnostics(z_intents, E_Ts, cone_vols, save_path=Path("runs/intent_diagnostics.png"))
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    if E_Ts.std() < 0.1:
        print("""
ROOT CAUSE: E[T] has no variation.

The tube is outputting nearly identical horizons for all z_intent values.
This means z_intent is NOT controlling commitment geometry.

Possible fixes:
1. Check if stop_logit head is actually connected to z_intent
2. Add explicit "diversity" objective during training
3. Train longer with higher KL on z_intent to spread the latent space
4. Add "intent knobs" - explicit directions in z-space that widen/lengthen
""")


if __name__ == "__main__":
    main()
