"""
Ablation: Can a tube network learn to depend on s0?

No z, no actor, no policy. Just:
  s0 → tube_net → (mu_knots, sigma_knots, stop_logit)

Train to fit actual trajectories from the environment.
Success = different s0 produce visibly different tubes.

V2: Added binding objective over full trajectories to incentivize longer horizons.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple, List, Dict
from pathlib import Path
import time


class SimpleTubeNet(nn.Module):
    """MLP that maps s0 directly to tube parameters. No z."""
    
    def __init__(self, state_dim: int, pred_dim: int = 2, M: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.M = M
        self.pred_dim = pred_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate heads for mu, sigma, stop
        self.mu_head = nn.Linear(hidden_dim, pred_dim * M)
        self.sigma_head = nn.Linear(hidden_dim, pred_dim * M)
        self.stop_head = nn.Linear(hidden_dim, 1)
        
        # Initialize sigma to reasonable values
        nn.init.constant_(self.sigma_head.bias, 0.0)  # log(1) = 0
        nn.init.constant_(self.stop_head.bias, -1.0)  # p_stop ≈ 0.27
    
    def forward(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s0: [B, state_dim]
        Returns:
            mu_knots: [B, M, pred_dim]
            logsig_knots: [B, M, pred_dim]
            stop_logit: [B]
        """
        h = self.encoder(s0)
        
        mu = self.mu_head(h).view(-1, self.M, self.pred_dim)
        logsig = self.sigma_head(h).view(-1, self.M, self.pred_dim)
        stop_logit = self.stop_head(h).squeeze(-1)
        
        return mu, logsig, stop_logit


def interp_knots(knots: torch.Tensor, T: int) -> torch.Tensor:
    """Linearly interpolate M knots to T timesteps. Returns [B, T, D]."""
    B, M, D = knots.shape
    device = knots.device
    
    knot_times = torch.linspace(0, 1, M, device=device)
    query_times = torch.linspace(0, 1, T, device=device)
    
    # For each query time, find bracketing knots
    idx = torch.searchsorted(knot_times, query_times).clamp(1, M - 1)
    t0 = knot_times[idx - 1]
    t1 = knot_times[idx]
    
    # Interpolation weight: [T]
    alpha = (query_times - t0) / (t1 - t0 + 1e-8)
    
    # Gather knot values: [B, T, D]
    v0 = knots[:, idx - 1, :]  # [B, T, D]
    v1 = knots[:, idx, :]
    
    # Interpolate
    alpha = alpha.view(1, T, 1)
    return v0 + alpha * (v1 - v0)


def truncated_geometric_weights(p_stop: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute horizon distribution weights."""
    device = p_stop.device
    t_idx = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    one_minus = (1.0 - p_stop).clamp(1e-6, 1.0 - 1e-6)
    
    # Survival to time t-1, then stop
    s = one_minus ** (t_idx - 1.0)
    w = s * p_stop
    
    # Tail mass at T
    tail = one_minus ** T
    w = w.clone()
    w[-1] = w[-1] + tail
    w = w / (w.sum() + 1e-8)
    
    E_T = (w * t_idx).sum()
    return w, E_T


def compute_tube_loss(
    tube_net: SimpleTubeNet,
    s0: torch.Tensor,
    trajectory: torch.Tensor,  # [T, pred_dim]
    k_sigma: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for fitting trajectory with tube.
    
    Returns dict with loss components.
    """
    T = trajectory.shape[0]
    
    # Get tube from s0
    mu_knots, logsig_knots, stop_logit = tube_net(s0.unsqueeze(0))
    
    # Interpolate to T steps
    mu_traj = interp_knots(mu_knots, T).squeeze(0)  # [T, D]
    logsig_traj = interp_knots(logsig_knots, T).squeeze(0)
    std_traj = torch.exp(logsig_traj).clamp(0.01, 10.0)
    
    # Horizon weights
    p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1 - 1e-4)
    w, E_T = truncated_geometric_weights(p_stop, T)
    
    # NLL at each timestep: 0.5 * ((y - mu)^2 / var + log(var))
    var_traj = std_traj ** 2
    sq_err = (trajectory - mu_traj) ** 2
    nll_t = 0.5 * (sq_err / var_traj + torch.log(var_traj)).sum(dim=-1)  # [T]
    
    # Expected NLL weighted by horizon
    exp_nll = (w * nll_t).sum()
    
    # Cone volume (weighted)
    cone_vol_t = torch.prod(std_traj, dim=-1)  # [T]
    cone_vol = (w * cone_vol_t).sum()
    
    # Bind rate: is trajectory within k*sigma? (weighted by horizon)
    inside_t = (torch.abs(trajectory - mu_traj) < k_sigma * std_traj).all(dim=-1).float()
    bind_rate = (w * inside_t).sum()
    
    # ===== NEW: Full trajectory binding =====
    # Check if the ENTIRE trajectory (up to expected horizon) stays within tube
    # This incentivizes looking ahead
    
    # Compute cumulative bind success: did we stay inside from t=0 to t?
    cum_inside = torch.cumprod(inside_t, dim=0)  # [T]
    
    # Full trajectory bind: weighted by horizon distribution
    # If E[T] = 8, we need to stay inside for 8 steps
    full_bind = (w * cum_inside).sum()
    
    # Horizon-weighted bind penalty: if we predict long horizon, we MUST stay inside
    # This creates pressure to either: (1) shorten horizon for hard cases, or (2) widen tube
    
    # Compute "bind until t" rate for each t
    # At t=1: just step 1 inside
    # At t=8: steps 1-8 all inside
    bind_until_t = cum_inside  # [T]
    
    # Expected bind success at the predicted horizon
    horizon_bind = (w * bind_until_t).sum()
    
    return {
        "nll": exp_nll,
        "cone_vol": cone_vol,
        "bind_rate": bind_rate,  # Per-step bind (weighted)
        "full_bind": full_bind,  # Cumulative bind (must stay inside whole time)
        "horizon_bind": horizon_bind,  # Bind success weighted by horizon
        "E_T": E_T,
        "mu_traj": mu_traj.detach(),
        "std_traj": std_traj.detach(),
        "inside_t": inside_t.detach(),
    }


class SimpleEnv:
    """
    Minimal 2D environment with different dynamics per rule.
    
    Rules determine how the agent moves:
    - Rule 0: Straight to goal
    - Rule 1: Curved path (clockwise)
    - Rule 2: Curved path (counter-clockwise)
    - Rule 3: Noisy path
    """
    
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_rules = 4
        self.reset()
    
    def reset(self) -> np.ndarray:
        # Random start and goal
        self.x = self.rng.uniform(0.1, 0.9)
        self.y = self.rng.uniform(0.1, 0.9)
        self.goal_x = self.rng.uniform(0.1, 0.9)
        self.goal_y = self.rng.uniform(0.1, 0.9)
        self.rule = self.rng.integers(0, self.n_rules)
        return self.observe()
    
    def observe(self) -> np.ndarray:
        """Return s0: [x, y, goal_x, goal_y, rule_onehot(4)]"""
        rule_oh = np.zeros(self.n_rules)
        rule_oh[self.rule] = 1.0
        return np.array([self.x, self.y, self.goal_x, self.goal_y, *rule_oh], dtype=np.float32)
    
    def generate_trajectory(self, T: int = 16) -> np.ndarray:
        """Generate trajectory based on rule."""
        traj = np.zeros((T, 2), dtype=np.float32)
        
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        
        for t in range(T):
            progress = (t + 1) / T
            
            if self.rule == 0:
                # Straight line
                traj[t, 0] = self.x + dx * progress
                traj[t, 1] = self.y + dy * progress
            
            elif self.rule == 1:
                # Curved clockwise
                angle = progress * np.pi / 2
                traj[t, 0] = self.x + dx * progress + 0.1 * np.sin(angle * 2)
                traj[t, 1] = self.y + dy * progress - 0.1 * (1 - np.cos(angle * 2))
            
            elif self.rule == 2:
                # Curved counter-clockwise
                angle = progress * np.pi / 2
                traj[t, 0] = self.x + dx * progress - 0.1 * np.sin(angle * 2)
                traj[t, 1] = self.y + dy * progress + 0.1 * (1 - np.cos(angle * 2))
            
            elif self.rule == 3:
                # Noisy
                noise_scale = 0.15
                traj[t, 0] = self.x + dx * progress + self.rng.normal(0, noise_scale * (1 - progress))
                traj[t, 1] = self.y + dy * progress + self.rng.normal(0, noise_scale * (1 - progress))
        
        return traj


def train_tube(
    n_steps: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
    target_bind: float = 0.85,
    w_cone: float = 0.1,
    w_horizon_bind: float = 1.0,
    horizon_bonus: float = 0.05,
    mode: str = "full_traj",  # "horizon_weighted", "full_traj", "horizon_match"
) -> Tuple[SimpleTubeNet, Dict[str, List[float]]]:
    """
    Train the tube network to fit trajectories.
    
    Modes:
    - horizon_weighted: bind weighted by predicted horizon distribution
    - full_traj: must bind ENTIRE trajectory (all T steps)
    - horizon_match: predict horizon that matches trajectory length, bind for that
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = SimpleEnv(seed=seed)
    tube_net = SimpleTubeNet(state_dim=8, pred_dim=2, M=8, hidden_dim=64)
    optimizer = optim.Adam(tube_net.parameters(), lr=lr)
    
    lambda_bind = 1.0
    
    history = {
        "nll": [],
        "cone_vol": [],
        "bind_rate": [],
        "full_bind": [],
        "E_T": [],
    }
    
    print(f"Training tube network (no z) - Mode: {mode}")
    print(f"  Target bind: {target_bind}, Cone weight: {w_cone}")
    
    for step in range(n_steps):
        s0 = env.reset()
        traj = env.generate_trajectory(T=16)
        T = len(traj)
        
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        result = compute_tube_loss(tube_net, s0_t, traj_t)
        
        loss = result["nll"]
        loss = loss + w_cone * result["cone_vol"]
        
        if mode == "full_traj":
            # Must bind ENTIRE trajectory - no horizon weighting
            # This forces the tube to cover all T steps
            all_inside = result["inside_t"]  # [T]
            full_traj_bind = all_inside.prod()  # 1 if all inside, 0 otherwise
            
            # Soft version: fraction of steps inside
            full_traj_bind_soft = all_inside.mean()
            
            bind_gap = target_bind - full_traj_bind_soft
            loss = loss + lambda_bind * bind_gap
            
            # Track full trajectory bind
            result["full_traj_bind"] = full_traj_bind_soft
            
        elif mode == "horizon_match":
            # The horizon SHOULD match the trajectory length (T)
            # Penalize deviation from T
            E_T = result["E_T"]
            horizon_gap = (E_T - T) ** 2 / T  # Normalized squared error
            loss = loss + 0.1 * horizon_gap
            
            # Still require binding for predicted horizon
            bind_gap = target_bind - result["horizon_bind"]
            loss = loss + lambda_bind * bind_gap
            
        else:  # horizon_weighted
            bind_gap = target_bind - result["horizon_bind"]
            loss = loss + lambda_bind * bind_gap
            
            if result["horizon_bind"] > target_bind and horizon_bonus > 0:
                loss = loss - horizon_bonus * result["E_T"]
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tube_net.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            lambda_bind = float(np.clip(lambda_bind + 0.01 * bind_gap.item(), 0.01, 10.0))
        
        # Track
        full_bind_val = result.get("full_traj_bind", result["horizon_bind"])
        history["nll"].append(float(result["nll"].item()))
        history["cone_vol"].append(float(result["cone_vol"].item()))
        history["bind_rate"].append(float(result["bind_rate"].item()))
        history["full_bind"].append(float(full_bind_val.item()) if torch.is_tensor(full_bind_val) else float(full_bind_val))
        history["E_T"].append(float(result["E_T"].item()))
        
        if step % 500 == 0:
            print(f"  Step {step}: NLL={result['nll']:.3f}, bind={full_bind_val:.3f}, "
                  f"E[T]={result['E_T']:.2f}, λ_bind={lambda_bind:.2f}")
    
    return tube_net, history


def evaluate_situation_dependence(tube_net: SimpleTubeNet, n_samples: int = 100, T_eval: int = 16):
    """Check if tube outputs vary with s0."""
    env = SimpleEnv(seed=42)
    
    E_Ts = []
    cone_vols = []
    rules = []
    stop_logits = []
    
    for _ in range(n_samples):
        s0 = env.reset()
        s0_t = torch.tensor(s0, dtype=torch.float32)
        
        with torch.no_grad():
            mu_knots, logsig_knots, stop_logit = tube_net(s0_t.unsqueeze(0))
        
        # Use truncated geometric for E[T] (consistent with training)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1 - 1e-4)
        _, E_T = truncated_geometric_weights(p_stop, T_eval)
        
        std_knots = torch.exp(logsig_knots)
        cone_vol = torch.prod(std_knots.mean(dim=1)).item()
        
        E_Ts.append(float(E_T.item()))
        cone_vols.append(np.log(cone_vol + 1e-8))
        rules.append(env.rule)
        stop_logits.append(float(stop_logit.item()))
    
    E_Ts = np.array(E_Ts)
    cone_vols = np.array(cone_vols)
    rules = np.array(rules)
    stop_logits = np.array(stop_logits)
    
    print("\n" + "=" * 60)
    print("SITUATION DEPENDENCE TEST")
    print("=" * 60)
    
    print(f"\nStop logit stats:")
    print(f"  Mean: {stop_logits.mean():.3f}, Std: {stop_logits.std():.3f}")
    print(f"  Range: [{stop_logits.min():.3f}, {stop_logits.max():.3f}]")
    
    print(f"\nE[T] variation (truncated geometric, T={T_eval}):")
    print(f"  Mean: {E_Ts.mean():.3f}, Std: {E_Ts.std():.3f}")
    print(f"  Range: [{E_Ts.min():.3f}, {E_Ts.max():.3f}]")
    
    print(f"\nLog(cone_vol) variation:")
    print(f"  Mean: {cone_vols.mean():.3f}, Std: {cone_vols.std():.3f}")
    
    print(f"\nPer-rule statistics:")
    for r in range(4):
        mask = rules == r
        if mask.sum() > 0:
            print(f"  Rule {r}: E[T]={E_Ts[mask].mean():.2f}±{E_Ts[mask].std():.2f}, "
                  f"cone={cone_vols[mask].mean():.2f}±{cone_vols[mask].std():.2f}, "
                  f"stop={stop_logits[mask].mean():.2f}±{stop_logits[mask].std():.2f}")
    
    # Success criteria: meaningful variation in either dimension
    E_T_varies = E_Ts.std() > 0.3
    cone_varies = cone_vols.std() > 0.2
    
    # Check if rules are differentiated
    rule_E_T_means = [E_Ts[rules == r].mean() for r in range(4)]
    rule_cone_means = [cone_vols[rules == r].mean() for r in range(4)]
    
    E_T_separates_rules = np.std(rule_E_T_means) > 0.2
    cone_separates_rules = np.std(rule_cone_means) > 0.2
    
    print(f"\n  E[T] separates rules: {E_T_separates_rules} (rule means std: {np.std(rule_E_T_means):.3f})")
    print(f"  Cone separates rules: {cone_separates_rules} (rule means std: {np.std(rule_cone_means):.3f})")
    
    success = E_T_varies or cone_varies or E_T_separates_rules or cone_separates_rules
    
    if success:
        print("\n✓ Tube outputs VARY with s0!")
    else:
        print("\n✗ Tube outputs are nearly constant - not learning situation dependence")
    
    return E_Ts, cone_vols, rules


def visualize_tubes(tube_net: SimpleTubeNet, save_path: Path = None):
    """Visualize tubes for different rules."""
    import matplotlib.pyplot as plt
    
    env = SimpleEnv(seed=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for rule in range(4):
        ax = axes[rule // 2, rule % 2]
        
        # Find an episode with this rule
        for _ in range(100):
            s0 = env.reset()
            if env.rule == rule:
                break
        
        traj = env.generate_trajectory(T=16)
        s0_t = torch.tensor(s0, dtype=torch.float32)
        traj_t = torch.tensor(traj, dtype=torch.float32)
        
        with torch.no_grad():
            result = compute_tube_loss(tube_net, s0_t, traj_t)
        
        mu = result["mu_traj"].numpy()
        std = result["std_traj"].numpy()
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        ax.plot(env.goal_x, env.goal_y, 'r*', markersize=15, label='Goal')
        
        # Plot tube (2-sigma ellipses at each timestep)
        for t in range(0, 16, 2):
            ellipse = plt.matplotlib.patches.Ellipse(
                (mu[t, 0], mu[t, 1]),
                width=4 * std[t, 0],  # 2-sigma on each side
                height=4 * std[t, 1],
                alpha=0.2,
                color='blue',
            )
            ax.add_patch(ellipse)
        
        ax.plot(mu[:, 0], mu[:, 1], 'b--', alpha=0.5, label='Tube mean')
        
        ax.set_title(f"Rule {rule}: E[T]={result['E_T']:.2f}, bind={result['bind_rate']:.2f}")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Tube Predictions by Rule (No z)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Run name suffix")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--horizon-bonus", type=float, default=0.05, help="Horizon bonus weight")
    parser.add_argument("--horizon-bind-weight", type=float, default=1.0, help="Horizon bind constraint weight")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--mode", type=str, default="full_traj", 
                       choices=["horizon_weighted", "full_traj", "horizon_match"],
                       help="Training mode for bind objective")
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"tube_ablation_{timestamp}"
    if args.name:
        run_name = f"tube_ablation_{args.name}_{timestamp}"
    
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")
    
    # Save config
    config = {
        "steps": args.steps,
        "horizon_bonus": args.horizon_bonus,
        "horizon_bind_weight": args.horizon_bind_weight,
        "seed": args.seed,
        "mode": args.mode,
    }
    import json
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Train
    tube_net, history = train_tube(
        n_steps=args.steps,
        lr=1e-3,
        seed=args.seed,
        w_horizon_bind=args.horizon_bind_weight,
        horizon_bonus=args.horizon_bonus,
        mode=args.mode,
    )
    
    # Evaluate
    E_Ts, cone_vols, rules = evaluate_situation_dependence(tube_net)
    
    # Visualize
    visualize_tubes(tube_net, save_path=out_dir / "tubes_by_rule.png")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(history["nll"])
    axes[0, 0].set_title("NLL")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["bind_rate"], label="Per-step")
    axes[0, 1].plot(history["full_bind"], label="Full trajectory")
    axes[0, 1].set_title("Bind Rate")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].legend()
    axes[0, 1].axhline(0.85, color='r', linestyle='--', alpha=0.5, label="Target")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history["E_T"])
    axes[0, 2].set_title("E[T] (Expected Horizon)")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history["cone_vol"])
    axes[1, 0].set_title("Cone Volume")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Per-rule E[T] distribution
    axes[1, 1].hist([E_Ts[rules == r] for r in range(4)], 
                    label=[f"Rule {r}" for r in range(4)], alpha=0.7)
    axes[1, 1].set_title("E[T] by Rule (Eval)")
    axes[1, 1].set_xlabel("E[T]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Per-rule cone vol distribution
    axes[1, 2].hist([cone_vols[rules == r] for r in range(4)],
                    label=[f"Rule {r}" for r in range(4)], alpha=0.7)
    axes[1, 2].set_title("Log(Cone Vol) by Rule (Eval)")
    axes[1, 2].set_xlabel("Log Cone Vol")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Tube Ablation: {run_name}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "training.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save evaluation results
    eval_results = {
        "E_T_mean": float(E_Ts.mean()),
        "E_T_std": float(E_Ts.std()),
        "cone_vol_mean": float(cone_vols.mean()),
        "cone_vol_std": float(cone_vols.std()),
        "per_rule_E_T": {r: float(E_Ts[rules == r].mean()) for r in range(4)},
        "per_rule_cone": {r: float(cone_vols[rules == r].mean()) for r in range(4)},
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
