"""
Standard plotting library for TAM experiments.

Provides three core plots that answer TAM's key questions:
1. Outcome vs Sharpness (Did commitment help?)
2. Calibration curve + scalar (Was uncertainty honest?)
3. Compute ROI (Was it worth it?)

Plus Commitment Atlas for visualizing z-space.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from actor import Actor
except ImportError:
    try:
        from ..actor import Actor  # type: ignore
    except ImportError:
        Actor = None  # type: ignore


def save_fig(fig, path: Path, dpi: int = 150):
    """Save a matplotlib figure and close it to prevent memory leaks."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def pca2(X: np.ndarray) -> np.ndarray:
    """Project data to 2D using PCA."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T  # [N,2]


def plot_outcome_vs_sharpness(
    agent: Actor,
    snapshots: List[Any],
    outcome_fn: Optional[Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]] = None,
    run_dir: Optional[Path] = None,
    prefix: str = "outcome_sharpness",
    k_star: float = 2.0,
) -> None:
    """
    Plot 1: Outcome vs Sharpness Tradeoff
    
    Answers: "Did tighter cones / longer commitments lead to better outcomes?"
    
    Args:
        agent: Trained actor
        snapshots: List of EvalSnapshot objects from evaluation
        outcome_fn: Function that computes episode outcome from (info, states, actions)
        run_dir: Directory to save plot
        prefix: Filename prefix
        k_star: k-sigma value to use for bind success calculation
    """
    if len(snapshots) == 0:
        print("Warning: No snapshots for outcome vs sharpness plot")
        return
    
    # Use last snapshot for per-episode data
    snap = snapshots[-1]
    
    # Extract per-episode data
    pareto_points = snap.pareto_points  # [N, 4] = [sharp_log_vol, cov_error, E_T, volatility]
    sharpness = pareto_points[:, 0]  # log cone volume
    E_T = pareto_points[:, 2]  # expected horizon
    volatility = pareto_points[:, 3]  # episode difficulty
    
    # Compute bind success from coverage
    # For each episode, check if coverage at k_star is close to nominal
    nominal_cov = _nominal_gaussian_coverage(k_star, D=2)  # assuming 2D predictions
    cov_error = pareto_points[:, 1]  # coverage error at k_star
    bind_success = (cov_error < 0.1).astype(float)  # threshold for "successful binding"
    
    # For outcome, extract from snapshots
    # Use J (control cost) as proxy - negative cost = outcome (higher is better)
    if hasattr(snap, 'J_points') and snap.J_points is not None:
        outcomes = -snap.J_points  # negative cost = outcome (higher is better)
    elif hasattr(snap, 'mean_J'):
        # If only mean available, use it for all points (less informative)
        outcomes = np.full(len(sharpness), -snap.mean_J)
    else:
        # Fallback: use inverse of sharpness as proxy
        outcomes = -sharpness
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Scatter plot: color by E[T], shape by bind success
    scatter = ax.scatter(
        sharpness,
        outcomes,
        c=E_T,
        s=50,
        alpha=0.6,
        cmap='viridis',
        marker='o',
        edgecolors='black' if bind_success.max() > 0 else None,
        linewidths=1.5 * bind_success,  # thicker outline for successful binds
    )
    
    ax.set_xlabel("Sharpness = weighted log cone volume (lower is tighter)")
    ax.set_ylabel("Task Outcome (higher is better)")
    ax.set_title("Outcome vs Sharpness Tradeoff\n(color = E[T], outline = bind success)")
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Expected Horizon E[T]")
    
    # Add baseline line (mean outcome)
    mean_outcome = np.mean(outcomes)
    ax.axhline(mean_outcome, linestyle="--", alpha=0.5, color='gray', label=f"Mean outcome: {mean_outcome:.3f}")
    
    # Add annotations
    bind_rate = bind_success.mean()
    mean_sharp = np.mean(sharpness)
    mean_et = np.mean(E_T)
    
    textstr = f"Bind success: {bind_rate:.2%}\nMean sharpness: {mean_sharp:.3f}\nMean E[T]: {mean_et:.2f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if run_dir is not None:
        save_fig(fig, run_dir / "fig" / f"{prefix}.png")
    else:
        plt.show()


def plot_calibration_curve(
    snapshots_lo: List[Any],
    snapshots_hi: List[Any],
    run_dir: Optional[Path] = None,
    prefix: str = "calibration",
    D: int = 2,
) -> None:
    """
    Plot 2: Calibration Curve + Scalar Error
    
    Answers: "Was the uncertainty honest?"
    
    Args:
        snapshots_lo: Snapshots with minimal reasoning (Hr=1)
        snapshots_hi: Snapshots with full reasoning (Hr=max)
        run_dir: Directory to save plot
        prefix: Filename prefix
        D: Prediction dimension (for nominal coverage calculation)
    """
    if len(snapshots_hi) == 0:
        print("Warning: No snapshots for calibration plot")
        return
    
    # Use last snapshot for current calibration state
    snap_lo = snapshots_lo[-1] if len(snapshots_lo) > 0 else None
    snap_hi = snapshots_hi[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ks = snap_hi.ks
    
    # Plot 1: Calibration curves
    if snap_lo is not None:
        emp_lo = [snap_lo.empirical_coverage[k] for k in ks]
        ax1.plot(ks, emp_lo, 'o-', alpha=0.7, label="Hr=1 (minimal)", linewidth=2)
    
    emp_hi = [snap_hi.empirical_coverage[k] for k in ks]
    ax1.plot(ks, emp_hi, 's-', alpha=0.7, label="Hr=max (full)", linewidth=2)
    
    # Nominal coverage line
    nom = [_nominal_gaussian_coverage(k, D) for k in ks]
    ax1.plot(ks, nom, '--', alpha=0.8, color='gray', linewidth=2, label="Nominal Gaussian")
    
    ax1.set_xlabel("k (tube radius in σ)")
    ax1.set_ylabel("Empirical coverage (fraction inside)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Calibration error (ECE-like scalar)
    if snap_lo is not None:
        error_lo = [abs(emp_lo[i] - nom[i]) for i in range(len(ks))]
        ax2.plot(ks, error_lo, 'o-', alpha=0.7, label="Hr=1", linewidth=2)
    
    error_hi = [abs(emp_hi[i] - nom[i]) for i in range(len(ks))]
    ax2.plot(ks, error_hi, 's-', alpha=0.7, label="Hr=max", linewidth=2)
    
    # Compute mean absolute error (ECE-like)
    mae_lo = np.mean(error_lo) if snap_lo is not None else 0.0
    mae_hi = np.mean(error_hi)
    
    ax2.axhline(mae_hi, linestyle=":", alpha=0.5, color='red', label=f"MAE (Hr=max): {mae_hi:.3f}")
    if snap_lo is not None:
        ax2.axhline(mae_lo, linestyle=":", alpha=0.5, color='blue', label=f"MAE (Hr=1): {mae_lo:.3f}")
    
    ax2.set_xlabel("k (tube radius in σ)")
    ax2.set_ylabel("|Empirical - Nominal| coverage error")
    ax2.set_title("Calibration Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation with summary
    textstr = f"MAE (Hr=max): {mae_hi:.3f}"
    if snap_lo is not None:
        textstr += f"\nMAE (Hr=1): {mae_lo:.3f}"
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if run_dir is not None:
        save_fig(fig, run_dir / "fig" / f"{prefix}.png")
        
        # Save calibration metrics
        import json
        metrics = {
            "mae_hr_max": float(mae_hi),
            "mae_hr_1": float(mae_lo) if snap_lo is not None else None,
            "calibration_curve": {
                "k": [float(k) for k in ks],
                "empirical_hr_max": [float(e) for e in emp_hi],
                "empirical_hr_1": [float(e) for e in emp_lo] if snap_lo is not None else None,
                "nominal": [float(n) for n in nom],
            }
        }
        with open(run_dir / "data" / "calibration_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        plt.show()


def plot_compute_roi(
    agent: Actor,
    run_dir: Optional[Path] = None,
    prefix: str = "compute_roi",
) -> None:
    """
    Plot 3: Compute Payback Curve (Reasoning ROI)
    
    Answers: "Was the compute worth it?"
    
    Args:
        agent: Trained actor with history
        run_dir: Directory to save plot
        prefix: Filename prefix
    """
    if not hasattr(agent, 'history'):
        print("Warning: Agent has no history for compute ROI plot")
        return
    
    h = agent.history
    
    # Extract reasoning compute and improvement
    if "E_Hr_train" not in h or "improve" not in h:
        print("Warning: Missing E_Hr_train or improve in history")
        return
    
    E_Hr = np.asarray(h["E_Hr_train"], dtype=np.float64)
    improve = np.asarray(h["improve"], dtype=np.float64)  # ΔNLL = NLL0 - NLLr
    
    # Episode difficulty proxy: use volatility, memory risk, or predicted NLL0
    if "volatility" in h:
        difficulty = np.asarray(h["volatility"], dtype=np.float64)
    elif "nll0" in h:
        difficulty = np.asarray(h["nll0"], dtype=np.float64)
    elif "mem_risk" in h:
        difficulty = np.asarray(h["mem_risk"], dtype=np.float64)
    else:
        # Fallback: use inverse of improve as difficulty proxy
        difficulty = -improve + improve.max()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Scatter plot: color by difficulty
    scatter = ax.scatter(
        E_Hr,
        improve,
        c=difficulty,
        s=30,
        alpha=0.6,
        cmap='RdYlGn_r',  # red = hard, green = easy
    )
    
    ax.set_xlabel("Reasoning Compute E[Hr] (expected reasoning steps)")
    ax.set_ylabel("Improvement ΔNLL = NLL0 - NLLr (higher is better)")
    ax.set_title("Compute ROI: Reasoning Payback\n(color = episode difficulty)")
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Episode Difficulty")
    
    # Add baseline: y=0 (no improvement)
    ax.axhline(0, linestyle="--", alpha=0.5, color='gray', label="No improvement")
    
    # Add trend line if enough points
    if len(E_Hr) > 10:
        z = np.polyfit(E_Hr, improve, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(E_Hr.min(), E_Hr.max(), 100)
        ax.plot(x_trend, p(x_trend), "--", alpha=0.7, color='blue', label=f"Trend: {z[0]:.3f}x + {z[1]:.3f}")
    
    # Add annotations
    mean_hr = np.mean(E_Hr)
    mean_improve = np.mean(improve)
    positive_ratio = (improve > 0).mean()
    
    textstr = f"Mean E[Hr]: {mean_hr:.2f}\nMean improvement: {mean_improve:.3f}\nPositive ratio: {positive_ratio:.2%}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, run_dir / "fig" / f"{prefix}.png")


def plot_commitment_atlas(
    agent: Actor,
    outcome_fn: Optional[Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]] = None,
    run_dir: Path = None,
    prefix: str = "commitment_atlas",
    max_points: int = 5000,
) -> None:
    """
    Plot 4: Commitment Atlas
    
    Visualizes z-space as a decision-grade map of reusable commitments.
    
    Args:
        agent: Trained actor with memory
        outcome_fn: Optional function to compute outcomes from memory entries
        run_dir: Directory to save plot (optional)
        prefix: Filename prefix
        max_points: Maximum points to plot (for performance)
    """
    if not hasattr(agent, 'mem') or len(agent.mem) == 0:
        print("Warning: No memory for commitment atlas")
        return
    
    # Extract memory data
    zs = np.stack([m["z"].numpy() for m in agent.mem], axis=0)  # [N, z_dim]
    soft_bind = np.array([m["soft_bind"] for m in agent.mem])
    E_T = np.array([m.get("E_T", 0.0) for m in agent.mem])
    
    # Compute outcomes if function provided, otherwise use soft_bind as proxy
    if outcome_fn is not None:
        # Try to extract states/actions from memory if available
        outcomes = np.array([m.get("outcome", soft_bind[i]) for i, m in enumerate(agent.mem)])
    else:
        outcomes = soft_bind  # Use bind success as outcome proxy
    
    # Subsample if too many points
    N = len(zs)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        zs = zs[idx]
        soft_bind = soft_bind[idx]
        E_T = E_T[idx]
        outcomes = outcomes[idx]
    
    # Project to 2D PCA
    Z2 = pca2(zs)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Scatter plot:
    # - color = outcome/success
    # - size = E[T] (confidence/horizon)
    # - edge color = bind success (thickness indicates reliability)
    
    scatter = ax.scatter(
        Z2[:, 0],
        Z2[:, 1],
        c=outcomes,
        s=50 + 200 * E_T / (E_T.max() + 1e-8),  # size proportional to E[T]
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidths=1.0 + 2.0 * soft_bind,  # thicker outline for successful binds
    )
    
    ax.set_xlabel("PC1 (z-space)")
    ax.set_ylabel("PC2 (z-space)")
    ax.set_title("Commitment Atlas\n(color = outcome, size = E[T], outline = bind success)")
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Task Outcome")
    
    # Add annotations
    mean_outcome = np.mean(outcomes)
    mean_bind = np.mean(soft_bind)
    mean_et = np.mean(E_T)
    
    textstr = f"Mean outcome: {mean_outcome:.3f}\nMean bind success: {mean_bind:.2%}\nMean E[T]: {mean_et:.2f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if run_dir is not None:
        save_fig(fig, run_dir / "fig" / f"{prefix}.png")
    else:
        plt.show()


def _nominal_gaussian_coverage(k: float, D: int) -> float:
    """Compute nominal Gaussian coverage for D dimensions."""
    import math
    p1 = math.erf(k / math.sqrt(2.0))
    return float(p1 ** D)


def plot_standard_dashboard(
    agent: Actor,
    snapshots_lo: List[Any],
    snapshots_hi: List[Any],
    run_dir: Path,
    outcome_fn: Optional[Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]] = None,
    prefix: str = "dashboard",
    k_star: float = 2.0,
    D: int = 2,
) -> None:
    """
    Generate all standard plots in one call.
    
    Args:
        agent: Trained actor
        snapshots_lo: Evaluation snapshots with minimal reasoning
        snapshots_hi: Evaluation snapshots with full reasoning
        run_dir: Directory to save plots
        outcome_fn: Optional function to compute episode outcomes
        prefix: Filename prefix
        k_star: k-sigma value for bind success
        D: Prediction dimension
    """
    print("Generating standard dashboard plots...")
    
    # Plot 1: Outcome vs Sharpness
    if len(snapshots_hi) > 0:
        plot_outcome_vs_sharpness(
            agent, snapshots_hi, outcome_fn or (lambda *args: 0.0),
            run_dir, prefix=f"{prefix}_outcome_sharpness", k_star=k_star
        )
    
    # Plot 2: Calibration
    plot_calibration_curve(snapshots_lo, snapshots_hi, run_dir, prefix=f"{prefix}_calibration", D=D)
    
    # Plot 3: Compute ROI
    plot_compute_roi(agent, run_dir, prefix=f"{prefix}_compute_roi")
    
    # Plot 4: Commitment Atlas
    plot_commitment_atlas(agent, outcome_fn, run_dir, prefix=f"{prefix}_atlas")
    
    print("Dashboard plots saved to", run_dir / "fig")
