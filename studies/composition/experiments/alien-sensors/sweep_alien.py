#!/usr/bin/env python3
"""
Sweep across all Alien levels: NONE → ALIEN_1 → ALIEN_2 → ALIEN_3

Generates a summary plot: mode agreement vs alien level.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from parent directories
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_experiment(alien_level: int, args) -> dict:
    """Run a single alien level experiment."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_alien.py"),
        "--alien-level", str(alien_level),
        "--actor-steps", str(args.actor_steps),
        "--functor-epochs", str(args.functor_epochs),
        "--pair-samples", str(args.pair_samples),
        "--eval-samples", str(args.eval_samples),
        "--z-dim", str(args.z_dim),
        "--alien-dim", str(args.alien_dim),
        "--seed", str(args.seed),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running Alien Level {alien_level}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Experiment failed for level {alien_level}")
        return None
    
    # Find the most recent run for this level
    runs_dir = Path(__file__).parent / "runs"
    level_names = ["NONE", "ALIEN_1", "ALIEN_2", "ALIEN_3"]
    level_name = level_names[alien_level]
    
    matching = sorted([d for d in runs_dir.iterdir() if d.name.startswith(level_name)], reverse=True)
    if not matching:
        return None
    
    summary_path = matching[0] / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def plot_sweep(summaries: dict, out_path: Path):
    """Plot summary across all alien levels."""
    levels = list(summaries.keys())
    level_names = ["NONE", "ALIEN_1", "ALIEN_2", "ALIEN_3"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mode agreement vs alien level
    ax = axes[0]
    agreements = [summaries[l]["mode_agreement"]["mean"] for l in levels]
    ax.bar(levels, agreements, color='#2ecc71', edgecolor='black')
    ax.axhline(0.5, color='red', linestyle='--', label='Chance')
    ax.axhline(0.85, color='blue', linestyle='--', label='Target')
    ax.set_xticks(levels)
    ax.set_xticklabels([level_names[l] for l in levels])
    ax.set_ylabel("Mode Agreement")
    ax.set_title("Intent Transfer vs Alien Level")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bind rate comparison
    ax = axes[1]
    x = np.arange(len(levels))
    width = 0.35
    
    trans_binds = [summaries[l]["transfer"]["bind_mean"] for l in levels]
    cem_binds = [summaries[l]["B_CEM"]["bind_mean"] for l in levels]
    
    ax.bar(x - width/2, trans_binds, width, label='Transfer', color='#2ecc71')
    ax.bar(x + width/2, cem_binds, width, label='B-CEM', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels([level_names[l] for l in levels])
    ax.set_ylabel("Bind Rate")
    ax.set_title("Transfer vs B-CEM")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Log volume comparison
    ax = axes[2]
    trans_vols = [summaries[l]["transfer"]["log_vol_mean"] for l in levels]
    cem_vols = [summaries[l]["B_CEM"]["log_vol_mean"] for l in levels]
    
    ax.bar(x - width/2, trans_vols, width, label='Transfer', color='#2ecc71')
    ax.bar(x + width/2, cem_vols, width, label='B-CEM', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels([level_names[l] for l in levels])
    ax.set_ylabel("Log Volume")
    ax.set_title("Tube Tightness")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Alien Sensors Sweep: Topological Bridge Sufficiency")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved sweep plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-steps", type=int, default=5000)
    parser.add_argument("--functor-epochs", type=int, default=100)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--alien-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=str, default="0,1,2,3", help="Comma-separated alien levels")
    args = parser.parse_args()
    
    levels = [int(x) for x in args.levels.split(",")]
    
    summaries = {}
    for level in levels:
        summary = run_experiment(level, args)
        if summary:
            summaries[level] = summary
    
    if summaries:
        out_path = Path(__file__).parent / "sweep_results.png"
        plot_sweep(summaries, out_path)
        
        # Save combined summary
        with open(Path(__file__).parent / "sweep_summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
        
        # Print summary table
        print(f"\n{'='*60}")
        print("SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"{'Level':<10} {'Agreement':>12} {'Trans Bind':>12} {'CEM Bind':>12}")
        print("-" * 50)
        for level in summaries:
            s = summaries[level]
            print(f"{['NONE','ALIEN_1','ALIEN_2','ALIEN_3'][level]:<10} "
                  f"{s['mode_agreement']['mean']:>11.1%} "
                  f"{s['transfer']['bind_mean']:>12.3f} "
                  f"{s['B_CEM']['bind_mean']:>12.3f}")


if __name__ == "__main__":
    main()
