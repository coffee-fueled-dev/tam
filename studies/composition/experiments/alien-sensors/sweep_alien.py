"""
Sweep across alien levels for the Alien Sensors Experiment.

Runs train_alien.py for each AlienLevel and collects results.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from alien_obs import AlienLevel


def run_experiment(
    alien_level: int,
    actor_steps: int,
    functor_epochs: int,
    pair_samples: int,
    eval_samples: int,
    z_dim: int,
    alien_dim: int,
    seed: int,
) -> dict:
    """Run a single experiment at given alien level."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_alien.py"),
        "--alien-level", str(alien_level),
        "--actor-steps", str(actor_steps),
        "--functor-epochs", str(functor_epochs),
        "--pair-samples", str(pair_samples),
        "--eval-samples", str(eval_samples),
        "--z-dim", str(z_dim),
        "--alien-dim", str(alien_dim),
        "--seed", str(seed),
    ]
    
    level_name = AlienLevel(alien_level).name
    print(f"\n{'='*60}")
    print(f"Running {level_name}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"  ❌ {level_name} failed!")
        return None
    
    # Find the most recent run for this level
    runs_dir = Path(__file__).parent / "runs"
    level_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(level_name)],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    
    if not level_runs:
        print(f"  ❌ No output found for {level_name}")
        return None
    
    latest_run = level_runs[0]
    summary_path = latest_run / "summary.json"
    
    if not summary_path.exists():
        print(f"  ❌ No summary.json in {latest_run}")
        return None
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    summary["run_dir"] = str(latest_run)
    summary["alien_level"] = level_name
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-steps", type=int, default=5000)
    parser.add_argument("--functor-epochs", type=int, default=100)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--alien-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=str, default="0,1,2,3",
                        help="Comma-separated list of alien levels to run")
    args = parser.parse_args()
    
    levels = [int(x) for x in args.levels.split(",")]
    
    print(f"\n{'='*60}")
    print("Alien Sensors Sweep (Minimal Actor)")
    print(f"Levels: {[AlienLevel(l).name for l in levels]}")
    print(f"{'='*60}")
    
    results = []
    
    for level in levels:
        summary = run_experiment(
            alien_level=level,
            actor_steps=args.actor_steps,
            functor_epochs=args.functor_epochs,
            pair_samples=args.pair_samples,
            eval_samples=args.eval_samples,
            z_dim=args.z_dim,
            alien_dim=args.alien_dim,
            seed=args.seed,
        )
        if summary:
            results.append(summary)
    
    # Print summary table with fairness metrics
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY (with Fairness Controls)")
    print(f"{'='*70}")
    print(f"\n{'Level':<10} {'Ceiling':<10} {'Transfer':<10} {'Lift':<10} {'Shuffle':<10} {'Bind Δ':<10}")
    print("-" * 70)
    
    for r in results:
        level = r.get("alien_level", "?")
        ceiling = r.get("bayes_ceiling", 0.5)
        transfer = r.get("mode_agreement", {}).get("mean", 0)
        lift = r.get("lift", 0)
        shuffle = r.get("shuffle_ablation", {}).get("mode_agreement", {}).get("mean", 0)
        trans_bind = r.get("transfer", {}).get("bind_mean", 0)
        cem_bind = r.get("B_CEM", {}).get("bind_mean", 0)
        bind_diff = abs(trans_bind - cem_bind)
        
        print(f"{level:<10} {ceiling:<10.1%} {transfer:<10.1%} {lift:<+10.1%} {shuffle:<10.1%} {bind_diff:<10.3f}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("FAIRNESS ANALYSIS")
    print(f"{'='*70}")
    
    print("\n1. Bayes Ceiling Check (is B's sensor mode-blind?):")
    for r in results:
        level = r["alien_level"]
        ceiling = r.get("bayes_ceiling", 0.5)
        mode_blind = ceiling < 0.6
        status = "✓ mode-blind" if mode_blind else "⚠ has mode info"
        print(f"   {level}: ceiling={ceiling:.1%} → {status}")
    
    print("\n2. Lift Check (does A add value?):")
    for r in results:
        level = r["alien_level"]
        lift = r.get("lift", 0)
        positive = lift > 0.05
        status = "✓ A adds value" if positive else "⚠ low lift"
        print(f"   {level}: lift={lift:+.1%} → {status}")
    
    print("\n3. Shuffle Ablation (does breaking correspondence hurt?):")
    for r in results:
        level = r["alien_level"]
        ceiling = r.get("bayes_ceiling", 0.5)
        shuffle = r.get("shuffle_ablation", {}).get("mode_agreement", {}).get("mean", 0)
        collapsed = shuffle < ceiling + 0.10
        status = "✓ collapsed" if collapsed else "⚠ still works (leakage?)"
        print(f"   {level}: shuffle={shuffle:.1%} vs ceiling={ceiling:.1%} → {status}")
    
    # Win conditions (updated)
    print(f"\n{'='*70}")
    print("STRONG RESULTS (all fairness checks pass)")
    print(f"{'='*70}")
    
    for r in results:
        level = r["alien_level"]
        ceiling = r.get("bayes_ceiling", 0.5)
        transfer = r.get("mode_agreement", {}).get("mean", 0)
        lift = r.get("lift", 0)
        shuffle = r.get("shuffle_ablation", {}).get("mode_agreement", {}).get("mean", 0)
        trans_bind = r.get("transfer", {}).get("bind_mean", 0)
        cem_bind = r.get("B_CEM", {}).get("bind_mean", 0)
        
        lift_ok = lift > 0.05
        shuffle_ok = shuffle < ceiling + 0.10
        ma_ok = transfer > 0.85
        bind_ok = abs(trans_bind - cem_bind) < 0.05
        
        all_ok = lift_ok and shuffle_ok and ma_ok and bind_ok
        status = "🎉 STRONG" if all_ok else "⚠ partial"
        checks = f"lift={lift:+.1%}, shuffle_Δ={shuffle-ceiling:+.1%}, ma={transfer:.1%}"
        print(f"  {status} {level}: {checks}")
    
    # Save sweep results
    sweep_out = Path(__file__).parent / "runs" / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(sweep_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {sweep_out}")


if __name__ == "__main__":
    main()
