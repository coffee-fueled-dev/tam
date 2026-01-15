#!/usr/bin/env python3
"""
Run all functor tests with visualizations.

Usage:
    python -m v2.tests.functors.run_all [--output-dir PATH] [--epochs N]
"""

import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run functor topology tests")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: artifacts/functor_tests/run_TIMESTAMP)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs per actor")
    parser.add_argument("--test", type=str, choices=["rank", "cycle", "composition", "triadic", "probe", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--functor-epochs", type=int, default=5000,
                        help="Training epochs for triadic functor alignment")
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"artifacts/functor_tests/run_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    results = {}
    
    if args.test in ("rank", "all"):
        print("\n" + "="*60)
        print("TEST 1: Rank Preservation")
        print("="*60)
        from .test_functor_rank import run_full_rank_test
        results["rank"] = run_full_rank_test(output_dir, n_epochs=args.epochs)
    
    if args.test in ("cycle", "all"):
        print("\n" + "="*60)
        print("TEST 2: Cycle Consistency")
        print("="*60)
        from .test_functor_cycle import run_cycle_test
        results["cycle"] = run_cycle_test(output_dir, n_epochs=args.epochs)
    
    if args.test in ("composition", "all"):
        print("\n" + "="*60)
        print("TEST 3: Functor Composition")
        print("="*60)
        from .test_functor_composition import run_composition_test
        results["composition"] = run_composition_test(output_dir, n_epochs=args.epochs)
    
    if args.test in ("triadic", "all"):
        print("\n" + "="*60)
        print("TEST 4: Triadic Coordination-Induced Gauge Alignment")
        print("="*60)
        from .test_triadic_alignment import run_triadic_alignment_test
        results["triadic"] = run_triadic_alignment_test(
            output_dir,
            n_actor_epochs=args.epochs,
            n_functor_epochs=args.functor_epochs,
        )
    
    if args.test in ("probe", "all"):
        print("\n" + "="*60)
        print("TEST 5: Reference Probe Gauge Alignment")
        print("="*60)
        from .test_reference_probe import run_reference_probe_test
        results["probe"] = run_reference_probe_test(
            output_dir,
            n_actor_epochs=args.epochs,
            n_functor_epochs=args.functor_epochs,
        )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if "rank" in results:
        print(f"Rank correlation: {results['rank'].get('rank_correlation', 'N/A'):.3f}")
    
    if "cycle" in results:
        print(f"Cycle similarity: {results['cycle'].get('mean_similarity', 'N/A'):.3f}")
    
    if "composition" in results:
        print(f"Composition similarity: {results['composition'].get('mean_cosine_sim', 'N/A'):.3f}")
        print(f"Composition holds: {results['composition'].get('composition_holds', 'N/A')}")
    
    if "triadic" in results:
        print(f"Triadic composition similarity: {results['triadic'].get('final_comp_sim', 'N/A'):.3f}")
        print(f"Triadic task accuracy: {results['triadic'].get('final_mean_acc', 'N/A'):.3f}")
        print(f"Gauge fixed by interaction: {results['triadic'].get('gauge_fixed', 'N/A')}")
    
    if "probe" in results:
        print(f"Probe composition similarity: {results['probe'].get('final_comp_sim', 'N/A'):.3f}")
        print(f"Probe agreement A→B: {results['probe'].get('probe_sim_AB', 'N/A'):.3f}")
        print(f"Probe agreement A→C: {results['probe'].get('probe_sim_AC', 'N/A'):.3f}")
        print(f"Gauge fixed by probes: {results['probe'].get('gauge_fixed', 'N/A')}")
    
    # Save summary
    import json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "epochs": args.epochs,
        "results": {
            k: {kk: vv for kk, vv in v.items() if not hasattr(vv, '__call__') and not isinstance(vv, object) or isinstance(vv, (int, float, str, bool, list, dict, type(None)))}
            for k, v in results.items()
        }
    }
    
    # Filter out non-serializable items
    def filter_serializable(obj):
        if isinstance(obj, dict):
            return {k: filter_serializable(v) for k, v in obj.items() 
                    if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [filter_serializable(x) for x in obj]
        else:
            return None
    
    summary_filtered = filter_serializable(summary)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_filtered, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
