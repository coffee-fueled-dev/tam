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
    parser.add_argument("--test", type=str, choices=["rank", "cycle", "composition", "triadic", "probe", "collision", "rotation_control", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--functor-epochs", type=int, default=5000,
                        help="Training epochs for triadic functor alignment")
    parser.add_argument("--K", type=int, default=3,
                        help="Number of modes/basins (shared)")
    parser.add_argument("--z-dim", type=int, default=2,
                        help="Latent dimension (shared)")
    parser.add_argument("--d", type=int, default=None,
                        help="State dimension (default: z_dim + 1)")
    parser.add_argument("--n-probes", type=int, default=None,
                        help="Number of probes (default: min(K_A,K_B,K_C)+1)")
    # Per-world overrides for asymmetric experiments
    parser.add_argument("--K-A", type=int, default=None, help="K for world A")
    parser.add_argument("--K-B", type=int, default=None, help="K for world B")
    parser.add_argument("--K-C", type=int, default=None, help="K for world C")
    parser.add_argument("--z-dim-A", type=int, default=None, help="z_dim for actor A")
    parser.add_argument("--z-dim-B", type=int, default=None, help="z_dim for actor B")
    parser.add_argument("--z-dim-C", type=int, default=None, help="z_dim for actor C")
    parser.add_argument("--d-A", type=int, default=None, help="d for world A")
    parser.add_argument("--d-B", type=int, default=None, help="d for world B")
    parser.add_argument("--d-C", type=int, default=None, help="d for world C")
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
            K=args.K,
            z_dim=args.z_dim,
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
            K=args.K,
            z_dim=args.z_dim,
            d=args.d,
            n_probes=args.n_probes,
            # Per-world overrides (None = use shared values)
            K_A=args.K_A, K_B=args.K_B, K_C=args.K_C,
            z_dim_A=args.z_dim_A, z_dim_B=args.z_dim_B, z_dim_C=args.z_dim_C,
            d_A=args.d_A, d_B=args.d_B, d_C=args.d_C,
        )
    
    if args.test == "collision":
        print("\n" + "="*60)
        print("TEST 6: Topological Collision")
        print("="*60)
        print("Testing functor behavior under topological collision:")
        print("  World A: K=2 (two adjacent basins)")
        print("  World B: K=3 (middle basin intervenes)")
        print("  World C: K=2 (same as A)")
        print()
        from .test_reference_probe import run_reference_probe_test
        # Use collision config: K_A=2, K_B=3, K_C=2
        results["collision"] = run_reference_probe_test(
            output_dir,
            n_actor_epochs=args.epochs,
            n_functor_epochs=args.functor_epochs,
            K=2,  # Default for A and C
            K_A=2, K_B=3, K_C=2,  # Collision topology
            z_dim=args.z_dim,
            d=args.d,
            n_probes=3,  # Use min(K) + 1 = 3 probes
        )
    
    if args.test == "rotation_control":
        print("\n" + "="*60)
        print("TEST 7: Rotation Control (MLP Architecture Check)")
        print("="*60)
        from .test_reference_probe import run_rotation_control_test
        d = args.d if args.d else max(3, args.z_dim + 1)
        results["rotation_control"] = run_rotation_control_test(
            output_dir,
            n_actor_epochs=args.epochs,
            n_functor_epochs=args.functor_epochs,
            z_dim=args.z_dim,
            K=args.K,
            d=d,
        )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if "rank" in results:
        val = results['rank'].get('rank_correlation', 'N/A')
        if isinstance(val, (int, float)):
            print(f"Rank correlation: {val:.3f}")
        else:
            print(f"Rank correlation: {val}")
    
    if "cycle" in results:
        val = results['cycle'].get('mean_similarity', 'N/A')
        if isinstance(val, (int, float)):
            print(f"Cycle similarity: {val:.3f}")
        else:
            print(f"Cycle similarity: {val}")
    
    if "composition" in results:
        val = results['composition'].get('mean_cosine_sim', 'N/A')
        if isinstance(val, (int, float)):
            print(f"Composition similarity: {val:.3f}")
        else:
            print(f"Composition similarity: {val}")
        print(f"Composition holds: {results['composition'].get('composition_holds', 'N/A')}")
    
    if "triadic" in results:
        val = results['triadic'].get('final_comp_sim', 'N/A')
        if isinstance(val, (int, float)):
            print(f"Triadic composition similarity: {val:.3f}")
        else:
            print(f"Triadic composition similarity: {val}")
        val = results['triadic'].get('final_mean_acc', 'N/A')
        if isinstance(val, (int, float)):
            print(f"Triadic task accuracy: {val:.3f}")
        else:
            print(f"Triadic task accuracy: {val}")
        print(f"Gauge fixed by interaction: {results['triadic'].get('gauge_fixed', 'N/A')}")
    
    if "probe" in results:
        probe_results = results['probe']
        comp_sim = probe_results.get('comp_sim', 'N/A')
        probe_sim = probe_results.get('probe_sim', 'N/A')
        comp_zscore = probe_results.get('comp_zscore', 'N/A')
        probe_pvalue = probe_results.get('probe_pvalue', 'N/A')
        probe_pvalue_at_res = probe_results.get('probe_pvalue_at_resolution', False)
        probe_adv = probe_results.get('probe_adv', 'N/A')
        null_is_valid = probe_results.get('null_is_valid', True)
        gauge_fixed = probe_results.get('gauge_fixed', 'N/A')
        
        if isinstance(comp_sim, (int, float)):
            print(f"Probe composition similarity: {comp_sim:.3f} (z={comp_zscore:.1f}σ)" if isinstance(comp_zscore, (int, float)) else f"Probe composition similarity: {comp_sim:.3f}")
        else:
            print(f"Probe composition similarity: {comp_sim}")
        
        if isinstance(probe_sim, (int, float)):
            stats_parts = []
            if isinstance(probe_adv, (int, float)):
                stats_parts.append(f"adv={probe_adv:+.3f}")
            if isinstance(probe_pvalue, (int, float)):
                # Report as inequality when at resolution limit
                if probe_pvalue_at_res:
                    stats_parts.append(f"p<{probe_pvalue:.4f}")
                else:
                    stats_parts.append(f"p={probe_pvalue:.4f}")
            stats_str = ", ".join(stats_parts)
            if stats_str:
                print(f"Probe agreement: {probe_sim:.3f} ({stats_str})")
            else:
                print(f"Probe agreement: {probe_sim:.3f}")
        else:
            print(f"Probe agreement: {probe_sim}")
        
        if not null_is_valid:
            print("⚠️ WARNING: Degenerate null distribution. Test may be invalid.")
        
        print(f"Gauge fixed by probes: {gauge_fixed}")
        
        # Topological stress (sphere-aware: folds/condition in ambient space are not meaningful)
        topo_health = probe_results.get('topo_health', 'N/A')
        print(f"Topological health: {topo_health}")
    
    if "collision" in results:
        print()
        print("TOPOLOGICAL COLLISION RESULTS:")
        col = results['collision']
        print(f"  Composition: {col.get('comp_sim', 'N/A'):.3f}" if isinstance(col.get('comp_sim'), (int, float)) else f"  Composition: {col.get('comp_sim', 'N/A')}")
        print(f"  Gauge fixed: {col.get('gauge_fixed', 'N/A')}")
        print(f"  Topo health (sphere-aware): {col.get('topo_health', 'N/A')}")
    
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
