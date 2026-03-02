#!/usr/bin/env python3
"""Run full CPDE v3.2 validation pipeline.

Usage:
    python scripts/run_full_validation.py
    python scripts/run_full_validation.py --n_samples 50000 --bootstrap 30
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fusionmind4.discovery import EnsembleCPDE
from fusionmind4.utils import FM3LitePhysicsEngine, PLASMA_VARS, N_VARS


def main():
    parser = argparse.ArgumentParser(description="CPDE v3.2 Full Validation")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood", action="store_true", help="Run OOD robustness test")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CPDE v3.2 — Causal Plasma Discovery Engine              ║")
    print("║  FusionMind 4.0 · Patent Family PF1                      ║")
    print("║  Dr. Mladen Mester · March 2026                          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Generate data
    print(f"\n[DATA] Generating FM3-Lite dataset ({args.n_samples} samples)...")
    engine = FM3LitePhysicsEngine(n_samples=args.n_samples, seed=args.seed)
    data, interventions = engine.generate()
    print(f"  Shape: {data.shape}, Vars: {N_VARS}")

    # Run CPDE
    print("\n" + "=" * 60)
    print("CPDE v3.2 — Ensemble Causal Discovery")
    print("=" * 60)

    cpde = EnsembleCPDE(config={
        "n_bootstrap": args.bootstrap,
        "threshold": args.threshold,
    })
    results = cpde.discover(data, interventional_data=interventions, seed=args.seed)

    # Print results
    m = results["metrics"]
    pc = results["physics_checks"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  F1 Score:   {results['f1']:.1%}")
    print(f"  Precision:  {results['precision']:.1%}")
    print(f"  Recall:     {results['recall']:.1%}")
    print(f"  SHD:        {results['shd']}")
    print(f"  TP/FP/FN:   {m['tp']}/{m['fp']}/{m['fn']}")
    print(f"  Edges:      {m['n_discovered']} discovered / {m['n_true']} true")

    print(f"\n  PHYSICS: {pc['passed']}/{pc['total']} passed")
    for k, v in pc.items():
        if k in ("passed", "total"):
            continue
        symbol = "✓" if v else "✗"
        print(f"    {symbol} {k}")

    # Print top edges
    print("\n  TOP EDGES:")
    from fusionmind4.utils.plasma_vars import build_ground_truth_adjacency
    gt = build_ground_truth_adjacency()
    sorted_edges = sorted(results["edge_details"].items(), key=lambda x: -x[1]["score"])
    for (i, j), d in sorted_edges[:12]:
        is_tp = gt[i, j] != 0
        status = "✓TP" if is_tp else "✗FP"
        name_i = PLASMA_VARS[i].name
        name_j = PLASMA_VARS[j].name
        print(f"    {status}  {name_i:>10}→{name_j:<12} score={d['score']:.3f} "
              f"[NT={d['nt']:.2f} GC={d['gc']:.0f} PC={d['pc']:.2f} "
              f"INT={d['int']:.2f} PHY={d['phy']:.1f}]")

    # OOD robustness
    if args.ood:
        print("\n" + "=" * 60)
        print("OOD ROBUSTNESS TEST")
        print("=" * 60)
        for noise_pct in [5, 10]:
            noise_data = engine.add_noise(data, noise_pct / 100)
            ood_results = cpde.discover(noise_data, interventional_data=interventions, seed=args.seed)
            print(f"  Noise={noise_pct}%: F1={ood_results['f1']:.1%} "
                  f"Pr={ood_results['precision']:.1%} "
                  f"Rc={ood_results['recall']:.1%}")

    print(f"\n{'=' * 60}")
    print(f"FINAL: F1={results['f1']:.1%} Physics={pc['passed']}/{pc['total']}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
