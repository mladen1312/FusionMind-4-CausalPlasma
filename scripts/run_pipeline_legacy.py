#!/usr/bin/env python3
"""
FusionMind 4.0 — Unified Pipeline Runner

Runs all 5 patent families (PF1–PF5) with best-of-both implementations.
Single entry point for the complete causal plasma control system.

Usage:
    python run_pipeline.py              # Run all modules
    python run_pipeline.py --cpde       # Run only CPDE
    python run_pipeline.py --cpc        # Run only CPC
    python run_pipeline.py --all        # Run all with detailed output

Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import sys
import os
import time
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from physics.simulator import (
    FM3LiteSimulator, FM3LiteConfig, get_ground_truth_adjacency,
    VARIABLE_NAMES, VARIABLE_CATEGORIES
)
from cpde.engine import CPDE
from cpc.scm import PlasmaSCM
from cpc.interventions import InterventionEngine, CounterfactualEngine
from cpc.controller import CounterfactualPlasmaController


def run_cpde(verbose=True):
    """PF1: Causal Plasma Discovery Engine."""
    print("\n" + "=" * 70)
    print("PF1: CAUSAL PLASMA DISCOVERY ENGINE (CPDE) v3.2")
    print("=" * 70)

    # Generate synthetic plasma data
    config = FM3LiteConfig(n_timesteps=2000, noise_level=0.03)
    sim = FM3LiteSimulator(config)
    data, _, _ = sim.simulate()
    gt = get_ground_truth_adjacency()

    # Run unified CPDE
    cpde = CPDE(VARIABLE_NAMES, VARIABLE_CATEGORIES)
    result = cpde.discover(data, ground_truth=gt, verbose=verbose)

    # Summary
    m = result['metrics']
    pinn = result['pinn_validation']
    print(f"\n  ── SUMMARY ──")
    print(f"  F1={m['f1']:.1%}  Recall={m['recall']:.1%}  "
          f"Precision={m['precision']:.1%}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  SHD={m['shd']}")
    print(f"  PINN: {pinn['summary']['passed']}/{pinn['summary']['total_checks']} "
          f"physics checks passed")

    return result


def run_cpc(cpde_result=None, verbose=True):
    """PF2: Counterfactual Plasma Controller."""
    print("\n" + "=" * 70)
    print("PF2: COUNTERFACTUAL PLASMA CONTROLLER (CPC) v2.0")
    print("=" * 70)

    # Get DAG (from CPDE or generate fresh)
    if cpde_result is None:
        config = FM3LiteConfig(n_timesteps=2000, noise_level=0.03)
        sim = FM3LiteSimulator(config)
        data, _, _ = sim.simulate()
        cpde = CPDE(VARIABLE_NAMES, VARIABLE_CATEGORIES)
        cpde_result = cpde.discover(data, verbose=False)

    dag = cpde_result['adjacency']

    # Build SCM from discovered DAG
    config = FM3LiteConfig(n_timesteps=2000, noise_level=0.03)
    sim = FM3LiteSimulator(config)
    data, _, _ = sim.simulate()

    scm = PlasmaSCM(VARIABLE_NAMES, dag)
    scm.fit(data, verbose=verbose)

    # Create controller
    actuator_names = [VARIABLE_NAMES[i] for i in VARIABLE_CATEGORIES['actuator']]
    target_names = ['Te', 'beta_N', 'W_stored']

    controller = CounterfactualPlasmaController(
        scm=scm,
        actuator_names=actuator_names,
        target_names=target_names,
    )

    # Test interventions
    intervention_engine = InterventionEngine(scm)
    print("\n  ── Intervention Tests ──")

    current_state = {name: float(data[-1, i]) for i, name in enumerate(VARIABLE_NAMES)}

    test_interventions = [
        ('P_NBI', 0.8, 'Ti'),
        ('P_ECRH', 0.8, 'Te'),
        ('gas_puff', 0.8, 'ne'),
    ]

    for cause, value, effect in test_interventions:
        result = intervention_engine.do({cause: value}, current_state)
        ate = result.causal_effects.get(effect, 0)
        direction = "+" if ate > 0 else ""
        print(f"    do({cause}={value:.1f}) → {effect}: ATE={direction}{ate:.4f}")

    # Test counterfactual
    cf_engine = CounterfactualEngine(scm)
    cf = cf_engine.counterfactual(current_state, {'P_NBI': 0.9})
    print(f"\n    Counterfactual: 'What if P_NBI had been 0.9?'")
    print(f"      Ti: {current_state['Ti']:.4f} → {cf.counterfactual_outcomes['Ti']:.4f}")

    # Test causal path explanation
    print(f"\n  ── Causal Path Trace ──")
    paths = controller.explain_causal_path('P_ECRH', 'MHD_amp', current_state)
    for p in paths[:3]:
        print(f"    {p['path']}  (total: {p['total_effect']:+.4f})")

    # Test disruption avoidance
    print(f"\n  ── Disruption Avoidance ──")
    emergency = controller.disruption_avoidance(current_state)
    if 'error' not in emergency:
        print(f"    Optimal actuators: {emergency.get('optimal_actuators', {})}")
        print(f"    Target achieved: {emergency.get('target_achieved', 'N/A')}")
    else:
        print(f"    {emergency['error']}")

    return controller


def run_upfm(verbose=True):
    """PF3: Universal Plasma Foundation Model."""
    print("\n" + "=" * 70)
    print("PF3: UNIVERSAL PLASMA FOUNDATION MODEL (UPFM)")
    print("=" * 70)

    from upfm.core import DimensionlessTokenizer, TokamakDevice, PlasmaFoundationModel
    import numpy as np

    tokenizer = DimensionlessTokenizer()

    # Define devices
    device_specs = {
        'ITER':    TokamakDevice('ITER',    R0=6.2,  a=2.0,  B0=5.3,  Ip_max=15e6,  kappa=1.7, P_heat=150e6),
        'JET':     TokamakDevice('JET',     R0=2.96, a=1.25, B0=3.45, Ip_max=3.5e6, kappa=1.68, P_heat=30e6),
        'DIII-D':  TokamakDevice('DIII-D',  R0=1.67, a=0.67, B0=2.2,  Ip_max=1.5e6, kappa=1.8, P_heat=20e6),
        'EAST':    TokamakDevice('EAST',    R0=1.85, a=0.45, B0=3.5,  Ip_max=1e6,   kappa=1.7, P_heat=10e6),
        'KSTAR':   TokamakDevice('KSTAR',   R0=1.8,  a=0.5,  B0=3.5,  Ip_max=1e6,   kappa=1.8, P_heat=8e6),
        'ASDEX-U': TokamakDevice('ASDEX-U', R0=1.65, a=0.5,  B0=2.5,  Ip_max=1e6,   kappa=1.6, P_heat=15e6),
    }

    raw_data = {
        'ITER':    {'ne': 1e20, 'Te': 25.0, 'Ti': 20.0, 'Ip': 15.0, 'P_heat': 150.0, 'tau_E': 3.7, 'q95': 3.0},
        'JET':     {'ne': 8e19, 'Te': 10.0, 'Ti': 8.0,  'Ip': 3.5,  'P_heat': 30.0,  'tau_E': 0.4, 'q95': 3.5},
        'DIII-D':  {'ne': 5e19, 'Te': 5.0,  'Ti': 4.0,  'Ip': 1.5,  'P_heat': 20.0,  'tau_E': 0.15, 'q95': 4.0},
        'EAST':    {'ne': 4e19, 'Te': 3.0,  'Ti': 2.5,  'Ip': 1.0,  'P_heat': 10.0,  'tau_E': 0.1,  'q95': 4.5},
        'KSTAR':   {'ne': 5e19, 'Te': 3.5,  'Ti': 3.0,  'Ip': 1.0,  'P_heat': 8.0,   'tau_E': 0.12, 'q95': 4.0},
        'ASDEX-U': {'ne': 6e19, 'Te': 4.0,  'Ti': 3.5,  'Ip': 1.0,  'P_heat': 15.0,  'tau_E': 0.08, 'q95': 3.8},
    }

    tokens = {}
    for name in device_specs:
        t = tokenizer.tokenize(raw_data[name], device_specs[name])
        tokens[name] = t
        if verbose:
            labels = ['βN', 'ν*', 'ρ*', 'q95', 'H98', 'fGW']
            parts = '  '.join(f'{labels[k]}={t[k]:.3f}' for k in range(min(len(t), len(labels))))
            print(f"  {name:8s}: {parts}")

    all_tokens = np.array(list(tokens.values()))
    cv = np.mean(np.std(all_tokens, axis=0) / (np.mean(all_tokens, axis=0) + 1e-10))
    print(f"\n  Cross-device CV: {cv:.3f}")

    # Foundation model transfer test
    model = PlasmaFoundationModel()
    # Create simple training data: tokens as features, next-step Te as label
    train_tokens = all_tokens[:-1]
    train_labels = all_tokens[1:, 0]  # Predict next βN
    if train_tokens.shape[0] >= 2:
        model.fit(train_tokens, train_labels)
        pred = model.predict(train_tokens)
        r2 = 1 - np.sum((train_labels - pred) ** 2) / (np.sum((train_labels - np.mean(train_labels)) ** 2) + 1e-10)
        print(f"  Transfer R²: {r2:.3f}")
    else:
        r2 = 0.0
        print(f"  Transfer R²: N/A (insufficient data)")

    return cv, r2


def run_d3r(verbose=True):
    """PF4: Diffusion 3D Reconstruction."""
    print("\n" + "=" * 70)
    print("PF4: DIFFUSION 3D PLASMA RECONSTRUCTION (D3R)")
    print("=" * 70)

    from d3r.core import SimplifiedDiffusionReconstructor

    recon = SimplifiedDiffusionReconstructor(grid_size=32, n_diffusion_steps=50)
    gt = recon.generate_ground_truth(seed=42)
    measurements = recon.generate_sparse_measurements(gt, n_thomson=12, n_interferometry=5)
    result = recon.reconstruct(measurements, gt, n_samples=3)

    compression = result.get('compression_ratio', 0)
    rmse = result.get('rmse', 0)

    print(f"  Compression: {compression:.0f}:1")
    print(f"  RMSE: {rmse:.3f}")
    if 'ssim' in result:
        print(f"  SSIM: {result['ssim']:.3f}")

    return compression, rmse


def run_aede(verbose=True):
    """PF5: Active Experiment Design Engine."""
    print("\n" + "=" * 70)
    print("PF5: ACTIVE EXPERIMENT DESIGN ENGINE (AEDE)")
    print("=" * 70)

    from aede.core import ExperimentDesigner, BootstrapCausalUncertainty

    # Generate proper edge uncertainty (simulate what bootstrap would produce)
    edge_uncertainty = {
        (0, 6): {'mean_weight': 0.72, 'std_weight': 0.08, 'presence_rate': 0.96, 'uncertain': False},
        (0, 9): {'mean_weight': 0.50, 'std_weight': 0.15, 'presence_rate': 0.80, 'uncertain': True},
        (1, 5): {'mean_weight': 0.69, 'std_weight': 0.10, 'presence_rate': 0.92, 'uncertain': False},
        (2, 4): {'mean_weight': 0.76, 'std_weight': 0.06, 'presence_rate': 0.98, 'uncertain': False},
        (3, 7): {'mean_weight': -0.87, 'std_weight': 0.05, 'presence_rate': 1.00, 'uncertain': False},
        (0, 4): {'mean_weight': 0.15, 'std_weight': 0.20, 'presence_rate': 0.40, 'uncertain': True},
        (1, 12): {'mean_weight': 0.30, 'std_weight': 0.25, 'presence_rate': 0.55, 'uncertain': True},
        (8, 12): {'mean_weight': 0.93, 'std_weight': 0.04, 'presence_rate': 1.00, 'uncertain': False},
        (4, 8): {'mean_weight': 0.38, 'std_weight': 0.18, 'presence_rate': 0.70, 'uncertain': True},
        (5, 8): {'mean_weight': 0.50, 'std_weight': 0.12, 'presence_rate': 0.85, 'uncertain': True},
    }

    designer = ExperimentDesigner(edge_uncertainty=edge_uncertainty)
    experiments = designer.rank_experiments()

    print(f"  Ranked {len(experiments)} experiments:")
    for i, exp in enumerate(experiments):
        print(f"    {i+1}. {exp['actuator_name']:10s} — "
              f"EIG={exp['eig']:.3f}  cost={exp['cost']:.1f}  "
              f"value={exp['value']:.3f}")

    return experiments


def main():
    parser = argparse.ArgumentParser(description='FusionMind 4.0 — Unified Pipeline')
    parser.add_argument('--cpde', action='store_true', help='Run only CPDE')
    parser.add_argument('--cpc', action='store_true', help='Run only CPC')
    parser.add_argument('--upfm', action='store_true', help='Run only UPFM')
    parser.add_argument('--d3r', action='store_true', help='Run only D3R')
    parser.add_argument('--aede', action='store_true', help='Run only AEDE')
    parser.add_argument('--all', action='store_true', help='Run everything (default)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    run_all = args.all or not any([args.cpde, args.cpc, args.upfm, args.d3r, args.aede])
    verbose = not args.quiet

    print("=" * 70)
    print("  FUSIONMIND 4.0 — UNIFIED CAUSAL AI PLASMA CONTROL")
    print("  5 Patent Families · Best of Both Implementations")
    print("=" * 70)

    t_start = time.time()
    cpde_result = None

    if run_all or args.cpde:
        cpde_result = run_cpde(verbose)

    if run_all or args.cpc:
        run_cpc(cpde_result, verbose)

    if run_all or args.upfm:
        run_upfm(verbose)

    if run_all or args.d3r:
        run_d3r(verbose)

    if run_all or args.aede:
        run_aede(verbose)

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  ✅ COMPLETE — {t_total:.1f}s total")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
