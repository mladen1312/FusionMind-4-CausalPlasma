#!/usr/bin/env python3
"""
Download/Regenerate Test Fixtures from FAIR-MAST S3
====================================================
Re-downloads real MAST tokamak data and generates C-Mod synthetic
data for repeatable test fixtures.

Usage:
    python scripts/download_fixtures.py              # All fixtures
    python scripts/download_fixtures.py --mast-only   # MAST only
    python scripts/download_fixtures.py --cmod-only   # C-Mod only

Requirements:
    pip install s3fs zarr

Source: UKAEA FAIR-MAST (https://s3.echo.stfc.ac.uk/mast/level1/shots/)
Citation: Jackson et al., IEEE Trans. Plasma Sci., 2025

Author: Dr. Mladen Mester
Date: March 2026
"""

import argparse
import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'tests', 'fixtures')

# MAST shots with known good data (M8/M9 campaigns)
MAST_SHOTS = [27880, 27881, 27882, 27885, 27886, 27887,
              27888, 27889, 27890, 27891, 27892, 27893]

VAR_NAMES = ['betan', 'betap', 'q95', 'q_axis', 'li', 'elongation',
             'ne_core', 'Te_core', 'P_NBI', 'D_alpha', 'Ip']
VAR_LABELS = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
              'ne_core', 'Te_core', 'P_NBI', 'D_alpha', 'Ip']


def download_mast_fixtures():
    """Download real FAIR-MAST data from S3 and save as numpy fixture."""
    import s3fs
    import zarr

    print("=" * 60)
    print("Downloading FAIR-MAST Real Data")
    print("=" * 60)
    print(f"  S3 endpoint: https://s3.echo.stfc.ac.uk")
    print(f"  Shots: {MAST_SHOTS}")
    print()

    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={'endpoint_url': 'https://s3.echo.stfc.ac.uk'}
    )

    all_shot_data = []
    shot_meta = []

    for shot_id in MAST_SHOTS:
        path = f'mast/level1/shots/{shot_id}.zarr'
        t0 = time.time()
        print(f"  Shot {shot_id}...", end="", flush=True)

        try:
            store = s3fs.S3Map(root=path, s3=fs, check=False)
            root = zarr.open(store, mode='r')
            efm = root['efm']
            efit_time = np.array(efm['all_times'])
            N = len(efit_time)

            row = np.full((N, 11), np.nan)

            # EFM signals
            for i, sig in enumerate(['betan', 'betap', 'q_95', 'q_axis', 'li', 'elongation']):
                if sig in efm:
                    arr = np.array(efm[sig])
                    if len(arr) == N:
                        row[:, i] = arr

            # Thomson ne_core
            if 'ayc' in root and 'ne' in root['ayc']:
                ne = np.array(root['ayc']['ne'])
                if ne.ndim == 2 and ne.shape[1] > 0:
                    core_ch = ne.shape[1] // 3
                    ne_core = ne[:, core_ch]
                    src_t = np.linspace(efit_time[0], efit_time[-1], len(ne_core))
                    row[:, 6] = np.interp(efit_time, src_t, ne_core)

            # Thomson Te_core
            if 'ayc' in root and 'te' in root['ayc']:
                te = np.array(root['ayc']['te'])
                if te.ndim == 2 and te.shape[1] > 0:
                    core_ch = te.shape[1] // 3
                    te_core = te[:, core_ch]
                    src_t = np.linspace(efit_time[0], efit_time[-1], len(te_core))
                    row[:, 7] = np.interp(efit_time, src_t, te_core)

            # NBI power
            if 'anb' in root and 'ss_sum_power' in root['anb']:
                pnbi = np.array(root['anb']['ss_sum_power'])
                if len(pnbi) > 0:
                    src_t = np.linspace(efit_time[0], efit_time[-1], len(pnbi))
                    row[:, 8] = np.interp(efit_time, src_t, pnbi)

            # D-alpha
            if 'ada' in root and 'dalpha_integrated' in root['ada']:
                da = np.array(root['ada']['dalpha_integrated'])
                if len(da) > 0:
                    src_t = np.linspace(efit_time[0], efit_time[-1], len(da))
                    row[:, 9] = np.interp(efit_time, src_t, da)

            # Plasma current
            if 'amc' in root and 'plasma_current' in root['amc']:
                ip = np.array(root['amc']['plasma_current'])
                if len(ip) > 0:
                    src_t = np.linspace(efit_time[0], efit_time[-1], len(ip))
                    row[:, 10] = np.interp(efit_time, src_t, ip)

            # Clean rows
            mask = np.sum(~np.isnan(row), axis=1) >= 6
            clean = row[mask]

            if len(clean) >= 10:
                all_shot_data.append(clean)
                shot_meta.append({
                    'shot_id': int(shot_id),
                    'n_timepoints': int(len(clean)),
                    'valid_vars': int(np.sum(~np.all(np.isnan(row), axis=0))),
                    'efit_time_range': [float(efit_time[0]), float(efit_time[-1])]
                })
                dt = time.time() - t0
                print(f" ✓ ({len(clean)} tp, {dt:.1f}s)")
            else:
                print(f" ✗ (too few valid rows)")

        except Exception as e:
            print(f" ✗ ({str(e)[:50]})")

    if not all_shot_data:
        print("\n  ERROR: No data loaded!")
        return False

    combined = np.vstack(all_shot_data)

    # Fill NaN with column median
    for j in range(combined.shape[1]):
        col = combined[:, j]
        nans = np.isnan(col)
        if nans.any() and not nans.all():
            combined[nans, j] = np.nanmedian(col)
        elif nans.all():
            combined[:, j] = 0.0

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    # Save data
    np.save(os.path.join(FIXTURE_DIR, 'mast_real_data.npy'), combined)
    with open(os.path.join(FIXTURE_DIR, 'mast_real_meta.json'), 'w') as f:
        json.dump({
            'var_names': VAR_NAMES,
            'var_labels': VAR_LABELS,
            'shots': shot_meta,
            'total_timepoints': int(combined.shape[0]),
            'n_vars': int(combined.shape[1]),
            'source': 'UKAEA FAIR-MAST S3 (https://s3.echo.stfc.ac.uk/mast/level1/shots/)',
            'download_date': time.strftime('%Y-%m-%d'),
            'citation': 'Jackson et al., IEEE Trans. Plasma Sci., 2025',
        }, f, indent=2)

    # Run CPDE to generate expected results
    print(f"\n  Running CPDE on {combined.shape[0]} timepoints...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fusionmind4.discovery import EnsembleCPDE

    norm = combined.copy()
    for j in range(norm.shape[1]):
        s = np.std(norm[:, j])
        if s > 0:
            norm[:, j] = (norm[:, j] - np.mean(norm[:, j])) / s

    config = {
        'n_bootstrap': 10, 'threshold': 0.18,
        'physics_weight': 0.25, 'notears_weight': 0.30,
        'granger_weight': 0.25, 'pc_weight': 0.20,
    }
    cpde = EnsembleCPDE(config, verbose=True)
    result = cpde.discover(norm, var_names=VAR_NAMES)

    dag = result['dag']
    edges = []
    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if dag[i, j] > 0:
                score = float(result.get('edge_details', {}).get(
                    (i, j), {}).get('score', dag[i, j]))
                edges.append({
                    'src': VAR_NAMES[i], 'tgt': VAR_NAMES[j],
                    'src_idx': int(i), 'tgt_idx': int(j),
                    'score': round(score, 4)
                })

    with open(os.path.join(FIXTURE_DIR, 'mast_expected_results.json'), 'w') as f:
        json.dump({
            'n_edges': int(dag.sum()),
            'edges': edges,
            'config': config,
            'dag_shape': list(dag.shape),
        }, f, indent=2)

    sz = os.path.getsize(os.path.join(FIXTURE_DIR, 'mast_real_data.npy'))
    print(f"\n  ✓ Saved MAST fixtures ({sz/1024:.1f} KB)")
    print(f"    {combined.shape[0]} timepoints, {len(shot_meta)} shots")
    return True


def generate_cmod_fixtures():
    """Generate Alcator C-Mod synthetic fixture."""
    print("\n" + "=" * 60)
    print("Generating Alcator C-Mod Synthetic Fixture")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'scripts'))
    from run_real_data import generate_synthetic_cmod

    data, disrupted, labels = generate_synthetic_cmod(n_shots=1876, seed=42)
    print(f"  Full dataset: {data.shape}")

    # Subset for git-friendly size
    rng = np.random.RandomState(42)
    idx = rng.choice(len(data), size=5000, replace=False)

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(FIXTURE_DIR, 'cmod_synthetic.npz'),
        data=data[idx],
        disrupted=disrupted[idx],
        full_data_shape=data.shape
    )

    # Run CPDE
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fusionmind4.discovery import EnsembleCPDE

    norm = data[idx].copy()
    var_names = list(labels.keys())
    for j in range(norm.shape[1]):
        s = np.std(norm[:, j])
        if s > 0:
            norm[:, j] = (norm[:, j] - np.mean(norm[:, j])) / s

    cpde = EnsembleCPDE({'n_bootstrap': 8, 'threshold': 0.25}, verbose=True)
    result = cpde.discover(norm, var_names=var_names)

    with open(os.path.join(FIXTURE_DIR, 'cmod_expected_results.json'), 'w') as f:
        json.dump({
            'var_names': var_names,
            'labels': labels,
            'n_shots': 1876,
            'subset_size': 5000,
            'cpde': {'n_edges': int(result['n_edges'])},
        }, f, indent=2)

    sz = os.path.getsize(os.path.join(FIXTURE_DIR, 'cmod_synthetic.npz'))
    print(f"\n  ✓ Saved C-Mod fixtures ({sz/1024:.1f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download/regenerate test fixtures")
    parser.add_argument('--mast-only', action='store_true')
    parser.add_argument('--cmod-only', action='store_true')
    args = parser.parse_args()

    both = not args.mast_only and not args.cmod_only

    if both or args.mast_only:
        download_mast_fixtures()
    if both or args.cmod_only:
        generate_cmod_fixtures()

    print("\n" + "=" * 60)
    print("✓ All fixtures generated!")
    print("=" * 60)
    print(f"  Location: {FIXTURE_DIR}")
    for f in sorted(os.listdir(FIXTURE_DIR)):
        sz = os.path.getsize(os.path.join(FIXTURE_DIR, f))
        print(f"    {f:<35} {sz/1024:>8.1f} KB")


if __name__ == '__main__':
    main()
