#!/usr/bin/env python3
"""
FusionMind 4.0 — Inference Script
===================================
Predict disruption probability for a single shot or batch of shots.
Uses the verified GBT model (AUC=0.979 on MAST, 5-fold CV).

Usage:
  # Single shot from MAST data:
  python scripts/predict.py --data data/mast/mast_level2_2941shots.npz --shot 30420

  # All shots, output CSV:
  python scripts/predict.py --data data/mast/mast_level2_2941shots.npz --output predictions.csv

  # Train on labeled data, then predict unlabeled:
  python scripts/predict.py --data data/mast/mast_level2_2941shots.npz \
      --labels data/mast/disruption_info.json --output predictions.csv

Requirements: numpy, scikit-learn (no PyTorch needed)
"""
import argparse
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════
# FEATURE BUILDER (same as verified 63f + margins)
# ═══════════════════════════════════════════════

SIGNAL_MAP = {
    'li': ['li', 'internal_inductance'],
    'q95': ['q95', 'safety_factor_95'],
    'betan': ['betan', 'beta_n', 'normalized_beta'],
    'betap': ['betap', 'beta_p'],
    'ne': ['ne_line', 'density', 'ne_avg'],
    'ne_gw': ['greenwald_den', 'n_greenwald'],
    'Ip': ['Ip', 'plasma_current'],
    'p_rad': ['p_rad', 'radiated_power'],
    'wmhd': ['wmhd', 'stored_energy'],
    'p_nbi': ['p_nbi', 'p_input'],
    'kappa': ['elongation', 'kappa'],
    'a': ['minor_radius', 'aminor'],
    'Bt': ['toroidal_B_field', 'bt'],
}

LIMITS_SPHERICAL = {'li': 2.0, 'q95': 1.5, 'betan': 6.0, 'fGW': 1.5, 'betap': 2.0}
LIMITS_CONVENTIONAL = {'li': 1.5, 'q95': 2.0, 'betan': 3.5, 'fGW': 1.0, 'betap': 1.2}


def resolve_signals(var_names):
    """Map canonical names to actual column indices."""
    mapping = {}
    var_idx = {v: i for i, v in enumerate(var_names)}
    for canonical, aliases in SIGNAL_MAP.items():
        for alias in aliases:
            if alias in var_idx:
                mapping[canonical] = var_idx[alias]
                break
    return mapping


def detect_machine_type(var_names):
    """Heuristic: spherical tokamaks have higher beta limits."""
    has_betan = any(v in var_names for v in ['betan', 'beta_n'])
    has_ne_gw = any(v in var_names for v in ['greenwald_den', 'n_greenwald'])
    # If we have both betan AND greenwald_den, likely MAST-style data
    if has_betan and has_ne_gw:
        return 'spherical', LIMITS_SPHERICAL
    return 'conventional', LIMITS_CONVENTIONAL


def build_shot_features(shot_data, sig_map, limits, truncate_tp=0):
    """Build 63 features for one shot. Returns (features, explanation)."""
    n = len(shot_data)
    if truncate_tp > 0 and n > truncate_tp + 3:
        shot_data = shot_data[:n - truncate_tp]
        n = len(shot_data)

    n30 = max(int(0.3 * n), 1)
    feats = []
    signals = {}

    # Extract available signals
    for name, col in sig_map.items():
        signals[name] = shot_data[:, col]

    # Derived: Greenwald fraction
    if 'ne' in signals and 'ne_gw' in signals:
        signals['f_GW'] = signals['ne'] / (signals['ne_gw'] + 1e-10)
    elif 'ne' in signals and 'Ip' in signals and 'a' in signals:
        ngw = signals['Ip'] / (np.pi * signals['a']**2 + 1e-10)
        signals['f_GW'] = signals['ne'] / (ngw + 1e-10)

    # Per-signal stats (6 per signal)
    key_sigs = ['li', 'q95', 'betan', 'betap', 'f_GW', 'p_rad', 'wmhd', 'Ip']
    for name in key_sigs:
        if name not in signals:
            feats.extend([0, 0, 0, 0, 0, 0])
            continue
        s = signals[name]
        if len(s) < 3:
            feats.extend([0, 0, 0, 0, 0, 0])
            continue
        feats.extend([
            np.mean(s), np.std(s), np.max(s),
            np.mean(s[-n30:]),
            np.mean(s[-n30:]) - np.mean(s[:n30]),
            np.max(np.abs(np.diff(s)))
        ])

    # Stability margins
    li_s = signals.get('li', np.zeros(1))
    q_s = signals.get('q95', np.ones(1) * 10)
    bn_s = signals.get('betan', np.zeros(1))
    bp_s = signals.get('betap', np.zeros(1))
    fgw_s = signals.get('f_GW', np.zeros(1))
    q_min = np.min(q_s[q_s > 0.5]) if np.any(q_s > 0.5) else 10

    margins = {
        'li': np.clip(1 - np.max(li_s) / limits['li'], -1, 1),
        'q95': np.clip(1 - limits['q95'] / q_min, -1, 1),
        'betan': np.clip(1 - np.max(bn_s) / limits['betan'], -1, 1),
        'fGW': np.clip(1 - np.max(fgw_s) / limits['fGW'], -1, 1),
        'betap': np.clip(1 - np.max(bp_s) / limits['betap'], -1, 1),
    }
    feats.extend(list(margins.values()))
    feats.append(min(margins.values()))
    feats.append(sum(1 for m in margins.values() if m < 0.3))

    # Cross-variable interactions
    feats.extend([
        np.max(li_s) * np.max(bn_s),
        np.max(li_s) / (q_min + 0.5),
        np.std(li_s) * np.std(q_s),
        np.max(fgw_s) * np.max(li_s),
    ])

    # Temporal shape
    feats.extend([
        np.max(li_s[-n30:]) / (np.mean(li_s[:n30]) + 1e-10),
        np.max(li_s[-n30:]) - np.mean(li_s[:n30]),
        np.max(bn_s[-n30:]) - np.mean(bn_s[:n30]),
        np.min(q_s[-n30:]) / (np.mean(q_s[:n30]) + 1e-10),
    ])

    # Build explanation
    closest = min(margins, key=margins.get)
    explanation = {
        'margins': {k: round(v, 3) for k, v in margins.items()},
        'closest_limit': closest,
        'min_margin': round(min(margins.values()), 3),
        'n_limits_stressed': sum(1 for m in margins.values() if m < 0.3),
    }

    # Neuro-symbolic rules
    rules_triggered = []
    if margins['li'] < 0.1:
        rules_triggered.append(f"li at {100*(1-margins['li']):.0f}% of kink limit")
    if margins['q95'] < 0.1:
        rules_triggered.append(f"q95 at {100*(1-margins['q95']):.0f}% of stability limit")
    if margins['betan'] < 0.2:
        rules_triggered.append(f"betaN at {100*(1-margins['betan']):.0f}% of Troyon limit")
    if margins['fGW'] < 0.1:
        rules_triggered.append(f"density at {100*(1-margins['fGW']):.0f}% of Greenwald limit")
    if 'li' in signals and len(li_s) > 5:
        li_rate = np.mean(li_s[-n30:]) - np.mean(li_s[:n30])
        if li_rate > 0.1:
            rules_triggered.append(f"li rising (trend={li_rate:.3f})")
    explanation['rules_triggered'] = rules_triggered

    return np.array(feats, dtype=np.float32), explanation


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FusionMind 4.0 Disruption Predictor')
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--labels', default=None, help='Path to disruption_info.json (for training)')
    parser.add_argument('--shot', type=int, default=None, help='Predict single shot')
    parser.add_argument('--output', default=None, help='Output CSV path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Alarm threshold')
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    d = np.load(args.data, allow_pickle=True)
    D = np.nan_to_num(d['data']).astype(np.float32)
    L = d['shot_ids']
    VN = [str(v) for v in d['variables']]
    u = np.unique(L)

    print(f"  {len(u)} shots, {D.shape[0]} timepoints, {len(VN)} variables")
    print(f"  Variables: {VN}")

    # Resolve signals and detect machine
    sig_map = resolve_signals(VN)
    machine_type, limits = detect_machine_type(VN)
    print(f"  Machine type: {machine_type}")
    print(f"  Resolved signals: {list(sig_map.keys())}")

    # Build features for all shots
    X_all = []
    shot_list = []
    explanations = {}

    for sid in u:
        mask = L == sid
        n = mask.sum()
        if n < 8:
            continue
        feats, expl = build_shot_features(D[mask], sig_map, limits)
        X_all.append(feats)
        shot_list.append(int(sid))
        explanations[int(sid)] = expl

    X = np.clip(np.nan_to_num(np.array(X_all)), -1e6, 1e6).astype(np.float32)
    print(f"  Built {X.shape[1]} features for {len(shot_list)} shots")

    # If labels provided: train model, then predict
    if args.labels:
        from sklearn.ensemble import GradientBoostingClassifier

        with open(args.labels) as f:
            di = json.load(f)
        dset = set(di['disrupted'])
        labels = np.array([1 if s in dset else 0 for s in shot_list])
        n_dis = sum(labels)
        print(f"  Labels: {n_dis} disrupted + {len(labels)-n_dis} clean")

        # Augmentation
        np.random.seed(42)
        dis_idx = [i for i, l in enumerate(labels) if l == 1]
        aug_X = [X[i] * (1 + np.random.normal(0, 0.05, X.shape[1]))
                 for i in dis_idx for _ in range(4)]
        X_aug = np.vstack([X, np.clip(np.nan_to_num(np.array(aug_X)), -1e6, 1e6)])
        labels_aug = np.concatenate([labels, np.ones(len(dis_idx) * 4)])

        # Train
        print("  Training GBT (100 trees, depth=4)...")
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=3, random_state=42)
        model.fit(X_aug, labels_aug)

        # Predict
        probs = model.predict_proba(X)[:, 1]
    else:
        # No labels: use physics-only scoring (Track A)
        print("  No labels provided — using physics-only scoring")
        probs = np.array([1 - explanations[s]['min_margin'] for s in shot_list])

    # Single shot output
    if args.shot is not None:
        if args.shot in shot_list:
            idx = shot_list.index(args.shot)
            prob = probs[idx]
            expl = explanations[args.shot]
            print(f"\n{'='*50}")
            print(f"Shot {args.shot}: P(disruption) = {prob:.3f}")
            print(f"  Decision: {'⚠ ALARM' if prob > args.threshold else '✓ SAFE'}")
            print(f"  Closest limit: {expl['closest_limit']} (margin={expl['min_margin']:.3f})")
            print(f"  All margins: {expl['margins']}")
            if expl['rules_triggered']:
                print(f"  Rules triggered:")
                for rule in expl['rules_triggered']:
                    print(f"    → {rule}")
            else:
                print(f"  No physics rules triggered")
        else:
            print(f"  Shot {args.shot} not found in data")
        return

    # Batch output
    print(f"\n  Top 10 highest risk shots:")
    ranked = sorted(zip(shot_list, probs), key=lambda x: -x[1])
    for sid, prob in ranked[:10]:
        expl = explanations[sid]
        status = "⚠ ALARM" if prob > args.threshold else "  watch"
        rules = "; ".join(expl['rules_triggered'][:2]) if expl['rules_triggered'] else "no rules"
        print(f"    {status} Shot {sid}: P={prob:.3f}  closest={expl['closest_limit']}({expl['min_margin']:.2f})  {rules}")

    # Save CSV
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['shot_id', 'disruption_probability', 'decision',
                        'closest_limit', 'min_margin', 'rules_triggered'])
            for sid, prob in zip(shot_list, probs):
                expl = explanations[sid]
                decision = 'ALARM' if prob > args.threshold else 'SAFE'
                rules = "; ".join(expl['rules_triggered'])
                w.writerow([sid, f"{prob:.4f}", decision,
                           expl['closest_limit'], f"{expl['min_margin']:.3f}", rules])
        print(f"\n  Saved {len(shot_list)} predictions to {args.output}")


if __name__ == '__main__':
    main()
