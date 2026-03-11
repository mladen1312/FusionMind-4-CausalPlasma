#!/usr/bin/env python3
"""
FusionMind 4.0 — Production Inference (6-Track Parallel)
=========================================================

Combines GBT prediction + physics margins + TTD estimation + uncertainty
into a single production-ready output.

Usage:
  python scripts/predict_production.py --data data/mast/mast_level2_2941shots.npz \
      --labels data/mast/disruption_info.json --shot 27000

  python scripts/predict_production.py --data data/mast/mast_level2_2941shots.npz \
      --labels data/mast/disruption_info.json --output predictions.json

Output per shot:
  {
    "shot_id": 27000,
    "disruption_probability": 0.929,
    "risk_level": "CRITICAL",
    "time_to_disruption_ms": 142,
    "uncertainty": 0.034,
    "closest_limit": "li",
    "min_margin": 0.05,
    "explanation": "li at 95% of kink limit, margin shrinking at 0.003/ms",
    "active_tracks": ["physics", "stats", "trajectory", "causal", "rates", "pairwise"],
    "recommendation": "ALARM"
  }

Requirements: numpy, scikit-learn (no PyTorch)
"""
import argparse, json, sys, os, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.predict import resolve_signals, detect_machine_type, build_shot_features

# Stability limits per machine type
LIMITS = {
    'spherical':    {'li': 2.0, 'q95': 1.5, 'betan': 6.0, 'fGW': 1.5, 'betap': 2.0},
    'conventional': {'li': 1.5, 'q95': 2.0, 'betan': 3.5, 'fGW': 1.0, 'betap': 1.2},
}

def estimate_ttd_from_margin(shot_data, sig_map, machine_type, dt_ms=10):
    """Estimate time-to-disruption from margin trajectory.
    
    Extrapolates the steepest-declining margin to find when it hits 0.
    TTD = current_margin / |d(margin)/dt|
    """
    limits = LIMITS[machine_type]
    best_ttd = None
    best_reason = 'no declining margin'
    
    # Check each variable's margin trajectory
    for var_name, limit_name, limit_val, invert in [
        ('li', 'li', limits['li'], False),
        ('q95', 'q95', limits['q95'], True),
        ('betan', 'betan', limits['betan'], False),
    ]:
        if var_name not in sig_map:
            continue
        col = sig_map[var_name]
        signal = shot_data[:, col]
        if len(signal) < 10:
            continue
        
        # Compute margin time series
        if invert:  # q95: lower = worse
            margin = 1 - limit_val / (signal + 0.5)
        else:
            margin = 1 - signal / limit_val
        margin = np.clip(margin, -2, 1)
        
        # Compute rate in last 30%
        n30 = max(int(0.3 * len(margin)), 5)
        late = margin[-n30:]
        t = np.arange(len(late)) * dt_ms
        
        if np.std(late) < 1e-10:
            continue
        
        slope = np.polyfit(t, late, 1)[0]  # margin/ms
        current = margin[-1]
        
        if slope >= 0 or current <= -1:
            continue  # Not declining or already past limit
        
        ttd = max(current / abs(slope), 0)
        ttd = min(ttd, 10000)
        
        if best_ttd is None or ttd < best_ttd:
            best_ttd = ttd
            best_reason = f'{var_name} margin={current:.3f}, slope={slope:.5f}/ms'
    
    return best_ttd, best_reason


def estimate_uncertainty(model, X_shot):
    """Uncertainty from GBT staged predictions (tree variance)."""
    if not hasattr(model, 'staged_predict_proba'):
        return 0.0
    
    X_clean = np.clip(np.nan_to_num(X_shot), -1e6, 1e6).reshape(1, -1).astype(np.float64)
    
    try:
        staged = list(model.staged_predict_proba(X_clean))
    except:
        return 0.0
    
    n = len(staged)
    if n < 10:
        return 0.0
    
    checkpoints = [max(0, int(n*f)-1) for f in [0.4, 0.6, 0.8, 0.9, 1.0]]
    preds = [staged[c][0, 1] for c in checkpoints if c < n]
    return float(np.std(preds))


def assign_risk(prob, uncertainty, explanation):
    """Risk level from probability + uncertainty + physics rules."""
    rules = explanation.get('rules_triggered', [])
    n_rules = len(rules)
    min_margin = explanation.get('min_margin', 1.0)
    
    if prob > 0.8 and n_rules >= 1:
        return 'CRITICAL', 'ALARM'
    elif prob > 0.6:
        return 'HIGH', 'ALARM'
    elif prob > 0.3 or n_rules >= 1:
        return 'MEDIUM', 'MONITOR'
    elif prob > 0.1:
        return 'LOW', 'MONITOR'
    else:
        return 'SAFE', 'SAFE'


def predict_shot_full(shot_data, sig_map, limits, model, machine_type, sid=None):
    """Full production prediction for one shot."""
    t0 = time.time()
    
    # 1. Build features + physics explanation
    feats, explanation = build_shot_features(shot_data, sig_map, limits)
    
    # 2. GBT probability
    X = np.clip(np.nan_to_num(feats), -1e6, 1e6).reshape(1, -1).astype(np.float32)
    prob = float(model.predict_proba(X)[0, 1])
    
    # 3. Uncertainty
    uncertainty = estimate_uncertainty(model, feats)
    
    # 4. TTD
    ttd, ttd_reason = estimate_ttd_from_margin(shot_data, sig_map, machine_type)
    
    # 5. Risk level
    risk_level, recommendation = assign_risk(prob, uncertainty, explanation)
    
    # 6. Active tracks (based on available signals)
    active = ['physics']  # Always on
    if len(sig_map) >= 3: active.extend(['stats', 'trajectory', 'rates'])
    if len(sig_map) >= 4: active.append('pairwise')
    if any(k in sig_map for k in ['li', 'q95', 'betan']): active.append('causal')
    
    # 7. Build explanation string
    rules = explanation.get('rules_triggered', [])
    closest = explanation.get('closest_limit', '?')
    min_m = explanation.get('min_margin', 0)
    
    if rules:
        expl_str = '; '.join(rules[:3])
    else:
        expl_str = f'{closest} margin = {min_m:.3f} (safe)'
    
    if ttd is not None:
        expl_str += f' — TTD ≈ {ttd:.0f}ms'
    
    elapsed_ms = (time.time() - t0) * 1000
    
    return {
        'shot_id': int(sid) if sid else 0,
        'disruption_probability': round(prob, 4),
        'risk_level': risk_level,
        'time_to_disruption_ms': round(ttd, 0) if ttd else None,
        'uncertainty': round(uncertainty, 4),
        'closest_limit': closest,
        'min_margin': round(min_m, 3),
        'margins': {k: round(v, 3) for k, v in explanation.get('margins', {}).items()},
        'explanation': expl_str,
        'rules_triggered': rules,
        'active_tracks': active,
        'recommendation': recommendation,
        'inference_ms': round(elapsed_ms, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='FusionMind 4.0 Production Predictor')
    parser.add_argument('--data', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--shot', type=int, default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--top', type=int, default=10, help='Show top N highest risk')
    args = parser.parse_args()
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Load data
    d = np.load(args.data, allow_pickle=True)
    D = np.nan_to_num(d['data']).astype(np.float32)
    L = d['shot_ids']; VN = [str(v) for v in d['variables']]
    with open(args.labels) as f: di = json.load(f)
    u = np.unique(L); dset = set(di['disrupted'])
    
    sig_map = resolve_signals(VN)
    machine_type, limits = detect_machine_type(VN)
    
    print(f"FusionMind 4.0 Production Predictor")
    print(f"  Data: {len(u)} shots, {len(VN)} vars, machine={machine_type}")
    print(f"  Signals resolved: {list(sig_map.keys())}")
    
    # Build features + train GBT
    X_all = []; labels = []; sid_list = []
    expl_cache = {}
    for sid in u:
        mask = L == sid; n = mask.sum()
        if n < 8: continue
        is_dis = int(sid) in dset
        feats, expl = build_shot_features(
            D[mask][:n-4] if (is_dis and n > 7) else D[mask], sig_map, limits)
        X_all.append(feats); labels.append(1 if is_dis else 0)
        sid_list.append(int(sid)); expl_cache[int(sid)] = expl
    
    X = np.clip(np.nan_to_num(np.array(X_all)), -1e6, 1e6).astype(np.float32)
    lb = np.array(labels)
    
    # Augment + train
    np.random.seed(42)
    dis_idx = [i for i, l in enumerate(lb) if l == 1]
    aug = [X[i]*(1+np.random.normal(0,0.05,X.shape[1])) for i in dis_idx for _ in range(4)]
    Xa = np.vstack([X, np.clip(np.nan_to_num(np.array(aug)), -1e6, 1e6)])
    la = np.concatenate([lb, np.ones(len(dis_idx)*4)])
    
    print(f"  Training: {sum(lb)}d + {len(lb)-sum(lb)}c + {len(aug)} augmented")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=3, random_state=42)
    model.fit(Xa, la)
    print(f"  Model ready ({model.n_estimators} trees)")
    
    # Predict
    if args.shot is not None:
        # Single shot
        if args.shot not in sid_list:
            print(f"  Shot {args.shot} not found"); return
        mask = L == args.shot
        result = predict_shot_full(D[mask], sig_map, limits, model, machine_type, args.shot)
        
        print(f"\n{'='*60}")
        risk_emoji = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢','SAFE':'✅'}
        e = risk_emoji.get(result['risk_level'], '?')
        print(f"  Shot {result['shot_id']}: {e} {result['risk_level']}")
        print(f"  P(disruption) = {result['disruption_probability']:.3f} ± {result['uncertainty']:.3f}")
        print(f"  Recommendation: {result['recommendation']}")
        if result['time_to_disruption_ms']:
            print(f"  TTD: ~{result['time_to_disruption_ms']:.0f} ms")
        print(f"  Closest limit: {result['closest_limit']} (margin={result['min_margin']:.3f})")
        print(f"  Explanation: {result['explanation']}")
        print(f"  Active tracks: {result['active_tracks']}")
        print(f"  Inference: {result['inference_ms']:.1f} ms")
        print(f"{'='*60}")
    else:
        # Batch
        results = []
        t0 = time.time()
        for sid in sid_list:
            mask = L == sid
            r = predict_shot_full(D[mask], sig_map, limits, model, machine_type, sid)
            results.append(r)
        elapsed = time.time() - t0
        
        print(f"\n  Predicted {len(results)} shots in {elapsed:.1f}s ({elapsed/len(results)*1000:.1f}ms/shot)")
        
        # Top N highest risk
        ranked = sorted(results, key=lambda r: -r['disruption_probability'])
        risk_emoji = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢','SAFE':'✅'}
        
        print(f"\n  Top {args.top} highest risk:")
        for r in ranked[:args.top]:
            e = risk_emoji.get(r['risk_level'], '?')
            ttd_str = f"TTD={r['time_to_disruption_ms']:.0f}ms" if r['time_to_disruption_ms'] else "TTD=N/A"
            print(f"    {e} {r['risk_level']:<8} Shot {r['shot_id']}: "
                  f"P={r['disruption_probability']:.3f}±{r['uncertainty']:.3f} "
                  f"{ttd_str} — {r['explanation'][:60]}")
        
        n_alarm = sum(1 for r in results if r['recommendation'] == 'ALARM')
        n_monitor = sum(1 for r in results if r['recommendation'] == 'MONITOR')
        n_safe = sum(1 for r in results if r['recommendation'] == 'SAFE')
        print(f"\n  Summary: {n_alarm} ALARM / {n_monitor} MONITOR / {n_safe} SAFE")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to {args.output}")


if __name__ == '__main__':
    main()
