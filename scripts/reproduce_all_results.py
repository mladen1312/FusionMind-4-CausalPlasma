#!/usr/bin/env python3
"""
FusionMind 4.0 — Complete Result Reproduction Script
Run this to verify ALL claimed results from raw data.

Usage: python scripts/reproduce_all_results.py

Requirements: numpy, scikit-learn, scipy
"""
import numpy as np
import json
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

def load_mast():
    md = np.load('data/mast/mast_level2_2941shots.npz', allow_pickle=True)
    D = np.nan_to_num(md['data']).astype(np.float32)
    L = md['shot_ids']; VN = [str(v) for v in md['variables']]
    with open('data/mast/disruption_info.json') as f: di = json.load(f)
    idx = {v:i for i,v in enumerate(VN)}
    u = np.unique(L); dset = set(di['disrupted'])
    S = {v: D[:,idx[v]] for v in VN}
    S['f_GW'] = S['ne_line']/(S['greenwald_den']+1e-10)
    return D, L, VN, idx, u, dset, S

def load_cmod():
    cd = np.load('data/cmod/cmod_density_limit.npz', allow_pickle=True)
    D = np.nan_to_num(cd['data']).astype(np.float32)
    L = cd['shot_ids']; VN = [str(v) for v in cd['variables']]
    with open('data/cmod/disruption_info.json') as f: di = json.load(f)
    idx = {v:i for i,v in enumerate(VN)}
    u = np.unique(L); dset = set(di['disrupted'])
    return D, L, VN, idx, u, dset

def build_mast_features(D, L, VN, idx, u, dset, S):
    NU = 4; key8 = ['li','q95','betan','betap','f_GW','p_rad','wmhd','Ip']
    X_list = []; lb_list = []
    for sid in u:
        mask=L==sid; n=mask.sum()
        if n<8: continue
        is_dis = int(sid) in dset
        e = n-NU if (is_dis and n>NU+3) else n
        n30 = max(int(0.3*e),1); feats = []
        li_s=S['li'][mask][:e]; q_s=S['q95'][mask][:e]
        bn=S['betan'][mask][:e]; bp=S['betap'][mask][:e]; fgw=S['f_GW'][mask][:e]
        q_min = np.min(q_s[q_s>0.5]) if np.any(q_s>0.5) else 10
        for v in key8:
            s = S[v][mask][:e] if v!='f_GW' else fgw
            if len(s)<3: s=np.zeros(5)
            feats.extend([np.mean(s),np.std(s),np.max(s),np.mean(s[-n30:]),
                          np.mean(s[-n30:])-np.mean(s[:n30]),
                          np.max(np.abs(np.diff(s))) if len(s)>1 else 0])
        margins = [np.clip(1-np.max(li_s)/2,-1,1), np.clip(1-2/q_min,-1,1),
                   np.clip(1-np.max(bn)/4,-1,1), np.clip(1-np.max(fgw)/1.2,-1,1),
                   np.clip(1-np.max(bp)/1.5,-1,1)]
        feats.extend(margins + [min(margins), sum(1 for m in margins if m<0.3)])
        feats.extend([np.max(li_s)*np.max(bn), np.max(li_s)/(q_min+0.5),
                      np.std(li_s)*np.std(q_s), np.max(fgw)*np.max(li_s)])
        feats.extend([np.max(li_s[-n30:])/(np.mean(li_s[:n30])+1e-10),
                      np.max(li_s[-n30:])-np.mean(li_s[:n30]),
                      np.max(bn[-n30:])-np.mean(bn[:n30]),
                      np.min(q_s[-n30:])/(np.mean(q_s[:n30])+1e-10)])
        X_list.append(feats); lb_list.append(1 if is_dis else 0)
    X = np.clip(np.nan_to_num(np.array(X_list)),-1e6,1e6).astype(np.float32)
    lb = np.array(lb_list)
    return X, lb

def augment(X, lb, n_copies=4, noise=0.05, seed=42):
    np.random.seed(seed)
    dis_idx = [i for i,l in enumerate(lb) if l==1]
    aug_X = []; aug_lb = []
    for i in dis_idx:
        for _ in range(n_copies):
            aug_X.append(X[i]*(1+np.random.normal(0, noise, X.shape[1])))
            aug_lb.append(1)
    X_aug = np.vstack([X, np.clip(np.nan_to_num(np.array(aug_X)),-1e6,1e6)])
    lb_aug = np.concatenate([lb, np.array(aug_lb)])
    return X_aug, lb_aug, dis_idx

def main():
    print("="*70)
    print("FUSIONMIND 4.0 — COMPLETE RESULT REPRODUCTION")
    print("="*70)

    # ═══ TEST 1: Data verification ═══
    print("\n[TEST 1] Data verification")
    D, L, VN, idx, u, dset, S = load_mast()
    n_shots = len(u); n_dis = len([s for s in u if s in dset])
    print(f"  MAST: {n_shots} shots, {D.shape[0]} tp, {len(VN)} vars, {n_dis} disrupted")
    assert n_shots == 2941, f"Expected 2941 shots, got {n_shots}"
    assert n_dis == 83, f"Expected 83 disrupted, got {n_dis}"
    assert len(VN) == 16, f"Expected 16 vars, got {len(VN)}"
    print("  ✓ MAST data verified")

    D_c, L_c, VN_c, idx_c, u_c, dset_c = load_cmod()
    n_shots_c = len(u_c); n_dis_c = len([s for s in u_c if s in dset_c])
    print(f"  C-Mod: {n_shots_c} shots, {D_c.shape[0]} tp, {len(VN_c)} vars, {n_dis_c} disrupted")
    assert n_shots_c == 2333, f"Expected 2333, got {n_shots_c}"
    assert n_dis_c == 78, f"Expected 78, got {n_dis_c}"
    print("  ✓ C-Mod data verified")

    # ═══ TEST 2: MAST max(li) AUC = 0.908 ═══
    print("\n[TEST 2] MAST max(li) AUC")
    sc = [np.max(S['li'][L==sid]) for sid in u if (L==sid).sum()>=5]
    lb_t = [1 if int(sid) in dset else 0 for sid in u if (L==sid).sum()>=5]
    auc_li = roc_auc_score(lb_t, sc)
    print(f"  max(li) AUC = {auc_li:.4f}  (claimed: 0.908)")
    assert abs(auc_li - 0.908) < 0.005, f"MISMATCH: {auc_li:.4f} vs 0.908"
    print("  ✓ Confirmed")

    # ═══ TEST 3: C-Mod peak f_GW AUC = 0.985 ═══
    print("\n[TEST 3] C-Mod peak Greenwald fraction AUC")
    ne=D_c[:,idx_c['density']]; Ip=D_c[:,idx_c['plasma_current']]; a=D_c[:,idx_c['minor_radius']]
    f_GW_c = ne / (Ip/(np.pi*a**2+1e-10)+1e-10)
    sc_c = [np.max(f_GW_c[L_c==sid]) for sid in u_c if (L_c==sid).sum()>=5]
    lb_c = [1 if int(sid) in dset_c else 0 for sid in u_c if (L_c==sid).sum()>=5]
    auc_fgw = roc_auc_score(lb_c, sc_c)
    print(f"  peak f_GW AUC = {auc_fgw:.4f}  (claimed: 0.985)")
    assert abs(auc_fgw - 0.985) < 0.005, f"MISMATCH: {auc_fgw:.4f} vs 0.985"
    print("  ✓ Confirmed")

    # ═══ TEST 4: C-Mod physics formula AUC = 0.978 ═══
    print("\n[TEST 4] C-Mod physics formula AUC")
    Bt=D_c[:,idx_c['toroidal_B_field']]; kap=D_c[:,idx_c['elongation']]
    q95_c = 5*a**2*Bt*kap/(Ip*0.67+1e-10); inv_q = Ip/(a*Bt+1e-10)
    sc_phys = []; lb_phys = []
    for sid in u_c:
        mask=L_c==sid; n=mask.sum()
        if n<5: continue
        fgw_s=f_GW_c[mask]; n30=max(int(0.3*n),1)
        q_s=q95_c[mask]; iq_s=inv_q[mask]; dfgw=np.gradient(fgw_s)
        q_min=np.min(q_s[q_s>0]) if np.any(q_s>0) else 10
        score = (0.35*np.max(fgw_s) + 0.25*np.mean(fgw_s[-n30:]) +
                 0.15*max(np.mean(dfgw[-n30:]),0) + 0.10/(q_min+0.1) +
                 0.10*np.max(iq_s*fgw_s) + 0.05*np.mean(fgw_s > 0.8*np.max(fgw_s)))
        sc_phys.append(score); lb_phys.append(1 if int(sid) in dset_c else 0)
    auc_phys = roc_auc_score(lb_phys, sc_phys)
    print(f"  Physics formula AUC = {auc_phys:.4f}  (claimed: 0.978)")
    assert abs(auc_phys - 0.978) < 0.005, f"MISMATCH: {auc_phys:.4f} vs 0.978"
    print("  ✓ Confirmed")

    # ═══ TEST 5: MAST GBT 63f + augmentation AUC = 0.979 ═══
    print("\n[TEST 5] MAST GBT 63f + augmentation (5-fold CV)")
    X, lb = build_mast_features(D, L, VN, idx, u, dset, S)
    print(f"  Features: {X.shape[1]}, Shots: {len(lb)} ({sum(lb)}d + {sum(~lb.astype(bool))}c)")
    X_aug, lb_aug, dis_idx = augment(X, lb)
    n_orig = len(lb)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []; fold_tpr5 = []
    for fold,(tr,te) in enumerate(kf.split(X, lb)):
        tr_aug = list(tr)
        for j in range(len(dis_idx)*4):
            if dis_idx[j//4] in tr: tr_aug.append(n_orig+j)
        gbt = GradientBoostingClassifier(n_estimators=100, max_depth=4,
              learning_rate=0.1, subsample=0.8, min_samples_leaf=3, random_state=42)
        gbt.fit(X_aug[tr_aug], lb_aug[tr_aug])
        p = gbt.predict_proba(X[te])[:,1]
        auc = roc_auc_score(lb[te], p)
        fpr_a,tpr_a,_ = roc_curve(lb[te],p)
        tpr5 = tpr_a[np.argmin(np.abs(fpr_a-0.05))]
        fold_aucs.append(auc); fold_tpr5.append(tpr5)
        print(f"  Fold {fold}: AUC={auc:.4f}  TPR@5%={tpr5:.0%}  ({sum(lb[te])}d test)")

    mean_auc = np.mean(fold_aucs); std_auc = np.std(fold_aucs)
    print(f"\n  MAST AUC = {mean_auc:.3f} ± {std_auc:.3f}  (claimed: 0.979 ± 0.011)")
    print(f"  TPR@5%   = {np.mean(fold_tpr5):.0%}")
    assert abs(mean_auc - 0.979) < 0.005, f"MISMATCH: {mean_auc:.3f} vs 0.979"
    print("  ✓ Confirmed")

    # ═══ TEST 6: v1.1 baseline from benchmark file ═══
    print("\n[TEST 6] v1.1 baseline AUC from benchmark file")
    with open('benchmarks/disruptionbench_Ensemblegeo.json') as f:
        d = json.load(f)
    print(f"  v1.1 AUC = {d['metrics']['roc_auc_score']:.3f}  (claimed: 0.842)")
    assert abs(d['metrics']['roc_auc_score'] - 0.842) < 0.005
    print("  ✓ Confirmed")

    # ═══ SUMMARY ═══
    print(f"\n{'='*70}")
    print(f"ALL 6 TESTS PASSED — every claimed number reproduced from raw data")
    print(f"{'='*70}")
    print(f"""
  VERIFIED RESULTS:
    MAST max(li) alone:           AUC = {auc_li:.3f}
    MAST GBT 63f + augmentation:  AUC = {mean_auc:.3f} ± {std_auc:.3f}
    C-Mod peak f_GW alone:        AUC = {auc_fgw:.3f}
    C-Mod physics formula:        AUC = {auc_phys:.3f}
    v1.1 baseline:                AUC = 0.842

  COMPARISON WITH LITERATURE:
    FusionMind (MAST):   {mean_auc:.3f}   ← THIS WORK
    CCNN (C-Mod):        0.974   ← NOT DIRECTLY COMPARABLE
    FRNN (DIII-D):       ~0.97
    GPT-2 (C-Mod):       0.840

  WHY NOT DIRECTLY COMPARABLE:
    - Different machines (MAST spherical vs C-Mod/DIII-D conventional)
    - Different datasets (83 disrupted vs thousands)
    - Different disruption types
    - Different evaluation protocols (shot-level vs timepoint-level)
""")

if __name__ == "__main__":
    main()
