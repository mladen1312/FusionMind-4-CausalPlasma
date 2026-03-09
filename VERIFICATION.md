# FusionMind 4.0 — Verification Guide

**Repo:** https://github.com/mladen1312/FusionMind-4-CausalPlasma

## How to verify every claimed result

### Step 0: Clone and check data
```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma

# Check data files exist
ls -lh data/mast/mast_level2_2941shots.npz   # 13M, 2941 MAST shots
ls -lh data/cmod/cmod_density_limit.npz       # 4.7M, 2333 C-Mod shots
ls -lh data/mast/disruption_info.json         # 83 disrupted + 249 clean labels
ls -lh data/cmod/disruption_info.json         # 78 disrupted + 2255 clean labels
```

### Step 1: Verify data contents
```python
import numpy as np, json

# MAST
md = np.load('data/mast/mast_level2_2941shots.npz', allow_pickle=True)
D = md['data']; L = md['shot_ids']; VN = [str(v) for v in md['variables']]
print(f"MAST: {len(np.unique(L))} shots, {D.shape[0]} timepoints, {len(VN)} vars")
print(f"Variables: {VN}")
# Expected: 2941 shots, 268667 timepoints, 16 variables
# Variables: betan, betap, betat, q95, elongation, li, wmhd, q_axis,
#            minor_radius, tribot, tritop, Ip, ne_line, greenwald_den, p_rad, p_nbi

with open('data/mast/disruption_info.json') as f: di = json.load(f)
print(f"Disrupted: {len(di['disrupted'])}, Clean: {len(di['clean'])}")
# Expected: 83 disrupted, 249 clean (expert labels)

# C-Mod
cd = np.load('data/cmod/cmod_density_limit.npz', allow_pickle=True)
D_c = cd['data']; L_c = cd['shot_ids']; VN_c = [str(v) for v in cd['variables']]
print(f"C-Mod: {len(np.unique(L_c))} shots, {D_c.shape[0]} tp, {len(VN_c)} vars")
print(f"Variables: {VN_c}")
# Expected: 2333 shots, 264385 tp, 6 variables
# Variables: density, elongation, minor_radius, plasma_current, toroidal_B_field, triangularity

with open('data/cmod/disruption_info.json') as f: di_c = json.load(f)
print(f"Disrupted: {len(di_c['disrupted'])}, Clean: {len(di_c['clean'])}")
# Expected: 78 disrupted, 2255 clean
```

### Step 2: Reproduce MAST AUC = 0.979 ± 0.011
```python
import numpy as np, json
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

md = np.load('data/mast/mast_level2_2941shots.npz', allow_pickle=True)
D = np.nan_to_num(md['data']).astype(np.float32)
L = md['shot_ids']; VN = [str(v) for v in md['variables']]
with open('data/mast/disruption_info.json') as f: di = json.load(f)
idx = {v:i for i,v in enumerate(VN)}
u = np.unique(L); dset = set(di['disrupted'])
S = {v: D[:,idx[v]] for v in VN}
S['f_GW'] = S['ne_line']/(S['greenwald_den']+1e-10)
NU = 4  # 40ms truncation (fair evaluation)

key8 = ['li','q95','betan','betap','f_GW','p_rad','wmhd','Ip']
X_list = []; lb_list = []
for sid in u:
    mask=L==sid; n=mask.sum()
    if n<8: continue
    is_dis = int(sid) in dset
    e = n-NU if (is_dis and n>NU+3) else n
    n30 = max(int(0.3*e),1)
    feats = []
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

# Augmentation: 4x disrupted copies with 5% noise
np.random.seed(42)
dis_idx = [i for i,l in enumerate(lb) if l==1]
aug_X = []; aug_lb = []
for i in dis_idx:
    for _ in range(4):
        aug_X.append(X[i]*(1+np.random.normal(0,0.05,X.shape[1])))
        aug_lb.append(1)
X_aug = np.vstack([X, np.clip(np.nan_to_num(np.array(aug_X)),-1e6,1e6)])
lb_aug = np.concatenate([lb, np.array(aug_lb)])
n_orig = len(lb)

# 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
for fold,(tr,te) in enumerate(kf.split(X, lb)):
    tr_aug = list(tr)
    for j in range(len(aug_X)):
        if dis_idx[j//4] in tr: tr_aug.append(n_orig+j)
    gbt = GradientBoostingClassifier(n_estimators=100, max_depth=4,
          learning_rate=0.1, subsample=0.8, min_samples_leaf=3, random_state=42)
    gbt.fit(X_aug[tr_aug], lb_aug[tr_aug])
    p = gbt.predict_proba(X[te])[:,1]
    auc = roc_auc_score(lb[te], p)
    fold_aucs.append(auc)
    print(f"Fold {fold}: AUC = {auc:.4f}")

print(f"\nMAST AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
# Expected: 0.979 ± 0.011
# Expected folds: [0.984, 0.988, 0.971, 0.960, 0.990]
```

### Step 3: Reproduce C-Mod physics AUC = 0.978
```python
import numpy as np, json
from sklearn.metrics import roc_auc_score

cd = np.load('data/cmod/cmod_density_limit.npz', allow_pickle=True)
D = np.nan_to_num(cd['data']).astype(np.float32)
L = cd['shot_ids']; VN = [str(v) for v in cd['variables']]
with open('data/cmod/disruption_info.json') as f: di = json.load(f)
idx = {v:i for i,v in enumerate(VN)}
u = np.unique(L); dset = set(di['disrupted'])

ne=D[:,idx['density']]; Ip=D[:,idx['plasma_current']]; a=D[:,idx['minor_radius']]
Bt=D[:,idx['toroidal_B_field']]; kappa=D[:,idx['elongation']]
f_GW = ne / (Ip/(np.pi*a**2+1e-10)+1e-10)
q95 = 5*a**2*Bt*kappa/(Ip*0.67+1e-10)
inv_q = Ip/(a*Bt+1e-10)

scores = []; labels = []
for sid in u:
    mask=L==sid; n=mask.sum()
    if n<5: continue
    fgw_s=f_GW[mask]; n30=max(int(0.3*n),1)
    q_s=q95[mask]; iq_s=inv_q[mask]; dfgw=np.gradient(fgw_s)
    q_min=np.min(q_s[q_s>0]) if np.any(q_s>0) else 10
    score = (0.35*np.max(fgw_s) + 0.25*np.mean(fgw_s[-n30:]) +
             0.15*max(np.mean(dfgw[-n30:]),0) + 0.10/(q_min+0.1) +
             0.10*np.max(iq_s*fgw_s) + 0.05*np.mean(fgw_s > 0.8*np.max(fgw_s)))
    scores.append(score); labels.append(1 if int(sid) in dset else 0)

auc = roc_auc_score(labels, scores)
print(f"C-Mod physics AUC = {auc:.3f}")
# Expected: 0.978
# NOTE: This is density-limit-only data. NOT comparable to CCNN (0.974) which
# tests on ALL disruption types. Greenwald fraction alone gives 0.985.
```

### Step 4: Reproduce MAST physics li AUC = 0.908
```python
# Same MAST data as Step 2
scores = []; labels = []
for sid in u:
    mask=L==sid; n=mask.sum()
    if n<5: continue
    scores.append(np.max(S['li'][mask]))
    labels.append(1 if int(sid) in dset else 0)
print(f"MAST max(li) AUC = {roc_auc_score(labels, scores):.3f}")
# Expected: 0.908
```

### Step 5: Verify v1.1 baseline (from benchmark files)
```python
import json
with open('benchmarks/disruptionbench_Ensemblegeo.json') as f:
    d = json.load(f)
print(f"v1.1 AUC = {d['metrics']['roc_auc_score']:.3f}")
# Expected: 0.842
```

## Key benchmark files to check

| File | Contains |
|------|----------|
| `benchmarks/best_model_mast_v2.json` | AUC=0.979 result details |
| `benchmarks/physics_vs_ml.json` | Physics vs ML comparison |
| `benchmarks/cmod_honest_assessment.json` | Why C-Mod is not comparable to CCNN |
| `benchmarks/disruptionbench_Ensemblegeo.json` | v1.1 baseline AUC=0.842 |
| `RESULTS.md` | Complete results summary |

## Data sources (publicly verifiable)

- **MAST data**: FAIR-MAST S3 archive at `s3://mast/level1/shots/` via `https://s3.echo.stfc.ac.uk`
  - GraphQL API: `https://mastapp.site/graphql`
  - 2941 shots with EFIT equilibrium reconstruction data
  - 16 variables from IMAS/IDS format
- **C-Mod data**: MIT PSFC Open Density Limit Database
  - Source: disruption-py package (MIT PSFC)
  - 2333 shots, 6 equilibrium variables
  - Contains ONLY density-limit disruptions

## Important caveats

1. **MAST 0.979 vs CCNN 0.974 are NOT directly comparable**
   - Different machines (MAST spherical vs C-Mod conventional)
   - Different datasets (83 disrupted vs hundreds)
   - Different disruption types (diverse vs all-types)
   - Different evaluation protocols (shot-level GBT vs timepoint-level CCNN)

2. **C-Mod 0.978 is trivially high** because dataset contains only density-limit
   disruptions. Peak Greenwald fraction ALONE gives AUC=0.985.

3. **83 disrupted shots** means ~16-17 per test fold → confidence intervals are wide
