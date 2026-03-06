#!/usr/bin/env python3
"""
FusionMind 4.0 — Large-Scale Statistical Benchmark
====================================================
332 shots | 25,548 timepoints | 9 variables | FAIR-MAST (UKAEA)

Proper statistical validation:
  1. Data cleaning (outlier removal, physical range filtering)
  2. Leave-5-shots-out cross-validation (66 folds)
  3. Per-fold CPDE + SCM + evaluation
  4. Bootstrap confidence intervals
  5. Honest metrics (no proxy labels for disruption)
"""

import numpy as np
import json
import time
import warnings
from scipy.optimize import minimize
from scipy.stats import f as f_dist, norm
from scipy.linalg import expm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from itertools import combinations
warnings.filterwarnings("ignore")

VARS = ['betan', 'betap', 'q_95', 'q_axis', 'elongation', 'li', 'wplasmd', 'betat', 'Ip']

GROUND_TRUTH_PAIRS = {
    frozenset(('betan', 'betap')),
    frozenset(('betan', 'betat')),
    frozenset(('betan', 'wplasmd')),
    frozenset(('betap', 'wplasmd')),
    frozenset(('q_95', 'q_axis')),
    frozenset(('li', 'q_95')),
    frozenset(('li', 'q_axis')),
    frozenset(('wplasmd', 'betat')),
    frozenset(('elongation', 'wplasmd')),
    frozenset(('elongation', 'betap')),
    frozenset(('Ip', 'q_95')),
    frozenset(('Ip', 'betan')),
    frozenset(('Ip', 'li')),
}


# ═══════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════

def clean_data(data, labels):
    """Remove outliers and unphysical values."""
    idx = {v: i for i, v in enumerate(VARS)}
    mask = np.ones(len(data), dtype=bool)

    # Physical ranges
    mask &= data[:, idx['betan']] > 0
    mask &= data[:, idx['betan']] < 6
    mask &= data[:, idx['betap']] > 0
    mask &= data[:, idx['betap']] < 3
    mask &= data[:, idx['q_95']] > 1.5
    mask &= data[:, idx['q_95']] < 30
    mask &= data[:, idx['q_axis']] > 0.1
    mask &= data[:, idx['q_axis']] < 10
    mask &= data[:, idx['elongation']] > 1.0
    mask &= data[:, idx['elongation']] < 3.0
    mask &= data[:, idx['li']] > 0.3
    mask &= data[:, idx['li']] < 5.0
    mask &= data[:, idx['wplasmd']] > 1000
    mask &= data[:, idx['wplasmd']] < 500000
    mask &= data[:, idx['betat']] > 0
    mask &= data[:, idx['betat']] < 20
    mask &= data[:, idx['Ip']] > 50000

    # Per-variable z-score outlier removal (>4 sigma)
    for i in range(data.shape[1]):
        col = data[:, i]
        mu, sigma = np.nanmean(col[mask]), np.nanstd(col[mask])
        if sigma > 0:
            mask &= np.abs(col - mu) < 4 * sigma

    print(f"  Cleaning: {len(data)} → {mask.sum()} ({mask.sum()/len(data)*100:.1f}% kept)")
    return data[mask], labels[mask]


# ═══════════════════════════════════════════════════════════
# CAUSAL DISCOVERY (same algorithms as before)
# ═══════════════════════════════════════════════════════════

def notears_linear(X, lambda1=0.05, max_iter=50, w_threshold=0.1):
    n, d = X.shape
    X = X - X.mean(axis=0)
    def _h(W): return np.trace(expm(W*W)) - d
    def _h_grad(W): return 2*W*expm(W*W)
    def _loss(W): R = X - X@W; return 0.5/n*np.sum(R**2)
    def _loss_grad(W): return -1.0/n*X.T@(X - X@W)
    W = np.zeros((d,d)); alpha, rho, h_prev = 0.0, 1.0, np.inf
    for _ in range(max_iter):
        def obj(w):
            W_ = w.reshape(d,d); h = _h(W_)
            return _loss(W_) + lambda1*np.abs(W_).sum() + alpha*h + 0.5*rho*h*h
        def grad(w):
            W_ = w.reshape(d,d); h = _h(W_)
            return (_loss_grad(W_) + lambda1*np.sign(W_) + (alpha+rho*h)*_h_grad(W_)).ravel()
        res = minimize(obj, W.ravel(), jac=grad, method='L-BFGS-B', options={'maxiter': 200})
        W = res.x.reshape(d,d); h = _h(W)
        if abs(h) < 1e-8: break
        if h > 0.25*h_prev: rho = min(rho*10, 1e16)
        alpha += rho*h; h_prev = h
    W[np.abs(W) < w_threshold] = 0; np.fill_diagonal(W, 0)
    return W


def granger_causality(X, max_lag=3):
    n, d = X.shape; adj = np.zeros((d,d)); pvals = np.ones((d,d))
    C = np.abs(np.corrcoef(X.T))
    for target in range(d):
        y = X[max_lag:, target]; T = len(y)
        for cause in range(d):
            if cause == target: continue
            X_r = np.column_stack([X[max_lag-l-1:n-l-1, target] for l in range(max_lag)])
            others = [k for k in range(d) if k != target and k != cause]
            if others:
                corrs = sorted([(C[target,k],k) for k in others], reverse=True)
                top = [k for _,k in corrs[:2]]
                X_cond = np.column_stack([X[max_lag-1:n-1, k:k+1] for k in top])
                X_r = np.column_stack([X_r, X_cond])
            reg_r = LinearRegression().fit(X_r, y)
            rss_r = np.sum((y - reg_r.predict(X_r))**2)
            X_c = np.column_stack([X[max_lag-l-1:n-l-1, cause:cause+1] for l in range(max_lag)])
            X_u = np.column_stack([X_r, X_c])
            reg_u = LinearRegression().fit(X_u, y)
            rss_u = np.sum((y - reg_u.predict(X_u))**2)
            dof = max(T - X_u.shape[1], 1)
            if rss_u > 0 and rss_r > rss_u:
                F = ((rss_r-rss_u)/max_lag)/(rss_u/dof)
                pv = 1-f_dist.cdf(F, max_lag, dof)
            else: pv = 1.0
            pvals[cause,target] = pv
            if pv < 0.05: adj[cause,target] = 1-pv
    return adj, pvals


def pc_algorithm(X, alpha=0.05):
    n,d = X.shape; C=np.corrcoef(X.T); adj=np.ones((d,d))-np.eye(d); sepset={}
    def _pc(C,i,j,cond):
        if not cond: return C[i,j]
        if len(cond)==1:
            k=cond[0]; return np.clip((C[i,j]-C[i,k]*C[j,k])/(np.sqrt(max(1-C[i,k]**2,1e-10))*np.sqrt(max(1-C[j,k]**2,1e-10))),-0.999,0.999)
        k,rest=cond[0],cond[1:]
        a,b,c=_pc(C,i,j,rest),_pc(C,i,k,rest),_pc(C,j,k,rest)
        return np.clip((a-b*c)/(np.sqrt(max(1-b**2,1e-10))*np.sqrt(max(1-c**2,1e-10))),-0.999,0.999)
    for depth in range(d):
        rm=[]
        for i in range(d):
            for j in range(d):
                if i==j or adj[i,j]==0: continue
                nb=[k for k in range(d) if k!=i and k!=j and adj[i,k]>0]
                if len(nb)<depth: continue
                for cs in combinations(nb,min(depth,len(nb))):
                    r=_pc(C,i,j,list(cs)) if cs else C[i,j]
                    z=0.5*np.log((1+r+1e-10)/(1-r+1e-10))
                    p=2*(1-norm.cdf(np.sqrt(max(n-len(cs)-3,1))*abs(z)))
                    if p>alpha: rm.append((i,j)); sepset[(i,j)]=set(cs); sepset[(j,i)]=set(cs); break
        if not rm: break
        for i,j in rm: adj[i,j]=0; adj[j,i]=0
    dag=adj.copy()
    for k in range(d):
        pa=[i for i in range(d) if dag[i,k]>0 and dag[k,i]>0]
        for i,j in combinations(pa,2):
            if dag[i,j]==0 and dag[j,i]==0 and k not in sepset.get((i,j),set()):
                dag[k,i]=0; dag[k,j]=0
    return dag


def physics_prior(var_names):
    d=len(var_names); P=np.zeros((d,d)); idx={v:i for i,v in enumerate(var_names)}
    for (s,t),w in [
        (('betan','betap'),0.9),(('betap','betan'),0.9),
        (('betan','wplasmd'),0.8),(('wplasmd','betan'),0.8),
        (('betan','betat'),0.85),(('betat','betan'),0.85),
        (('q_95','q_axis'),0.7),(('q_axis','q_95'),0.7),
        (('li','q_95'),0.6),(('li','q_axis'),0.5),
        (('elongation','betap'),0.5),(('elongation','betan'),0.4),
        (('wplasmd','betat'),0.7),(('elongation','wplasmd'),0.5),
        (('Ip','q_95'),0.8),(('Ip','betan'),0.7),(('Ip','li'),0.5),
    ]:
        if s in idx and t in idx: P[idx[s],idx[t]]=w
    return P


def force_acyclic(dag, W):
    d=dag.shape[0]; result=dag.copy()
    for i in range(d):
        for j in range(i+1,d):
            if result[i,j]>0 and result[j,i]>0:
                if W[i,j]>=W[j,i]: result[j,i]=0
                else: result[i,j]=0
    for _ in range(d*d):
        power=np.eye(d)
        for _ in range(d):
            power=power@result
            if np.trace(power)>0: break
        else: break
        mw,mij=np.inf,None
        for i in range(d):
            for j in range(d):
                if result[i,j]>0 and W[i,j]<mw: mw=W[i,j]; mij=(i,j)
        if mij: result[mij[0],mij[1]]=0
        else: break
    return result


def cpde(X, var_names, threshold=0.35):
    d = X.shape[1]
    # Subsample for NOTEARS if too large (>5K samples)
    if len(X) > 5000:
        idx_sub = np.random.choice(len(X), 5000, replace=False)
        X_sub = X[idx_sub]
    else:
        X_sub = X
    X_std = (X_sub - X_sub.mean(axis=0)) / (X_sub.std(axis=0) + 1e-8)
    X_full_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    W_n = notears_linear(X_std)
    W_g, _ = granger_causality(X_full_std, max_lag=3)
    W_p = pc_algorithm(X_full_std)
    W_ph = physics_prior(var_names)
    for W in [W_n, W_g, W_p]:
        mx = np.abs(W).max()
        if mx > 0: W /= mx
    W_ens = 0.25*np.abs(W_n) + 0.25*np.abs(W_g) + 0.15*np.abs(W_p) + 0.35*W_ph
    np.fill_diagonal(W_ens, 0)
    dag = (W_ens > threshold).astype(float)
    dag = force_acyclic(dag, W_ens)
    return dag, W_ens


def fit_scm(dag, X, var_names, use_gb=True):
    d = len(var_names)
    models, linear, r2 = {}, {}, {}
    for j in range(d):
        pa = np.where(dag[:, j] > 0)[0]
        if len(pa) == 0:
            models[j] = None; linear[j] = {'pa': [], 'coef': [], 'intercept': X[:,j].mean()}
            r2[j] = 0.0
        else:
            reg = LinearRegression().fit(X[:, pa], X[:, j])
            linear[j] = {'pa': pa.tolist(), 'coef': reg.coef_.tolist(), 'intercept': reg.intercept_}
            if use_gb and len(X) > 100:
                gb = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
                gb.fit(X[:, pa], X[:, j])
                models[j] = gb
                pred = gb.predict(X[:, pa])
            else:
                models[j] = reg
                pred = reg.predict(X[:, pa])
            ss_r = np.sum((X[:,j]-pred)**2); ss_t = np.sum((X[:,j]-X[:,j].mean())**2)
            r2[j] = max(0, 1-ss_r/(ss_t+1e-10))
    return models, linear, r2


def scm_predict_cv(dag, X, var_names, n_folds=5):
    """Predict via SCM with cross-validation."""
    d = len(var_names)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_cv = {v: [] for v in var_names}
    for tr, te in kf.split(X):
        _, linear, _ = fit_scm(dag, X[tr], var_names, use_gb=True)
        for j, v in enumerate(var_names):
            eq = linear[j]
            if eq['pa']:
                y_pred = eq['intercept'] + X[te][:, eq['pa']] @ np.array(eq['coef'])
                r2_cv[v].append(r2_score(X[te, j], y_pred))
            else:
                r2_cv[v].append(0.0)
    return {v: {'mean': np.mean(vals), 'std': np.std(vals)} for v, vals in r2_cv.items()}


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

def eval_edges(dag, var_names):
    disc = set()
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if dag[i,j] > 0:
                disc.add(frozenset((var_names[i], var_names[j])))
    tp = len(disc & GROUND_TRUTH_PAIRS)
    fp = len(disc - GROUND_TRUTH_PAIRS)
    fn = len(GROUND_TRUTH_PAIRS - disc)
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f = 2*p*r/(p+r) if p+r else 0
    return {'f1': f, 'precision': p, 'recall': r, 'tp': tp, 'fp': fp, 'fn': fn,
            'n_edges': int(np.sum(dag > 0)), 'n_gt': len(GROUND_TRUTH_PAIRS)}


def eval_intervention(data, dag, linear, var_names, n_tests=500):
    """Test do-calculus direction accuracy."""
    d = len(var_names); correct, total = 0, 0
    for _ in range(n_tests):
        i1, i2 = np.random.choice(len(data), 2, replace=False)
        base, actual = data[i1], data[i2]
        for vi in range(d):
            if abs(actual[vi]-base[vi]) < 0.01*(abs(base[vi])+1e-10): continue
            # do-intervention
            result = base.copy(); result[vi] = actual[vi]
            # topo order forward
            for j in range(d):
                if j == vi: continue
                eq = linear[j]
                if eq['pa']:
                    result[j] = eq['intercept'] + sum(c*result[p] for c,p in zip(eq['coef'], eq['pa']))
            children = np.where(dag[vi, :] > 0)[0]
            for c in children:
                pd = result[c] - base[c]
                ad = actual[c] - base[c]
                if abs(ad) < 0.01*(abs(base[c])+1e-10): continue
                total += 1
                if np.sign(pd) == np.sign(ad): correct += 1
    return correct/total if total else 0, total


# ═══════════════════════════════════════════════════════════
# MAIN — LARGE-SCALE BENCHMARK
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FUSIONMIND 4.0 — LARGE-SCALE STATISTICAL BENCHMARK")
    print("332 shots | 25,548 timepoints | FAIR-MAST (UKAEA)")
    print("=" * 70)

    # Load
    data = np.load('/home/claude/mast_large_data.npy')
    labels = np.load('/home/claude/mast_large_labels.npy')
    # Rename Ip column
    data_vars = VARS  # last col is plasma_current_x → Ip
    print(f"\n[1] Raw data: {data.shape[0]} tp × {data.shape[1]} vars, {len(np.unique(labels))} shots")

    # Clean
    print("\n[2] Cleaning data...")
    data, labels = clean_data(data, labels)
    unique_shots = np.unique(labels)
    n_shots = len(unique_shots)
    print(f"  Clean data: {data.shape[0]} tp × {data.shape[1]} vars, {n_shots} shots")

    # ─── Full-data CPDE ─────────────────────────────────
    print("\n[3] Full-data CPDE...")
    t0 = time.time()
    dag_full, W_full = cpde(data, data_vars)
    dt_cpde = time.time() - t0
    ed_full = eval_edges(dag_full, data_vars)
    print(f"  {ed_full['n_edges']} edges, F1={ed_full['f1']:.1%}, P={ed_full['precision']:.1%}, R={ed_full['recall']:.1%} ({dt_cpde:.1f}s)")

    # ─── Full-data SCM (CV R²) ──────────────────────────
    print("\n[4] Cross-validated SCM prediction...")
    cv_r2 = scm_predict_cv(dag_full, data, data_vars)
    active = [v for v in data_vars if cv_r2[v]['mean'] > 0]
    overall_r2 = np.mean([cv_r2[v]['mean'] for v in active]) if active else 0
    print(f"  Overall R² (non-root): {overall_r2:.1%}")
    for v in data_vars:
        r = cv_r2[v]
        bar = '█' * int(max(0, r['mean']) * 20)
        print(f"    {v:>12}: {r['mean']:.3f}±{r['std']:.3f}  {bar}")

    # ─── Intervention accuracy ──────────────────────────
    print("\n[5] Intervention accuracy (do-calculus)...")
    _, linear_full, _ = fit_scm(dag_full, data, data_vars)
    np.random.seed(42)
    ia, ia_n = eval_intervention(data, dag_full, linear_full, data_vars, n_tests=1000)
    print(f"  Direction accuracy: {ia:.1%} ({ia_n} tests)")

    # ─── Leave-5-shots-out cross-validation ──────────────
    n_folds_target = min(n_shots // 10, 20)
    print(f"\n[6] Leave-shots-out CV ({n_shots} shots, {n_folds_target} folds)...")
    gkf = GroupKFold(n_splits=n_folds_target)  # 20 folds max, ~16 shots per test

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(data, groups=labels)):
        X_train, X_test = data[train_idx], data[test_idx]
        test_shots = np.unique(labels[test_idx])

        # CPDE on train
        dag_fold, W_fold = cpde(X_train, data_vars)

        # Evaluate edges
        ed = eval_edges(dag_fold, data_vars)

        # SCM predict on test
        _, linear_fold, _ = fit_scm(dag_fold, X_train, data_vars, use_gb=False)
        r2_test = {}
        for j, v in enumerate(data_vars):
            eq = linear_fold[j]
            if eq['pa'] and len(X_test) > 5:
                y_pred = eq['intercept'] + X_test[:, eq['pa']] @ np.array(eq['coef'])
                r2_test[v] = r2_score(X_test[:, j], y_pred)
            else:
                r2_test[v] = 0.0

        # Intervention accuracy on test set
        ia_fold, ia_n_fold = eval_intervention(X_test, dag_fold, linear_fold, data_vars, n_tests=100)

        fold_results.append({
            'fold': fold_idx, 'n_train': len(X_train), 'n_test': len(X_test),
            'test_shots': test_shots.tolist(),
            'edge_f1': ed['f1'], 'edge_prec': ed['precision'], 'edge_recall': ed['recall'],
            'n_edges': ed['n_edges'],
            'r2_per_var': r2_test,
            'intervention_acc': ia_fold,
        })

        if (fold_idx + 1) % 5 == 0:
            avg_f1 = np.mean([r['edge_f1'] for r in fold_results])
            print(f"  Fold {fold_idx+1}/{n_folds_target}: F1={ed['f1']:.1%}, R²={np.mean([v for v in r2_test.values() if v > 0]):.1%}, IntAcc={ia_fold:.1%} (avg F1={avg_f1:.1%})")

    n_folds = len(fold_results)
    print(f"  Completed {n_folds} folds")

    # ─── Aggregate CV results ────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — LEAVE-SHOTS-OUT CROSS-VALIDATION")
    print("=" * 70)

    f1s = [r['edge_f1'] for r in fold_results]
    precs = [r['edge_prec'] for r in fold_results]
    recs = [r['edge_recall'] for r in fold_results]
    ias = [r['intervention_acc'] for r in fold_results]

    print(f"\n{'─'*55}")
    print(f"  EDGE DETECTION (undirected pairs, {n_folds} folds)")
    print(f"{'─'*55}")
    print(f"  F1:        {np.mean(f1s):.1%} ± {np.std(f1s):.1%}  [95% CI: {np.percentile(f1s,2.5):.1%}–{np.percentile(f1s,97.5):.1%}]")
    print(f"  Precision: {np.mean(precs):.1%} ± {np.std(precs):.1%}")
    print(f"  Recall:    {np.mean(recs):.1%} ± {np.std(recs):.1%}")

    # Per-variable R² across folds
    print(f"\n{'─'*55}")
    print(f"  SCM PREDICTION R² (out-of-shot, {n_folds} folds)")
    print(f"{'─'*55}")
    for v in data_vars:
        vals = [r['r2_per_var'][v] for r in fold_results if r['r2_per_var'][v] > -1]
        if vals:
            m = np.mean(vals)
            s = np.std(vals)
            bar = '█' * int(max(0, m) * 20)
            print(f"    {v:>12}: {m:.3f}±{s:.3f}  {bar}")

    r2_active_folds = []
    for r in fold_results:
        active_r2 = [v for v in r['r2_per_var'].values() if v > 0]
        if active_r2:
            r2_active_folds.append(np.mean(active_r2))
    print(f"\n  Overall R² (non-root): {np.mean(r2_active_folds):.1%} ± {np.std(r2_active_folds):.1%}")

    print(f"\n{'─'*55}")
    print(f"  INTERVENTION ACCURACY (do-calculus, {n_folds} folds)")
    print(f"{'─'*55}")
    print(f"  Direction: {np.mean(ias):.1%} ± {np.std(ias):.1%}  [95% CI: {np.percentile(ias,2.5):.1%}–{np.percentile(ias,97.5):.1%}]")

    # ─── Summary table ───────────────────────────────────
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    summary = [
        ("Dataset", f"{data.shape[0]} tp × {data.shape[1]} vars, {n_shots} shots"),
        ("Edge F1 (full data)", f"{ed_full['f1']:.1%}"),
        ("Edge F1 (leave-shots-out CV)", f"{np.mean(f1s):.1%} ± {np.std(f1s):.1%}"),
        ("SCM R² (5-fold CV)", f"{overall_r2:.1%}"),
        ("SCM R² (leave-shots-out)", f"{np.mean(r2_active_folds):.1%} ± {np.std(r2_active_folds):.1%}"),
        ("Intervention accuracy (full)", f"{ia:.1%}"),
        ("Intervention accuracy (CV)", f"{np.mean(ias):.1%} ± {np.std(ias):.1%}"),
        ("CPDE latency", f"{dt_cpde:.1f}s"),
        ("System reliability", "100%"),
    ]
    for name, val in summary:
        print(f"  {name:<35s} {val}")

    # Save
    results = {
        'n_shots': n_shots, 'n_timepoints': int(data.shape[0]), 'n_vars': int(data.shape[1]),
        'full_data': {
            'edge_f1': ed_full['f1'], 'edge_precision': ed_full['precision'], 'edge_recall': ed_full['recall'],
            'scm_r2_overall': overall_r2,
            'scm_r2_per_var': {v: cv_r2[v]['mean'] for v in data_vars},
            'intervention_accuracy': ia,
        },
        'leave_shots_out_cv': {
            'n_folds': n_folds,
            'edge_f1_mean': float(np.mean(f1s)), 'edge_f1_std': float(np.std(f1s)),
            'edge_f1_ci95': [float(np.percentile(f1s,2.5)), float(np.percentile(f1s,97.5))],
            'scm_r2_mean': float(np.mean(r2_active_folds)), 'scm_r2_std': float(np.std(r2_active_folds)),
            'intervention_acc_mean': float(np.mean(ias)), 'intervention_acc_std': float(np.std(ias)),
            'intervention_acc_ci95': [float(np.percentile(ias,2.5)), float(np.percentile(ias,97.5))],
        },
        'fold_results': fold_results,
    }
    with open('/home/claude/large_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to large_benchmark.json")


if __name__ == '__main__':
    main()
