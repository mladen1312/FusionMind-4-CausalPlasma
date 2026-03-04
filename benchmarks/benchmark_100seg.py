#!/usr/bin/env python3
"""
FusionMind 4.0 — Statistical Benchmark on 100 Time Segments
Real FAIR-MAST Tokamak Data from UKAEA S3

Measures: CPDE accuracy, stability, timing, SCM quality across 100 segments.
"""

import numpy as np
import time
import json
import warnings
from scipy.optimize import minimize
from scipy.stats import f as f_dist, norm
from scipy.linalg import expm
from sklearn.linear_model import LinearRegression
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: FAIR-MAST DATA DOWNLOAD
# ============================================================================

MAST_SHOTS = [
    # Original 15
    30420, 30421, 30422, 30423, 30424, 30425, 30426, 30427, 30428,
    30430, 30439, 30440, 30441, 30443, 30444,
    # Extended range
    30400, 30404, 30405, 30406, 30407, 30409, 30410, 30411, 30412, 30413,
    30416, 30417, 30418, 30419, 30445, 30448, 30449, 30450,
    # 27xxx era (longer discharges)
    27000, 27001, 27002, 27003, 27004, 27005, 27010, 27011, 27012, 27013, 27014,
]

PLASMA_VARS = ['betan', 'betap', 'q_95', 'q_axis', 'elongation', 'li', 'wplasmd', 'betat']

# Display names for reporting
VAR_DISPLAY = {'betan': 'βN', 'betap': 'βp', 'q_95': 'q95', 'q_axis': 'q_axis',
               'elongation': 'κ', 'li': 'li', 'wplasmd': 'Wstored', 'betat': 'βt'}

EXPECTED_EDGES = {
    ('betan', 'betap'), ('betap', 'betan'),
    ('betan', 'wplasmd'), ('wplasmd', 'betan'),
    ('betan', 'betat'), ('betat', 'betan'),
    ('betap', 'wplasmd'), ('wplasmd', 'betap'),
    ('q_95', 'q_axis'), ('q_axis', 'q_95'),
    ('elongation', 'betap'), ('betap', 'elongation'),
    ('li', 'q_95'), ('q_95', 'li'),
    ('li', 'q_axis'), ('q_axis', 'li'),
    ('elongation', 'betan'), ('betan', 'elongation'),
    ('wplasmd', 'betat'), ('betat', 'wplasmd'),
}

# For F1 calculation: accept EITHER direction as a true positive
# (DAG can only have one direction per pair)
EXPECTED_PAIRS = {
    frozenset(('betan', 'betap')),
    frozenset(('betan', 'wplasmd')),
    frozenset(('betan', 'betat')),
    frozenset(('betap', 'wplasmd')),
    frozenset(('q_95', 'q_axis')),
    frozenset(('elongation', 'betap')),
    frozenset(('li', 'q_95')),
    frozenset(('li', 'q_axis')),
    frozenset(('elongation', 'betan')),
    frozenset(('wplasmd', 'betat')),
    frozenset(('li', 'betat')),       # li affects stability → energy
    frozenset(('elongation', 'wplasmd')),  # shape affects stored energy
}


def download_mast_data(shots=None):
    """Download and assemble multi-shot MAST data."""
    import s3fs
    import zarr

    if shots is None:
        shots = MAST_SHOTS

    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={
            'endpoint_url': 'https://s3.echo.stfc.ac.uk',
            'region_name': 'us-east-1',
        }
    )

    all_data = []
    shot_labels = []

    for sid in shots:
        try:
            path = f'mast/level1/shots/{sid}.zarr'
            store = s3fs.S3Map(root=path, s3=fs, check=False)
            root = zarr.open(store, mode='r')

            if 'efm' not in root:
                continue

            efm = root['efm']
            times = np.array(efm['all_times'])

            mask = times > 0.01
            if mask.sum() < 10:
                continue

            shot_data = {}
            ok = True
            for var in PLASMA_VARS:
                if var not in efm:
                    ok = False
                    break
                arr = np.array(efm[var])[mask]
                if np.all(np.isnan(arr)) or len(arr) < 10:
                    ok = False
                    break
                shot_data[var] = arr

            if not ok:
                continue

            matrix = np.column_stack([shot_data[v] for v in PLASMA_VARS])
            valid = ~np.any(np.isnan(matrix) | np.isinf(matrix), axis=1)
            matrix = matrix[valid]

            if len(matrix) >= 10:
                all_data.append(matrix)
                shot_labels.extend([sid] * len(matrix))
                print(f"  Shot {sid}: {len(matrix)} valid timepoints")

        except Exception as e:
            print(f"  Shot {sid}: FAIL ({str(e)[:60]})")

    if not all_data:
        raise RuntimeError("No MAST data downloaded")

    combined = np.vstack(all_data)
    print(f"\n  TOTAL: {combined.shape[0]} timepoints x {combined.shape[1]} variables "
          f"from {len(all_data)} shots")

    return combined, np.array(shot_labels), PLASMA_VARS


# ============================================================================
# PART 2: CAUSAL DISCOVERY ALGORITHMS
# ============================================================================

def notears_linear(X, lambda1=0.1, max_iter=100, h_tol=1e-8, w_threshold=0.3):
    """
    Two-phase NOTEARS: 
    Phase 1: L1-regularized regression (Lasso) for each variable
    Phase 2: Break cycles by removing weakest edges
    This avoids the augmented Lagrangian over-penalization issue.
    """
    from sklearn.linear_model import Lasso
    n, d = X.shape
    X = X - X.mean(axis=0)

    # Phase 1: Lasso regression for each target variable
    W = np.zeros((d, d))
    for j in range(d):
        mask_j = np.ones(d, dtype=bool)
        mask_j[j] = False
        lasso = Lasso(alpha=lambda1 / (2 * n), max_iter=10000, tol=1e-5)
        lasso.fit(X[:, mask_j], X[:, j])
        coefs = np.zeros(d)
        coefs[mask_j] = lasso.coef_
        W[:, j] = coefs

    # Threshold small weights
    W[np.abs(W) < w_threshold] = 0
    np.fill_diagonal(W, 0)

    # Phase 2: Break cycles by removing weakest edges
    W = _force_dag(W)

    return W


def _force_dag(W):
    """Remove weakest edges to make W a DAG."""
    d = W.shape[0]
    result = W.copy()
    for _ in range(d * d):
        # Check acyclicity
        M = (np.abs(result) > 0).astype(float)
        power = np.eye(d)
        has_cycle = False
        for _ in range(d):
            power = power @ M
            if np.trace(power) > 0:
                has_cycle = True
                break
        if not has_cycle:
            break
        # Remove weakest edge
        min_w, min_ij = np.inf, None
        for i in range(d):
            for j in range(d):
                if abs(result[i, j]) > 0 and abs(result[i, j]) < min_w:
                    min_w = abs(result[i, j])
                    min_ij = (i, j)
        if min_ij:
            result[min_ij[0], min_ij[1]] = 0
        else:
            break
    return result


def granger_causality(X, max_lag=3):
    """Pairwise Granger causality with F-test p-values."""
    n, d = X.shape
    adj = np.zeros((d, d))
    pvals = np.ones((d, d))

    for target in range(d):
        y = X[max_lag:, target]
        T = len(y)
        for cause in range(d):
            if cause == target:
                continue
            X_r = np.column_stack([X[max_lag-l-1:n-l-1, target] for l in range(max_lag)])
            reg_r = LinearRegression().fit(X_r, y)
            rss_r = np.sum((y - reg_r.predict(X_r)) ** 2)

            X_u = np.column_stack([X_r] + [X[max_lag-l-1:n-l-1, cause:cause+1] for l in range(max_lag)])
            reg_u = LinearRegression().fit(X_u, y)
            rss_u = np.sum((y - reg_u.predict(X_u)) ** 2)

            p = max_lag
            if rss_u > 0 and rss_r > rss_u:
                F_stat = ((rss_r - rss_u) / p) / (rss_u / max(T - 2 * max_lag, 1))
                p_value = 1.0 - f_dist.cdf(F_stat, p, max(T - 2 * max_lag, 1))
            else:
                p_value = 1.0

            pvals[cause, target] = p_value
            if p_value < 0.05:
                adj[cause, target] = 1.0 - p_value

    return adj, pvals


def _partial_corr(C, i, j, cond):
    if len(cond) == 0:
        return C[i, j]
    if len(cond) == 1:
        k = cond[0]
        num = C[i, j] - C[i, k] * C[j, k]
        den = np.sqrt(max(1 - C[i, k]**2, 1e-10)) * np.sqrt(max(1 - C[j, k]**2, 1e-10))
        return np.clip(num / den, -0.999, 0.999)
    k = cond[0]
    rest = cond[1:]
    r_ij = _partial_corr(C, i, j, rest)
    r_ik = _partial_corr(C, i, k, rest)
    r_jk = _partial_corr(C, j, k, rest)
    num = r_ij - r_ik * r_jk
    den = np.sqrt(max(1 - r_ik**2, 1e-10)) * np.sqrt(max(1 - r_jk**2, 1e-10))
    return np.clip(num / den, -0.999, 0.999)


def pc_algorithm(X, alpha=0.05):
    """PC algorithm with Fisher-z test and Meek's rules."""
    n, d = X.shape
    C = np.corrcoef(X.T)
    adj = np.ones((d, d)) - np.eye(d)
    sepset = {}

    for depth in range(d):
        removals = []
        for i in range(d):
            for j in range(d):
                if i == j or adj[i, j] == 0:
                    continue
                neighbors = [k for k in range(d) if k != i and k != j and adj[i, k] > 0]
                if len(neighbors) < depth:
                    continue
                for cond_set in combinations(neighbors, min(depth, len(neighbors))):
                    r = _partial_corr(C, i, j, list(cond_set)) if cond_set else C[i, j]
                    z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
                    z_stat = np.sqrt(max(n - len(cond_set) - 3, 1)) * abs(z)
                    p_value = 2 * (1 - norm.cdf(z_stat))
                    if p_value > alpha:
                        removals.append((i, j))
                        sepset[(i, j)] = set(cond_set)
                        sepset[(j, i)] = set(cond_set)
                        break
        if not removals:
            break
        for i, j in removals:
            adj[i, j] = 0
            adj[j, i] = 0

    # V-structures
    dag = adj.copy()
    for k in range(d):
        parents = [i for i in range(d) if dag[i, k] > 0 and dag[k, i] > 0]
        for i, j in combinations(parents, 2):
            if dag[i, j] == 0 and dag[j, i] == 0:
                sep = sepset.get((i, j), set())
                if k not in sep:
                    dag[k, i] = 0
                    dag[k, j] = 0

    # Meek R1-R2
    changed = True
    while changed:
        changed = False
        for i in range(d):
            for j in range(d):
                if dag[i, j] > 0 and dag[j, i] > 0:
                    for k in range(d):
                        if k != i and k != j:
                            if dag[i, k] > 0 and dag[k, i] == 0 and dag[k, j] > 0 and dag[j, k] == 0:
                                dag[j, i] = 0
                                changed = True
    return dag


# ============================================================================
# PART 3: ENSEMBLE CPDE + SCM
# ============================================================================

def physics_prior_matrix(var_names):
    d = len(var_names)
    prior = np.zeros((d, d))
    idx = {v: i for i, v in enumerate(var_names)}
    edges = [
        ('betan', 'betap', 0.9), ('betap', 'betan', 0.9),
        ('betan', 'wplasmd', 0.8), ('wplasmd', 'betan', 0.8),
        ('betan', 'betat', 0.85), ('betat', 'betan', 0.85),
        ('q_95', 'q_axis', 0.7), ('q_axis', 'q_95', 0.7),
        ('li', 'q_95', 0.6), ('li', 'q_axis', 0.5),
        ('elongation', 'betap', 0.5), ('elongation', 'betan', 0.4),
        ('wplasmd', 'betat', 0.7),
    ]
    for src, tgt, w in edges:
        if src in idx and tgt in idx:
            prior[idx[src], idx[tgt]] = w
    return prior


def _force_acyclic(dag, weights):
    d = dag.shape[0]
    result = dag.copy()
    for _ in range(100):
        power = np.eye(d)
        cycle = False
        for _ in range(d):
            power = power @ result
            if np.trace(power) > 0:
                cycle = True
                break
        if not cycle:
            break
        min_w, min_ij = np.inf, None
        for i in range(d):
            for j in range(d):
                if result[i, j] > 0 and weights[i, j] < min_w:
                    min_w = weights[i, j]
                    min_ij = (i, j)
        if min_ij:
            result[min_ij[0], min_ij[1]] = 0
        else:
            break
    return result


def ensemble_cpde(X, var_names, threshold=0.2):
    """Ensemble CPDE: NOTEARS + Granger + PC + physics priors."""
    d = X.shape[1]
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    W_notears = notears_linear(X_std, lambda1=0.1, w_threshold=0.15)
    W_granger, pvals = granger_causality(X_std, max_lag=2)
    W_pc = pc_algorithm(X_std, alpha=0.05)
    W_physics = physics_prior_matrix(var_names)

    for W in [W_notears, W_granger, W_pc]:
        mx = np.abs(W).max()
        if mx > 0:
            W /= mx

    W_ensemble = (0.35 * np.abs(W_notears) + 0.25 * np.abs(W_granger) +
                  0.20 * np.abs(W_pc) + 0.20 * W_physics)
    np.fill_diagonal(W_ensemble, 0)

    dag = (W_ensemble > threshold).astype(float)
    dag = _force_dag(dag)

    edges = []
    for i in range(d):
        for j in range(d):
            if dag[i, j] > 0:
                edges.append({
                    'source': var_names[i], 'target': var_names[j],
                    'weight': float(W_ensemble[i, j]),
                    'notears': float(abs(W_notears[i, j])),
                    'granger': float(abs(W_granger[i, j])),
                    'pc': float(abs(W_pc[i, j])),
                    'physics': float(W_physics[i, j]),
                    'granger_pval': float(pvals[i, j]),
                })

    return dag, edges, {'W_notears': W_notears, 'W_granger': W_granger,
                        'W_pc': W_pc, 'W_ensemble': W_ensemble}


class PlasmaSCM:
    def __init__(self, dag, var_names):
        self.dag, self.var_names = dag, var_names
        self.d = len(var_names)
        self.equations, self.residuals, self.r2_scores = {}, {}, {}

    def fit(self, X):
        for j in range(self.d):
            parents = np.where(self.dag[:, j] > 0)[0]
            if len(parents) == 0:
                self.equations[j] = {'parents': [], 'coefs': [], 'intercept': X[:, j].mean()}
                self.residuals[j] = X[:, j] - X[:, j].mean()
                self.r2_scores[j] = 0.0
            else:
                reg = LinearRegression().fit(X[:, parents], X[:, j])
                pred = reg.predict(X[:, parents])
                self.equations[j] = {'parents': parents.tolist(), 'coefs': reg.coef_.tolist(),
                                     'intercept': reg.intercept_}
                self.residuals[j] = X[:, j] - pred
                ss_res = np.sum((X[:, j] - pred) ** 2)
                ss_tot = np.sum((X[:, j] - X[:, j].mean()) ** 2)
                self.r2_scores[j] = max(0, 1 - ss_res / (ss_tot + 1e-10))

    def _topo_order(self):
        visited, order = set(), []
        def dfs(n):
            if n in visited: return
            visited.add(n)
            for p in range(self.d):
                if self.dag[p, n] > 0: dfs(p)
            order.append(n)
        for i in range(self.d): dfs(i)
        return order

    def do(self, interventions, baseline):
        result = baseline.copy()
        idx = {v: i for i, v in enumerate(self.var_names)}
        for var, val in interventions.items():
            if var in idx: result[idx[var]] = val
        for j in self._topo_order():
            if self.var_names[j] in interventions: continue
            eq = self.equations[j]
            if eq['parents']:
                result[j] = eq['intercept'] + sum(c * result[p] for c, p in zip(eq['coefs'], eq['parents']))
        return result

    def counterfactual(self, factual, interventions):
        noise = {}
        for j in range(self.d):
            eq = self.equations[j]
            if eq['parents']:
                pred = eq['intercept'] + sum(c * factual[p] for c, p in zip(eq['coefs'], eq['parents']))
                noise[j] = factual[j] - pred
            else:
                noise[j] = factual[j] - eq['intercept']
        result = factual.copy()
        idx = {v: i for i, v in enumerate(self.var_names)}
        for var, val in interventions.items():
            if var in idx: result[idx[var]] = val
        for j in self._topo_order():
            if self.var_names[j] in interventions: continue
            eq = self.equations[j]
            if eq['parents']:
                result[j] = eq['intercept'] + sum(c * result[p] for c, p in zip(eq['coefs'], eq['parents'])) + noise[j]
            else:
                result[j] = eq['intercept'] + noise[j]
        return result


# ============================================================================
# PART 4: BENCHMARK ENGINE
# ============================================================================

@dataclass
class SegmentResult:
    segment_id: int
    shot_ids: list
    n_timepoints: int
    n_edges: int
    edge_set: set
    edges_detail: list
    dag: object
    timing_s: float
    scm_r2_mean: float
    scm_r2_per_var: dict
    do_ok: bool
    cf_ok: bool
    acyclic: bool


def create_segments(data, shot_labels, n_segments=100, window_size=500):
    """Create segments using overlapping windows on pooled data."""
    segments = []
    n_total = len(data)

    if n_total < window_size:
        raise ValueError(f"Not enough data: {n_total} < window_size={window_size}")

    # Calculate stride for desired number of segments
    max_start = n_total - window_size
    stride = max(1, max_start // (n_segments - 1)) if n_segments > 1 else max_start

    starts = np.linspace(0, max_start, n_segments, dtype=int)

    for i, start in enumerate(starts):
        end = start + window_size
        chunk = data[start:end]
        chunk_shots = list(np.unique(shot_labels[start:end]))
        segments.append({
            'data': chunk,
            'shots': chunk_shots,
            'n_tp': window_size,
        })

    print(f"  Created {len(segments)} segments (window={window_size}, "
          f"stride~{stride}, total_data={n_total})")
    return segments


def is_acyclic(dag):
    d = dag.shape[0]
    M = (dag > 0).astype(float)
    power = np.eye(d)
    for _ in range(d):
        power = power @ M
        if np.trace(power) > 0:
            return False
    return True


def run_benchmark(data, shot_labels, var_names, n_segments=100):
    print("\n" + "=" * 70)
    print("FUSIONMIND 4.0 — STATISTICAL BENCHMARK (100 SEGMENTS)")
    print("=" * 70)

    segments = create_segments(data, shot_labels, n_segments=n_segments)
    results = []

    for i, seg in enumerate(segments):
        X = seg['data']
        print(f"\r  Segment {i+1:3d}/{len(segments)} (n={seg['n_tp']}, shot={seg['shots'][0]})", end="", flush=True)

        t0 = time.time()
        try:
            dag, edges, details = ensemble_cpde(X, var_names)
            dt = time.time() - t0

            scm = PlasmaSCM(dag, var_names)
            scm.fit(X)

            baseline = X.mean(axis=0)
            try:
                dr = scm.do({'betan': baseline[0] * 1.5}, baseline)
                do_ok = not np.any(np.isnan(dr)) and not np.any(np.isinf(dr))
            except:
                do_ok = False
            try:
                cr = scm.counterfactual(X[0], {'betan': X[0, 0] * 0.8})
                cf_ok = not np.any(np.isnan(cr)) and not np.any(np.isinf(cr))
            except:
                cf_ok = False

            edge_set = {(e['source'], e['target']) for e in edges}
            r2_per = {var_names[j]: scm.r2_scores.get(j, 0) for j in range(len(var_names))}

            results.append(SegmentResult(
                segment_id=i, shot_ids=seg['shots'], n_timepoints=seg['n_tp'],
                n_edges=len(edges), edge_set=edge_set, edges_detail=edges, dag=dag,
                timing_s=dt, scm_r2_mean=np.mean(list(scm.r2_scores.values())),
                scm_r2_per_var=r2_per, do_ok=do_ok, cf_ok=cf_ok, acyclic=is_acyclic(dag),
            ))
        except Exception as e:
            results.append(SegmentResult(
                segment_id=i, shot_ids=seg['shots'], n_timepoints=seg['n_tp'],
                n_edges=0, edge_set=set(), edges_detail=[], dag=np.zeros((len(var_names), len(var_names))),
                timing_s=time.time() - t0, scm_r2_mean=0, scm_r2_per_var={},
                do_ok=False, cf_ok=False, acyclic=True,
            ))

    print()
    return results


# ============================================================================
# PART 5: STATISTICAL ANALYSIS & REPORT
# ============================================================================

def compute_and_print_report(results, var_names):
    n = len(results)
    n_success = sum(1 for r in results if r.n_edges > 0)
    n_acyclic = sum(1 for r in results if r.acyclic)
    n_do_ok = sum(1 for r in results if r.do_ok)
    n_cf_ok = sum(1 for r in results if r.cf_ok)

    f1s, precs, recs = [], [], []
    for r in results:
        disc_pairs = {frozenset((e['source'], e['target'])) for e in r.edges_detail}
        disc_directed = {(e['source'], e['target']) for e in r.edges_detail}
        # TP: discovered pair matches expected pair (either direction)
        tp = len(disc_pairs & EXPECTED_PAIRS)
        fp = len(disc_pairs - EXPECTED_PAIRS)
        fn = len(EXPECTED_PAIRS - disc_pairs)
        p = tp / (tp + fp) if tp + fp > 0 else 0
        rc = tp / (tp + fn) if tp + fn > 0 else 0
        f = 2 * p * rc / (p + rc) if p + rc > 0 else 0
        f1s.append(f); precs.append(p); recs.append(rc)

    timings = [r.timing_s for r in results if r.n_edges > 0]
    r2s = [r.scm_r2_mean for r in results if r.scm_r2_mean > 0]

    edge_freq = {}
    for r in results:
        for e in r.edge_set:
            edge_freq[e] = edge_freq.get(e, 0) + 1
    stable = sorted(edge_freq.items(), key=lambda x: -x[1])

    shds = []
    for i in range(1, len(results)):
        if results[i].n_edges > 0 and results[i-1].n_edges > 0:
            shds.append(np.sum(np.abs(results[i].dag - results[i-1].dag)))

    r2_per_var = {v: [] for v in var_names}
    for r in results:
        for v in var_names:
            if v in r.scm_r2_per_var:
                r2_per_var[v].append(r.scm_r2_per_var[v])

    # Edge contribution analysis per algorithm
    algo_contrib = {'notears': [], 'granger': [], 'pc': [], 'physics': []}
    for r in results:
        for e in r.edges_detail:
            for algo in algo_contrib:
                algo_contrib[algo].append(e.get(algo, 0))

    print("\n" + "=" * 70)
    print("BENCHMARK IZVJESTAJ — FUSIONMIND 4.0 CPDE")
    print("Pravi FAIR-MAST tokamak podaci | 100 vremenskih segmenata")
    print("=" * 70)

    print(f"\n{'─'*55}")
    print(f"  {'SUSTAV':^50}")
    print(f"{'─'*55}")
    print(f"  Segmenata:        {n}")
    print(f"  Uspjesnih:        {n_success} ({n_success/n*100:.1f}%)")
    print(f"  Aciklicni DAG:    {n_acyclic} ({n_acyclic/n*100:.1f}%)")
    print(f"  do() OK:          {n_do_ok} ({n_do_ok/n*100:.1f}%)")
    print(f"  Counterfactual:   {n_cf_ok} ({n_cf_ok/n*100:.1f}%)")

    print(f"\n{'─'*55}")
    print(f"  {'KAUZALNA DETEKCIJA':^50}")
    print(f"{'─'*55}")
    print(f"  F1 score:         {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"  F1 median:        {np.median(f1s):.3f}")
    print(f"  F1 raspon:        [{np.min(f1s):.3f}, {np.max(f1s):.3f}]")
    print(f"  Precision:        {np.mean(precs):.3f} +/- {np.std(precs):.3f}")
    print(f"  Recall:           {np.mean(recs):.3f} +/- {np.std(recs):.3f}")
    ec = [r.n_edges for r in results]
    print(f"  Bridova/segment:  {np.mean(ec):.1f} +/- {np.std(ec):.1f}")

    print(f"\n{'─'*55}")
    print(f"  {'DOPRINOS ALGORITAMA (srednji udio po bridu)':^50}")
    print(f"{'─'*55}")
    for algo in ['notears', 'granger', 'pc', 'physics']:
        vals = algo_contrib[algo]
        if vals:
            print(f"  {algo:>12}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    print(f"\n{'─'*55}")
    print(f"  {'STABILNOST':^50}")
    print(f"{'─'*55}")
    if shds:
        print(f"  SHD (konsekutivni): {np.mean(shds):.2f} +/- {np.std(shds):.2f}")
    print(f"\n  Top stabilni bridovi (N/{n} segmenata):")
    for (src, tgt), count in stable[:15]:
        pct = count / n * 100
        bar = '█' * int(pct / 5)
        exp = '✓' if frozenset((src, tgt)) in EXPECTED_PAIRS else '○'
        print(f"    {exp} {src:>12} -> {tgt:<12}  {count:3d} ({pct:5.1f}%) {bar}")

    print(f"\n{'─'*55}")
    print(f"  {'SCM KVALITETA':^50}")
    print(f"{'─'*55}")
    if r2s:
        print(f"  Srednji R2:       {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")
    print(f"\n  R2 po varijabli:")
    for var, vals in sorted(r2_per_var.items(), key=lambda x: -np.mean(x[1]) if x[1] else 0):
        if vals:
            bar = '█' * int(np.mean(vals) * 20)
            print(f"    {var:>12}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}  {bar}")

    print(f"\n{'─'*55}")
    print(f"  {'PERFORMANSE':^50}")
    print(f"{'─'*55}")
    if timings:
        print(f"  Srednje vrijeme:  {np.mean(timings):.2f}s +/- {np.std(timings):.2f}s")
        print(f"  95. percentil:    {np.percentile(timings, 95):.2f}s")
        print(f"  Maksimum:         {np.max(timings):.2f}s")
        print(f"  Ukupno:           {sum(timings):.1f}s")

    print(f"\n{'─'*55}")
    print(f"  {'95% CONFIDENCE INTERVALS':^50}")
    print(f"{'─'*55}")
    for name, vals in [('F1', f1s), ('Precision', precs), ('Recall', recs)]:
        m = np.mean(vals)
        se = np.std(vals) / np.sqrt(n)
        print(f"  {name:>12}: {m:.3f}  [{m-1.96*se:.3f}, {m+1.96*se:.3f}]")

    print("\n" + "=" * 70)

    # Save JSON
    save = {
        'n_segments': n, 'success_rate': n_success/n, 'acyclicity_rate': n_acyclic/n,
        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
        'f1_median': float(np.median(f1s)),
        'precision_mean': float(np.mean(precs)), 'precision_std': float(np.std(precs)),
        'recall_mean': float(np.mean(recs)), 'recall_std': float(np.std(recs)),
        'scm_r2_mean': float(np.mean(r2s)) if r2s else 0,
        'timing_mean': float(np.mean(timings)) if timings else 0,
        'timing_p95': float(np.percentile(timings, 95)) if timings else 0,
        'do_rate': n_do_ok/n, 'cf_rate': n_cf_ok/n,
        'shd_mean': float(np.mean(shds)) if shds else 0,
        'stable_edges': [(f"{s}->{t}", c) for (s, t), c in stable[:20]],
        'algo_contributions': {a: float(np.mean(v)) if v else 0 for a, v in algo_contrib.items()},
    }
    with open('/home/claude/benchmark_results.json', 'w') as f:
        json.dump(save, f, indent=2)

    return save


def main():
    print("=" * 70)
    print("FUSIONMIND 4.0 — 100-SEGMENT BENCHMARK NA PRAVIM MAST PODACIMA")
    print("=" * 70)

    print("\n[1/3] DOWNLOAD FAIR-MAST PODATAKA S UKAEA S3...")
    data, shot_labels, var_names = download_mast_data()

    print("\n[2/3] POKRETANJE CPDE NA 100 SEGMENATA...")
    results = run_benchmark(data, shot_labels, var_names, n_segments=100)

    print("\n[3/3] STATISTICKA ANALIZA...")
    stats = compute_and_print_report(results, var_names)

    return stats


if __name__ == '__main__':
    main()
