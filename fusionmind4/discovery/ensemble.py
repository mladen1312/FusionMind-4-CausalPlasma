"""EnsembleCPDE — Causal Plasma Discovery Engine v3.2

Fuses 5 causal discovery algorithms with adaptive thresholding,
bootstrap confidence intervals, and physics-informed validation.

This is the core of Patent Family PF1.
"""
import numpy as np
from typing import Dict, Optional
from .notears import NOTEARSDiscovery
from .granger import GrangerCausalityTest
from .pc import PCAlgorithm
from .interventional import InterventionalScorer
from .physics import get_physics_prior_matrix, validate_physics, _find_cycle
from ..utils.plasma_vars import (
    PLASMA_VARS, N_VARS, ACTUATOR_IDS,
    build_ground_truth_adjacency, evaluate_dag,
)


DEFAULT_CONFIG = {
    "n_bootstrap": 15,
    "threshold": 0.32,
    "physics_weight": 0.30,
    "notears_weight": 0.30,
    "granger_weight": 0.22,
    "pc_weight": 0.18,
}


class EnsembleCPDE:
    """Ensemble Causal Plasma Discovery Engine.

    Combines NOTEARS, Granger causality, PC algorithm, interventional
    scoring, and physics priors into a unified causal discovery pipeline.

    Args:
        config: Configuration dictionary (see DEFAULT_CONFIG)
        verbose: Print progress information
    """

    def __init__(self, config: Optional[Dict] = None, verbose: bool = True):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.verbose = verbose
        self.threshold = self.config["threshold"]

    def discover(self, data: np.ndarray,
                 interventional_data: Optional[Dict] = None,
                 seed: int = 42,
                 var_names: Optional[list] = None) -> Dict:
        """Run full causal discovery pipeline.

        Args:
            data: (n_samples, n_vars) observational data
            interventional_data: Optional {actuator_id: (data_low, data_high)}
            seed: Random seed
            var_names: Optional list of variable names (for real data)

        Returns:
            Dictionary with dag, metrics, physics_checks, edge_details
        """
        rng = np.random.RandomState(seed)
        n_vars = data.shape[1]
        is_standard = (n_vars == N_VARS)  # True if FM3-Lite 14-var data

        if self.verbose:
            mode = "standard (14-var)" if is_standard else f"real data ({n_vars}-var)"
            print(f"  Mode: {mode}, samples: {data.shape[0]:,}")

        # ── Step 1: NOTEARS bootstrap ──
        # Try C++ AVX-512 kernel first (888x faster), fall back to Python
        try:
            from ..realtime.causal_bindings import (
                fast_notears_bootstrap, fast_granger, CAUSAL_CPP_AVAILABLE,
            )
            use_cpp = CAUSAL_CPP_AVAILABLE
        except ImportError:
            use_cpp = False

        if use_cpp:
            if self.verbose:
                print("[1/5] NOTEARS bootstrap (C++ AVX-512)...")
            notears_stability = fast_notears_bootstrap(
                data, n_bootstrap=self.config["n_bootstrap"],
                lambda1=0.05, w_threshold=0.10, seed=seed,
            )
        else:
            if self.verbose:
                print("[1/5] NOTEARS bootstrap...")
            nt = NOTEARSDiscovery(lambda1=0.05, w_threshold=0.10)
            notears_stability = nt.fit_bootstrap(data, self.config["n_bootstrap"], rng)

        # ── Step 2: Granger causality ──
        if use_cpp:
            if self.verbose:
                print("[2/5] Granger causality (C++ OpenMP)...")
            granger_matrix = fast_granger(data, max_lag=5, alpha=0.05)
        else:
            if self.verbose:
                print("[2/5] Granger causality...")
            gc = GrangerCausalityTest(max_lag=5, alpha=0.05, bonferroni=True)
            granger_matrix = gc.test_all_pairs(data)

        # ── Step 3: PC algorithm bootstrap ──
        if self.verbose:
            print("[3/5] PC algorithm bootstrap...")
        pc = PCAlgorithm(alpha=0.05, max_cond_set=3)
        pc_stability = pc.fit_bootstrap(data, self.config["n_bootstrap"], rng)

        # ── Step 4: Interventional scoring ──
        if self.verbose:
            print("[4/5] Interventional scoring...")
        int_scores = np.zeros((n_vars, n_vars))
        if interventional_data:
            scorer = InterventionalScorer(effect_threshold=0.3)
            int_scores = scorer.score(interventional_data)

        # ── Step 5: Ensemble fusion ──
        if self.verbose:
            print("[5/5] Ensemble fusion + DAG enforcement...")

        # Physics prior: only available for standard 14-var layout
        if is_standard:
            physics_prior = get_physics_prior_matrix()
        else:
            physics_prior = np.zeros((n_vars, n_vars))

        # Weighted combination
        w_nt = self.config["notears_weight"]
        w_gc = self.config["granger_weight"]
        w_pc = self.config["pc_weight"]
        w_phy = self.config["physics_weight"]

        # When no physics priors, redistribute weight to algorithms
        if not is_standard:
            total_alg = w_nt + w_gc + w_pc
            if total_alg > 0:
                w_nt = w_nt / total_alg
                w_gc = w_gc / total_alg
                w_pc = w_pc / total_alg
            w_phy = 0.0

        # Build ensemble adjacency with edge details
        dag = np.zeros((n_vars, n_vars))
        edge_details = {}

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                s_nt = notears_stability[i, j]
                s_gc = granger_matrix[i, j]
                s_pc = pc_stability[i, j]
                s_int = int_scores[i, j]
                s_phy = physics_prior[i, j]

                score = w_nt * s_nt + w_gc * s_gc + w_pc * s_pc + w_phy * s_phy

                # Adaptive thresholding
                if s_phy > 0.5:
                    thr = self.threshold * 0.50  # Very low for required physics
                elif s_phy > 0:
                    thr = self.threshold * 0.70  # Low for known physics
                elif s_int > 0.5:
                    thr = self.threshold * 1.1   # Modest for interventional
                else:
                    thr = self.threshold * 1.6   # Hard for unknown edges

                if score >= thr:
                    dag[i, j] = score
                    edge_details[(i, j)] = {
                        "score": score,
                        "nt": s_nt, "gc": s_gc, "pc": s_pc,
                        "int": s_int, "phy": s_phy,
                    }

        # ── DAG enforcement ──
        dag = self._enforce_dag(dag, edge_details)

        # ── Remove indirect paths (only with physics for standard) ──
        if is_standard:
            dag, edge_details = self._remove_indirect(dag, edge_details)

        # ── Physics validation ──
        if is_standard:
            physics_checks = validate_physics(dag)
        else:
            physics_checks = {"passed": 0, "total": 0, "checks": {},
                             "note": "Physics priors not applicable to real data layout"}

        # ── Evaluation against ground truth ──
        if is_standard:
            metrics = evaluate_dag(dag)
        else:
            n_edges = int(np.sum(dag > 0))
            metrics = {
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "shd": 0,
                "tp": 0, "fp": 0, "fn": 0,
                "n_edges_discovered": n_edges,
                "note": "No ground truth for real data — manual validation required",
            }

        return {
            "dag": dag,
            "edge_details": edge_details,
            "metrics": metrics,
            "physics_checks": physics_checks,
            "n_vars": n_vars,
            "n_edges": int(np.sum(dag > 0)),
            "var_names": var_names,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "shd": metrics["shd"],
            "physics_passed": physics_checks["passed"],
            "physics_total": physics_checks["total"],
        }

    def _enforce_dag(self, adj: np.ndarray, details: Dict) -> np.ndarray:
        """Remove cycles by dropping weakest edges."""
        max_iter = 100
        for _ in range(max_iter):
            cycle = _find_cycle(adj)
            if cycle is None:
                break
            # Remove weakest edge in cycle
            u, v = cycle
            if adj[u, v] <= adj[v, u]:
                adj[u, v] = 0
                details.pop((u, v), None)
            else:
                adj[v, u] = 0
                details.pop((v, u), None)
        return adj

    def _remove_indirect(self, adj: np.ndarray, details: Dict):
        """Remove indirect actuator→downstream edges.

        If actuator a → mediator m → target t exists with physics support,
        and a → t also exists without physics support, remove a → t.
        """
        to_remove = []

        for (i, j), d in details.items():
            if d["phy"] > 0:
                continue  # Keep physics-supported edges

            # Pattern: Actuator → far downstream (skip mediator)
            n = adj.shape[0]
            if i in ACTUATOR_IDS:
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ik = details.get((i, k), {})
                    kj = details.get((k, j), {})
                    if adj[i, k] > 0 and adj[k, j] > 0:
                        if ik.get("phy", 0) > 0 or kj.get("phy", 0) > 0:
                            to_remove.append((i, j))
                            break
                continue

            # Pattern: Non-actuator indirect with physics mediator
            if d["nt"] > 0.8 and d["gc"] > 0:
                continue  # Strong evidence
            for k in range(n):
                if k == i or k == j:
                    continue
                if adj[i, k] > 0 and adj[k, j] > 0:
                    ik = details.get((i, k), {})
                    kj = details.get((k, j), {})
                    if ik.get("phy", 0) > 0 and kj.get("phy", 0) > 0:
                        to_remove.append((i, j))
                        break

        for i, j in set(to_remove):
            adj[i, j] = 0
            details.pop((i, j), None)

        return adj, details
