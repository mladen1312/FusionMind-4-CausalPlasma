"""PC Algorithm — Constraint-based causal discovery.

Discovers causal skeleton via conditional independence testing,
then orients edges using v-structure detection and Meek's rules R1-R4.

Upgrade v4.3: Added Meek's orientation rules for maximal edge orientation,
stable-PC variant for order-independence, Fisher z-test with proper p-values.
"""
import numpy as np
from typing import Optional, Set, Dict, Tuple
from itertools import combinations


class PCAlgorithm:
    """PC algorithm for constraint-based causal discovery.

    Args:
        alpha: Significance threshold for independence tests
        max_cond_set: Maximum conditioning set size
        stable: If True, use stable-PC variant (order-independent)
    """

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3,
                 stable: bool = True):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.stable = stable

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Run PC algorithm on data.

        Args:
            X: (n_samples, n_vars) data matrix

        Returns:
            dag: (n_vars, n_vars) partially oriented adjacency matrix
        """
        n, d = X.shape
        # Start with complete undirected graph
        skeleton = np.ones((d, d)) - np.eye(d)
        sep_sets = {}

        # Phase 1: Skeleton discovery
        skeleton, sep_sets = self._discover_skeleton(X, skeleton, sep_sets)

        # Phase 2: Orient v-structures
        dag = self._orient_v_structures(skeleton, sep_sets, d)

        # Phase 3: Apply Meek's rules R1-R4
        dag = self._apply_meek_rules(dag, d)

        return dag

    def _discover_skeleton(self, X: np.ndarray, skeleton: np.ndarray,
                           sep_sets: dict) -> Tuple[np.ndarray, dict]:
        """Phase 1: Remove edges via conditional independence tests.

        If stable=True, edges are collected for removal but only applied
        after testing all pairs at each conditioning set size. This makes
        the result invariant to variable ordering.
        """
        d = X.shape[1]

        for cond_size in range(self.max_cond_set + 1):
            if self.stable:
                # Stable-PC: collect removals, apply after full sweep
                removals = []
                for i in range(d):
                    for j in range(i + 1, d):
                        if skeleton[i, j] == 0:
                            continue
                        result = self._test_edge(X, i, j, skeleton, cond_size)
                        if result is not None:
                            removals.append((i, j, result))
                # Apply removals
                for i, j, sep in removals:
                    skeleton[i, j] = 0
                    skeleton[j, i] = 0
                    sep_sets[(i, j)] = sep
                    sep_sets[(j, i)] = sep
            else:
                # Standard PC: remove immediately
                for i in range(d):
                    for j in range(i + 1, d):
                        if skeleton[i, j] == 0:
                            continue
                        result = self._test_edge(X, i, j, skeleton, cond_size)
                        if result is not None:
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            sep_sets[(i, j)] = result
                            sep_sets[(j, i)] = result

        return skeleton, sep_sets

    def _test_edge(self, X: np.ndarray, i: int, j: int,
                   skeleton: np.ndarray, cond_size: int) -> Optional[Set]:
        """Test if edge i-j can be removed given conditioning set of given size.

        Returns the separating set if independent, None otherwise.
        """
        d = X.shape[1]
        # Adjacencies of i (excluding j)
        adj_i = set(k for k in range(d) if k != i and k != j and skeleton[i, k] > 0)

        if len(adj_i) < cond_size:
            return None

        for cond_set in combinations(adj_i, cond_size):
            cond_list = list(cond_set)
            pval = self._fisher_z_test(X, i, j, cond_list)
            if pval > self.alpha:
                return set(cond_set)

        return None

    def _orient_v_structures(self, skeleton: np.ndarray, sep_sets: dict,
                             d: int) -> np.ndarray:
        """Phase 2: Orient v-structures i -> k <- j if k not in sep(i,j)."""
        dag = skeleton.copy()

        for i in range(d):
            for j in range(i + 1, d):
                if skeleton[i, j] > 0:
                    continue  # i and j are adjacent, skip
                for k in range(d):
                    if skeleton[i, k] > 0 and skeleton[j, k] > 0:
                        sep = sep_sets.get((i, j), set())
                        if k not in sep:
                            # Orient i -> k <- j
                            dag[k, i] = 0  # Remove k->i
                            dag[k, j] = 0  # Remove k->j

        return dag

    def _apply_meek_rules(self, dag: np.ndarray, d: int) -> np.ndarray:
        """Phase 3: Apply Meek's orientation rules R1-R4 until convergence.

        These rules maximally orient edges without creating new v-structures
        or directed cycles.
        """
        changed = True
        max_iter = d * d  # Safety limit
        iteration = 0

        while changed and iteration < max_iter:
            changed = False
            iteration += 1

            # R1: If i -> j - k (and i not adjacent to k), orient j -> k
            for j in range(d):
                for k in range(d):
                    if j == k:
                        continue
                    if not self._is_undirected(dag, j, k):
                        continue
                    for i in range(d):
                        if i == j or i == k:
                            continue
                        if self._is_directed(dag, i, j) and dag[i, k] == 0 and dag[k, i] == 0:
                            dag[k, j] = 0  # Orient j -> k
                            changed = True

            # R2: If i -> k -> j and i - j, orient i -> j
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    if not self._is_undirected(dag, i, j):
                        continue
                    for k in range(d):
                        if k == i or k == j:
                            continue
                        if self._is_directed(dag, i, k) and self._is_directed(dag, k, j):
                            dag[j, i] = 0  # Orient i -> j
                            changed = True

            # R3: If i - k -> j and i - l -> j and i - j, and k not adj l,
            #     orient i -> j
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    if not self._is_undirected(dag, i, j):
                        continue
                    # Find k,l such that i-k->j and i-l->j and k not adj l
                    k_list = []
                    for k in range(d):
                        if k == i or k == j:
                            continue
                        if self._is_undirected(dag, i, k) and self._is_directed(dag, k, j):
                            k_list.append(k)
                    # Check pairs
                    for ki in range(len(k_list)):
                        for li in range(ki + 1, len(k_list)):
                            k, l = k_list[ki], k_list[li]
                            if dag[k, l] == 0 and dag[l, k] == 0:  # Not adjacent
                                dag[j, i] = 0  # Orient i -> j
                                changed = True

            # R4: If i - k -> l -> j and i - j, orient i -> j
            for i in range(d):
                for j in range(d):
                    if i == j:
                        continue
                    if not self._is_undirected(dag, i, j):
                        continue
                    for k in range(d):
                        if k == i or k == j:
                            continue
                        if not self._is_undirected(dag, i, k):
                            continue
                        for l in range(d):
                            if l == i or l == j or l == k:
                                continue
                            if self._is_directed(dag, k, l) and self._is_directed(dag, l, j):
                                dag[j, i] = 0  # Orient i -> j
                                changed = True

        return dag

    @staticmethod
    def _is_directed(dag: np.ndarray, i: int, j: int) -> bool:
        """Check if there is a directed edge i -> j."""
        return bool(dag[i, j] > 0 and dag[j, i] == 0)

    @staticmethod
    def _is_undirected(dag: np.ndarray, i: int, j: int) -> bool:
        """Check if there is an undirected edge i - j."""
        return bool(dag[i, j] > 0 and dag[j, i] > 0)

    def fit_bootstrap(self, X: np.ndarray, n_bootstrap: int = 15,
                      rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Bootstrap PC algorithm for edge stability."""
        if rng is None:
            rng = np.random.RandomState(42)

        n = X.shape[0]
        stability = np.zeros((X.shape[1], X.shape[1]))

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            pc = PCAlgorithm(
                alpha=self.alpha * (0.5 + rng.random()),
                max_cond_set=self.max_cond_set,
                stable=self.stable,
            )
            dag = pc.fit(X[idx])
            stability += (np.abs(dag) > 0).astype(float)

        return stability / n_bootstrap

    def _fisher_z_test(self, X: np.ndarray, i: int, j: int,
                       cond: list) -> float:
        """Fisher z-test for conditional independence.

        Tests H0: rho(X_i, X_j | X_cond) = 0
        using Fisher's z-transform of the partial correlation.

        Returns:
            p-value (two-sided)
        """
        pcorr = self._partial_correlation(X, i, j, cond)
        n = X.shape[0]
        k = len(cond)

        # Effective sample size for Fisher z
        n_eff = n - k - 3
        if n_eff < 4:
            return 1.0  # Not enough data

        # Fisher z-transform
        # z = 0.5 * ln((1+r)/(1-r)) * sqrt(n-k-3)
        r = np.clip(pcorr, -0.9999, 0.9999)
        z = 0.5 * np.log((1 + r) / (1 - r))
        z_stat = np.abs(z) * np.sqrt(n_eff)

        # Two-sided p-value from standard normal
        # P(|Z| > z_stat) = 2 * (1 - Phi(z_stat))
        try:
            from scipy.stats import norm
            pval = 2.0 * (1.0 - norm.cdf(z_stat))
        except ImportError:
            # Fallback: use approximation of normal CDF tail
            pval = 2.0 * np.exp(-0.5 * z_stat ** 2) / (z_stat * np.sqrt(2 * np.pi) + 1e-10)
            pval = min(pval, 1.0)

        return float(pval)

    # Keep old method as fallback alias
    def _test_conditional_independence(self, X, i, j, neighbors, cond_size):
        """Legacy method: test conditional independence."""
        if cond_size == 0:
            return self._fisher_z_test(X, i, j, []) > self.alpha

        if len(neighbors) < cond_size:
            return False

        for cond_set in combinations(neighbors[:min(len(neighbors), 6)], cond_size):
            pval = self._fisher_z_test(X, i, j, list(cond_set))
            if pval > self.alpha:
                return True
        return False

    @staticmethod
    def _partial_correlation(X, i, j, cond):
        """Compute partial correlation of X_i and X_j given X_cond."""
        if not cond:
            return np.corrcoef(X[:, i], X[:, j])[0, 1]

        Z = X[:, cond]
        Z = np.column_stack([Z, np.ones(len(Z))])

        try:
            beta_i = np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
            beta_j = np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
            res_i = X[:, i] - Z @ beta_i
            res_j = X[:, j] - Z @ beta_j
            corr = np.corrcoef(res_i, res_j)[0, 1]
            return corr if np.isfinite(corr) else 0.0
        except np.linalg.LinAlgError:
            return 0.0
