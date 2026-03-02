"""PC Algorithm — Constraint-based causal discovery.

Discovers causal skeleton via conditional independence testing,
then orients edges using v-structure detection.
"""
import numpy as np
from typing import Optional


class PCAlgorithm:
    """PC algorithm for constraint-based causal discovery.

    Args:
        alpha: Significance threshold for independence tests
        max_cond_set: Maximum conditioning set size
    """

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3):
        self.alpha = alpha
        self.max_cond_set = max_cond_set

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

        # Phase 1: Skeleton discovery (remove edges)
        for cond_size in range(self.max_cond_set + 1):
            for i in range(d):
                for j in range(i + 1, d):
                    if skeleton[i, j] == 0:
                        continue
                    # Find neighbors
                    neighbors = [k for k in range(d) if k != i and k != j
                                 and (skeleton[i, k] > 0 or skeleton[j, k] > 0)]
                    # Test conditional independence
                    if self._test_conditional_independence(
                        X, i, j, neighbors, cond_size
                    ):
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        sep_sets[(i, j)] = set(neighbors[:cond_size])

        # Phase 2: Orient v-structures (i → k ← j if k not in sep(i,j))
        dag = skeleton.copy()
        for i in range(d):
            for j in range(i + 1, d):
                if skeleton[i, j] > 0:
                    continue  # i and j are adjacent
                for k in range(d):
                    if skeleton[i, k] > 0 and skeleton[j, k] > 0:
                        sep = sep_sets.get((min(i, j), max(i, j)), set())
                        if k not in sep:
                            # Orient as v-structure: i → k ← j
                            dag[k, i] = 0  # Remove k→i
                            dag[k, j] = 0  # Remove k→j

        return dag

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
            )
            dag = pc.fit(X[idx])
            stability += (np.abs(dag) > 0).astype(float)

        return stability / n_bootstrap

    def _test_conditional_independence(self, X, i, j, neighbors, cond_size):
        """Test if X_i ⊥ X_j | X_S using partial correlation."""
        if cond_size == 0:
            # Marginal correlation test
            corr = np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
            n = X.shape[0]
            # Fisher z-transform threshold
            z_thresh = 1.96 / np.sqrt(n - 3) if n > 6 else 0.5
            return corr < z_thresh * 3

        if len(neighbors) < cond_size:
            return False

        # Test with subsets of conditioning variables
        from itertools import combinations
        for cond_set in combinations(neighbors[:min(len(neighbors), 6)], cond_size):
            cond_set = list(cond_set)
            pcorr = self._partial_correlation(X, i, j, cond_set)
            n = X.shape[0]
            z_thresh = 1.96 / np.sqrt(n - len(cond_set) - 3) if n > len(cond_set) + 6 else 0.5
            if abs(pcorr) < z_thresh * 3:
                return True

        return False

    @staticmethod
    def _partial_correlation(X, i, j, cond):
        """Compute partial correlation of X_i and X_j given X_cond."""
        if not cond:
            return np.corrcoef(X[:, i], X[:, j])[0, 1]

        # Regress out conditioning variables
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
