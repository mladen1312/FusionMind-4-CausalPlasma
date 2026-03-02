"""NOTEARS — Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian.

Learns a DAG from observational data using continuous optimization.
Based on Zheng et al. (2018) with L1 regularization and soft thresholding.
"""
import numpy as np
from typing import Optional


class NOTEARSDiscovery:
    """NOTEARS-based DAG structure learning.

    Uses coordinate descent with iterative soft thresholding to learn
    a weighted adjacency matrix W such that:
        X = X @ W + noise
    subject to the DAG constraint: h(W) = tr(e^{W∘W}) - d = 0

    Args:
        lambda1: L1 regularization strength
        max_iter: Maximum optimization iterations
        w_threshold: Edge weight threshold for pruning
    """

    def __init__(self, lambda1: float = 0.05, max_iter: int = 50,
                 w_threshold: float = 0.10):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.w_threshold = w_threshold

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Learn DAG structure from data.

        Args:
            X: (n_samples, n_vars) data matrix

        Returns:
            W: (n_vars, n_vars) weighted adjacency matrix
        """
        n, d = X.shape
        X = X - X.mean(axis=0)

        # Initialize
        W = np.zeros((d, d))
        XtX = X.T @ X / n

        for iteration in range(self.max_iter):
            W_old = W.copy()

            # Coordinate descent over each edge
            for j in range(d):
                for i in range(d):
                    if i == j:
                        continue

                    # Compute gradient for W[i,j]
                    residual = X[:, j] - X @ W[:, j]
                    grad = -X[:, i].T @ residual / n

                    # Soft thresholding (L1 proximal)
                    step_size = 1.0 / (XtX[i, i] + 1e-8)
                    proposal = W[i, j] - step_size * grad
                    W[i, j] = np.sign(proposal) * max(
                        abs(proposal) - self.lambda1 * step_size, 0
                    )

            # Check convergence
            change = np.max(np.abs(W - W_old))
            if change < 1e-6:
                break

        # Threshold small weights
        W[np.abs(W) < self.w_threshold] = 0

        return W

    def fit_bootstrap(self, X: np.ndarray, n_bootstrap: int = 15,
                      rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Bootstrap NOTEARS for edge stability estimation.

        Args:
            X: (n_samples, n_vars) data matrix
            n_bootstrap: Number of bootstrap iterations
            rng: Random state for reproducibility

        Returns:
            stability: (n_vars, n_vars) fraction of bootstraps where edge appears
        """
        if rng is None:
            rng = np.random.RandomState(42)

        n = X.shape[0]
        stability = np.zeros((X.shape[1], X.shape[1]))

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            # Randomize hyperparameters slightly
            nt = NOTEARSDiscovery(
                lambda1=self.lambda1 * (0.5 + rng.random()),
                w_threshold=self.w_threshold * (0.6 + 0.8 * rng.random()),
                max_iter=self.max_iter,
            )
            W = nt.fit(X[idx])
            stability += (np.abs(W) > 0).astype(float)

        return stability / n_bootstrap
