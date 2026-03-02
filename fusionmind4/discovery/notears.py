"""NOTEARS — Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian.

Learns a DAG from observational data using continuous optimization.
Based on Zheng et al. (2018) with augmented Lagrangian enforcement of
the acyclicity constraint: h(W) = tr(e^{W∘W}) - d = 0.

UPGRADE v3.2: Proper DAG constraint via augmented Lagrangian.
ADDITION: DYNOTEARS for temporal causal discovery from time series.
"""
import numpy as np
from typing import Optional, Tuple
from scipy.linalg import expm


class NOTEARSDiscovery:
    """NOTEARS-based DAG structure learning with proper acyclicity constraint.

    Minimizes:
        min_W  ½n ||X - XW||²_F + λ₁||W||₁
        s.t.   h(W) = tr(e^{W∘W}) - d = 0   (DAG constraint)

    Solved via augmented Lagrangian:
        L(W, α, ρ) = loss(W) + λ₁||W||₁ + α·h(W) + ½ρ·h(W)²

    Args:
        lambda1: L1 regularization strength
        max_iter: Maximum augmented Lagrangian outer iterations
        h_tol: Tolerance for acyclicity constraint h(W) ≈ 0
        w_threshold: Edge weight threshold for final pruning
        rho_max: Maximum penalty parameter
    """

    def __init__(self, lambda1: float = 0.05, max_iter: int = 50,
                 h_tol: float = 1e-8, w_threshold: float = 0.10,
                 rho_max: float = 1e16):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold
        self.rho_max = rho_max

    @staticmethod
    def _h(W: np.ndarray) -> float:
        """Acyclicity constraint: h(W) = tr(e^{W∘W}) - d.

        h(W) = 0  iff  W encodes a DAG (no directed cycles).
        """
        d = W.shape[0]
        M = W * W  # Hadamard product
        E = expm(M)
        return float(np.trace(E) - d)

    @staticmethod
    def _h_grad(W: np.ndarray) -> np.ndarray:
        """Gradient of h(W) w.r.t. W: ∇h = (e^{W∘W})ᵀ ∘ 2W."""
        M = W * W
        E = expm(M)
        return E.T * 2 * W

    def _loss(self, W: np.ndarray, X: np.ndarray) -> float:
        """Least-squares loss: ½n ||X - XW||²_F."""
        n = X.shape[0]
        R = X - X @ W
        return 0.5 / n * np.sum(R ** 2)

    def _loss_grad(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Gradient of least-squares loss."""
        n = X.shape[0]
        return -1.0 / n * X.T @ (X - X @ W)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Learn DAG structure using augmented Lagrangian.

        Args:
            X: (n_samples, n_vars) data matrix

        Returns:
            W: (n_vars, n_vars) weighted adjacency matrix (DAG)
        """
        n, d = X.shape
        X = X - X.mean(axis=0)

        W = np.zeros((d, d))
        alpha = 0.0        # Lagrange multiplier
        rho = 1.0          # Penalty parameter
        h_prev = np.inf

        for outer in range(self.max_iter):
            W = self._inner_solve(W, X, alpha, rho, n_inner=30)

            h_val = self._h(W)

            if h_val > 0.25 * h_prev:
                rho = min(10 * rho, self.rho_max)
            else:
                alpha += rho * h_val

            h_prev = h_val

            if abs(h_val) < self.h_tol:
                break

        # Threshold small weights
        W[np.abs(W) < self.w_threshold] = 0
        return W

    def _inner_solve(self, W: np.ndarray, X: np.ndarray,
                     alpha: float, rho: float,
                     n_inner: int = 30) -> np.ndarray:
        """Proximal gradient descent for inner augmented Lagrangian problem."""
        d = W.shape[0]
        n = X.shape[0]
        XtX = X.T @ X / n

        for _ in range(n_inner):
            W_old = W.copy()

            # Clip weights to prevent expm overflow
            W = np.clip(W, -5.0, 5.0)

            grad_loss = self._loss_grad(W, X)
            h_val = self._h(W)
            if not np.isfinite(h_val):
                W *= 0.5  # Scale down if overflow
                continue
            grad_h = self._h_grad(W)
            if not np.all(np.isfinite(grad_h)):
                W *= 0.5
                continue
            grad_smooth = grad_loss + (alpha + rho * h_val) * grad_h

            for i in range(d):
                for j in range(d):
                    if i == j:
                        W[i, j] = 0
                        continue
                    step = 1.0 / (XtX[i, i] + rho * 2 * abs(W[i, j]) + 1e-8)
                    step = min(step, 0.1)  # Cap step size
                    proposal = W[i, j] - step * grad_smooth[i, j]
                    W[i, j] = np.sign(proposal) * max(
                        abs(proposal) - self.lambda1 * step, 0
                    )

            W = np.clip(W, -5.0, 5.0)
            change = np.max(np.abs(W - W_old))
            if change < 1e-7:
                break

        return W

    def fit_bootstrap(self, X: np.ndarray, n_bootstrap: int = 15,
                      rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Bootstrap NOTEARS for edge stability estimation.

        Returns:
            stability: (n_vars, n_vars) fraction of bootstraps where edge appears
        """
        if rng is None:
            rng = np.random.RandomState(42)

        n = X.shape[0]
        stability = np.zeros((X.shape[1], X.shape[1]))

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            nt = NOTEARSDiscovery(
                lambda1=self.lambda1 * (0.5 + rng.random()),
                w_threshold=self.w_threshold * (0.6 + 0.8 * rng.random()),
                max_iter=self.max_iter,
            )
            W = nt.fit(X[idx])
            stability += (np.abs(W) > 0).astype(float)

        return stability / n_bootstrap


# ============================================================================
# DYNOTEARS — Temporal Causal Discovery
# ============================================================================

class DYNOTEARSDiscovery:
    """DYNOTEARS — DAG learning from time series data.

    Extends NOTEARS to temporal data by jointly learning:
    - Contemporaneous DAG (W): instantaneous causal effects
    - Lagged effects (A_lag): effects from past timesteps

    Model: X_t = X_t @ W + Σ_k X_{t-k} @ A_k + noise
    Subject to: h(W) = 0  (only contemporaneous part must be DAG)

    Based on Pamfil et al. (2020) "DYNOTEARS: Structure Learning
    from Time-Series Data".

    Args:
        lambda_w: L1 regularization for contemporaneous edges
        lambda_a: L1 regularization for lagged edges
        max_lag: Maximum number of time lags to consider
        max_iter: Maximum augmented Lagrangian iterations
        w_threshold: Edge pruning threshold
    """

    def __init__(self, lambda_w: float = 0.05, lambda_a: float = 0.05,
                 max_lag: int = 3, max_iter: int = 50,
                 w_threshold: float = 0.10):
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_lag = max_lag
        self.max_iter = max_iter
        self.w_threshold = w_threshold

    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Learn temporal causal structure.

        Args:
            X: (T, d) time series data matrix

        Returns:
            W: (d, d) contemporaneous DAG
            A: (max_lag*d, d) lagged effects stacked [A_1; A_2; ...]
        """
        T, d = X.shape
        X_centered = X - X.mean(axis=0)

        Y, Z = self._build_lagged_matrices(X_centered)
        n = Y.shape[0]

        if n < d + 5:
            nt = NOTEARSDiscovery(lambda1=self.lambda_w,
                                  w_threshold=self.w_threshold,
                                  max_iter=self.max_iter)
            W = nt.fit(X_centered)
            A = np.zeros((self.max_lag * d, d))
            return W, A

        W = np.zeros((d, d))
        A = np.zeros((self.max_lag * d, d))
        alpha = 0.0
        rho = 1.0

        for outer in range(self.max_iter):
            W, A = self._inner_solve(W, A, Y, Z, alpha, rho, n_inner=25)
            h_val = NOTEARSDiscovery._h(W)

            if abs(h_val) < 1e-8:
                break
            alpha += rho * h_val
            rho = min(10 * rho, 1e16)

        W[np.abs(W) < self.w_threshold] = 0
        A[np.abs(A) < self.w_threshold] = 0
        return W, A

    def _build_lagged_matrices(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build target Y and lagged design matrix Z."""
        T, d = X.shape
        Y = X[self.max_lag:]
        Z_parts = []
        for lag in range(1, self.max_lag + 1):
            Z_parts.append(X[self.max_lag - lag: T - lag])
        Z = np.hstack(Z_parts)
        return Y, Z

    def _inner_solve(self, W, A, Y, Z, alpha, rho, n_inner=25):
        """Joint optimization of W and A."""
        n, d = Y.shape

        for _ in range(n_inner):
            W_old = W.copy()
            R = Y - Y @ W - Z @ A

            # Update W (with DAG constraint)
            grad_W = -Y.T @ R / n
            h_val = NOTEARSDiscovery._h(W)
            grad_h = NOTEARSDiscovery._h_grad(W)
            grad_W += (alpha + rho * h_val) * grad_h

            YtY = Y.T @ Y / n
            for i in range(d):
                for j in range(d):
                    if i == j:
                        W[i, j] = 0
                        continue
                    step = 1.0 / (YtY[i, i] + rho + 1e-8)
                    proposal = W[i, j] - step * grad_W[i, j]
                    W[i, j] = np.sign(proposal) * max(
                        abs(proposal) - self.lambda_w * step, 0)

            # Update A (no DAG constraint — lags are inherently acyclic)
            grad_A = -Z.T @ R / n
            ZtZ_diag = np.sum(Z ** 2, axis=0) / n

            for i in range(A.shape[0]):
                for j in range(d):
                    step = 1.0 / (ZtZ_diag[i] + 1e-8)
                    proposal = A[i, j] - step * grad_A[i, j]
                    A[i, j] = np.sign(proposal) * max(
                        abs(proposal) - self.lambda_a * step, 0)

            change = np.max(np.abs(W - W_old))
            if change < 1e-7:
                break

        return W, A

    def get_lagged_edge(self, A: np.ndarray, lag: int,
                        var_from: int, var_to: int) -> float:
        """Get weight of lagged edge: var_from(t-lag) → var_to(t)."""
        d = A.shape[1]
        idx = (lag - 1) * d + var_from
        if idx < A.shape[0]:
            return float(A[idx, var_to])
        return 0.0

    def get_temporal_summary(self, W: np.ndarray, A: np.ndarray,
                              var_names: list) -> dict:
        """Summarize temporal causal structure."""
        d = len(var_names)
        contemporaneous = []
        lagged = []

        for i in range(d):
            for j in range(d):
                if abs(W[i, j]) > 0:
                    contemporaneous.append({
                        'from': var_names[i], 'to': var_names[j],
                        'weight': float(W[i, j]), 'lag': 0
                    })

        for lag in range(1, self.max_lag + 1):
            for i in range(d):
                for j in range(d):
                    w = self.get_lagged_edge(A, lag, i, j)
                    if abs(w) > 0:
                        lagged.append({
                            'from': var_names[i], 'to': var_names[j],
                            'weight': w, 'lag': lag
                        })

        return {
            'contemporaneous': contemporaneous,
            'lagged': lagged,
            'n_contemporaneous': len(contemporaneous),
            'n_lagged': len(lagged),
        }
