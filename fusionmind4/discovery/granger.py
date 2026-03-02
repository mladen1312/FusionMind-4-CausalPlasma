"""Granger Causality testing for temporal causal discovery.

Tests whether past values of X improve prediction of Y beyond Y's own past,
with lag selection and Bonferroni correction for multiple testing.
"""
import numpy as np
from typing import Optional


class GrangerCausalityTest:
    """Granger causality testing with lag selection.

    Args:
        max_lag: Maximum lag to test
        alpha: Significance level
        bonferroni: Whether to apply Bonferroni correction
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.05,
                 bonferroni: bool = True):
        self.max_lag = max_lag
        self.alpha = alpha
        self.bonferroni = bonferroni

    def test_all_pairs(self, data: np.ndarray) -> np.ndarray:
        """Test Granger causality for all variable pairs.

        Args:
            data: (n_samples, n_vars) time series data

        Returns:
            gc_matrix: (n_vars, n_vars) binary matrix, 1 if i Granger-causes j
        """
        n, d = data.shape
        gc_matrix = np.zeros((d, d))

        # Bonferroni correction
        n_tests = d * (d - 1)
        alpha_adj = self.alpha / n_tests if self.bonferroni else self.alpha

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if self._granger_test(data[:, i], data[:, j], alpha_adj):
                    gc_matrix[i, j] = 1

        return gc_matrix

    def _granger_test(self, x: np.ndarray, y: np.ndarray,
                      alpha: float) -> bool:
        """Test if x Granger-causes y using F-test.

        Compares restricted model (y ~ y_lags) to unrestricted (y ~ y_lags + x_lags).
        """
        n = len(y)
        if n < 2 * self.max_lag + 10:
            return False

        best_lag = self._select_lag(x, y)
        if best_lag == 0:
            return False

        # Build lag matrices
        Y = y[best_lag:]
        n_eff = len(Y)

        # Restricted model: Y ~ Y_lags
        Y_lags = np.column_stack([y[best_lag - k - 1:n - k - 1] for k in range(best_lag)])
        Y_lags = np.column_stack([Y_lags, np.ones(n_eff)])

        # Unrestricted model: Y ~ Y_lags + X_lags
        X_lags = np.column_stack([x[best_lag - k - 1:n - k - 1] for k in range(best_lag)])
        XY_lags = np.column_stack([Y_lags, X_lags])

        # Fit OLS
        rss_r = self._rss(Y_lags, Y)
        rss_u = self._rss(XY_lags, Y)

        # F-test
        p_r = Y_lags.shape[1]
        p_u = XY_lags.shape[1]
        df1 = p_u - p_r
        df2 = n_eff - p_u

        if df2 <= 0 or rss_u <= 0:
            return False

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

        # Approximate p-value using F-distribution CDF
        # Using simple threshold instead of scipy for minimal deps
        f_critical = self._f_critical(df1, df2, alpha)
        return f_stat > f_critical

    def _select_lag(self, x: np.ndarray, y: np.ndarray) -> int:
        """Select optimal lag using BIC."""
        n = len(y)
        best_bic = np.inf
        best_lag = 1

        for lag in range(1, self.max_lag + 1):
            if n - lag < lag + 5:
                break
            Y = y[lag:]
            Y_lags = np.column_stack([y[lag - k - 1:n - k - 1] for k in range(lag)])
            X_lags = np.column_stack([x[lag - k - 1:n - k - 1] for k in range(lag)])
            design = np.column_stack([Y_lags, X_lags, np.ones(len(Y))])
            rss = self._rss(design, Y)
            k = design.shape[1]
            n_eff = len(Y)
            bic = n_eff * np.log(rss / n_eff + 1e-10) + k * np.log(n_eff)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag

        return best_lag

    @staticmethod
    def _rss(X: np.ndarray, y: np.ndarray) -> float:
        """Residual sum of squares from OLS fit."""
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            return float(np.sum(residuals ** 2))
        except np.linalg.LinAlgError:
            return np.inf

    @staticmethod
    def _f_critical(df1: int, df2: int, alpha: float) -> float:
        """Approximate F-critical value (avoids scipy dependency)."""
        # Simple approximation for common cases
        if alpha <= 0.001:
            return 10.0 + 5.0 / max(df1, 1)
        elif alpha <= 0.01:
            return 6.0 + 3.0 / max(df1, 1)
        elif alpha <= 0.05:
            return 3.8 + 1.5 / max(df1, 1)
        else:
            return 2.5
