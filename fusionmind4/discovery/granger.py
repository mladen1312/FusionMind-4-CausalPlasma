"""Granger Causality testing for temporal causal discovery.

Tests whether past values of X improve prediction of Y beyond Y's own past,
with proper F-distribution p-values, conditional Granger controlling for
confounders, BIC lag selection, variance decomposition, and frequency-domain
spectral Granger causality.

Upgrade v4.3: Replaced approximate F-critical with scipy.stats.f,
added conditional Granger, variance decomposition, SpectralGrangerCausality.
"""
import numpy as np
from typing import Optional, Dict, List, Tuple

try:
    from scipy.stats import f as f_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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

    def test_all_pairs_pvalues(self, data: np.ndarray) -> np.ndarray:
        """Return p-value matrix for all Granger causality tests.

        Args:
            data: (n_samples, n_vars) time series data

        Returns:
            pval_matrix: (n_vars, n_vars) p-value matrix
        """
        n, d = data.shape
        pval_matrix = np.ones((d, d))

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                pval_matrix[i, j] = self._granger_pvalue(data[:, i], data[:, j])

        return pval_matrix

    def _granger_test(self, x: np.ndarray, y: np.ndarray,
                      alpha: float) -> bool:
        """Test if x Granger-causes y using F-test."""
        pval = self._granger_pvalue(x, y)
        return pval < alpha

    def _granger_pvalue(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute p-value for Granger causality test of x -> y."""
        n = len(y)
        if n < 2 * self.max_lag + 10:
            return 1.0

        best_lag = self._select_lag_bic(x, y)
        if best_lag == 0:
            return 1.0

        Y = y[best_lag:]
        n_eff = len(Y)

        # Restricted model: Y ~ Y_lags + const
        Y_lags = np.column_stack([y[best_lag - k - 1:n - k - 1] for k in range(best_lag)])
        Y_lags = np.column_stack([Y_lags, np.ones(n_eff)])

        # Unrestricted model: Y ~ Y_lags + X_lags + const
        X_lags = np.column_stack([x[best_lag - k - 1:n - k - 1] for k in range(best_lag)])
        XY_lags = np.column_stack([Y_lags, X_lags])

        rss_r = self._rss(Y_lags, Y)
        rss_u = self._rss(XY_lags, Y)

        p_r = Y_lags.shape[1]
        p_u = XY_lags.shape[1]
        df1 = p_u - p_r
        df2 = n_eff - p_u

        if df2 <= 0 or rss_u <= 0:
            return 1.0

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

        if HAS_SCIPY:
            pval = 1.0 - f_dist.cdf(f_stat, df1, df2)
        else:
            f_critical = self._f_critical_approx(df1, df2, 0.05)
            pval = 0.01 if f_stat > f_critical else 0.5

        return float(pval)

    def _select_lag_bic(self, x: np.ndarray, y: np.ndarray) -> int:
        """Select optimal lag using BIC (Bayesian Information Criterion)."""
        n = len(y)
        best_bic = np.inf
        best_lag = 1

        for lag in range(1, self.max_lag + 1):
            if n - lag < lag + 5:
                break
            Y = y[lag:]
            n_eff = len(Y)
            Y_lags = np.column_stack([y[lag - k - 1:n - k - 1] for k in range(lag)])
            X_lags = np.column_stack([x[lag - k - 1:n - k - 1] for k in range(lag)])
            design = np.column_stack([Y_lags, X_lags, np.ones(n_eff)])
            rss = self._rss(design, Y)
            k = design.shape[1]
            bic = n_eff * np.log(rss / n_eff + 1e-10) + k * np.log(n_eff)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag

        return best_lag

    # Backward-compatible alias
    _select_lag = _select_lag_bic

    def variance_decomposition(self, x: np.ndarray, y: np.ndarray,
                                lag: Optional[int] = None) -> Dict[str, float]:
        """Compute variance decomposition: how much of Y's variance is
        explained by X's lags beyond Y's own lags.

        Returns:
            dict with 'r2_restricted', 'r2_unrestricted', 'incremental_r2'
        """
        n = len(y)
        if lag is None:
            lag = self._select_lag_bic(x, y)
        if lag == 0:
            lag = 1

        Y = y[lag:]
        n_eff = len(Y)
        var_y = np.var(Y)

        if var_y < 1e-15:
            return {'r2_restricted': 0.0, 'r2_unrestricted': 0.0,
                    'incremental_r2': 0.0}

        Y_lags = np.column_stack([y[lag - k - 1:n - k - 1] for k in range(lag)])
        Y_lags_c = np.column_stack([Y_lags, np.ones(n_eff)])
        rss_r = self._rss(Y_lags_c, Y)
        r2_r = 1.0 - rss_r / (var_y * n_eff)

        X_lags = np.column_stack([x[lag - k - 1:n - k - 1] for k in range(lag)])
        XY_lags = np.column_stack([Y_lags_c, X_lags])
        rss_u = self._rss(XY_lags, Y)
        r2_u = 1.0 - rss_u / (var_y * n_eff)

        return {
            'r2_restricted': max(0.0, float(r2_r)),
            'r2_unrestricted': max(0.0, float(r2_u)),
            'incremental_r2': max(0.0, float(r2_u - r2_r)),
        }

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
    def _f_critical_approx(df1: int, df2: int, alpha: float) -> float:
        """Approximate F-critical value (fallback when scipy unavailable)."""
        if alpha <= 0.001:
            return 10.0 + 5.0 / max(df1, 1)
        elif alpha <= 0.01:
            return 6.0 + 3.0 / max(df1, 1)
        elif alpha <= 0.05:
            return 3.8 + 1.5 / max(df1, 1)
        else:
            return 2.5


class ConditionalGrangerTest:
    """Conditional Granger causality: test X -> Y controlling for confounders Z.

    Standard Granger can find spurious causation when X and Y share a
    common cause Z. Conditional Granger controls for this.

    Model comparison:
        Restricted:   Y_t = f(Y_{t-1..L}, Z_{t-1..L})
        Unrestricted: Y_t = f(Y_{t-1..L}, Z_{t-1..L}, X_{t-1..L})
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha

    def test(self, x: np.ndarray, y: np.ndarray,
             z: np.ndarray, lag: Optional[int] = None) -> Dict:
        """Test X -> Y | Z.

        Args:
            x: (n_samples,) cause variable
            y: (n_samples,) effect variable
            z: (n_samples,) or (n_samples, n_confounders) conditioning variables

        Returns:
            dict with 'significant', 'p_value', 'f_stat'
        """
        n = len(y)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        n_z = z.shape[1]

        if lag is None:
            gc = GrangerCausalityTest(max_lag=self.max_lag)
            lag = gc._select_lag_bic(x, y)
        if lag == 0:
            lag = 1

        if n < lag + 10:
            return {'significant': False, 'p_value': 1.0, 'f_stat': 0.0}

        Y = y[lag:]
        n_eff = len(Y)

        # Build restricted design: Y_lags + Z_lags + const
        Y_lags = np.column_stack([y[lag - k - 1:n - k - 1] for k in range(lag)])
        Z_lags_list = []
        for zi in range(n_z):
            for k in range(lag):
                Z_lags_list.append(z[lag - k - 1:n - k - 1, zi])
        Z_lags = np.column_stack(Z_lags_list) if Z_lags_list else np.empty((n_eff, 0))

        restricted = np.column_stack([Y_lags, Z_lags, np.ones(n_eff)])

        # Unrestricted: + X_lags
        X_lags = np.column_stack([x[lag - k - 1:n - k - 1] for k in range(lag)])
        unrestricted = np.column_stack([restricted, X_lags])

        rss_r = GrangerCausalityTest._rss(restricted, Y)
        rss_u = GrangerCausalityTest._rss(unrestricted, Y)

        df1 = lag
        df2 = n_eff - unrestricted.shape[1]

        if df2 <= 0 or rss_u <= 0:
            return {'significant': False, 'p_value': 1.0, 'f_stat': 0.0}

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

        if HAS_SCIPY:
            pval = 1.0 - f_dist.cdf(f_stat, df1, df2)
        else:
            f_crit = GrangerCausalityTest._f_critical_approx(df1, df2, self.alpha)
            pval = 0.01 if f_stat > f_crit else 0.5

        return {
            'significant': pval < self.alpha,
            'p_value': float(pval),
            'f_stat': float(f_stat),
        }


class SpectralGrangerCausality:
    """Frequency-domain Granger causality via spectral transfer function.

    Decomposes Granger causality across frequencies, revealing whether
    X causes Y at fast timescales (MHD) or slow timescales (transport).

    Uses VAR model -> transfer function -> spectral decomposition.
    """

    def __init__(self, max_lag: int = 5, n_freqs: int = 128):
        self.max_lag = max_lag
        self.n_freqs = n_freqs

    def fit(self, x: np.ndarray, y: np.ndarray,
            lag: Optional[int] = None) -> Dict:
        """Compute spectral Granger causality from x -> y.

        Uses Geweke (1982) decomposition: fit full bivariate VAR and
        restricted univariate AR, compare spectral densities.

        Returns:
            dict with 'frequencies', 'spectral_gc', 'total_gc',
            'peak_frequency', 'peak_gc'
        """
        n = len(y)
        if lag is None:
            gc = GrangerCausalityTest(max_lag=self.max_lag)
            lag = gc._select_lag_bic(x, y)
        if lag == 0:
            lag = 1

        # Full bivariate VAR(lag) model
        data = np.column_stack([y, x])
        A_mats, sigma = self._fit_var(data, lag)

        # Restricted univariate AR(lag) model — y on its own lags only
        sigma_r = self._fit_univariate_ar(y, lag)

        freqs = np.linspace(0, 0.5, self.n_freqs)
        spectral_gc = np.zeros(self.n_freqs)

        for fi, f in enumerate(freqs):
            # Full model spectral density of y
            H = self._transfer_function(A_mats, f)
            S = H @ sigma @ H.conj().T
            s_yy_full = np.abs(S[0, 0])

            # Restricted model spectral density of y (univariate AR)
            h_r = self._univariate_transfer(self._ar_coeffs, f)
            s_yy_restricted = sigma_r / (np.abs(h_r) ** 2 + 1e-30)

            if s_yy_full > 1e-15 and s_yy_restricted > 1e-15:
                ratio = s_yy_restricted / s_yy_full
                if ratio > 1.0:
                    spectral_gc[fi] = np.log(ratio)

        total_gc = float(np.trapezoid(spectral_gc, freqs))
        peak_idx = np.argmax(spectral_gc)

        return {
            'frequencies': freqs,
            'spectral_gc': spectral_gc,
            'total_gc': total_gc,
            'peak_frequency': float(freqs[peak_idx]),
            'peak_gc': float(spectral_gc[peak_idx]),
        }

    def _fit_univariate_ar(self, y: np.ndarray, lag: int) -> float:
        """Fit univariate AR(p) model, return residual variance."""
        n = len(y)
        Y = y[lag:]
        X_parts = [y[lag - k - 1:n - k - 1].reshape(-1, 1) for k in range(lag)]
        X = np.column_stack(X_parts + [np.ones((len(Y), 1))])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        self._ar_coeffs = beta[:lag]  # Store for transfer function
        residuals = Y - X @ beta
        return float(np.var(residuals))

    def _univariate_transfer(self, ar_coeffs: np.ndarray, f: float) -> complex:
        """Univariate AR transfer function: 1 - sum_k a_k e^{-2pi*i*f*k}"""
        h = 1.0 + 0j
        for k, a in enumerate(ar_coeffs):
            h -= a * np.exp(-2j * np.pi * f * (k + 1))
        return h
        peak_idx = np.argmax(spectral_gc)

        return {
            'frequencies': freqs,
            'spectral_gc': spectral_gc,
            'total_gc': total_gc,
            'peak_frequency': float(freqs[peak_idx]),
            'peak_gc': float(spectral_gc[peak_idx]),
        }

    def _fit_var(self, data: np.ndarray, lag: int) -> Tuple:
        """Fit VAR(p) model via OLS."""
        n, d = data.shape
        Y = data[lag:]
        n_eff = len(Y)

        X_parts = []
        for k in range(lag):
            X_parts.append(data[lag - k - 1:n - k - 1])
        X = np.column_stack(X_parts + [np.ones((n_eff, 1))])

        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return [np.zeros((d, d))] * lag, np.eye(d)

        residuals = Y - X @ beta
        sigma = (residuals.T @ residuals) / n_eff

        A_mats = []
        for k in range(lag):
            A_mats.append(beta[k * d:(k + 1) * d].T)

        return A_mats, sigma

    def _transfer_function(self, A_mats: List, f: float) -> np.ndarray:
        """H(f) = (I - sum_k A_k e^{-2pi*i*f*k})^{-1}"""
        d = A_mats[0].shape[0]
        H_inv = np.eye(d, dtype=complex)
        for k, A in enumerate(A_mats):
            H_inv -= A * np.exp(-2j * np.pi * f * (k + 1))
        try:
            return np.linalg.inv(H_inv)
        except np.linalg.LinAlgError:
            return np.eye(d, dtype=complex)

    def _transfer_function_restricted(self, A_mats: List, f: float) -> np.ndarray:
        """Transfer function with x->y coupling removed."""
        d = A_mats[0].shape[0]
        H_inv = np.eye(d, dtype=complex)
        for k, A in enumerate(A_mats):
            A_r = A.copy()
            A_r[0, 1] = 0
            H_inv -= A_r * np.exp(-2j * np.pi * f * (k + 1))
        try:
            return np.linalg.inv(H_inv)
        except np.linalg.LinAlgError:
            return np.eye(d, dtype=complex)
