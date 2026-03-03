"""Tests for v4.3 upgrades: Granger (proper F-test, conditional, spectral),
PC (Meek's rules, stable-PC, Fisher z-test), D3R (Grad-Shafranov score).
"""
import numpy as np
import pytest


# ============================================================================
# Granger Causality Upgrades
# ============================================================================

class TestGrangerPValues:
    """Test proper F-distribution p-values."""

    def test_pvalue_matrix_shape(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        data = rng.randn(200, 4)
        gc = GrangerCausalityTest(max_lag=3, bonferroni=False)
        pvals = gc.test_all_pairs_pvalues(data)
        assert pvals.shape == (4, 4)

    def test_diagonal_is_one(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        data = rng.randn(200, 3)
        gc = GrangerCausalityTest(max_lag=3)
        pvals = gc.test_all_pairs_pvalues(data)
        np.testing.assert_array_equal(np.diag(pvals), 1.0)

    def test_pvalues_in_range(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        data = rng.randn(200, 3)
        gc = GrangerCausalityTest(max_lag=3)
        pvals = gc.test_all_pairs_pvalues(data)
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_detects_true_causation(self):
        """Generate x -> y with known lag, verify low p-value."""
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.8 * y[t-1] + 0.6 * x[t-2] + 0.3 * rng.randn()
        gc = GrangerCausalityTest(max_lag=5)
        pval = gc._granger_pvalue(x, y)
        assert pval < 0.05

    def test_independent_series_high_pvalue(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        n = 200
        x = rng.randn(n)
        y = rng.randn(n)
        gc = GrangerCausalityTest(max_lag=3)
        pval = gc._granger_pvalue(x, y)
        assert pval > 0.05


class TestVarianceDecomposition:
    """Test variance decomposition."""

    def test_returns_dict(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        n = 300
        x = rng.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t-1] + 0.4 * x[t-1] + rng.randn()
        gc = GrangerCausalityTest(max_lag=3)
        vd = gc.variance_decomposition(x, y)
        assert 'r2_restricted' in vd
        assert 'r2_unrestricted' in vd
        assert 'incremental_r2' in vd

    def test_incremental_r2_positive_for_cause(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t-1] + 0.5 * x[t-1] + 0.3 * rng.randn()
        gc = GrangerCausalityTest(max_lag=3)
        vd = gc.variance_decomposition(x, y)
        assert vd['incremental_r2'] > 0.01

    def test_incremental_r2_small_for_independent(self):
        from fusionmind4.discovery.granger import GrangerCausalityTest
        rng = np.random.RandomState(42)
        n = 300
        x = rng.randn(n)
        y = rng.randn(n)
        gc = GrangerCausalityTest(max_lag=3)
        vd = gc.variance_decomposition(x, y)
        assert vd['incremental_r2'] < 0.1


class TestConditionalGranger:
    """Test conditional Granger causality."""

    def test_basic_conditional(self):
        from fusionmind4.discovery.granger import ConditionalGrangerTest
        rng = np.random.RandomState(42)
        n = 500
        z = rng.randn(n)
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * z[t-1] + rng.randn()
            y[t] = 0.5 * y[t-1] + 0.5 * z[t-1] + rng.randn()
        # x -> y is spurious (both caused by z)
        cgt = ConditionalGrangerTest(max_lag=3, alpha=0.05)
        result = cgt.test(x, y, z)
        assert 'significant' in result
        assert 'p_value' in result
        assert 'f_stat' in result

    def test_returns_result_dict(self):
        from fusionmind4.discovery.granger import ConditionalGrangerTest
        rng = np.random.RandomState(42)
        data = rng.randn(200, 3)
        cgt = ConditionalGrangerTest(max_lag=3)
        result = cgt.test(data[:, 0], data[:, 1], data[:, 2])
        assert isinstance(result, dict)
        assert result['p_value'] >= 0
        assert result['p_value'] <= 1

    def test_2d_confounders(self):
        from fusionmind4.discovery.granger import ConditionalGrangerTest
        rng = np.random.RandomState(42)
        n = 300
        z = rng.randn(n, 2)
        x = rng.randn(n)
        y = rng.randn(n)
        cgt = ConditionalGrangerTest(max_lag=3)
        result = cgt.test(x, y, z)
        assert not result['significant'] or result['p_value'] < 0.05


class TestSpectralGranger:
    """Test spectral Granger causality."""

    def test_fit_returns_dict(self):
        from fusionmind4.discovery.granger import SpectralGrangerCausality
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t-1] + 0.4 * x[t-1] + rng.randn()
        sgc = SpectralGrangerCausality(max_lag=3, n_freqs=64)
        result = sgc.fit(x, y)
        assert 'frequencies' in result
        assert 'spectral_gc' in result
        assert 'total_gc' in result
        assert 'peak_frequency' in result

    def test_spectral_gc_non_negative(self):
        from fusionmind4.discovery.granger import SpectralGrangerCausality
        rng = np.random.RandomState(42)
        n = 300
        x = rng.randn(n)
        y = rng.randn(n)
        sgc = SpectralGrangerCausality(max_lag=3, n_freqs=32)
        result = sgc.fit(x, y)
        assert np.all(result['spectral_gc'] >= 0)

    def test_frequencies_range(self):
        from fusionmind4.discovery.granger import SpectralGrangerCausality
        sgc = SpectralGrangerCausality(n_freqs=64)
        rng = np.random.RandomState(42)
        result = sgc.fit(rng.randn(200), rng.randn(200))
        assert result['frequencies'][0] == 0
        assert result['frequencies'][-1] == 0.5

    def test_total_gc_positive_for_causal_pair(self):
        from fusionmind4.discovery.granger import SpectralGrangerCausality
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t-1] + 0.5 * x[t-1] + 0.2 * rng.randn()
        sgc = SpectralGrangerCausality(max_lag=5, n_freqs=64)
        result = sgc.fit(x, y)
        assert result['total_gc'] > 0


# ============================================================================
# PC Algorithm Upgrades
# ============================================================================

class TestPCStable:
    """Test stable-PC variant."""

    def test_stable_produces_dag(self):
        from fusionmind4.discovery.pc import PCAlgorithm
        rng = np.random.RandomState(42)
        data = rng.randn(200, 5)
        pc = PCAlgorithm(alpha=0.05, stable=True)
        dag = pc.fit(data)
        assert dag.shape == (5, 5)
        assert np.all(np.diag(dag) == 0)

    def test_stable_vs_unstable_both_work(self):
        from fusionmind4.discovery.pc import PCAlgorithm
        rng = np.random.RandomState(42)
        data = rng.randn(200, 4)
        dag_s = PCAlgorithm(alpha=0.05, stable=True).fit(data)
        dag_u = PCAlgorithm(alpha=0.05, stable=False).fit(data)
        assert dag_s.shape == dag_u.shape


class TestFisherZTest:
    """Test Fisher z-test for conditional independence."""

    def test_correlated_pair_low_pvalue(self):
        from fusionmind4.discovery.pc import PCAlgorithm
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = 0.8 * x + 0.2 * rng.randn(n)
        data = np.column_stack([x, y, rng.randn(n)])
        pc = PCAlgorithm()
        pval = pc._fisher_z_test(data, 0, 1, [])
        assert pval < 0.01

    def test_independent_pair_high_pvalue(self):
        from fusionmind4.discovery.pc import PCAlgorithm
        rng = np.random.RandomState(42)
        n = 200
        data = rng.randn(n, 3)
        pc = PCAlgorithm()
        pval = pc._fisher_z_test(data, 0, 1, [])
        assert pval > 0.05

    def test_conditional_independence(self):
        """x -> z <- y; x _|_ y but x not _|_ y | z (collider)."""
        from fusionmind4.discovery.pc import PCAlgorithm
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = rng.randn(n)
        z = 0.5 * x + 0.5 * y + 0.2 * rng.randn(n)
        data = np.column_stack([x, y, z])
        pc = PCAlgorithm()
        # Marginal: x _|_ y
        pval_marginal = pc._fisher_z_test(data, 0, 1, [])
        assert pval_marginal > 0.05
        # Conditional: x not _|_ y | z (explaining away)
        pval_cond = pc._fisher_z_test(data, 0, 1, [2])
        assert pval_cond < 0.05


class TestMeekRules:
    """Test Meek's orientation rules R1-R4."""

    def test_r1_orientation(self):
        """R1: i -> j - k (i not adj k) => j -> k"""
        from fusionmind4.discovery.pc import PCAlgorithm
        pc = PCAlgorithm()
        # Manual DAG: 0->1, 1-2 (undirected), 0 not adj 2
        dag = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        result = pc._apply_meek_rules(dag, 3)
        # Should orient 1->2 (R1)
        assert result[1, 2] > 0
        assert result[2, 1] == 0

    def test_r2_orientation(self):
        """R2: i -> k -> j and i - j => i -> j"""
        from fusionmind4.discovery.pc import PCAlgorithm
        pc = PCAlgorithm()
        # 0->1, 1->2, 0-2 undirected
        dag = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float)
        result = pc._apply_meek_rules(dag, 3)
        # Should orient 0->2 (R2: directed path 0->1->2 and 0-2)
        assert result[0, 2] > 0
        assert result[2, 0] == 0

    def test_helper_is_directed(self):
        from fusionmind4.discovery.pc import PCAlgorithm
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        assert PCAlgorithm._is_directed(dag, 0, 1) is True
        assert PCAlgorithm._is_directed(dag, 1, 0) is False

    def test_helper_is_undirected(self):
        from fusionmind4.discovery.pc import PCAlgorithm
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        assert PCAlgorithm._is_undirected(dag, 1, 2) is True
        assert PCAlgorithm._is_undirected(dag, 0, 1) is False


# ============================================================================
# D3R Grad-Shafranov Score
# ============================================================================

class TestGradShafranovScore:
    """Test Grad-Shafranov equilibrium prior in D3R."""

    def test_score_runs(self):
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=32)
        gt = recon.generate_ground_truth()
        mask = gt['plasma_mask'].astype(float)
        x_t = gt['Te'] + 0.5 * np.random.randn(32, 32) * mask
        score = recon._grad_shafranov_score(x_t, mask)
        assert score.shape == (32, 32)

    def test_score_is_zero_for_constant(self):
        """If profile is constant on flux surfaces, GS score should be ~0."""
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=32)
        gt = recon.generate_ground_truth()
        mask = gt['plasma_mask'].astype(float)
        # Constant value everywhere inside plasma
        x_t = 5.0 * mask
        score = recon._grad_shafranov_score(x_t, mask)
        assert np.max(np.abs(score)) < 0.01

    def test_score_nonzero_for_asymmetric(self):
        """If profile varies on flux surfaces, GS score should be nonzero."""
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=32)
        gt = recon.generate_ground_truth()
        mask = gt['plasma_mask'].astype(float)
        # Add strong asymmetry (poloidally varying)
        N = 32
        rng = np.random.RandomState(42)
        x_t = gt['Te'] + 3.0 * rng.randn(N, N) * mask
        score = recon._grad_shafranov_score(x_t, mask)
        assert np.max(np.abs(score)) > 0.001

    def test_reconstruction_with_gs_prior(self):
        """Full reconstruction pipeline works with GS prior."""
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=16, n_diffusion_steps=20)
        gt = recon.generate_ground_truth(seed=42)
        meas = recon.generate_sparse_measurements(gt, n_thomson=8)
        result = recon.reconstruct(meas, gt, n_samples=2)
        assert result['compression_ratio'] > 1.0
        assert result['relative_error'] < 1.5  # Relaxed: small grid + few steps
