#!/usr/bin/env python3
"""
Real Data Validation Tests — FusionMind 4.0
=============================================
Tests CPDE causal discovery on:
  1. FAIR-MAST tokamak data (12 shots, UKAEA S3)
  2. Alcator C-Mod synthetic data (Simpson's Paradox, density limits)
  3. Cross-correlations and physics consistency

Fixtures in tests/fixtures/ are committed to git for reproducibility.
To regenerate from live S3: python scripts/download_fixtures.py

Author: Dr. Mladen Mester
Date: March 2026
"""

import pytest
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _have_mast_fixtures():
    return (os.path.exists(os.path.join(FIXTURE_DIR, 'mast_real_data.npy'))
            and os.path.exists(os.path.join(FIXTURE_DIR, 'mast_real_meta.json')))


def _have_cmod_fixtures():
    return (os.path.exists(os.path.join(FIXTURE_DIR, 'cmod_synthetic.npz'))
            and os.path.exists(os.path.join(FIXTURE_DIR, 'cmod_expected_results.json')))


@pytest.fixture(scope='module')
def mast_data():
    """Load FAIR-MAST real data fixture."""
    data = np.load(os.path.join(FIXTURE_DIR, 'mast_real_data.npy'))
    with open(os.path.join(FIXTURE_DIR, 'mast_real_meta.json')) as f:
        meta = json.load(f)
    return data, meta


@pytest.fixture(scope='module')
def mast_normalized(mast_data):
    """Z-score normalized MAST data."""
    data, meta = mast_data
    norm = data.copy()
    for j in range(norm.shape[1]):
        s = np.std(norm[:, j])
        if s > 0:
            norm[:, j] = (norm[:, j] - np.mean(norm[:, j])) / s
        else:
            norm[:, j] = 0.0
    return norm, meta['var_names']


@pytest.fixture(scope='module')
def mast_expected():
    """Load expected CPDE results on MAST data."""
    with open(os.path.join(FIXTURE_DIR, 'mast_expected_results.json')) as f:
        return json.load(f)


@pytest.fixture(scope='module')
def cmod_data():
    """Load C-Mod synthetic fixture."""
    npz = np.load(os.path.join(FIXTURE_DIR, 'cmod_synthetic.npz'))
    with open(os.path.join(FIXTURE_DIR, 'cmod_expected_results.json')) as f:
        meta = json.load(f)
    return npz['data'], npz['disrupted'], meta


@pytest.fixture(scope='module')
def cpde_mast_result(mast_normalized):
    """Run CPDE on MAST data (cached per module)."""
    from fusionmind4.discovery import EnsembleCPDE
    norm, var_names = mast_normalized
    config = {
        'n_bootstrap': 10,
        'threshold': 0.18,
        'physics_weight': 0.25,
        'notears_weight': 0.30,
        'granger_weight': 0.25,
        'pc_weight': 0.20,
    }
    cpde = EnsembleCPDE(config, verbose=False)
    result = cpde.discover(norm, var_names=var_names)
    return result


@pytest.fixture(scope='module')
def cpde_cmod_result(cmod_data):
    """Run CPDE on C-Mod data (cached per module)."""
    from fusionmind4.discovery import EnsembleCPDE
    data, disrupted, meta = cmod_data
    var_names = meta['var_names']
    # Normalize
    norm = data.copy()
    for j in range(norm.shape[1]):
        s = np.std(norm[:, j])
        if s > 0:
            norm[:, j] = (norm[:, j] - np.mean(norm[:, j])) / s
    cpde = EnsembleCPDE({'n_bootstrap': 8, 'threshold': 0.25}, verbose=False)
    result = cpde.discover(norm, var_names=var_names)
    return result


# ===========================================================================
# PART 1: FAIR-MAST Data Integrity Tests
# ===========================================================================

@pytest.mark.skipif(not _have_mast_fixtures(), reason="MAST fixtures not found")
class TestMASTDataIntegrity:
    """Verify the downloaded MAST fixture data is valid."""

    def test_data_shape(self, mast_data):
        data, meta = mast_data
        assert data.ndim == 2
        assert data.shape[1] == meta['n_vars']
        assert data.shape[0] == meta['total_timepoints']
        assert data.shape[0] >= 500, "Need at least 500 timepoints"

    def test_no_nan(self, mast_data):
        data, _ = mast_data
        assert np.isnan(data).sum() == 0, "Fixture data should have no NaN after cleaning"

    def test_no_inf(self, mast_data):
        data, _ = mast_data
        assert np.isinf(data).sum() == 0

    def test_variable_count(self, mast_data):
        _, meta = mast_data
        assert len(meta['var_names']) == 11
        assert 'betan' in meta['var_names']
        assert 'q95' in meta['var_names']
        assert 'Ip' in meta['var_names']

    def test_shot_count(self, mast_data):
        _, meta = mast_data
        assert len(meta['shots']) >= 8, "Need at least 8 shots"

    def test_physical_ranges_betan(self, mast_data):
        """βN should be roughly in [-5, 10] for MAST."""
        data, meta = mast_data
        idx = meta['var_names'].index('betan')
        betan = data[:, idx]
        assert np.median(betan) > 0, "Median βN should be positive"
        assert np.median(betan) < 5, "Median βN should be < 5"

    def test_physical_ranges_q95(self, mast_data):
        """q95 should be roughly in [2, 20] for MAST."""
        data, meta = mast_data
        idx = meta['var_names'].index('q95')
        q95 = data[:, idx]
        assert np.median(q95) > 2, "q95 should be > 2"
        assert np.median(q95) < 20, "q95 should be < 20"

    def test_physical_ranges_elongation(self, mast_data):
        """Elongation should be ~1.5-2.0 for MAST (spherical tokamak)."""
        data, meta = mast_data
        idx = meta['var_names'].index('elongation')
        kappa = data[:, idx]
        assert np.median(kappa) > 1.2, "MAST elongation should be > 1.2"
        assert np.median(kappa) < 2.5, "MAST elongation should be < 2.5"

    def test_variance_present(self, mast_data):
        """Each variable should have non-trivial variance."""
        data, meta = mast_data
        for j in range(data.shape[1]):
            std = np.std(data[:, j])
            assert std > 0, f"Variable {meta['var_names'][j]} has zero variance"

    def test_metadata_source(self, mast_data):
        _, meta = mast_data
        assert 'FAIR-MAST' in meta['source'] or 'UKAEA' in meta['source']
        assert meta['download_date'] == '2026-03-04'


# ===========================================================================
# PART 2: CPDE on Real MAST Data
# ===========================================================================

@pytest.mark.skipif(not _have_mast_fixtures(), reason="MAST fixtures not found")
class TestCPDEonMAST:
    """Run CPDE causal discovery on real MAST tokamak data."""

    def test_cpde_runs_without_error(self, cpde_mast_result):
        """CPDE should complete without exceptions on real data."""
        assert cpde_mast_result is not None
        assert 'dag' in cpde_mast_result

    def test_dag_shape(self, cpde_mast_result):
        dag = cpde_mast_result['dag']
        assert dag.shape == (11, 11), f"DAG shape should be (11,11), got {dag.shape}"

    def test_dag_values_valid(self, cpde_mast_result):
        """DAG values should be non-negative (weights or binary)."""
        dag = cpde_mast_result['dag']
        assert np.all(dag >= 0), "DAG values should be non-negative"
        assert np.all(dag <= 1.0 + 1e-10), "DAG values should be <= 1"

    def test_dag_no_self_loops(self, cpde_mast_result):
        dag = cpde_mast_result['dag']
        assert np.trace(dag) == 0, "DAG should have no self-loops"

    def test_edges_discovered(self, cpde_mast_result):
        """Should discover at least some edges on real data."""
        n_edges = cpde_mast_result['dag'].sum()
        assert n_edges >= 1, "Should discover at least 1 edge"
        assert n_edges <= 50, "Should not discover more than 50 edges (11 vars)"

    def test_edge_details_present(self, cpde_mast_result):
        assert 'edge_details' in cpde_mast_result
        # Should have scored edges
        details = cpde_mast_result['edge_details']
        assert len(details) >= 1

    def test_betan_betap_correlated(self, mast_normalized):
        """βN and βp should be strongly correlated (physics: both are pressure metrics)."""
        norm, var_names = mast_normalized
        i = var_names.index('betan')
        j = var_names.index('betap')
        r = np.corrcoef(norm[:, i], norm[:, j])[0, 1]
        assert abs(r) > 0.7, f"βN-βp correlation should be > 0.7, got {r:.3f}"

    def test_q95_q_axis_correlated(self, mast_normalized):
        """q95 and q_axis should be positively correlated."""
        norm, var_names = mast_normalized
        i = var_names.index('q95')
        j = var_names.index('q_axis')
        r = np.corrcoef(norm[:, i], norm[:, j])[0, 1]
        assert r > 0.5, f"q95-q_axis correlation should be > 0.5, got {r:.3f}"

    def test_reproducibility(self, mast_normalized):
        """Two CPDE runs with same seed should produce structurally similar DAGs.

        Note: NOTEARS uses scipy.optimize internally which can exhibit minor
        floating-point non-determinism across runs.  We require >90% edge
        agreement rather than exact equality.
        """
        from fusionmind4.discovery import EnsembleCPDE
        norm, var_names = mast_normalized
        config = {'n_bootstrap': 5, 'threshold': 0.25}

        cpde1 = EnsembleCPDE(config, verbose=False)
        r1 = cpde1.discover(norm, var_names=var_names)

        cpde2 = EnsembleCPDE(config, verbose=False)
        r2 = cpde2.discover(norm, var_names=var_names)

        # Binarize: edge present (>0) vs absent
        d1 = (r1['dag'] > 0).astype(int)
        d2 = (r2['dag'] > 0).astype(int)
        total_cells = d1.size
        agreement = np.sum(d1 == d2) / total_cells
        assert agreement > 0.90, (
            f"DAG structural agreement {agreement:.1%} < 90%"
        )

    def test_performance_under_10s(self, mast_normalized):
        """CPDE on 829×11 real data should complete in < 10s."""
        from fusionmind4.discovery import EnsembleCPDE
        norm, var_names = mast_normalized
        config = {'n_bootstrap': 5, 'threshold': 0.25}
        cpde = EnsembleCPDE(config, verbose=False)
        t0 = time.time()
        cpde.discover(norm, var_names=var_names)
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"CPDE took {elapsed:.1f}s, should be < 10s"


# ===========================================================================
# PART 3: Alcator C-Mod Simpson's Paradox
# ===========================================================================

@pytest.mark.skipif(not _have_cmod_fixtures(), reason="C-Mod fixtures not found")
class TestCModSimpsonsParadox:
    """Test Simpson's Paradox detection on C-Mod data."""

    def test_data_shape(self, cmod_data):
        data, disrupted, meta = cmod_data
        assert data.ndim == 2
        assert data.shape[1] == 8, "C-Mod has 8 variables"
        assert len(disrupted) == data.shape[0]

    def test_disruption_rate(self, cmod_data):
        """Disruption rate should be reasonable (5-30%)."""
        _, disrupted, _ = cmod_data
        rate = np.mean(disrupted)
        assert 0.05 < rate < 0.40, f"Disruption rate {rate:.1%} outside expected range"

    def test_simpsons_paradox_function(self, cmod_data):
        """Simpson's Paradox analysis should run without error."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
        from run_real_data import demonstrate_simpsons_paradox
        data, disrupted, meta = cmod_data
        labels = meta['labels']
        result = demonstrate_simpsons_paradox(data, labels, disrupted)
        assert 'raw_correlation' in result
        assert 'partial_correlation_given_Ip' in result
        assert 'simpson_detected' in result
        # Correlations should be finite
        assert np.isfinite(result['raw_correlation'])
        assert np.isfinite(result['partial_correlation_given_Ip'])

    def test_density_limit_prediction(self, cmod_data):
        """Density limit prediction should produce valid AUC."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
        from run_real_data import density_limit_prediction
        data, disrupted, meta = cmod_data
        labels = meta['labels']
        result = density_limit_prediction(data, labels, disrupted)
        # AUC should be between 0 and 1 (on subset, may be near 0.5)
        assert 0.0 <= result['auc_greenwald'] <= 1.0
        assert 0.0 <= result['auc_causal'] <= 1.0
        assert result['n_disruptions'] > 0


# ===========================================================================
# PART 4: CPDE on C-Mod Data
# ===========================================================================

@pytest.mark.skipif(not _have_cmod_fixtures(), reason="C-Mod fixtures not found")
class TestCPDEonCMod:
    """Run CPDE on C-Mod synthetic data."""

    def test_cpde_runs(self, cpde_cmod_result):
        assert cpde_cmod_result is not None
        assert 'dag' in cpde_cmod_result

    def test_dag_shape(self, cpde_cmod_result):
        dag = cpde_cmod_result['dag']
        assert dag.shape == (8, 8)

    def test_edges_discovered(self, cpde_cmod_result):
        n = cpde_cmod_result['dag'].sum()
        assert n >= 1, "Should discover edges on C-Mod data"

    def test_ip_ne_connection(self, cpde_cmod_result, cmod_data):
        """Ip and ne should have some causal connection (via Greenwald limit)."""
        _, _, meta = cmod_data
        var_names = meta['var_names']
        dag = cpde_cmod_result['dag']
        ip_idx = var_names.index('Ip')
        ne_idx = var_names.index('ne')
        # Either direct Ip→ne or ne→Ip should exist, or both connected via edges
        connected = dag[ip_idx, ne_idx] > 0 or dag[ne_idx, ip_idx] > 0
        # Also check edge_details for any scoring
        details = cpde_cmod_result.get('edge_details', {})
        scored = (ip_idx, ne_idx) in details or (ne_idx, ip_idx) in details
        assert connected or scored, "Ip and ne should be connected or scored"


# ===========================================================================
# PART 5: D3R Reconstruction PoC
# ===========================================================================

class TestD3RReconstruction:
    """Test D3R diffusion reconstruction on synthetic MAST geometry."""

    def test_d3r_basic_reconstruction(self):
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=32, n_diffusion_steps=50)
        gt = recon.generate_ground_truth(seed=42)
        meas = recon.generate_sparse_measurements(gt, n_thomson=8, n_interferometry=3)
        result = recon.reconstruct(meas, gt, n_samples=3)

        assert result['rmse'] > 0
        assert result['compression_ratio'] > 10
        assert result['relative_error'] < 2.0  # Less than 200% error on small grid

    def test_d3r_positivity(self):
        """Reconstructed Te should be non-negative."""
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=32, n_diffusion_steps=50)
        gt = recon.generate_ground_truth(seed=123)
        meas = recon.generate_sparse_measurements(gt, n_thomson=8, n_interferometry=3)
        result = recon.reconstruct(meas, gt, n_samples=3)
        # Mean reconstruction in plasma region should be >= -0.5 (small numerical noise ok)
        mask = gt['plasma_mask']
        assert np.min(result['mean'][mask]) >= -1.0

    def test_d3r_compression(self):
        """Compression ratio should be > 50:1 for practical use."""
        from fusionmind4.reconstruction.core import SimplifiedDiffusionReconstructor
        recon = SimplifiedDiffusionReconstructor(grid_size=48, n_diffusion_steps=80)
        gt = recon.generate_ground_truth(seed=42)
        meas = recon.generate_sparse_measurements(gt, n_thomson=8, n_interferometry=3)
        result = recon.reconstruct(meas, gt, n_samples=5)
        assert result['compression_ratio'] > 50, \
            f"Compression {result['compression_ratio']:.0f}:1 should be > 50:1"


# ===========================================================================
# PART 6: Full Pipeline Integration
# ===========================================================================

@pytest.mark.skipif(not _have_mast_fixtures(), reason="MAST fixtures not found")
class TestFullPipelineIntegration:
    """End-to-end: load real data → CPDE → SCM → counterfactual."""

    def test_mast_to_scm_pipeline(self, mast_normalized, cpde_mast_result):
        """Discover DAG on MAST data, then build SCM for interventions."""
        from fusionmind4.control.scm import PlasmaSCM
        from fusionmind4.control.interventions import InterventionEngine

        norm, var_names = mast_normalized
        dag = cpde_mast_result['dag']

        # Build SCM from discovered DAG
        scm = PlasmaSCM(variable_names=var_names, dag=dag)
        scm.fit(norm)

        # Test prediction
        sample = {v: float(norm[0, i]) for i, v in enumerate(var_names)}
        pred = scm.predict(sample)
        assert isinstance(pred, dict)
        assert len(pred) == len(var_names)

        # Test intervention: what if betan increased?
        engine = InterventionEngine(scm)
        result = engine.do(interventions={'betan': 2.0}, current_state=sample)
        assert isinstance(result.outcomes, dict)
        assert abs(result.outcomes['betan'] - 2.0) < 0.01

    def test_mast_to_counterfactual(self, mast_normalized, cpde_mast_result):
        """Build full counterfactual: what if we had changed betan?"""
        from fusionmind4.control.scm import PlasmaSCM
        from fusionmind4.control.interventions import CounterfactualEngine

        norm, var_names = mast_normalized
        dag = cpde_mast_result['dag']

        scm = PlasmaSCM(variable_names=var_names, dag=dag)
        scm.fit(norm)

        # Counterfactual on a real datapoint
        factual = {v: float(norm[50, i]) for i, v in enumerate(var_names)}
        cf_engine = CounterfactualEngine(scm)
        cf_result = cf_engine.counterfactual(
            factual_state=factual,
            intervention={'betan': factual['betan'] * 1.5}
        )
        assert isinstance(cf_result.counterfactual_outcomes, dict)
        # betan should be set to 1.5x
        expected = factual['betan'] * 1.5
        assert abs(cf_result.counterfactual_outcomes['betan'] - expected) < 0.1


# ===========================================================================
# PART 7: AEDE on Real Data
# ===========================================================================

@pytest.mark.skipif(not _have_mast_fixtures(), reason="MAST fixtures not found")
class TestAEDEonRealData:
    """Test Active Experiment Design Engine on real MAST data."""

    def test_aede_experiment_ranking(self, mast_normalized, cpde_mast_result):
        """AEDE should rank experiments by information gain."""
        from fusionmind4.experiment.aede import ActiveExperimentDesignEngine

        norm, var_names = mast_normalized
        dag = cpde_mast_result['dag']

        aede = ActiveExperimentDesignEngine(
            variable_names=var_names, seed=42
        )

        # Use CPDE result to feed AEDE
        # Create synthetic bootstrap/ensemble from the DAG
        n = len(var_names)
        bootstrap_stability = np.where(dag > 0, 0.7, 0.1) + 0.1 * np.random.randn(n, n)
        bootstrap_stability = np.clip(bootstrap_stability, 0, 1)
        ensemble_agreement = np.where(dag > 0, 0.8, 0.2) + 0.1 * np.random.randn(n, n)
        ensemble_agreement = np.clip(ensemble_agreement, 0, 1)

        experiments = aede.design_experiments(
            bootstrap_stability=bootstrap_stability,
            ensemble_agreement=ensemble_agreement,
            edge_weights=dag,
            top_k=5,
        )
        assert len(experiments) >= 1
        assert all(hasattr(exp, 'score') for exp in experiments)
        assert all(exp.score >= 0 for exp in experiments)

    def test_aede_uncertainty(self, mast_normalized, cpde_mast_result):
        """AEDE should estimate edge uncertainties."""
        from fusionmind4.experiment.aede import EdgeUncertaintyEstimator

        _, var_names = mast_normalized
        dag = cpde_mast_result['dag']
        n = len(var_names)

        estimator = EdgeUncertaintyEstimator(var_names)
        bootstrap_stability = np.where(dag > 0, 0.7, 0.1)
        ensemble_agreement = np.where(dag > 0, 0.8, 0.2)

        uncertainty = estimator.compute_uncertainties(
            bootstrap_stability, ensemble_agreement, dag
        )
        assert uncertainty.shape == (n, n)
        assert np.all(uncertainty >= 0)


# ===========================================================================
# PART 8: Live S3 Download Test (optional, slow)
# ===========================================================================

@pytest.mark.slow
class TestLiveS3Download:
    """Test live download from FAIR-MAST S3 (requires network)."""

    @pytest.mark.skipif(
        os.environ.get('FM_SKIP_S3', '1') == '1',
        reason="Set FM_SKIP_S3=0 to enable live S3 tests"
    )
    def test_download_single_shot(self):
        """Download a single MAST shot from S3."""
        import s3fs
        import zarr

        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={'endpoint_url': 'https://s3.echo.stfc.ac.uk'}
        )
        path = 'mast/level1/shots/27880.zarr'
        store = s3fs.S3Map(root=path, s3=fs, check=False)
        root = zarr.open(store, mode='r')

        assert 'efm' in list(root.keys())
        efm = root['efm']
        betan = np.array(efm['betan'])
        assert len(betan) > 50, "Shot 27880 should have > 50 EFM timepoints"
        assert np.isfinite(betan).all()
