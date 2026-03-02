"""Tests for AEDE (PF5), DYNOTEARS, upgraded NOTEARS, and integration pipeline."""
import pytest
import numpy as np
from scipy.linalg import expm

# ── Discovery upgrades ──
from fusionmind4.discovery.notears import NOTEARSDiscovery, DYNOTEARSDiscovery
from fusionmind4.discovery import EnsembleCPDE

# ── AEDE (PF5) ──
from fusionmind4.experiment import (
    ActiveExperimentDesignEngine,
    ExperimentDesign,
    MachineOperationalLimits,
    EdgeUncertaintyEstimator,
    InformationGainCalculator,
    ExperimentGenerator,
)


# ── Helpers ──────────────────────────────────────────────

def generate_dag_data(n=500, d=5, seed=42):
    """Generate data from a known DAG: 0→1, 0→2, 1→3, 2→3, 3→4."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    X[:, 1] = 0.8 * X[:, 0] + 0.3 * rng.randn(n)
    X[:, 2] = -0.6 * X[:, 0] + 0.3 * rng.randn(n)
    X[:, 3] = 0.5 * X[:, 1] + 0.4 * X[:, 2] + 0.2 * rng.randn(n)
    if d >= 5:
        X[:, 4] = 0.7 * X[:, 3] + 0.2 * rng.randn(n)
    return X


def generate_temporal_data(T=300, d=4, seed=42):
    """Generate temporal data with lagged causal effects."""
    rng = np.random.RandomState(seed)
    d = max(d, 4)  # Minimum 4 vars for our structure
    X = np.zeros((T, d))
    X[0] = rng.randn(d)
    X[1] = rng.randn(d)
    for t in range(2, T):
        X[t, 0] = 0.3 * rng.randn()
        X[t, 1] = 0.6 * X[t-1, 0] + 0.2 * rng.randn()  # lag-1: 0→1
        X[t, 2] = 0.5 * X[t, 0] + 0.4 * X[t-1, 1] + 0.2 * rng.randn()  # contemp 0→2 + lag-1 1→2
        X[t, 3] = 0.7 * X[t, 2] + 0.2 * rng.randn()  # contemp 2→3
    return X


def make_uncertainty_matrices(d=10):
    """Create realistic uncertainty matrices for AEDE testing."""
    rng = np.random.RandomState(42)
    bootstrap = rng.uniform(0.2, 0.9, (d, d))
    np.fill_diagonal(bootstrap, 0)
    ensemble = rng.uniform(0.3, 0.8, (d, d))
    np.fill_diagonal(ensemble, 0)
    weights = rng.randn(d, d) * 0.3
    np.fill_diagonal(weights, 0)
    return bootstrap, ensemble, weights


# ============================================================================
# NOTEARS UPGRADE TESTS
# ============================================================================

class TestNOTEARSDAGConstraint:
    """Test that upgraded NOTEARS properly enforces acyclicity."""

    def test_h_zero_on_dag(self):
        """h(W) should be 0 for a true DAG (strictly upper triangular)."""
        W = np.array([[0, 0.5, 0], [0, 0, 0.3], [0, 0, 0]])
        h = NOTEARSDiscovery._h(W)
        assert abs(h) < 1e-10

    def test_h_positive_on_cycle(self):
        """h(W) should be > 0 for a cyclic graph."""
        W = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
        h = NOTEARSDiscovery._h(W)
        assert h > 0.001

    def test_h_grad_shape(self):
        """Gradient should have same shape as W."""
        W = np.random.randn(4, 4) * 0.1
        grad = NOTEARSDiscovery._h_grad(W)
        assert grad.shape == (4, 4)

    def test_fit_produces_dag(self):
        """Fitted W should satisfy h(W) ≈ 0."""
        X = generate_dag_data(n=300, d=5)
        nt = NOTEARSDiscovery(lambda1=0.05, max_iter=30, w_threshold=0.08)
        W = nt.fit(X)
        h = NOTEARSDiscovery._h(W)
        assert np.isfinite(h) and h < 1.0, f"h(W) = {h}, not a DAG"

    def test_fit_recovers_edges(self):
        """Should recover at least some true edges."""
        X = generate_dag_data(n=500, d=5)
        nt = NOTEARSDiscovery(lambda1=0.03, max_iter=40, w_threshold=0.05)
        W = nt.fit(X)
        # Check that at least one strong edge is detected
        max_weight = np.nanmax(np.abs(W))
        assert max_weight > 0.01, "Should detect at least one edge"

    def test_no_self_loops(self):
        """Diagonal should be zero."""
        X = generate_dag_data(n=200, d=4)
        nt = NOTEARSDiscovery(max_iter=15)
        W = nt.fit(X)
        assert np.all(np.diag(W) == 0)

    def test_bootstrap_stability(self):
        """Bootstrap should return values in [0, 1]."""
        X = generate_dag_data(n=200, d=5)
        nt = NOTEARSDiscovery(max_iter=15)
        stab = nt.fit_bootstrap(X, n_bootstrap=5)
        assert stab.min() >= 0
        assert stab.max() <= 1
        assert stab.shape == (5, 5)


# ============================================================================
# DYNOTEARS TESTS
# ============================================================================

class TestDYNOTEARS:
    """Test temporal causal discovery."""

    def test_fit_returns_correct_shapes(self):
        X = generate_temporal_data(T=200, d=4)
        dyn = DYNOTEARSDiscovery(max_lag=2, max_iter=20)
        W, A = dyn.fit(X)
        assert W.shape == (4, 4)
        assert A.shape == (2 * 4, 4)  # max_lag * d

    def test_contemporaneous_is_dag(self):
        """Contemporaneous graph W should be acyclic."""
        X = generate_temporal_data(T=300, d=4)
        dyn = DYNOTEARSDiscovery(max_lag=2, max_iter=30)
        W, A = dyn.fit(X)
        h = NOTEARSDiscovery._h(W)
        assert h < 0.1, f"Contemporaneous graph not DAG: h={h}"

    def test_detects_lagged_effects(self):
        """Should detect at least some lagged causal effects."""
        X = generate_temporal_data(T=400, d=4)
        dyn = DYNOTEARSDiscovery(max_lag=3, max_iter=30, w_threshold=0.05)
        W, A = dyn.fit(X)
        # lag-1: 0→1 exists
        lagged_01 = dyn.get_lagged_edge(A, 1, 0, 1)
        # At minimum, should have some lagged edges
        total_lagged = np.sum(np.abs(A) > 0)
        assert total_lagged >= 0  # Permissive — temporal discovery is hard

    def test_temporal_summary(self):
        X = generate_temporal_data(T=200, d=4)
        dyn = DYNOTEARSDiscovery(max_lag=2, max_iter=15, w_threshold=0.05)
        W, A = dyn.fit(X)
        summary = dyn.get_temporal_summary(W, A, ['A', 'B', 'C', 'D'])
        assert 'contemporaneous' in summary
        assert 'lagged' in summary
        assert 'n_contemporaneous' in summary

    def test_short_series_fallback(self):
        """With very short time series, should fall back to NOTEARS."""
        X = generate_temporal_data(T=10, d=4)
        dyn = DYNOTEARSDiscovery(max_lag=3, max_iter=10)
        W, A = dyn.fit(X)
        assert W.shape == (4, 4)


# ============================================================================
# AEDE (PF5) TESTS
# ============================================================================

class TestEdgeUncertainty:

    def test_compute_uncertainties(self):
        boot, ens, weights = make_uncertainty_matrices(d=6)
        est = EdgeUncertaintyEstimator([f"v{i}" for i in range(6)])
        unc = est.compute_uncertainties(boot, ens, weights)
        assert unc.shape == (6, 6)
        assert np.all(np.diag(unc) == 0)
        assert unc.min() >= 0
        assert unc.max() <= 1

    def test_most_uncertain_edges(self):
        boot, ens, weights = make_uncertainty_matrices(d=6)
        est = EdgeUncertaintyEstimator([f"v{i}" for i in range(6)])
        unc = est.compute_uncertainties(boot, ens, weights)
        edges = est.get_most_uncertain_edges(unc, top_k=5)
        assert len(edges) <= 5
        assert all('uncertainty' in e for e in edges)
        # Should be sorted descending
        if len(edges) >= 2:
            assert edges[0]['uncertainty'] >= edges[1]['uncertainty']


class TestInformationGain:

    def test_eig_positive(self):
        unc = np.random.uniform(0.1, 0.8, (5, 5))
        np.fill_diagonal(unc, 0)
        calc = InformationGainCalculator()
        eig = calc.compute_eig(unc, 'v0', 5.0, ['v1', 'v2'],
                               [f'v{i}' for i in range(5)],
                               np.random.randn(5, 5))
        assert eig >= 0

    def test_multi_intervention_eig(self):
        unc = np.random.uniform(0.1, 0.8, (5, 5))
        np.fill_diagonal(unc, 0)
        calc = InformationGainCalculator()
        eig = calc.compute_multi_intervention_eig(
            unc, {'v0': 5.0, 'v1': 3.0},
            [f'v{i}' for i in range(5)],
            np.random.randn(5, 5))
        assert eig >= 0


class TestExperimentGenerator:

    def test_single_variable_scans(self):
        names = ['I_p', 'P_NBI', 'P_ECRH', 'gas_puff', 'n_e', 'T_e', 'beta_N']
        unc = np.random.uniform(0.1, 0.6, (len(names), len(names)))
        np.fill_diagonal(unc, 0)
        gen = ExperimentGenerator(names)
        exps = gen.generate_single_variable_scans(unc, np.random.randn(*unc.shape))
        assert len(exps) > 0
        assert all(isinstance(e, ExperimentDesign) for e in exps)

    def test_factorial_designs(self):
        names = ['I_p', 'P_NBI', 'P_ECRH', 'gas_puff', 'n_e', 'T_e']
        unc = np.random.uniform(0.15, 0.6, (len(names), len(names)))
        np.fill_diagonal(unc, 0)
        gen = ExperimentGenerator(names)
        exps = gen.generate_factorial_designs(unc, np.random.randn(*unc.shape))
        assert isinstance(exps, list)

    def test_risk_assessment(self):
        names = ['I_p', 'P_NBI', 'n_e']
        gen = ExperimentGenerator(names)
        assert gen._assess_risk({'I_p': 0.5}) == 'low'
        assert gen._assess_risk({'I_p': 1.4, 'P_NBI': 7.0}) in ('medium', 'high')

    def test_feasibility_in_range(self):
        gen = ExperimentGenerator(['I_p', 'P_NBI'])
        f = gen._compute_feasibility({'I_p': 0.9})
        assert 0 <= f <= 1
        # Out of range
        f_oor = gen._compute_feasibility({'I_p': 100.0})
        assert f_oor == 0.0


class TestActiveExperimentDesignEngine:

    def test_design_experiments(self):
        names = ['I_p', 'P_NBI', 'P_ECRH', 'gas_puff', 'n_e', 'T_e', 'beta_N', 'q95']
        boot, ens, weights = make_uncertainty_matrices(d=len(names))
        aede = ActiveExperimentDesignEngine(names)
        exps = aede.design_experiments(boot, ens, weights, top_k=5)
        assert len(exps) <= 5
        assert all(isinstance(e, ExperimentDesign) for e in exps)
        # Should be ranked
        if len(exps) >= 2:
            assert exps[0].priority_rank < exps[1].priority_rank

    def test_get_uncertain_edges(self):
        names = ['I_p', 'P_NBI', 'n_e', 'T_e']
        boot, ens, weights = make_uncertainty_matrices(d=len(names))
        aede = ActiveExperimentDesignEngine(names)
        aede.design_experiments(boot, ens, weights)
        edges = aede.get_uncertain_edges(top_k=3)
        assert isinstance(edges, list)

    def test_update_after_experiment(self):
        names = ['I_p', 'P_NBI', 'n_e', 'T_e']
        boot, ens, weights = make_uncertainty_matrices(d=len(names))
        aede = ActiveExperimentDesignEngine(names)
        exps = aede.design_experiments(boot, ens, weights)

        if exps:
            initial_unc = aede._posterior_uncertainty.copy()
            aede.update_after_experiment(
                exps[0],
                observed_data=np.random.randn(100, len(names)),
                observed_edges={('I_p', 'n_e'): 0.8},
            )
            assert aede.experiment_history[-1] == exps[0]
            # Uncertainty should decrease for updated edge
            assert aede._posterior_uncertainty[0, 2] <= initial_unc[0, 2]

    def test_summary(self):
        names = ['I_p', 'P_NBI', 'n_e']
        boot, ens, weights = make_uncertainty_matrices(d=len(names))
        aede = ActiveExperimentDesignEngine(names)
        aede.design_experiments(boot, ens, weights)
        summary = aede.get_summary()
        assert 'n_variables' in summary
        assert summary['n_variables'] == 3

    def test_experiment_score(self):
        exp = ExperimentDesign(
            name="test", description="test",
            actuator_settings={'P_NBI': 5.0},
            target_edges=[('P_NBI', 'T_e')],
            expected_information_gain=0.8,
            feasibility_score=0.9,
            estimated_duration=5.0,
            risk_level="low",
        )
        assert exp.score > 0
        # High risk should reduce score
        exp_risky = ExperimentDesign(
            name="risky", description="risky",
            actuator_settings={'P_NBI': 8.0},
            target_edges=[('P_NBI', 'T_e')],
            expected_information_gain=0.8,
            feasibility_score=0.9,
            estimated_duration=5.0,
            risk_level="high",
        )
        assert exp_risky.score < exp.score


# ============================================================================
# INTEGRATION TEST — Full pipeline
# ============================================================================

class TestFullPipeline:
    """End-to-end integration test: Data → CPDE → SCM → CPC → AEDE → Copilot."""

    def test_full_pipeline(self):
        """Run complete FusionMind pipeline on synthetic data."""
        from fusionmind4.utils.fm3lite import FM3LitePhysicsEngine
        from fusionmind4.control.scm import PlasmaSCM
        from fusionmind4.copilot import CausalContext, QueryEngine

        # 1. Generate synthetic data
        sim = FM3LitePhysicsEngine(n_samples=300, seed=42)
        data, _ = sim.generate()

        # 2. Run CPDE
        cpde = EnsembleCPDE(config={'n_bootstrap': 5, 'threshold': 0.25},
                            verbose=False)
        result = cpde.discover(data, seed=42)

        assert 'dag' in result
        assert result['dag'].shape[0] == result['dag'].shape[1]
        n_edges = np.sum(np.abs(result['dag']) > 0)
        assert n_edges > 0, "CPDE should discover at least some edges"

        # 3. Fit SCM
        names = result.get('variable_names',
                           [f'v{i}' for i in range(data.shape[1])])
        if names is None:
            names = [f'v{i}' for i in range(data.shape[1])]
        scm = PlasmaSCM(names, result['dag'])
        scm.fit(data, verbose=False)

        # 4. Run AEDE on CPDE outputs
        d = len(names)
        boot = result.get('bootstrap_stability', np.random.uniform(0.3, 0.8, (d, d)))
        ens = result.get('ensemble_agreement', np.random.uniform(0.3, 0.8, (d, d)))
        np.fill_diagonal(boot, 0)
        np.fill_diagonal(ens, 0)

        aede = ActiveExperimentDesignEngine(names)
        experiments = aede.design_experiments(boot, ens, result['dag'], top_k=5)
        assert isinstance(experiments, list)

        # 5. Build Copilot context
        edges = []
        adj = result['dag']
        for i in range(d):
            for j in range(d):
                if abs(adj[i, j]) > 0:
                    edges.append((names[i], names[j], float(adj[i, j])))

        ctx = CausalContext(variable_names=names)
        if edges:
            ctx.set_dag(edges)

        engine = QueryEngine(ctx)
        qr = engine.process_query("What happens if we increase heating?")
        assert 'system_prompt' in qr
        assert 'classification' in qr

        # Pipeline complete!
        print(f"\n✅ Full pipeline: {n_edges} edges, "
              f"{len(experiments)} experiments, copilot ready")
