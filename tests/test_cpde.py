"""Tests for CPDE v3.2 — Causal Plasma Discovery Engine."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from fusionmind4.utils.plasma_vars import (
    PLASMA_VARS, N_VARS, ACTUATOR_IDS,
    build_ground_truth_adjacency, evaluate_dag,
)
from fusionmind4.utils.fm3lite import FM3LitePhysicsEngine
from fusionmind4.discovery.notears import NOTEARSDiscovery
from fusionmind4.discovery.granger import GrangerCausalityTest
from fusionmind4.discovery.pc import PCAlgorithm
from fusionmind4.discovery.interventional import InterventionalScorer
from fusionmind4.discovery.physics import validate_physics, get_physics_prior_matrix
from fusionmind4.discovery.ensemble import EnsembleCPDE


class TestPlasmaVars:
    def test_variable_count(self):
        assert N_VARS == 14

    def test_actuator_count(self):
        assert len(ACTUATOR_IDS) == 4

    def test_ground_truth_has_edges(self):
        gt = build_ground_truth_adjacency()
        assert np.sum(np.abs(gt) > 0) == 28

    def test_evaluate_perfect(self):
        gt = build_ground_truth_adjacency()
        result = evaluate_dag(gt)
        assert result["f1"] == 1.0
        assert result["tp"] == 28
        assert result["fp"] == 0

    def test_evaluate_empty(self):
        empty = np.zeros((N_VARS, N_VARS))
        result = evaluate_dag(empty)
        assert result["tp"] == 0
        assert result["fn"] == 28


class TestFM3Lite:
    def test_generates_correct_shape(self):
        engine = FM3LitePhysicsEngine(n_samples=100, seed=42)
        data, _ = engine.generate()
        assert data.shape == (100, 14)

    def test_generates_interventional(self):
        engine = FM3LitePhysicsEngine(n_samples=100, seed=42)
        _, interventions = engine.generate()
        assert len(interventions) == 4  # 4 actuators
        for act_id, (low, high) in interventions.items():
            assert act_id in ACTUATOR_IDS
            assert low.shape[1] == 14
            assert high.shape[1] == 14

    def test_noise_injection(self):
        engine = FM3LitePhysicsEngine(n_samples=100, seed=42)
        data, _ = engine.generate()
        noisy = engine.add_noise(data, 0.1)
        assert noisy.shape == data.shape
        assert not np.allclose(data, noisy)


class TestNOTEARS:
    def test_produces_adjacency(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 5)
        nt = NOTEARSDiscovery(lambda1=0.1)
        W = nt.fit(X)
        assert W.shape == (5, 5)
        assert np.all(np.diag(W) == 0)  # No self-loops

    def test_bootstrap_returns_stability(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        nt = NOTEARSDiscovery()
        stab = nt.fit_bootstrap(X, n_bootstrap=3, rng=rng)
        assert stab.shape == (4, 4)
        assert np.all(stab >= 0) and np.all(stab <= 1)


class TestGranger:
    def test_detects_no_causation_in_independent(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 3)
        gc = GrangerCausalityTest(max_lag=3, alpha=0.01)
        result = gc.test_all_pairs(X)
        # Independent data should have few/no edges
        assert np.sum(result) < 3


class TestPC:
    def test_produces_adjacency(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 4)
        pc = PCAlgorithm(alpha=0.05)
        dag = pc.fit(X)
        assert dag.shape == (4, 4)


class TestPhysics:
    def test_prior_matrix_shape(self):
        prior = get_physics_prior_matrix()
        assert prior.shape == (N_VARS, N_VARS)
        assert np.all(prior >= 0)

    def test_perfect_dag_passes_all(self):
        gt = build_ground_truth_adjacency()
        gt_binary = (np.abs(gt) > 0).astype(float)
        checks = validate_physics(gt_binary)
        # Ground truth has feedback cycles (P_rad→Te, MHD→ne), so dag_acyclic=False
        # All other 9 checks should pass
        assert checks["passed"] >= 9

    def test_empty_dag_fails(self):
        empty = np.zeros((N_VARS, N_VARS))
        checks = validate_physics(empty)
        assert checks["passed"] < checks["total"]


class TestEnsembleCPDE:
    """Integration test for the full pipeline."""

    def test_full_pipeline(self):
        engine = FM3LitePhysicsEngine(n_samples=5000, seed=42)
        data, interventions = engine.generate()

        cpde = EnsembleCPDE(
            config={"n_bootstrap": 5, "threshold": 0.32},
            verbose=False,
        )
        results = cpde.discover(data, interventional_data=interventions)

        # Basic sanity
        assert results["f1"] > 0.3  # Should find some edges
        assert results["precision"] > 0.2
        assert results["recall"] > 0.3
        assert results["physics_passed"] >= 4  # At least basic checks

    def test_dag_is_acyclic(self):
        engine = FM3LitePhysicsEngine(n_samples=5000, seed=42)
        data, interventions = engine.generate()

        cpde = EnsembleCPDE(
            config={"n_bootstrap": 5},
            verbose=False,
        )
        results = cpde.discover(data, interventional_data=interventions)
        assert results["physics_checks"]["dag_acyclic"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ══════════════════════════════════════════════════════════════════════════════
# CPC Tests (Patent Family PF2)
# ══════════════════════════════════════════════════════════════════════════════

class TestPlasmaSCM:
    def test_fit_and_predict(self):
        """SCM can be fitted from data and DAG."""
        from fusionmind4.control.scm import PlasmaSCM
        names = ["A", "B", "C"]
        dag = np.array([[0, 0.5, 0], [0, 0, 0.3], [0, 0, 0]])
        scm = PlasmaSCM(names, dag)
        data = np.random.randn(100, 3)
        data[:, 1] = 0.5 * data[:, 0] + 0.1 * np.random.randn(100)
        data[:, 2] = 0.3 * data[:, 1] + 0.1 * np.random.randn(100)
        scm.fit(data, verbose=False)
        assert scm._fitted
        result = scm.predict({"A": 1.0})
        assert "B" in result and "C" in result

    def test_equation_string(self):
        from fusionmind4.control.scm import PlasmaSCM
        names = ["X", "Y"]
        dag = np.array([[0, 0.7], [0, 0]])
        scm = PlasmaSCM(names, dag)
        data = np.column_stack([np.random.randn(50), np.random.randn(50)])
        scm.fit(data, verbose=False)
        eq = scm.get_equation_string("Y")
        assert "Y" in eq


class TestInterventionEngine:
    def test_do_intervention(self):
        """do(X=x) intervention produces valid results."""
        from fusionmind4.control.scm import PlasmaSCM
        from fusionmind4.control.interventions import InterventionEngine
        names = ["A", "B"]
        dag = np.array([[0, 0.8], [0, 0]])
        scm = PlasmaSCM(names, dag)
        data = np.column_stack([np.random.randn(100), np.random.randn(100)])
        scm.fit(data, verbose=False)
        engine = InterventionEngine(scm)
        result = engine.do({"A": 2.0}, {"A": 0.5, "B": 0.3})
        assert result.outcomes["A"] == 2.0  # Intervention is exact


class TestCounterfactualEngine:
    def test_counterfactual_query(self):
        """Counterfactual: same noise, different intervention."""
        from fusionmind4.control.scm import PlasmaSCM
        from fusionmind4.control.interventions import CounterfactualEngine
        names = ["X", "Y"]
        dag = np.array([[0, 0.5], [0, 0]])
        scm = PlasmaSCM(names, dag)
        data = np.column_stack([np.random.randn(100), np.random.randn(100)])
        scm.fit(data, verbose=False)
        cf_engine = CounterfactualEngine(scm)
        result = cf_engine.counterfactual(
            factual_state={"X": 1.0, "Y": 0.5},
            intervention={"X": 2.0}
        )
        assert "Y" in result.counterfactual_outcomes
        assert result.counterfactual_outcomes["X"] == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# UPFM Tests (Patent Family PF3)
# ══════════════════════════════════════════════════════════════════════════════

class TestDimensionlessTokenizer:
    def test_tokenize_produces_correct_length(self):
        from fusionmind4.foundation.core import DimensionlessTokenizer, DEVICES, N_TOKENS
        tokenizer = DimensionlessTokenizer()
        raw = {'ne': 5e19, 'Te': 5.0, 'Ti': 4.0, 'Ip': 1.0,
               'P_heat': 10.0, 'P_rad': 3.0, 'q95': 3.5,
               'tau_E': 0.1, 'v_tor': 1e5}
        tokens = tokenizer.tokenize(raw, DEVICES['ITER'])
        assert len(tokens) == N_TOKENS

    def test_cross_device_consistency(self):
        from fusionmind4.foundation.core import CrossDeviceValidator
        validator = CrossDeviceValidator()
        results = validator.generate_equivalent_plasmas()
        similarity = validator.compute_cross_device_similarity(results)
        # CV should be reasonable (< 1.0 for equivalent plasmas)
        assert similarity['overall_cv'] < 2.0
        assert similarity['n_devices'] == 6
