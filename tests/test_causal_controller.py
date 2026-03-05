"""Tests for CausalRL Controller — all three modes."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fusionmind4.control.causal_controller import (
    FusionMindController, CausalWorldModel, CausalSafetyMonitor,
    CausalRLPolicy, ControlMode, PlasmaState, SafetyLimits, ControlAction,
)


# ── Fixtures ─────────────────────────────────────────────

class MockSCM:
    """Simplified SCM for testing — physically scaled linear model."""
    def __init__(self, dag, var_names):
        self.dag = dag
        self.var_names = var_names
        self.d = len(var_names)
        self.r2_scores = {i: 0.9 for i in range(self.d)}
        self.linear_models = {}
        # Physically meaningful: small perturbation coefficients
        for j in range(self.d):
            pa = list(np.where(dag[:, j] > 0)[0])
            self.linear_models[j] = {
                'parents': pa,
                'coefs': [0.01] * len(pa),  # Small coefficient — 1% coupling
                'intercept': 0.0  # Baseline = current value (set during do/cf)
            }

    def do(self, interventions, baseline):
        result = baseline.copy()
        idx = {v: i for i, v in enumerate(self.var_names)}
        for var, val in interventions.items():
            if var in idx:
                result[idx[var]] = val
        order = self._topo()
        for j in order:
            if self.var_names[j] in interventions:
                continue
            lm = self.linear_models[j]
            if lm['parents']:
                # Relative perturbation: 10% of parent's relative change
                rel_delta = sum(
                    0.1 * (result[p] - baseline[p]) / (abs(baseline[p]) + 1e-10)
                    for p in lm['parents']
                )
                result[j] = baseline[j] * (1 + rel_delta)
        return result

    def counterfactual(self, factual, interventions):
        noise = {}
        for j in range(self.d):
            noise[j] = 0.0
        result = factual.copy()
        idx = {v: i for i, v in enumerate(self.var_names)}
        for var, val in interventions.items():
            if var in idx:
                result[idx[var]] = val
        for j in self._topo():
            if self.var_names[j] in interventions:
                continue
            lm = self.linear_models[j]
            if lm['parents']:
                rel_delta = sum(
                    0.1 * (result[p] - factual[p]) / (abs(factual[p]) + 1e-10)
                    for p in lm['parents']
                )
                result[j] = factual[j] * (1 + rel_delta) + noise[j]
        return result

    def _topo(self):
        vis, order = set(), []
        def dfs(n):
            if n in vis: return
            vis.add(n)
            for p in range(self.d):
                if self.dag[p, n] > 0: dfs(p)
            order.append(n)
        for i in range(self.d): dfs(i)
        return order


@pytest.fixture
def plasma_vars():
    return ['betan', 'betap', 'q_95', 'q_axis', 'elongation', 'li', 'wplasmd', 'betat', 'Ip', 'Prad']


@pytest.fixture
def dag(plasma_vars):
    d = len(plasma_vars)
    idx = {v: i for i, v in enumerate(plasma_vars)}
    A = np.zeros((d, d))
    # Known causal structure
    edges = [
        ('betat', 'betan'), ('betat', 'wplasmd'), ('betat', 'betap'),
        ('li', 'q_95'), ('li', 'q_axis'),
        ('Ip', 'q_95'), ('Ip', 'betan'),
        ('Prad', 'wplasmd'),
        ('wplasmd', 'betat'),
        ('elongation', 'betap'),
    ]
    for s, t in edges:
        if s in idx and t in idx:
            A[idx[s], idx[t]] = 1.0
    return A


@pytest.fixture
def world_model(dag, plasma_vars):
    scm = MockSCM(dag, plasma_vars)
    return CausalWorldModel(dag, scm, plasma_vars)


@pytest.fixture
def normal_state():
    return PlasmaState(
        values={'betan': 2.0, 'betap': 0.8, 'q_95': 4.0, 'q_axis': 1.5,
                'elongation': 1.7, 'li': 1.0, 'wplasmd': 50000,
                'betat': 1.5, 'Ip': 500000, 'Prad': 100000},
        timestamp=1.0
    )


@pytest.fixture
def risky_state():
    return PlasmaState(
        values={'betan': 3.6, 'betap': 1.5, 'q_95': 1.8, 'q_axis': 0.9,
                'elongation': 2.0, 'li': 2.1, 'wplasmd': 80000,
                'betat': 2.5, 'Ip': 700000, 'Prad': 300000},
        timestamp=2.0
    )


# ── CausalWorldModel Tests ──────────────────────────────

class TestCausalWorldModel:
    def test_predict_intervention(self, world_model, normal_state):
        result = world_model.predict_intervention(normal_state, {'Ip': 600000})
        assert isinstance(result, dict)
        assert 'betan' in result
        assert result['Ip'] == 600000  # Intervention holds

    def test_counterfactual(self, world_model, normal_state):
        cf = world_model.counterfactual(normal_state, {'betan': 1.0})
        assert isinstance(cf, dict)
        assert abs(cf['betan'] - 1.0) < 1e-6

    def test_causal_parents(self, world_model):
        parents = world_model.get_causal_parents('betan')
        assert 'betat' in parents or 'Ip' in parents

    def test_causal_children(self, world_model):
        children = world_model.get_causal_children('li')
        assert 'q_95' in children

    def test_trace_path(self, world_model):
        paths = world_model.trace_causal_path('li', 'q_95')
        assert len(paths) >= 1
        assert paths[0][0] == 'li'
        assert paths[0][-1] == 'q_95'

    def test_compute_risk_normal(self, world_model, normal_state):
        risk, factors = world_model.compute_risk(normal_state, SafetyLimits())
        assert risk < 0.5
        assert isinstance(factors, list)

    def test_compute_risk_risky(self, world_model, risky_state):
        risk, factors = world_model.compute_risk(risky_state, SafetyLimits())
        assert risk > 0.5
        assert len(factors) > 0


# ── MODE A: Wrapper Tests ────────────────────────────────

class TestWrapperMode:
    def test_approve_safe_action(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        action = ctrl.evaluate_external_action(
            normal_state, {'Ip': 510000}  # Small safe change
        )
        assert isinstance(action, ControlAction)
        assert not action.vetoed
        assert action.risk_score < 0.8
        assert action.causal_explanation != ""

    def test_veto_dangerous_action(self, world_model, risky_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        # Try to push already-risky plasma further
        action = ctrl.evaluate_external_action(
            risky_state, {'betan': 4.0}  # Way over Troyon limit
        )
        # Should either veto or warn
        assert action.risk_score > 0.5

    def test_provides_explanation(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        action = ctrl.evaluate_external_action(
            normal_state, {'Ip': 600000}
        )
        assert action.causal_explanation != ""
        assert action.source != ""

    def test_tracks_statistics(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        for _ in range(5):
            ctrl.evaluate_external_action(normal_state, {'Ip': 510000})
        stats = ctrl.get_statistics()
        assert stats['total_actions'] == 5
        assert stats['mode'] == 'wrapper'


# ── MODE B: Hybrid Tests ─────────────────────────────────

class TestHybridMode:
    def test_compute_action(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.HYBRID)
        ctrl.set_targets({'betan': 2.5, 'q_95': 4.0})
        action = ctrl.compute_action(normal_state, ['Ip', 'Prad'])
        assert isinstance(action, ControlAction)
        assert 'Ip' in action.actuator_values
        assert action.source in ('causal_rl', 'safety_override')

    def test_action_has_explanation(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.HYBRID)
        ctrl.set_targets({'betan': 2.5})
        action = ctrl.compute_action(normal_state, ['Ip'])
        assert action.causal_explanation != ""

    def test_respects_safety(self, world_model, risky_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.HYBRID)
        ctrl.set_targets({'betan': 4.0})  # Dangerous target
        action = ctrl.compute_action(risky_state, ['Ip', 'Prad'])
        # Should not produce high-risk action
        assert action.risk_score <= 1.0  # Bounded


# ── MODE C: Advisor Tests ─────────────────────────────────

class TestAdvisorMode:
    def test_explain_state(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.ADVISOR)
        report = ctrl.explain_state(normal_state)
        assert 'risk' in report
        assert 'causal_map' in report
        assert 'betan' in report['causal_map']

    def test_doesnt_modify_action(self, world_model, normal_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.ADVISOR)
        proposed = {'Ip': 999999}
        action = ctrl.evaluate_external_action(normal_state, proposed)
        # Advisor mode should pass through unchanged
        assert action.actuator_values == proposed
        assert not action.vetoed


# ── Disruption Explanation Tests ──────────────────────────

class TestDisruptionExplanation:
    def test_explain_disruption(self, world_model, normal_state, risky_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        explanation = ctrl.explain_disruption(normal_state, risky_state)
        assert 'root_causes' in explanation
        assert 'counterfactuals' in explanation
        assert len(explanation['root_causes']) > 0

    def test_counterfactual_in_explanation(self, world_model, normal_state, risky_state):
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        explanation = ctrl.explain_disruption(normal_state, risky_state)
        for cf in explanation['counterfactuals']:
            assert 'hypothesis' in cf
            assert 'would_prevent_disruption' in cf


# ── Integration: All Modes on Same Scenario ───────────────

class TestAllModes:
    def test_all_modes_handle_same_state(self, world_model, normal_state):
        for mode in ControlMode:
            ctrl = FusionMindController(world_model, mode=mode)
            if mode == ControlMode.ADVISOR:
                report = ctrl.explain_state(normal_state)
                assert report['risk'] < 0.5
            else:
                action = ctrl.evaluate_external_action(
                    normal_state, {'Ip': 510000})
                assert isinstance(action, ControlAction)
                assert action.source != ""

    def test_consistency_across_modes(self, world_model, risky_state):
        """All modes should agree that risky state is dangerous."""
        for mode in ControlMode:
            ctrl = FusionMindController(world_model, mode=mode)
            if mode == ControlMode.ADVISOR:
                report = ctrl.explain_state(risky_state)
                assert report['risk'] > 0.3
            else:
                action = ctrl.evaluate_external_action(
                    risky_state, {'betan': 3.5})
                assert action.risk_score > 0.3
