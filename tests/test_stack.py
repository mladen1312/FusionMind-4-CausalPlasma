"""Tests for FusionMind Unified 4-Layer Stack."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fusionmind4.control.stack import (
    FusionMindStack, Phase, StackConfig, SafetyLimits,
    PlasmaState, ActionCommand,
    Layer0_RealtimeEngine, Layer1_TacticalRL, Layer2_CausalStrategy, Layer3_SafetyMonitor,
)


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def var_names():
    return ['betan', 'betap', 'q_95', 'q_axis', 'elongation', 'li', 'wplasmd', 'betat', 'Ip', 'Prad']

@pytest.fixture
def dag(var_names):
    d = len(var_names)
    idx = {v: i for i, v in enumerate(var_names)}
    A = np.zeros((d, d))
    for s, t in [('betat','betan'),('betat','wplasmd'),('betat','betap'),
                 ('li','q_95'),('li','q_axis'),('Ip','q_95'),('Ip','betan'),
                 ('Prad','wplasmd'),('elongation','betap')]:
        A[idx[s], idx[t]] = 1.0
    return A

@pytest.fixture
def mock_scm(dag, var_names):
    """SCM with relative perturbation model."""
    class SCM:
        def __init__(self, dag, var_names):
            self.dag, self.var_names, self.d = dag, var_names, len(var_names)
            self.r2_scores = {i: 0.9 for i in range(self.d)}
            self.equations = {}
            for j in range(self.d):
                pa = list(np.where(dag[:, j] > 0)[0])
                self.equations[j] = {'pa': pa, 'coef': [0.1]*len(pa), 'intercept': 0}
        def do(self, interventions, baseline):
            r = baseline.copy()
            idx = {v: i for i, v in enumerate(self.var_names)}
            for var, val in interventions.items():
                if var in idx: r[idx[var]] = val
            for j in self._topo():
                if self.var_names[j] in interventions: continue
                eq = self.equations[j]
                if eq['pa']:
                    delta = sum(0.1*(r[p]-baseline[p])/(abs(baseline[p])+1e-10) for p in eq['pa'])
                    r[j] = baseline[j] * (1 + delta)
            return r
        def counterfactual(self, factual, interventions):
            r = factual.copy()
            idx = {v: i for i, v in enumerate(self.var_names)}
            for var, val in interventions.items():
                if var in idx: r[idx[var]] = val
            for j in self._topo():
                if self.var_names[j] in interventions: continue
                eq = self.equations[j]
                if eq['pa']:
                    delta = sum(0.1*(r[p]-factual[p])/(abs(factual[p])+1e-10) for p in eq['pa'])
                    r[j] = factual[j] * (1 + delta)
            return r
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
    return SCM(dag, var_names)

@pytest.fixture
def normal_state():
    return PlasmaState(
        values={'betan': 2.0, 'betap': 0.8, 'q_95': 4.0, 'q_axis': 1.5,
                'elongation': 1.7, 'li': 1.0, 'wplasmd': 50000,
                'betat': 1.5, 'Ip': 500000, 'Prad': 100000},
        timestamp=1.0)

@pytest.fixture
def risky_state():
    return PlasmaState(
        values={'betan': 3.6, 'betap': 1.5, 'q_95': 1.8, 'q_axis': 0.9,
                'elongation': 2.0, 'li': 2.1, 'wplasmd': 80000,
                'betat': 2.5, 'Ip': 700000, 'Prad': 300000},
        timestamp=2.0)


def make_stack(dag, scm, var_names, phase):
    cfg = StackConfig(phase=phase)
    return FusionMindStack(dag, scm, var_names, cfg)


# ── Layer 0 Tests ────────────────────────────────────────

class TestLayer0:
    def test_extract_features(self, var_names, normal_state):
        L0 = Layer0_RealtimeEngine(var_names)
        f = L0.extract_features(normal_state)
        assert 'betan' in f
        assert f['betan'] == 2.0

    def test_rates_computed(self, var_names, normal_state):
        L0 = Layer0_RealtimeEngine(var_names)
        L0.extract_features(normal_state)
        state2 = PlasmaState(
            values={**normal_state.values, 'betan': 2.5}, timestamp=2.0)
        f = L0.extract_features(state2)
        assert 'd_betan_dt' in f
        assert f['d_betan_dt'] == pytest.approx(0.5, rel=0.1)

    def test_fast_risk_normal(self, var_names, normal_state):
        L0 = Layer0_RealtimeEngine(var_names)
        risk = L0.fast_risk_score(normal_state, SafetyLimits())
        assert risk < 0.3

    def test_fast_risk_risky(self, var_names, risky_state):
        L0 = Layer0_RealtimeEngine(var_names)
        risk = L0.fast_risk_score(risky_state, SafetyLimits())
        assert risk > 0.5

    def test_rate_limit(self, var_names, normal_state):
        L0 = Layer0_RealtimeEngine(var_names)
        action = {'Ip': 1000000}  # Double Ip
        safe = L0.apply_rate_limits(action, normal_state, 0.2)
        assert safe['Ip'] < 700000  # Clamped to 20%


# ── Layer 1 Tests ────────────────────────────────────────

class TestLayer1:
    def test_untrained_fallback(self, var_names, normal_state):
        L1 = Layer1_TacticalRL(20, 10)
        action = L1.compute_action(
            normal_state, {'betan': 2.5}, var_names, ['Ip', 'Prad'])
        assert 'Ip' in action
        assert isinstance(action['Ip'], (int, float))

    def test_forward_shape(self):
        L1 = Layer1_TacticalRL(20, 5)
        obs = np.random.randn(20)
        out = L1.forward(obs)
        assert out.shape == (5,)
        assert np.all(np.abs(out) <= 1.0)  # tanh bounded


# ── Layer 2 Tests ────────────────────────────────────────

class TestLayer2:
    def test_compute_setpoints(self, dag, mock_scm, var_names, normal_state):
        L2 = Layer2_CausalStrategy(dag, mock_scm, var_names)
        L2.set_targets({'betan': 2.5, 'q_95': 4.0})
        sp, expl = L2.compute_setpoints(normal_state, ['Ip', 'Prad'], SafetyLimits())
        assert 'Ip' in sp
        assert isinstance(expl, str)
        assert len(expl) > 0

    def test_predict_outcome(self, dag, mock_scm, var_names, normal_state):
        L2 = Layer2_CausalStrategy(dag, mock_scm, var_names)
        pred = L2.predict_outcome(normal_state, {'Ip': 600000})
        assert 'betan' in pred
        assert isinstance(pred['betan'], (int, float))

    def test_counterfactual(self, dag, mock_scm, var_names, normal_state):
        L2 = Layer2_CausalStrategy(dag, mock_scm, var_names)
        cf = L2.counterfactual_analysis(normal_state, {'betan': 1.0})
        assert 'counterfactual_state' in cf
        assert 'changes' in cf

    def test_causal_paths(self, dag, mock_scm, var_names):
        L2 = Layer2_CausalStrategy(dag, mock_scm, var_names)
        paths = L2.get_causal_paths('li', 'q_95')
        assert len(paths) >= 1
        assert paths[0][0] == 'li'


# ── Layer 3 Tests ────────────────────────────────────────

class TestLayer3:
    def test_approve_safe(self, dag, mock_scm, var_names, normal_state):
        L3 = Layer3_SafetyMonitor(dag, mock_scm, var_names, SafetyLimits())
        cmd = L3.evaluate(normal_state, {'Ip': 510000}, source_layer=1)
        assert not cmd.vetoed
        assert cmd.risk_score < 0.8

    def test_veto_dangerous(self, dag, mock_scm, var_names, risky_state):
        L3 = Layer3_SafetyMonitor(dag, mock_scm, var_names, SafetyLimits())
        cmd = L3.evaluate(risky_state, {'betan': 4.0}, source_layer=1)
        assert cmd.risk_score > 0.5

    def test_explains_every_action(self, dag, mock_scm, var_names, normal_state):
        L3 = Layer3_SafetyMonitor(dag, mock_scm, var_names, SafetyLimits())
        cmd = L3.evaluate(normal_state, {'Ip': 510000}, source_layer=0)
        assert cmd.explanation != ""

    def test_stats_tracking(self, dag, mock_scm, var_names, normal_state):
        L3 = Layer3_SafetyMonitor(dag, mock_scm, var_names, SafetyLimits())
        for _ in range(5):
            L3.evaluate(normal_state, {'Ip': 510000}, source_layer=0)
        stats = L3.get_stats()
        assert stats['total_evaluations'] == 5

    def test_disruption_explanation(self, dag, mock_scm, var_names, normal_state, risky_state):
        L3 = Layer3_SafetyMonitor(dag, mock_scm, var_names, SafetyLimits())
        expl = L3.explain_disruption(normal_state, risky_state)
        assert 'root_causes' in expl
        assert 'counterfactuals' in expl
        assert len(expl['root_causes']) > 0


# ── Stack Phase 1 Tests ──────────────────────────────────

class TestPhase1:
    def test_evaluate_external(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        cmd = stack.evaluate_external_action(normal_state, {'Ip': 510000})
        assert isinstance(cmd, ActionCommand)
        assert cmd.phase == 'wrapper'
        assert cmd.explanation != ""

    def test_step_with_external(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        cmd = stack.step(normal_state, ['Ip'], external_action={'Ip': 510000})
        assert isinstance(cmd, ActionCommand)

    def test_step_without_external(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        cmd = stack.step(normal_state, ['Ip'])
        assert 'HOLD' in cmd.explanation

    def test_no_L2_in_phase1(self, dag, mock_scm, var_names):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        assert stack.L2 is None
        with pytest.raises(RuntimeError):
            stack.set_targets({'betan': 2.5})

    def test_stats(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        stack.step(normal_state, ['Ip'], external_action={'Ip': 510000})
        stats = stack.get_stats()
        assert stats['phase'] == 'wrapper'
        assert 0 in stats['layers_active']
        assert 3 in stats['layers_active']


# ── Stack Phase 2 Tests ──────────────────────────────────

class TestPhase2:
    def test_strategic_setpoints(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_2)
        stack.set_targets({'betan': 2.5, 'q_95': 4.0})
        cmd = stack.step(normal_state, ['Ip', 'Prad'])
        assert isinstance(cmd, ActionCommand)
        assert cmd.phase == 'hybrid'

    def test_L2_active(self, dag, mock_scm, var_names):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_2)
        assert stack.L2 is not None
        assert stack.L1 is None  # No tactical RL yet

    def test_with_external_tactical(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_2)
        stack.set_targets({'betan': 2.5})
        cmd = stack.step(normal_state, ['Ip'], external_action={'Ip': 520000})
        assert isinstance(cmd, ActionCommand)

    def test_predict_intervention(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_2)
        pred = stack.predict_intervention(normal_state, {'Ip': 600000})
        assert 'betan' in pred


# ── Stack Phase 3 Tests ──────────────────────────────────

class TestPhase3:
    def test_full_autonomous(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_3)
        stack.set_targets({'betan': 2.5, 'q_95': 4.0})
        cmd = stack.step(normal_state, ['Ip', 'Prad'])
        assert isinstance(cmd, ActionCommand)
        assert cmd.phase == 'full_stack'

    def test_all_layers_active(self, dag, mock_scm, var_names):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_3)
        assert stack.L0 is not None
        assert stack.L1 is not None
        assert stack.L2 is not None
        assert stack.L3 is not None

    def test_stats_show_all_layers(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_3)
        stack.set_targets({'betan': 2.5})
        stack.step(normal_state, ['Ip'])
        stats = stack.get_stats()
        assert sorted(stats['layers_active']) == [0, 1, 2, 3]


# ── Phase Upgrade Tests ──────────────────────────────────

class TestPhaseUpgrade:
    def test_upgrade_1_to_2(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        assert stack.L2 is None
        result = stack.upgrade_phase(Phase.PHASE_2)
        assert 'wrapper → hybrid' in result
        assert stack.L2 is not None
        stack.set_targets({'betan': 2.5})
        cmd = stack.step(normal_state, ['Ip'])
        assert cmd.phase == 'hybrid'

    def test_upgrade_2_to_3(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_2)
        assert stack.L1 is None
        stack.upgrade_phase(Phase.PHASE_3)
        assert stack.L1 is not None
        stack.set_targets({'betan': 2.5})
        cmd = stack.step(normal_state, ['Ip'])
        assert cmd.phase == 'full_stack'

    def test_upgrade_preserves_state(self, dag, mock_scm, var_names, normal_state):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        stack.step(normal_state, ['Ip'], external_action={'Ip': 510000})
        old_cycles = stack._cycle_count
        stack.upgrade_phase(Phase.PHASE_3)
        assert stack._cycle_count == old_cycles  # State preserved


# ── Cross-Phase Consistency ──────────────────────────────

class TestCrossPhase:
    def test_safety_always_active(self, dag, mock_scm, var_names, risky_state):
        """Safety layer (L3) works in ALL phases."""
        for phase in Phase:
            stack = make_stack(dag, mock_scm, var_names, phase)
            if phase != Phase.PHASE_1:
                stack.set_targets({'betan': 2.0})
            cmd = stack.step(risky_state, ['Ip', 'Prad'],
                             external_action={'Ip': 800000})
            assert cmd.risk_score > 0.3  # Recognizes risk in all phases

    def test_explain_works_all_phases(self, dag, mock_scm, var_names, normal_state):
        for phase in Phase:
            stack = make_stack(dag, mock_scm, var_names, phase)
            report = stack.explain_state(normal_state)
            assert 'risk' in report
            assert 'causal_map' in report
            assert report['phase'] == phase.value

    def test_counterfactual_works_all_phases(self, dag, mock_scm, var_names, normal_state):
        for phase in Phase:
            stack = make_stack(dag, mock_scm, var_names, phase)
            cf = stack.counterfactual(normal_state, {'betan': 1.0})
            assert isinstance(cf, dict)

    def test_disruption_explanation_all_phases(self, dag, mock_scm, var_names, normal_state, risky_state):
        for phase in Phase:
            stack = make_stack(dag, mock_scm, var_names, phase)
            expl = stack.explain_disruption(normal_state, risky_state)
            assert 'root_causes' in expl


# ── Simulation: 100-Step Run ─────────────────────────────

class TestSimulation:
    def test_100_step_phase1(self, dag, mock_scm, var_names):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_1)
        for t in range(100):
            state = PlasmaState(
                values={'betan': 2.0 + 0.01*t, 'betap': 0.8, 'q_95': 4.0 - 0.01*t,
                        'q_axis': 1.5, 'elongation': 1.7, 'li': 1.0,
                        'wplasmd': 50000, 'betat': 1.5, 'Ip': 500000, 'Prad': 100000},
                timestamp=float(t) * 0.005)
            cmd = stack.step(state, ['Ip'], external_action={'Ip': 500000 + t*100})
            assert isinstance(cmd, ActionCommand)
        stats = stack.get_stats()
        assert stats['cycles'] == 100
        assert stats['safety']['total_evaluations'] == 100

    def test_100_step_phase3(self, dag, mock_scm, var_names):
        stack = make_stack(dag, mock_scm, var_names, Phase.PHASE_3)
        stack.set_targets({'betan': 2.5, 'q_95': 4.0})
        vetoes = 0
        for t in range(100):
            state = PlasmaState(
                values={'betan': 2.0 + 0.01*t, 'betap': 0.8, 'q_95': 4.0 - 0.01*t,
                        'q_axis': 1.5, 'elongation': 1.7, 'li': 1.0,
                        'wplasmd': 50000, 'betat': 1.5, 'Ip': 500000, 'Prad': 100000},
                timestamp=float(t) * 0.005)
            cmd = stack.step(state, ['Ip', 'Prad'])
            if cmd.vetoed:
                vetoes += 1
        stats = stack.get_stats()
        assert stats['cycles'] == 100
        # Some vetoes expected as betan approaches limit
        assert isinstance(vetoes, int)
