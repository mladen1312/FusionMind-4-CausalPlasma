"""
FusionMind 4.0 — Full Integration Test
Uses MAST test fixtures. Tests: CPDE → SCM → do-calculus → Python Stack → C++ Stack
"""
import pytest
import numpy as np
import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

def _have_fixtures():
    return os.path.exists(os.path.join(FIXTURE_DIR, 'mast_real_data.npy'))

@pytest.fixture(scope="module")
def mast_env():
    """Load MAST data + run CPDE + fit SCM — shared across all tests."""
    if not _have_fixtures():
        pytest.skip("MAST fixtures not found")

    data = np.load(os.path.join(FIXTURE_DIR, 'mast_real_data.npy'))
    with open(os.path.join(FIXTURE_DIR, 'mast_real_meta.json')) as f:
        meta = json.load(f)
    vn = meta['var_names']

    # CPDE
    from fusionmind4.discovery.ensemble import EnsembleCPDE
    cpde = EnsembleCPDE(config={'n_bootstrap': 5, 'threshold': 0.18}, verbose=False)
    cpde_result = cpde.discover(data, var_names=vn)
    dag = cpde_result['dag']

    # SCM + engines
    from fusionmind4.control.scm import PlasmaSCM
    from fusionmind4.control.interventions import InterventionEngine, CounterfactualEngine
    scm = PlasmaSCM(vn, dag)
    scm.fit(data)
    do_engine = InterventionEngine(scm)
    cf_engine = CounterfactualEngine(scm)

    # SimpleSCM for Stack (has do/counterfactual on arrays)
    from fusionmind4.control.stack import FusionMindStack
    _, simple_scm = FusionMindStack._build_causal_model(data, vn)

    return {
        'data': data, 'var_names': vn, 'dag': dag,
        'scm': scm, 'do_engine': do_engine, 'cf_engine': cf_engine,
        'simple_scm': simple_scm, 'cpde_result': cpde_result,
    }


# ── CPDE ────────────────────────────────────────────────

@pytest.mark.skipif(not _have_fixtures(), reason="No fixtures")
class TestCPDE:
    def test_dag_shape(self, mast_env):
        assert mast_env['dag'].shape == (len(mast_env['var_names']), len(mast_env['var_names']))
    def test_edges_found(self, mast_env):
        assert mast_env['cpde_result']['n_edges'] > 0
    def test_no_self_loops(self, mast_env):
        assert np.all(np.diag(mast_env['dag']) == 0)


# ── SCM + do-calculus + counterfactual ──────────────────

@pytest.mark.skipif(not _have_fixtures(), reason="No fixtures")
class TestSCM:
    def test_fits(self, mast_env):
        assert len(mast_env['scm'].equations) == len(mast_env['var_names'])

    def test_do_intervention(self, mast_env):
        vn = mast_env['var_names']
        data = mast_env['data']
        state = {v: float(data[50, i]) for i, v in enumerate(vn)}
        result = mast_env['do_engine'].do({vn[0]: state[vn[0]] * 1.1}, current_state=state)
        assert hasattr(result, 'outcomes')
        assert vn[0] in result.outcomes

    def test_counterfactual(self, mast_env):
        vn = mast_env['var_names']
        data = mast_env['data']
        state = {v: float(data[50, i]) for i, v in enumerate(vn)}
        result = mast_env['cf_engine'].counterfactual(state, {vn[0]: state[vn[0]] * 0.9})
        assert result is not None


# ── Python Stack ────────────────────────────────────────

@pytest.mark.skipif(not _have_fixtures(), reason="No fixtures")
class TestPythonStack:
    def _make(self, mast_env, phase):
        from fusionmind4.control.stack import FusionMindStack, Phase, StackConfig, PlasmaState
        p = {'wrapper': Phase.PHASE_1, 'hybrid': Phase.PHASE_2, 'full_stack': Phase.PHASE_3}[phase]
        return FusionMindStack(mast_env['dag'], mast_env['simple_scm'], 
                                mast_env['var_names'], StackConfig(phase=p))

    def _state(self, mast_env, idx=100):
        from fusionmind4.control.stack import PlasmaState
        vn = mast_env['var_names']
        return PlasmaState(values={v: float(mast_env['data'][idx, i]) for i, v in enumerate(vn)}, timestamp=1.0)

    def test_phase1(self, mast_env):
        stack = self._make(mast_env, 'wrapper')
        state = self._state(mast_env)
        vn = mast_env['var_names']
        cmd = stack.step(state, vn[:2], external_action={vn[0]: state.get(vn[0]) * 1.02})
        assert isinstance(cmd.explanation, str)

    def test_phase2(self, mast_env):
        stack = self._make(mast_env, 'hybrid')
        stack.set_targets({mast_env['var_names'][0]: 2.0})
        cmd = stack.step(self._state(mast_env), mast_env['var_names'][:2])
        assert cmd.phase == 'hybrid'

    def test_phase3(self, mast_env):
        stack = self._make(mast_env, 'full_stack')
        stack.set_targets({mast_env['var_names'][0]: 2.0})
        cmd = stack.step(self._state(mast_env), mast_env['var_names'][:2])
        assert cmd.phase == 'full_stack'

    def test_upgrade(self, mast_env):
        stack = self._make(mast_env, 'wrapper')
        assert stack.L2 is None
        from fusionmind4.control.stack import Phase
        stack.upgrade_phase(Phase.PHASE_3)
        assert stack.L1 is not None and stack.L2 is not None

    def test_explain(self, mast_env):
        stack = self._make(mast_env, 'wrapper')
        report = stack.explain_state(self._state(mast_env))
        assert 'risk' in report

    def test_100_steps(self, mast_env):
        stack = self._make(mast_env, 'full_stack')
        stack.set_targets({mast_env['var_names'][0]: 2.0})
        vn = mast_env['var_names']
        from fusionmind4.control.stack import PlasmaState
        for i in range(100):
            s = PlasmaState(values={v: float(mast_env['data'][i, j]) for j, v in enumerate(vn)}, timestamp=i*0.005)
            stack.step(s, vn[:2])
        assert stack.get_stats()['cycles'] == 100


# ── C++ Stack ──────────────────────────────────────────

@pytest.mark.skipif(not _have_fixtures(), reason="No fixtures")
class TestCppStack:
    @pytest.fixture(autouse=True)
    def check(self):
        from fusionmind4.realtime.stack_bindings import CPP_STACK_AVAILABLE
        if not CPP_STACK_AVAILABLE:
            pytest.skip("C++ not compiled")

    def _make(self, mast_env, phase):
        from fusionmind4.realtime.stack_bindings import CppStack
        vn = mast_env['var_names']
        stack = CppStack(len(vn), phase=phase, var_names=vn)
        stack.load_scm(mast_env['dag'].astype(float), mast_env['simple_scm'])
        stack.load_safety_limits()
        return stack

    def test_phase1(self, mast_env):
        stack = self._make(mast_env, 1)
        vn = mast_env['var_names']
        vals = {v: float(mast_env['data'][100, i]) for i, v in enumerate(vn)}
        r = stack.step(vals, 1.0, vn[:2], external_action={vn[0]: vals[vn[0]] * 1.01})
        assert r.decision in ('APPROVE', 'WARN', 'VETO')

    def test_phase3(self, mast_env):
        stack = self._make(mast_env, 3)
        vn = mast_env['var_names']
        stack.set_setpoints({vn[0]: 2.0})
        vals = {v: float(mast_env['data'][100, i]) for i, v in enumerate(vn)}
        r = stack.step(vals, 1.0, vn[:2])
        assert r.decision in ('APPROVE', 'WARN', 'VETO')

    def test_do_calculus(self, mast_env):
        stack = self._make(mast_env, 1)
        vn = mast_env['var_names']
        vals = {v: float(mast_env['data'][100, i]) for i, v in enumerate(vn)}
        pred = stack.do_intervention(vals, {vn[0]: vals[vn[0]] * 1.1})
        assert not np.isnan(pred[vn[0]])

    def test_counterfactual(self, mast_env):
        stack = self._make(mast_env, 1)
        vn = mast_env['var_names']
        vals = {v: float(mast_env['data'][100, i]) for i, v in enumerate(vn)}
        cf = stack.counterfactual(vals, {vn[0]: vals[vn[0]] * 0.8})
        assert not np.isnan(cf[vn[1]])

    def test_latency(self, mast_env):
        stack = self._make(mast_env, 3)
        vn = mast_env['var_names']
        stack.set_setpoints({vn[0]: 2.0})
        bench = stack.benchmark(n_cycles=500, actuator_names=vn[:2])
        assert bench['p50_ns'] < 5000  # 5μs generous limit for CI

    def test_100_cycles(self, mast_env):
        stack = self._make(mast_env, 3)
        vn = mast_env['var_names']
        stack.set_setpoints({vn[0]: 2.0})
        for i in range(100):
            vals = {v: float(mast_env['data'][i, j]) for j, v in enumerate(vn)}
            stack.step(vals, i * 0.005, vn[:2])
        assert stack.get_stats()['cycles'] >= 100

    def test_phase_switch(self, mast_env):
        stack = self._make(mast_env, 1)
        assert stack.get_stats()['phase'] == 1
        stack.set_phase(3)
        vn = mast_env['var_names']
        stack.set_setpoints({vn[0]: 2.0})
        vals = {v: float(mast_env['data'][100, i]) for i, v in enumerate(vn)}
        r = stack.step(vals, 1.0, vn[:2])
        assert stack.get_stats()['phase'] == 3
