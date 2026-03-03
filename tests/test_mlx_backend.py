"""
Tests for FusionMind 4.0 MLX Backend
======================================

All tests skip gracefully if MLX is not installed (Linux CI, etc.).
On macOS with Apple Silicon + MLX, they verify:
- NeuralSCM_MLX fit + predict + do_intervention + counterfactual
- NOTEARS_MLX DAG discovery
- DYNOTEARS_MLX temporal discovery
- PPOPolicy_MLX forward + sample + log_prob
- CausalRL_MLX integration
- Copilot server status check

Run: pytest tests/test_mlx_backend.py -v
"""

import numpy as np
import pytest
import sys


# ═══════════════════════════════════════════════════════════════════════
# Skip all tests if MLX not available
# ═══════════════════════════════════════════════════════════════════════

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_OK = True
except ImportError:
    MLX_OK = False

pytestmark = pytest.mark.skipif(not MLX_OK, reason="MLX not installed (not Apple Silicon)")


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_dag():
    """3-var DAG: X0 → X1, X0 → X2, X1 → X2."""
    dag = np.zeros((3, 3))
    dag[0, 1] = 0.5   # X0 → X1
    dag[0, 2] = 0.3   # X0 → X2
    dag[1, 2] = 0.7   # X1 → X2
    return dag


@pytest.fixture
def simple_data(simple_dag):
    """Generate data from the DAG."""
    rng = np.random.RandomState(42)
    n = 500
    X0 = rng.randn(n)
    X1 = 0.5 * X0 + 0.1 * rng.randn(n)
    X2 = 0.3 * X0 + 0.7 * X1 + 0.1 * rng.randn(n)
    return np.column_stack([X0, X1, X2])


@pytest.fixture
def var_names():
    return ["X0", "X1", "X2"]


# ═══════════════════════════════════════════════════════════════════════
# NeuralSCM_MLX Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNeuralSCM_MLX:

    def test_init(self, var_names, simple_dag):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        assert scm.n_vars == 3
        assert len(scm.equations) == 3

    def test_fit(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        losses = scm.fit(simple_data, n_epochs=100, lr=1e-3, verbose=False)
        assert len(losses) == 3
        # Endogenous variables should have low loss
        assert losses["X1"] < 0.1
        assert losses["X2"] < 0.5

    def test_predict(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=100, verbose=False)
        result = scm.predict({"X0": 1.0})
        assert "X1" in result
        assert "X2" in result
        # X1 should be close to 0.5 * 1.0 = 0.5
        assert abs(result["X1"] - 0.5) < 0.5

    def test_do_intervention(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=100, verbose=False)
        state = {"X0": 1.0, "X1": 0.5, "X2": 0.8}
        # do(X1 = 2.0) should change X2 but not X0
        result = scm.do_intervention(state, {"X1": 2.0})
        assert result["X0"] == 1.0  # Unaffected
        assert result["X1"] == 2.0  # Set by intervention
        assert result["X2"] != state["X2"]  # Changed by propagation

    def test_counterfactual(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=100, verbose=False)
        factual = {"X0": 1.0, "X1": 0.5, "X2": 0.8}
        cf = scm.counterfactual(factual, {"X1": 2.0})
        assert "X2" in cf
        assert cf["X1"] == 2.0

    def test_predict_batch(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=50, verbose=False)
        result = scm.predict_batch(simple_data[:10])
        assert result.shape == (10, 3)

    def test_do_batch(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=50, verbose=False)
        result = scm.do_batch(simple_data[:10], {"X1": 2.0})
        assert result.shape == (10, 3)
        assert np.allclose(result[:, 1], 2.0)

    def test_jacobian(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=100, verbose=False)
        J = scm.jacobian({"X0": 1.0, "X1": 0.5, "X2": 0.8})
        assert J.shape == (3, 3)
        # X0 → X1 should have positive Jacobian
        assert J[0, 1] > 0

    def test_online_update(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        scm.fit(simple_data, n_epochs=50, verbose=False)
        # Online update should not crash
        scm.online_update(simple_data[:50], n_epochs=10)

    def test_summary(self, var_names, simple_dag):
        from fusionmind4.mlx_backend.neural_scm_mlx import NeuralSCM_MLX
        scm = NeuralSCM_MLX(var_names, simple_dag, hidden_dim=16)
        s = scm.summary()
        assert "MLX" in s
        assert "X0" in s


# ═══════════════════════════════════════════════════════════════════════
# NOTEARS_MLX Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNOTEARS_MLX:

    def test_basic_discovery(self, simple_data):
        from fusionmind4.mlx_backend.notears_mlx import NOTEARS_MLX
        notears = NOTEARS_MLX(lambda1=0.05, max_iter=20, inner_iter=100,
                               w_threshold=0.15, lr=0.005)
        W = notears.fit(simple_data, verbose=False)
        assert W.shape == (3, 3)
        # Diagonal should be zero
        assert np.allclose(np.diag(W), 0)
        # Should discover at least X0 → X2 (strongest effect)
        assert np.count_nonzero(W) >= 1

    def test_acyclicity(self, simple_data):
        from fusionmind4.mlx_backend.notears_mlx import NOTEARS_MLX, _matrix_power_series
        notears = NOTEARS_MLX(lambda1=0.05, max_iter=20, inner_iter=100)
        W = notears.fit(simple_data, verbose=False)
        # Check acyclicity: h(W) ≈ 0
        W_mx = mx.array(W.astype(np.float32))
        M = W_mx * W_mx
        E = _matrix_power_series(M, 12)
        h = float(mx.trace(E).item()) - W.shape[0]
        assert h < 0.1  # Approximately DAG

    def test_matrix_expm_accuracy(self):
        from fusionmind4.mlx_backend.notears_mlx import _matrix_power_series
        # Compare Taylor series with known result
        M = mx.array([[0.1, 0.0], [0.0, 0.2]])
        E = _matrix_power_series(M, 20)
        E_np = np.array(E)
        expected = np.diag([np.exp(0.1), np.exp(0.2)])
        assert np.allclose(E_np, expected, atol=1e-6)


class TestDYNOTEARS_MLX:

    def test_temporal_discovery(self):
        from fusionmind4.mlx_backend.notears_mlx import DYNOTEARS_MLX
        rng = np.random.RandomState(42)
        T, d = 300, 3
        data = np.zeros((T, d))
        data[0] = rng.randn(d)
        for t in range(1, T):
            data[t, 0] = 0.3 * data[t-1, 0] + 0.1 * rng.randn()
            data[t, 1] = 0.5 * data[t-1, 0] + 0.2 * data[t-1, 1] + 0.1 * rng.randn()
            data[t, 2] = 0.4 * data[t, 0] + 0.3 * data[t, 1] + 0.1 * rng.randn()

        dyno = DYNOTEARS_MLX(lambda1=0.05, lambda_a=0.05, max_iter=15,
                              inner_iter=80, lr=0.005)
        W, A = dyno.fit(data, verbose=False)
        assert W.shape == (3, 3)
        assert A.shape == (3, 3)


# ═══════════════════════════════════════════════════════════════════════
# PPO Policy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPPOPolicy_MLX:

    def test_forward(self):
        from fusionmind4.mlx_backend.policy_mlx import PPOPolicy_MLX, PPOConfig_MLX
        config = PPOConfig_MLX(hidden_dim=32, n_hidden=2)
        policy = PPOPolicy_MLX(obs_dim=10, act_dim=4, config=config)
        obs = mx.array(np.random.randn(5, 10).astype(np.float32))
        mean, std = policy(obs)
        assert mean.shape == (5, 4)
        assert std.shape == (5, 4)
        # mean should be in [-1, 1] due to tanh
        mean_np = np.array(mean)
        assert np.all(mean_np >= -1.01) and np.all(mean_np <= 1.01)

    def test_sample_action(self):
        from fusionmind4.mlx_backend.policy_mlx import PPOPolicy_MLX, PPOConfig_MLX
        config = PPOConfig_MLX(hidden_dim=32)
        policy = PPOPolicy_MLX(obs_dim=10, act_dim=4, config=config)
        obs = np.random.randn(10).astype(np.float32)
        action, log_prob = policy.sample_action(obs)
        assert action.shape == (4,)
        assert np.all(action >= 0) and np.all(action <= 1)
        assert isinstance(log_prob, float)

    def test_log_prob(self):
        from fusionmind4.mlx_backend.policy_mlx import PPOPolicy_MLX, PPOConfig_MLX
        config = PPOConfig_MLX(hidden_dim=32)
        policy = PPOPolicy_MLX(obs_dim=10, act_dim=4, config=config)
        obs = mx.array(np.random.randn(5, 10).astype(np.float32))
        actions = mx.array(np.random.randn(5, 4).astype(np.float32))
        lp = policy.log_prob(obs, actions)
        assert lp.shape == (5,)

    def test_value_network(self):
        from fusionmind4.mlx_backend.policy_mlx import ValueNetwork_MLX, PPOConfig_MLX
        config = PPOConfig_MLX(hidden_dim=32)
        vf = ValueNetwork_MLX(obs_dim=10, config=config)
        obs = np.random.randn(10).astype(np.float32)
        v = vf.value_np(obs)
        assert isinstance(v, float)


# ═══════════════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════════════

class TestCausalRL_MLX_Integration:

    def test_init(self, var_names, simple_dag):
        from fusionmind4.mlx_backend.causal_rl_mlx import CausalRL_MLX
        agent = CausalRL_MLX(var_names, simple_dag)
        assert agent.n_vars == 3
        # X0 is exogenous (no parents)
        assert "X0" in agent.actuator_names

    def test_fit_world_model(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.causal_rl_mlx import CausalRL_MLX, CausalRLConfig
        config = CausalRLConfig(scm_fit_epochs=50, scm_hidden_dim=16)
        agent = CausalRL_MLX(var_names, simple_dag, config=config)
        agent.fit_world_model(simple_data, verbose=False)
        assert agent.scm._fitted

    def test_causal_reward(self, var_names, simple_dag, simple_data):
        from fusionmind4.mlx_backend.causal_rl_mlx import CausalRL_MLX, CausalRLConfig
        config = CausalRLConfig(scm_fit_epochs=50, scm_hidden_dim=16)
        agent = CausalRL_MLX(var_names, simple_dag, config=config)
        agent.fit_world_model(simple_data, verbose=False)

        state = {"X0": 1.0, "X1": 0.5, "X2": 0.8}
        next_state = {"X0": 1.0, "X1": 0.6, "X2": 0.9}
        action = np.array([0.5])
        base_r, causal_b, safety_p = agent.compute_causal_reward(
            state, next_state, action
        )
        assert isinstance(base_r, float)
        assert isinstance(causal_b, float)
        assert isinstance(safety_p, float)

    def test_summary(self, var_names, simple_dag):
        from fusionmind4.mlx_backend.causal_rl_mlx import CausalRL_MLX
        agent = CausalRL_MLX(var_names, simple_dag)
        s = agent.summary()
        assert "CausalShield-RL" in s


# ═══════════════════════════════════════════════════════════════════════
# Copilot Server Status
# ═══════════════════════════════════════════════════════════════════════

class TestCopilotServer:

    def test_status_check(self):
        from fusionmind4.mlx_backend.copilot_server import check_vllm_mlx_status
        status = check_vllm_mlx_status()
        assert "vllm_mlx_installed" in status


# ═══════════════════════════════════════════════════════════════════════
# Backend Info
# ═══════════════════════════════════════════════════════════════════════

class TestBackendInfo:

    def test_get_info(self):
        from fusionmind4.mlx_backend import get_backend_info
        info = get_backend_info()
        assert info["mlx_available"] is True
        assert "default_device" in info
