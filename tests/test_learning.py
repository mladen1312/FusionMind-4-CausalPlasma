"""Tests for the FusionMind 4.0 Learning Module (PF7: CausalShield-RL)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionmind4.utils.fm3lite import FM3LitePhysicsEngine
from fusionmind4.utils.plasma_vars import (
    VAR_NAMES, N_VARS, ACTUATOR_IDS,
    build_ground_truth_adjacency, evaluate_dag
)
from fusionmind4.learning.neural_scm import NeuralSCM, NeuralEquation
from fusionmind4.learning.gym_plasma_env import GymPlasmaEnv, FM3LiteStep
from fusionmind4.learning.causal_reward import CausalRewardShaper
from fusionmind4.learning.causal_rl_hybrid import CausalRLHybrid, PolicyNetwork, PPOConfig


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def fm3_data():
    """Generate FM3Lite data for testing."""
    engine = FM3LitePhysicsEngine(n_samples=2000, seed=42)
    data, interventional = engine.generate()
    return data, interventional


@pytest.fixture
def ground_truth_dag():
    """Ground truth causal graph."""
    return build_ground_truth_adjacency()


@pytest.fixture
def small_neural_scm(ground_truth_dag):
    """Small Neural SCM fitted on synthetic data."""
    engine = FM3LitePhysicsEngine(n_samples=1000, seed=42)
    data, _ = engine.generate()
    
    scm = NeuralSCM(VAR_NAMES, ground_truth_dag, hidden_dim=8, lr=1e-3)
    scm.fit(data, n_epochs=100, verbose=False)
    return scm, data


# ═══════════════════════════════════════════════════════════════════════
# Neural SCM Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNeuralEquation:
    """Test individual neural structural equations."""
    
    def test_creation(self):
        eq = NeuralEquation("Te", ["P_ECRH", "P_NBI"], hidden_dim=8)
        assert eq.variable == "Te"
        assert len(eq.parents) == 2
    
    def test_forward(self):
        eq = NeuralEquation("Te", ["P_ECRH", "P_NBI"], hidden_dim=8)
        val = eq.forward(np.array([0.5, 0.3]))
        assert isinstance(val, float)
        assert np.isfinite(val)
    
    def test_forward_batch(self):
        eq = NeuralEquation("Te", ["P_ECRH"], hidden_dim=8)
        data = np.random.randn(100, 1)
        vals = eq.forward_batch(data)
        assert vals.shape == (100,)
        assert np.all(np.isfinite(vals))
    
    def test_fit_reduces_loss(self):
        eq = NeuralEquation("y", ["x1", "x2"], hidden_dim=16, lr=1e-3)
        
        # Generate linear data
        x = np.random.randn(500, 2)
        y = 0.7 * x[:, 0] + 0.3 * x[:, 1] + np.random.randn(500) * 0.1
        
        loss = eq.fit(x, y, n_epochs=300)
        assert loss < 0.5, f"Loss should decrease significantly, got {loss:.4f}"
    
    def test_exogenous_variable(self):
        eq = NeuralEquation("P_NBI", [], hidden_dim=4)
        val = eq.forward(np.array([]))
        assert isinstance(val, float)


class TestNeuralSCM:
    """Test the full Neural SCM."""
    
    def test_fit_and_predict(self, small_neural_scm):
        scm, data = small_neural_scm
        assert scm._fitted
        
        # Predict from first data point
        state = {name: float(data[0, i]) for i, name in enumerate(VAR_NAMES)}
        result = scm.predict(state)
        assert len(result) == N_VARS
        assert all(np.isfinite(v) for v in result.values())
    
    def test_do_intervention(self, small_neural_scm):
        scm, data = small_neural_scm
        
        state = {name: float(data[0, i]) for i, name in enumerate(VAR_NAMES)}
        
        # Intervene on P_ECRH
        result_low = scm.do_intervention(state, {'P_ECRH': 0.3})
        result_high = scm.do_intervention(state, {'P_ECRH': 1.5})
        
        # P_ECRH → Te should be positive, so higher ECRH → higher Te
        assert result_high['P_ECRH'] == 1.5
        assert result_low['P_ECRH'] == 0.3
        # Te should change (direction depends on learned weights)
        assert result_high['Te'] != result_low['Te']
    
    def test_counterfactual(self, small_neural_scm):
        scm, data = small_neural_scm
        
        factual = {name: float(data[0, i]) for i, name in enumerate(VAR_NAMES)}
        cf_result = scm.counterfactual(factual, {'P_ECRH': 1.5})
        
        assert len(cf_result) == N_VARS
        assert cf_result['P_ECRH'] == 1.5  # Intervention was applied
    
    def test_batch_predict(self, small_neural_scm):
        scm, data = small_neural_scm
        
        predictions = scm.predict_batch(data[:50])
        assert predictions.shape == (50, N_VARS)
        assert np.all(np.isfinite(predictions))
    
    def test_jacobian(self, small_neural_scm):
        scm, data = small_neural_scm
        
        state = {name: float(data[0, i]) for i, name in enumerate(VAR_NAMES)}
        J = scm.jacobian(state)
        
        assert J.shape == (N_VARS, N_VARS)
        assert np.all(np.isfinite(J))
        
        # Actuators (exogenous) should not be affected by perturbations to endogenous vars
        for j in range(4):  # Actuator columns
            for i in range(4, N_VARS):  # Endogenous rows
                assert abs(J[i, j]) < 1e-3, f"J[{i},{j}]={J[i,j]} — actuator shouldn't change"
    
    def test_online_update(self, small_neural_scm):
        scm, data = small_neural_scm
        
        # Should not raise
        new_data = data[:200] + np.random.randn(200, N_VARS) * 0.01
        scm.online_update(new_data, n_epochs=20)
    
    def test_causal_effect_matrix(self, small_neural_scm):
        scm, data = small_neural_scm
        
        state = {name: float(data[0, i]) for i, name in enumerate(VAR_NAMES)}
        effects = scm.get_causal_effect_matrix(state)
        
        assert effects.shape == (N_VARS, N_VARS)
        assert np.all(np.isfinite(effects))
    
    def test_summary(self, small_neural_scm):
        scm, _ = small_neural_scm
        s = scm.summary()
        assert "Neural SCM Summary" in s
        assert "Total parameters" in s


# ═══════════════════════════════════════════════════════════════════════
# Gym Environment Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGymPlasmaEnv:
    """Test the Gymnasium environment."""
    
    def test_reset(self):
        env = GymPlasmaEnv(max_steps=100)
        obs, info = env.reset(seed=42)
        assert obs.shape == (N_VARS,)
        assert np.all(np.isfinite(obs))
    
    def test_step(self):
        env = GymPlasmaEnv(max_steps=100)
        obs, _ = env.reset(seed=42)
        
        action = np.array([0.5, 0.5, 0.5, 0.5])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert next_obs.shape == (N_VARS,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert np.all(np.isfinite(next_obs))
    
    def test_episode_runs(self):
        env = GymPlasmaEnv(max_steps=50)
        obs, _ = env.reset(seed=42)
        
        total_reward = 0
        for _ in range(50):
            action = np.random.uniform(0, 1, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        assert np.isfinite(total_reward)
    
    def test_state_dict(self):
        env = GymPlasmaEnv()
        env.reset(seed=42)
        
        state_dict = env.get_state_dict()
        assert len(state_dict) == N_VARS
        assert 'Te' in state_dict
        assert 'P_NBI' in state_dict
    
    def test_render(self):
        env = GymPlasmaEnv()
        env.reset(seed=42)
        output = env.render()
        assert "Step 0" in output


class TestFM3LiteStep:
    """Test the single-step physics model."""
    
    def test_step_preserves_shape(self):
        physics = FM3LiteStep()
        state = np.random.uniform(0.1, 1.0, N_VARS)
        action = np.array([1.0, 0.5, 0.3, 1.0])
        
        next_state, info = physics.step(state, action)
        assert next_state.shape == (N_VARS,)
        assert np.all(np.isfinite(next_state))
    
    def test_actuators_applied(self):
        physics = FM3LiteStep()
        state = np.random.uniform(0.1, 1.0, N_VARS)
        action = np.array([2.0, 1.5, 0.8, 1.2])
        
        next_state, _ = physics.step(state, action)
        assert next_state[0] == 2.0  # P_NBI
        assert next_state[1] == 1.5  # P_ECRH
    
    def test_disruption_detection(self):
        physics = FM3LiteStep()
        # Create state near disruption
        state = np.zeros(N_VARS)
        state[8] = 5.0   # Very high betaN
        state[12] = 3.0  # Very high MHD
        
        _, info = physics.step(state, np.array([1, 1, 1, 1]))
        assert info['disruption_proximity'] > 0.5


# ═══════════════════════════════════════════════════════════════════════
# Causal Reward Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCausalRewardShaper:
    """Test causal reward shaping."""
    
    def test_creation(self, ground_truth_dag):
        shaper = CausalRewardShaper(ground_truth_dag, VAR_NAMES)
        assert len(shaper.allowed_paths) == N_VARS
    
    def test_shape_reward(self, ground_truth_dag):
        shaper = CausalRewardShaper(ground_truth_dag, VAR_NAMES)
        
        state = np.random.uniform(0.1, 1.0, N_VARS)
        action = np.array([0.5, 0.5, 0.5, 0.5])
        next_state = state + np.random.randn(N_VARS) * 0.01
        
        result = shaper.shape_reward(state, action, next_state, base_reward=-0.5)
        
        assert 'total' in result
        assert 'base' in result
        assert 'causal_bonus' in result
        assert 'safety_penalty' in result
        assert np.isfinite(result['total'])
    
    def test_safety_penalty_triggered(self, ground_truth_dag):
        shaper = CausalRewardShaper(ground_truth_dag, VAR_NAMES)
        
        # State violating safety bounds (low q)
        state = np.ones(N_VARS)
        state[7] = 0.5  # q way too low
        next_state = state.copy()
        
        result = shaper.shape_reward(state, np.zeros(4), next_state, base_reward=0)
        assert result['safety_penalty'] > 0
    
    def test_explanation(self, ground_truth_dag):
        shaper = CausalRewardShaper(ground_truth_dag, VAR_NAMES)
        
        state = np.ones(N_VARS)
        next_state = state.copy()
        next_state[0] = 2.0  # Increased P_NBI
        next_state[6] = 1.5  # Ti increased (NBI heats ions)
        
        explanation = shaper.get_action_explanation(
            np.array([1, 0.5, 0.5, 0.5]), state, next_state
        )
        assert "Causal Action Explanation" in explanation


# ═══════════════════════════════════════════════════════════════════════
# CausalRLHybrid Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCausalRLHybrid:
    """Test the full CausalShield-RL integration."""
    
    def test_full_pipeline(self, fm3_data):
        """End-to-end test: discover → fit → train → evaluate."""
        data, interventional = fm3_data
        
        hybrid = CausalRLHybrid(config={
            'scm_hidden_dim': 8,
            'policy_hidden': 32,
        })
        
        # Phase 1: Causal Discovery
        results = hybrid.discover_causal_graph(
            data, interventional, var_names=VAR_NAMES, verbose=False
        )
        assert results['n_edges'] > 10
        assert results['f1'] > 0.3
        
        # Phase 2: Neural SCM
        losses = hybrid.fit_world_model(data, n_epochs=50, verbose=False)
        assert len(losses) == N_VARS
        assert all(np.isfinite(v) for v in losses.values())
        
        # Phase 3: RL Training (very short)
        history = hybrid.train(
            n_episodes=10, rollout_steps=50,
            verbose=False, eval_every=5
        )
        assert len(history['episode_rewards']) == 10
        
        # Phase 4: Evaluation
        metrics = hybrid.evaluate(n_episodes=3, verbose=False)
        assert 'mean_reward' in metrics
        assert 'disruption_rate' in metrics
        assert np.isfinite(metrics['mean_reward'])
    
    def test_act(self, fm3_data):
        data, interventional = fm3_data
        
        hybrid = CausalRLHybrid(config={'scm_hidden_dim': 8, 'policy_hidden': 32})
        hybrid.discover_causal_graph(data, interventional, verbose=False)
        hybrid.fit_world_model(data, n_epochs=50, verbose=False)
        hybrid.train(n_episodes=5, rollout_steps=20, verbose=False, eval_every=5)
        
        obs = np.random.uniform(0, 1, N_VARS).astype(np.float32)
        action = hybrid.act(obs)
        
        assert action.shape == (4,)
        assert np.all(action >= 0) and np.all(action <= 1)
    
    def test_act_with_explanation(self, fm3_data):
        data, interventional = fm3_data
        
        hybrid = CausalRLHybrid(config={'scm_hidden_dim': 8, 'policy_hidden': 32})
        hybrid.discover_causal_graph(data, interventional, verbose=False)
        hybrid.fit_world_model(data, n_epochs=50, verbose=False)
        hybrid.train(n_episodes=5, rollout_steps=20, verbose=False, eval_every=5)
        
        obs = np.random.uniform(0, 1, N_VARS).astype(np.float32)
        result = hybrid.act_with_explanation(obs)
        
        assert 'action' in result
        assert 'actuator_values' in result
    
    def test_online_update(self, fm3_data):
        data, interventional = fm3_data
        
        hybrid = CausalRLHybrid(config={'scm_hidden_dim': 8, 'policy_hidden': 32})
        hybrid.discover_causal_graph(data, interventional, verbose=False)
        hybrid.fit_world_model(data, n_epochs=50, verbose=False)
        hybrid.train(n_episodes=5, rollout_steps=20, verbose=False, eval_every=5)
        
        new_data = data[:200]
        hybrid.online_update(new_data, verbose=False)
    
    def test_summary(self, fm3_data):
        data, interventional = fm3_data
        
        hybrid = CausalRLHybrid(config={'scm_hidden_dim': 8, 'policy_hidden': 32})
        hybrid.discover_causal_graph(data, interventional, verbose=False)
        hybrid.fit_world_model(data, n_epochs=50, verbose=False)
        hybrid.train(n_episodes=5, rollout_steps=20, verbose=False, eval_every=5)
        
        summary = hybrid.summary()
        assert "CausalShield-RL" in summary
        assert "✓ Fitted" in summary


class TestPolicyNetwork:
    """Test the PPO policy network."""
    
    def test_forward(self):
        config = PPOConfig(hidden_dim=16)
        policy = PolicyNetwork(N_VARS, 4, config)
        
        obs = np.random.randn(N_VARS)
        mean, std = policy.forward(obs)
        
        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert np.all(std > 0)
    
    def test_sample_action(self):
        config = PPOConfig(hidden_dim=16)
        policy = PolicyNetwork(N_VARS, 4, config)
        
        obs = np.random.randn(N_VARS)
        action, log_prob = policy.sample_action(obs)
        
        assert action.shape == (4,)
        assert np.all(action >= 0) and np.all(action <= 1)
        assert np.isfinite(log_prob)
    
    def test_value(self):
        config = PPOConfig(hidden_dim=16)
        policy = PolicyNetwork(N_VARS, 4, config)
        
        obs = np.random.randn(N_VARS)
        val = policy.value(obs)
        assert isinstance(val, float)
        assert np.isfinite(val)


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
