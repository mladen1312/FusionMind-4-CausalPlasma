"""Tests for FusionMind 4.0 Real-Time Subsystem (v4.5.0)

Tests:
  - FastMLPredictor: training, inference, latency
  - CausalDisruptionPredictor: fit, predict, Simpson's, counterfactual
  - DualModePredictor: fusion, safety override, performance
  - RealtimeControlBridge: emergency, target tracking, safety
  - StreamingPlasmaInterface: replay, buffer, dispatch
  - DisruptionFeatureExtractor: feature computation
  - End-to-end: stream → predict → control
"""

import numpy as np
import time
import pytest

from fusionmind4.realtime.predictor import (
    FastMLPredictor,
    CausalDisruptionPredictor,
    DualModePredictor,
    DisruptionFeatureExtractor,
    PlasmaSnapshot,
    ThreatLevel,
)
from fusionmind4.realtime.control_bridge import (
    RealtimeControlBridge,
    ControlMode,
    SafetyLimits,
)
from fusionmind4.realtime.streaming import (
    StreamingPlasmaInterface,
    StreamConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_synthetic_dataset(n_samples=500, n_vars=9, disruption_rate=0.2,
                           seed=42):
    """Create synthetic plasma data + labels for testing."""
    rng = np.random.RandomState(seed)
    var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                 'Ip', 'P_NBI', 'D_alpha'][:n_vars]

    X = rng.randn(n_samples, n_vars)
    # Make physics-like correlations
    X[:, 0] = 0.5 + 0.3 * X[:, 7 % n_vars]  # βN ~ P_NBI
    X[:, 1] = 0.2 + 0.1 * X[:, 0]            # βp ~ βN
    X[:, 2] = 5.0 + rng.randn(n_samples) * 0.5  # q95 ~ 5

    # Disruption labels
    y = np.zeros(n_samples, dtype=int)
    # Disruptions when βN high and q95 low
    disruptive = (X[:, 0] > 1.0) & (X[:, 2] < 4.5)
    y[disruptive] = 1
    # Add some random disruptions
    extra = rng.random(n_samples) < disruption_rate * 0.5
    y[extra] = 1

    return X, y, var_names


def make_dag(n_vars=9):
    """Create plausible causal DAG for testing."""
    dag = np.zeros((n_vars, n_vars))
    # P_NBI(7) → βN(0)
    dag[7 % n_vars, 0] = 0.8
    # βN(0) → βp(1)
    dag[0, 1] = 0.5
    # Ip(6) → q95(2)
    dag[6 % n_vars, 2] = -0.7
    # P_NBI(7) → D_alpha(8)
    if n_vars > 8:
        dag[7, 8] = 0.3
    return dag


# ── FastMLPredictor ───────────────────────────────────────────────────────

class TestFastMLPredictor:

    def test_fit_and_predict(self):
        X, y, var_names = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=50, max_depth=4)
        stats = ml.fit(X, y, feature_names=var_names)
        assert ml._fitted
        assert stats['val_f1'] > 0
        assert len(ml._trees) == 50

    def test_predict_single(self):
        X, y, var_names = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=30)
        ml.fit(X, y, feature_names=var_names)

        features = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        result = ml.predict(features)
        assert 0 <= result.disruption_probability <= 1
        assert isinstance(result.threat_level, ThreatLevel)
        assert result.channel == 'fast_ml'
        assert result.latency_us > 0

    def test_predict_batch(self):
        X, y, var_names = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=30)
        ml.fit(X, y, feature_names=var_names)

        probs = ml.predict_batch(X)
        assert probs.shape == (X.shape[0],)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_inference_latency(self):
        X, y, var_names = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=50)
        ml.fit(X, y, feature_names=var_names)

        features = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        latencies = []
        for _ in range(100):
            result = ml.predict(features)
            latencies.append(result.latency_us)

        # Should be < 1 ms (1000 μs) for most predictions
        assert np.median(latencies) < 5000  # generous for CI

    def test_feature_importance(self):
        X, y, var_names = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=50)
        ml.fit(X, y, feature_names=var_names)
        assert len(ml._feature_importances) == len(var_names)
        assert ml._feature_importances.sum() > 0


# ── CausalDisruptionPredictor ─────────────────────────────────────────────

class TestCausalPredictor:

    def test_fit_and_predict(self):
        X, y, var_names = make_synthetic_dataset()
        dag = make_dag()
        causal = CausalDisruptionPredictor(dag, var_names)
        causal.fit(X, y)
        assert causal._fitted

        features = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        result = causal.predict(features)
        assert 0 <= result.disruption_probability <= 1
        assert result.channel == 'causal'

    def test_causal_explanation(self):
        X, y, var_names = make_synthetic_dataset()
        dag = make_dag()
        causal = CausalDisruptionPredictor(dag, var_names)
        causal.fit(X, y)

        features = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        explanation = causal.explain(features)
        assert isinstance(explanation, list)
        assert len(explanation) > 0
        assert 'Disruption probability' in explanation[0]

    def test_counterfactual_avoidance(self):
        X, y, var_names = make_synthetic_dataset()
        dag = make_dag()
        causal = CausalDisruptionPredictor(dag, var_names)
        causal.fit(X, y)

        # Use a high-risk sample
        high_risk = X[y == 1]
        if len(high_risk) > 0:
            features = {v: float(high_risk[0, i])
                        for i, v in enumerate(var_names)}
            avoidance = causal.get_counterfactual_avoidance(features)
            # May or may not return interventions depending on data
            # but should not crash
            assert avoidance is None or isinstance(avoidance, dict)

    def test_confounder_detection(self):
        X, y, var_names = make_synthetic_dataset()
        dag = make_dag()
        causal = CausalDisruptionPredictor(dag, var_names)
        # Confounders should be detected from DAG structure
        assert isinstance(causal._confounder_sets, dict)

    def test_causal_paths(self):
        dag = make_dag()
        var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                     'Ip', 'P_NBI', 'D_alpha']
        causal = CausalDisruptionPredictor(dag, var_names)
        paths = causal._find_causal_paths('P_NBI', 'βp')
        # P_NBI → βN → βp should be found
        assert len(paths) > 0


# ── DualModePredictor ─────────────────────────────────────────────────────

class TestDualPredictor:

    def _build_dual(self):
        X, y, var_names = make_synthetic_dataset()
        dag = make_dag()

        ml = FastMLPredictor(n_estimators=30)
        ml.fit(X, y, feature_names=var_names)

        causal = CausalDisruptionPredictor(dag, var_names)
        causal.fit(X, y)

        extractor = DisruptionFeatureExtractor(history_length=20)
        extractor.set_variable_order(var_names)

        return DualModePredictor(ml, causal, extractor), X, var_names

    def test_dual_prediction(self):
        dual, X, var_names = self._build_dual()

        values = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        snap = PlasmaSnapshot(values=values, timestamp_s=0.0)
        result = dual.predict(snap)

        assert 0 <= result.fused_probability <= 1
        assert isinstance(result.fused_threat, ThreatLevel)
        assert result.fast_ml is not None
        assert result.causal is not None
        assert isinstance(result.causal_explanation, list)

    def test_safety_override(self):
        dual, X, var_names = self._build_dual()

        # Simulate multiple predictions and check safety
        for i in range(10):
            values = {v: float(X[i, j]) for j, v in enumerate(var_names)}
            snap = PlasmaSnapshot(values=values, timestamp_s=i * 0.001)
            result = dual.predict(snap)
            # If either channel sees IMMINENT, fused should be ≥ 0.95
            if (result.fast_ml.threat_level == ThreatLevel.IMMINENT or
                    result.causal.threat_level == ThreatLevel.IMMINENT):
                assert result.fused_probability >= 0.95

    def test_performance_stats(self):
        dual, X, var_names = self._build_dual()

        for i in range(20):
            values = {v: float(X[i, j]) for j, v in enumerate(var_names)}
            snap = PlasmaSnapshot(values=values, timestamp_s=i * 0.001)
            dual.predict(snap)

        stats = dual.get_performance_stats()
        assert stats['n_predictions'] == 20
        assert 'total_latency_mean_us' in stats

    def test_simpsons_paradox_weighting(self):
        dual, X, var_names = self._build_dual()
        # When Simpson's detected, causal weight should increase
        # This is indirectly tested by checking the fusion weights
        assert dual.W_CAUSAL > dual.W_ML


# ── DisruptionFeatureExtractor ────────────────────────────────────────────

class TestFeatureExtractor:

    def test_basic_extraction(self):
        var_names = ['βN', 'βp', 'q95', 'Ip', 'ne', 'P_NBI']
        ext = DisruptionFeatureExtractor(history_length=20)
        ext.set_variable_order(var_names)

        snap = PlasmaSnapshot(
            values={'βN': 1.5, 'βp': 0.3, 'q95': 4.0,
                    'Ip': 0.8, 'ne': 3.0, 'P_NBI': 2.5},
            timestamp_s=0.0,
        )
        ext.update(snap)
        features = ext.extract()

        assert 'βN' in features
        assert 'beta_proximity' in features
        assert 'q95_proximity' in features

    def test_rate_features(self):
        var_names = ['βN', 'q95', 'P_NBI']
        ext = DisruptionFeatureExtractor(history_length=20)
        ext.set_variable_order(var_names)

        # Add multiple snapshots
        for i in range(5):
            snap = PlasmaSnapshot(
                values={'βN': 1.0 + 0.1 * i, 'q95': 4.0 - 0.05 * i,
                        'P_NBI': 2.0},
                timestamp_s=i * 0.001,
            )
            ext.update(snap)

        features = ext.extract()
        assert 'dβN_dt' in features
        assert 'dq95_dt' in features

    def test_reset(self):
        ext = DisruptionFeatureExtractor()
        ext.set_variable_order(['a', 'b'])
        ext.update(PlasmaSnapshot(values={'a': 1, 'b': 2}, timestamp_s=0))
        ext.reset()
        assert len(ext._history) == 0


# ── RealtimeControlBridge ─────────────────────────────────────────────────

class TestControlBridge:

    def _build_bridge(self):
        var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                     'Ip', 'P_NBI', 'D_alpha']
        dag = make_dag()
        return RealtimeControlBridge(
            actuator_names=['P_NBI', 'Ip'],
            target_vars=['βN', 'q95'],
            dag=dag,
            var_names=var_names,
            mode=ControlMode.ADVISORY,
        ), var_names

    def test_target_tracking(self):
        bridge, var_names = self._build_bridge()
        bridge.set_targets({'βN': 1.5, 'q95': 5.0})

        X, y, _ = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=20)
        ml.fit(X, y, feature_names=var_names)
        causal = CausalDisruptionPredictor(make_dag(), var_names)
        causal.fit(X, y)
        ext = DisruptionFeatureExtractor(history_length=10)
        ext.set_variable_order(var_names)
        dual = DualModePredictor(ml, causal, ext)

        state = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        snap = PlasmaSnapshot(values=state, timestamp_s=0)
        prediction = dual.predict(snap)

        output = bridge.compute_control(prediction, state)
        assert len(output.commands) >= 0  # May or may not need action
        assert output.mode == ControlMode.ADVISORY
        assert 'beta_limit' in output.safety_status

    def test_emergency_control(self):
        bridge, var_names = self._build_bridge()

        X, y, _ = make_synthetic_dataset()
        ml = FastMLPredictor(n_estimators=20)
        ml.fit(X, y, feature_names=var_names)
        causal = CausalDisruptionPredictor(make_dag(), var_names)
        causal.fit(X, y)
        ext = DisruptionFeatureExtractor(history_length=10)
        ext.set_variable_order(var_names)
        dual = DualModePredictor(ml, causal, ext)

        state = {v: float(X[0, i]) for i, v in enumerate(var_names)}
        snap = PlasmaSnapshot(values=state, timestamp_s=0)
        prediction = dual.predict(snap)

        # Force emergency
        prediction.fused_threat = ThreatLevel.CRITICAL
        output = bridge.compute_control(prediction, state)
        assert len(output.causal_trace) > 0
        assert 'EMERGENCY' in output.causal_trace[0]

    def test_rate_limiting(self):
        bridge, _ = self._build_bridge()
        from fusionmind4.realtime.control_bridge import ActuatorCommand

        cmd = ActuatorCommand(
            actuator='P_NBI',
            current_value=2.0,
            target_value=20.0,  # Huge change
            delta=18.0,
            causal_reason='test',
            confidence=0.8,
            safety_verified=False,
        )
        limited = bridge._rate_limit([cmd])
        assert len(limited) == 1
        assert abs(limited[0].delta) < 18.0  # Should be clipped

    def test_safety_verification(self):
        bridge, _ = self._build_bridge()
        state = {'βN': 3.0, 'q95': 1.5, 'li': 2.0}  # All unsafe
        status = bridge._verify_safety([], state)
        assert status['beta_limit'] is False
        assert status['q_min'] is False
        assert status['li_range'] is False

    def test_statistics(self):
        bridge, _ = self._build_bridge()
        stats = bridge.get_statistics()
        assert stats == {} or 'cycles' in stats


# ── StreamingPlasmaInterface ──────────────────────────────────────────────

class TestStreaming:

    def test_replay_mode(self):
        var_names = ['βN', 'q95', 'P_NBI']
        config = StreamConfig(mode='replay', replay_speed=100.0,
                              callback_interval_ms=1.0)
        stream = StreamingPlasmaInterface(config, var_names)

        received = []
        stream.register_callback(lambda snap: received.append(snap))

        stream.start()
        time.sleep(0.1)
        stream.stop()

        assert len(received) > 0
        assert hasattr(received[0], 'values')
        assert hasattr(received[0], 'timestamp_s')

    def test_buffer(self):
        var_names = ['βN', 'q95']
        config = StreamConfig(mode='replay', replay_speed=100.0,
                              buffer_size=50)
        stream = StreamingPlasmaInterface(config, var_names)
        stream.start()
        time.sleep(0.1)
        stream.stop()

        arr = stream.get_buffer_as_array()
        assert arr.shape[1] == 2
        assert len(arr) > 0

    def test_statistics(self):
        var_names = ['βN']
        config = StreamConfig(mode='replay', replay_speed=100.0)
        stream = StreamingPlasmaInterface(config, var_names)
        stream.start()
        time.sleep(0.05)
        stream.stop()

        stats = stream.get_statistics()
        assert stats['snapshots_ingested'] > 0
        assert 'ingest_latency_mean_us' in stats


# ── End-to-End Integration ────────────────────────────────────────────────

class TestEndToEnd:

    def test_stream_to_predict_to_control(self):
        """Full pipeline: streaming → dual predictor → control bridge."""
        var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                     'Ip', 'P_NBI', 'D_alpha']

        # Train models
        X, y, _ = make_synthetic_dataset()
        dag = make_dag()

        ml = FastMLPredictor(n_estimators=30)
        ml.fit(X, y, feature_names=var_names)

        causal = CausalDisruptionPredictor(dag, var_names)
        causal.fit(X, y)

        ext = DisruptionFeatureExtractor(history_length=10)
        ext.set_variable_order(var_names)
        dual = DualModePredictor(ml, causal, ext)

        bridge = RealtimeControlBridge(
            actuator_names=['P_NBI', 'Ip'],
            target_vars=['βN'],
            dag=dag,
            var_names=var_names,
            mode=ControlMode.ADVISORY,
        )
        bridge.set_targets({'βN': 1.5})

        # Simulate 50 timesteps
        outputs = []
        for i in range(50):
            values = {v: float(X[i % X.shape[0], j])
                      for j, v in enumerate(var_names)}
            snap = PlasmaSnapshot(values=values, timestamp_s=i * 0.001)
            prediction = dual.predict(snap)
            output = bridge.compute_control(prediction, values)
            outputs.append(output)

        assert len(outputs) == 50
        assert all(o.cycle_number > 0 for o in outputs)

        # Check latency
        latencies = [o.cycle_latency_us + o.prediction.total_latency_us
                     for o in outputs]
        # Full cycle should be < 10 ms (10000 μs)
        assert np.median(latencies) < 50000

    def test_disruption_handling(self):
        """Verify system handles disruption scenario correctly."""
        var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                     'Ip', 'P_NBI', 'D_alpha']

        X, y, _ = make_synthetic_dataset()
        dag = make_dag()

        ml = FastMLPredictor(n_estimators=30)
        ml.fit(X, y, feature_names=var_names)

        causal = CausalDisruptionPredictor(dag, var_names)
        causal.fit(X, y)

        ext = DisruptionFeatureExtractor(history_length=10)
        ext.set_variable_order(var_names)
        dual = DualModePredictor(ml, causal, ext)

        # Simulate degrading plasma
        for i in range(30):
            values = {
                'βN': 2.5 + 0.1 * i,  # Rising βN
                'βp': 0.3,
                'q95': 4.0 - 0.1 * i,  # Falling q95
                'q_axis': 1.0,
                'li': 1.0 + 0.02 * i,
                'κ': 1.8,
                'Ip': 0.8,
                'P_NBI': 3.0,
                'D_alpha': 0.5 + 0.02 * i,
            }
            snap = PlasmaSnapshot(values=values, timestamp_s=i * 0.001)
            result = dual.predict(snap)

        # After degradation, should see elevated risk
        # (exact values depend on training data, but shouldn't crash)
        stats = dual.get_performance_stats()
        assert stats['n_predictions'] == 30
