"""Tests for Dynamic Overseer and Track D (Fast Diagnostics)."""
import numpy as np
import pytest
from fusionmind4.control.dynamic_overseer import DynamicOverseer, TrackOutput, OverseerDecision
from fusionmind4.control.track_fast import TrackD_FastDiagnostics


class TestDynamicOverseer:
    def setup_method(self):
        self.overseer = DynamicOverseer(
            history_len=10, disagreement_threshold=0.20,
            alarm_threshold=0.5, warning_threshold=0.3)
    
    def _make_tracks(self, pA=0.1, pB=0.1, pC=0.1, pD=0.1):
        return {
            'A': TrackOutput('A', pA, 0.85),
            'B': TrackOutput('B', pB, 0.80),
            'C': TrackOutput('C', pC, 0.95),
            'D': TrackOutput('D', pD, 0.90),
        }
    
    def test_low_disagreement_uses_best_track(self):
        tracks = self._make_tracks(0.1, 0.1, 0.1, 0.1)
        decision = self.overseer.decide(tracks)
        assert isinstance(decision, OverseerDecision)
        assert 0 <= decision.final_prob <= 1
        assert not decision.correction_applied
    
    def test_high_disagreement_trusts_physics(self):
        tracks = self._make_tracks(pA=0.9, pB=0.1, pC=0.7, pD=0.8)
        decision = self.overseer.decide(tracks)
        assert decision.correction_applied
        assert decision.best_track in ('C', 'D')
    
    def test_history_accumulates(self):
        for _ in range(5):
            self.overseer.decide(self._make_tracks(0.2, 0.2, 0.2, 0.2))
        assert len(self.overseer.history) == 5
    
    def test_reset_clears_state(self):
        self.overseer.decide(self._make_tracks())
        self.overseer.reset()
        assert len(self.overseer.history) == 0
    
    def test_safety_override(self):
        """If any track says alarm, overseer should not wash it out."""
        self.overseer.decide(self._make_tracks(0.1, 0.1, 0.1, 0.1))  # Build history
        tracks = self._make_tracks(pA=0.1, pB=0.1, pC=0.1, pD=0.8)
        decision = self.overseer.decide(tracks)
        assert decision.final_prob > 0.3  # Should not be washed out to 0.1
    
    def test_warning_levels(self):
        d1 = self.overseer.decide(self._make_tracks(0.05, 0.05, 0.05, 0.05))
        assert d1.warning_level == 0  # SAFE
        
        self.overseer.reset()
        d2 = self.overseer.decide(self._make_tracks(0.35, 0.35, 0.35, 0.35))
        assert d2.warning_level >= 2  # WARNING or ALARM
    
    def test_track_weights_sum_to_one(self):
        decision = self.overseer.decide(self._make_tracks(0.3, 0.2, 0.4, 0.5))
        total = sum(decision.track_weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_physics_priority(self):
        """Physics tracks (C, D) should get higher weight."""
        decision = self.overseer.decide(self._make_tracks(0.3, 0.3, 0.3, 0.3))
        assert decision.track_weights['C'] >= decision.track_weights['A']


class TestTrackD:
    def setup_method(self):
        self.td = TrackD_FastDiagnostics(z_threshold=2.5, calibration_fraction=0.4)
    
    def test_calibration_phase(self):
        signals = {'li_rate': 0.1, 'betap_rate': 0.05, 'mhd_n2': 0.01, 
                   'dalpha': 0.5, 'betan_rate': 0.02, 'q95_rate': 0.01}
        result = self.td.update(signals)
        assert result['phase'] == 'calibrating'
        assert result['prob'] == 0.0
    
    def test_detection_after_calibration(self):
        # Feed normal signals to calibrate
        for _ in range(15):
            signals = {'li_rate': 0.1 + np.random.normal(0, 0.01),
                      'betap_rate': 0.05, 'mhd_n2': 0.01, 'dalpha': 0.5,
                      'betan_rate': 0.02, 'q95_rate': 0.01}
            self.td.update(signals)
        
        self.td.force_calibrate()
        
        # Feed anomalous signal
        anomalous = {'li_rate': 2.0, 'betap_rate': 1.5, 'mhd_n2': 0.5,
                    'dalpha': 5.0, 'betan_rate': 0.5, 'q95_rate': 0.3}
        result = self.td._detect(anomalous)
        assert result['prob'] > 0.3
        assert result['n_anomalous'] >= 2
    
    def test_normal_signals_no_alarm(self):
        np.random.seed(42)
        for _ in range(15):
            self.td.update({'li_rate': 0.1 + np.random.normal(0, 0.02),
                           'betap_rate': 0.05 + np.random.normal(0, 0.01),
                           'mhd_n2': 0.01 + np.random.normal(0, 0.003),
                           'dalpha': 0.5 + np.random.normal(0, 0.05),
                           'betan_rate': 0.02 + np.random.normal(0, 0.005),
                           'q95_rate': 0.01 + np.random.normal(0, 0.003)})
        self.td.force_calibrate()
        
        # Signal within 1σ of mean — should not alarm
        result = self.td._detect({'li_rate': 0.11, 'betap_rate': 0.055, 'mhd_n2': 0.012,
                                  'dalpha': 0.52, 'betan_rate': 0.023, 'q95_rate': 0.012})
        assert result['prob'] < 0.3
    
    def test_reset(self):
        self.td.update({'li_rate': 0.1, 'betap_rate': 0.05, 'mhd_n2': 0.01,
                       'dalpha': 0.5, 'betan_rate': 0.02, 'q95_rate': 0.01})
        self.td.reset()
        assert not self.td._calibrated
        assert self.td.baseline is None
    
    def test_compute_signals(self):
        vars = ['betan','betap','q_95','q_axis','elongation','li',
                'wplasmd','betat','Ip','dalpha','mhd_n2','ne_line']
        current = np.random.rand(12)
        prev = np.random.rand(12)
        signals = self.td.compute_signals(current, prev, vars)
        assert 'li_rate' in signals
        assert 'mhd_n2' in signals
        assert 'dalpha' in signals
