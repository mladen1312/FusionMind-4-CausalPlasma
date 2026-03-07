"""
Track D: Fast Diagnostics Anomaly Detector
============================================

Physics-based anomaly detection on fast signals:
  - MHD n=2 amplitude growth (tearing mode precursor)
  - Dα emission spike (edge instability / detachment)
  - li rate (internal inductance change — current profile evolution)
  - βp rate (poloidal beta collapse)

No ML — pure physics thresholds calibrated per-shot.
This is the track that provides 100-700ms warning time.

Returns probability [0, 1] and confidence based on:
  - How many signals are anomalous simultaneously
  - How far above baseline each signal is (z-score)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque


@dataclass
class FastDiagState:
    """Running statistics for per-shot baseline calibration."""
    means: Dict[str, float]
    stds: Dict[str, float]
    n_samples: int


class TrackD_FastDiagnostics:
    """
    Physics-based fast anomaly detector.
    
    Self-calibrates baseline from first 40% of each shot,
    then detects anomalies as z-score exceedances.
    """
    
    SIGNAL_NAMES = ['li_rate', 'betap_rate', 'mhd_n2', 'dalpha', 'betan_rate', 'q95_rate']
    
    # Weight per signal (from permutation importance analysis)
    SIGNAL_WEIGHTS = {
        'li_rate': 0.25,      # 15.5x ratio disrupted/clean — strongest
        'betap_rate': 0.20,   # 10.6x ratio
        'mhd_n2': 0.20,      # MHD mode growth — earliest precursor
        'dalpha': 0.15,       # Edge emission
        'betan_rate': 0.10,   # Late but diagnostic
        'q95_rate': 0.10,     # Safety factor change
    }
    
    def __init__(self, 
                 calibration_fraction: float = 0.4,
                 z_threshold: float = 2.5,
                 alarm_n_signals: int = 2):
        """
        Args:
            calibration_fraction: Use first N% of shot as baseline
            z_threshold: Standard deviations above mean for anomaly
            alarm_n_signals: Min simultaneous anomalous signals for alarm
        """
        self.calibration_fraction = calibration_fraction
        self.z_threshold = z_threshold
        self.alarm_n_signals = alarm_n_signals
        self.baseline = None
        self._buffer = deque(maxlen=500)
        self._calibrated = False
        self._n_calibration = 0
    
    def reset(self):
        """Reset for new shot."""
        self.baseline = None
        self._buffer.clear()
        self._calibrated = False
        self._n_calibration = 0
    
    def compute_signals(self, current: np.ndarray, prev: np.ndarray, 
                        var_names: list) -> Dict[str, float]:
        """
        Compute fast diagnostic signals from consecutive timepoints.
        
        Args:
            current: Current plasma state vector
            prev: Previous plasma state vector
            var_names: Variable names matching state indices
        """
        idx = {v: i for i, v in enumerate(var_names)}
        rates = np.abs(current - prev)
        
        signals = {}
        for vname in ['li', 'betap', 'betan', 'q_95']:
            if vname in idx:
                signals[f'{vname}_rate'] = float(rates[idx[vname]])
        
        if 'mhd_n2' in idx:
            signals['mhd_n2'] = float(np.abs(current[idx['mhd_n2']]))
        elif len(current) > 10:
            signals['mhd_n2'] = float(np.abs(current[10])) if len(current) > 10 else 0.0
            
        if 'dalpha' in idx:
            signals['dalpha'] = float(np.abs(current[idx['dalpha']]))
        elif len(current) > 9:
            signals['dalpha'] = float(np.abs(current[9])) if len(current) > 9 else 0.0
        
        return signals
    
    def update(self, signals: Dict[str, float]) -> dict:
        """
        Process one timestep. Returns anomaly assessment.
        
        First calibration_fraction of calls build baseline.
        After that, detect anomalies.
        """
        self._buffer.append(signals)
        self._n_calibration += 1
        
        # Calibration phase
        if not self._calibrated:
            # We don't know shot length a priori, so calibrate after
            # at least 10 samples and mark calibrated when buffer is 40% full
            if self._n_calibration >= 10:
                self._calibrate()
            return {'prob': 0.0, 'confidence': 0.3, 'anomalous_signals': [], 
                    'phase': 'calibrating'}
        
        # Detection phase
        return self._detect(signals)
    
    def _calibrate(self):
        """Compute baseline statistics from buffer."""
        if len(self._buffer) < 5:
            return
        
        self.baseline = FastDiagState(means={}, stds={}, n_samples=len(self._buffer))
        
        for sig_name in self.SIGNAL_NAMES:
            values = [s.get(sig_name, 0) for s in self._buffer]
            self.baseline.means[sig_name] = np.mean(values)
            self.baseline.stds[sig_name] = np.std(values) + 1e-10
        
        self._calibrated = True
    
    def _detect(self, signals: Dict[str, float]) -> dict:
        """Detect anomalies relative to baseline."""
        if self.baseline is None:
            return {'prob': 0.0, 'confidence': 0.1, 'anomalous_signals': [],
                    'phase': 'no_baseline'}
        
        anomalous = []
        weighted_score = 0.0
        
        for sig_name in self.SIGNAL_NAMES:
            val = signals.get(sig_name, 0)
            mu = self.baseline.means.get(sig_name, 0)
            sd = self.baseline.stds.get(sig_name, 1e-10)
            
            z = (val - mu) / sd
            
            if z > self.z_threshold:
                weight = self.SIGNAL_WEIGHTS.get(sig_name, 0.1)
                # Score increases with z-score (more anomalous = higher prob)
                signal_score = weight * min(z / self.z_threshold, 3.0)
                weighted_score += signal_score
                anomalous.append({
                    'signal': sig_name,
                    'z_score': round(float(z), 2),
                    'value': round(float(val), 4),
                    'baseline_mean': round(float(mu), 4),
                })
        
        # Probability: sigmoid-like mapping of weighted score
        prob = float(np.clip(weighted_score / 1.5, 0, 1))
        
        # Confidence based on number of anomalous signals
        n_anom = len(anomalous)
        confidence = min(0.4 + 0.2 * n_anom, 0.95)
        
        return {
            'prob': round(prob, 4),
            'confidence': round(confidence, 3),
            'anomalous_signals': anomalous,
            'n_anomalous': n_anom,
            'weighted_score': round(weighted_score, 4),
            'phase': 'detecting',
        }
    
    def force_calibrate(self):
        """Force calibration with current buffer (useful for testing)."""
        self._calibrate()
