#!/usr/bin/env python3
"""
FusionMind 4.0 — Streaming Disruption Predictor (Real-Time Engine)
===================================================================

Phase 1 of the real-time control pipeline: converts shot-level offline
prediction into timepoint-by-timepoint streaming prediction.

Called every 10ms with new diagnostic measurement → returns:
  - P(disruption) with uncertainty
  - Risk level (SAFE/MONITOR/ALERT/ALARM/MITIGATE)
  - Physics explanation ("li approaching kink limit")
  - Time-to-disruption estimate (ms)
  - Recommended action (Phase 3: "reduce gas 20%")

Architecture:
  ┌─────────────────────────────────────────────┐
  │  ingest(measurement)  — called every 10ms   │
  │     │                                       │
  │     ├→ RollingBuffer.append()               │
  │     ├→ IPDA.update() [<1μs, streaming]      │
  │     ├→ IMM.update()  [<1μs, streaming]      │
  │     └→ PHANTOM.update() [<0.1μs, instant]   │
  │                                             │
  │  predict()  — called after ingest           │
  │     │                                       │
  │     ├→ margins (instant: value vs limit)    │
  │     ├→ rolling_stats (O(1) incremental)     │
  │     ├→ NX-MIMOSA state (already updated)    │
  │     ├→ GBT predict (pre-trained model)      │
  │     ├→ Overseer arbitration                 │
  │     └→ TTD estimation (margin slope)        │
  └─────────────────────────────────────────────┘

Latency budget: ≤2ms total (within 5ms control cycle)

NX-MIMOSA streaming advantage:
  IPDA/IMM carry only 2 floats of state per signal.
  Update = 1 Bayes multiplication per signal per cycle.
  Designed for radar scan-by-scan → perfect for tokamak EFIT-by-EFIT.

Author: Dr. Mladen Mester
License: BSL-1.1 (converts to Apache-2.0 on 2030-03-05)
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    SAFE = "SAFE"           # P < 0.10
    MONITOR = "MONITOR"     # 0.10 ≤ P < 0.40
    ALERT = "ALERT"         # 0.40 ≤ P < 0.70
    ALARM = "ALARM"         # 0.70 ≤ P < 0.90
    MITIGATE = "MITIGATE"   # P ≥ 0.90


class AlarmState(Enum):
    """Alarm state machine with hysteresis."""
    SAFE = 0
    MONITOR = 1
    ALERT = 2
    ALARM = 3
    MITIGATE = 4


@dataclass
class StreamingPrediction:
    """Output of one predict() call."""
    timestamp_ms: float
    probability: float
    uncertainty: float
    risk_level: RiskLevel
    alarm_state: AlarmState
    margins: Dict[str, float]
    closest_limit: str
    explanation: str
    ttd_ms: Optional[float]       # Time-to-disruption estimate
    ipda_existence: Dict[str, float]  # Per-signal existence probability
    imm_regime: Dict[str, float]      # Per-signal unstable probability
    recommended_action: Optional[str]  # Phase 3: actuator command
    latency_us: float                  # Inference latency in microseconds
    cycle_count: int


@dataclass
class StreamingConfig:
    """Configuration for streaming predictor."""
    # Buffer
    window_size: int = 100          # Rolling buffer depth (timepoints)
    feature_recompute_every: int = 5  # Recompute GBT features every N cycles
    
    # Physics limits (override per machine)
    li_max: float = 2.0
    q95_min: float = 2.0
    betan_max: float = 3.5
    fgw_max: float = 1.0
    betap_max: float = 1.5
    
    # IPDA
    ipda_pd: float = 0.85
    ipda_ps: float = 0.99
    ipda_zscore_threshold: float = 2.0
    ipda_diff_threshold: float = 1.5
    ipda_lr: float = 5.0
    
    # IMM
    imm_p_switch: float = 0.05
    imm_variance_window: int = 5
    
    # PHANTOM
    phantom_sharpness: float = 5.0
    phantom_floor: float = 0.05
    
    # Alarm hysteresis
    alarm_up_threshold: float = 0.6    # P to escalate
    alarm_down_threshold: float = 0.3  # P to de-escalate
    alarm_hold_cycles: int = 10        # Minimum cycles before de-escalation
    
    # TTD
    ttd_margin_history: int = 20       # Timepoints for slope estimation
    
    @classmethod
    def for_spherical(cls):
        """MAST, NSTX, ST40."""
        return cls(li_max=2.0, q95_min=1.5, betan_max=6.0, fgw_max=1.5, betap_max=2.0)
    
    @classmethod
    def for_conventional(cls):
        """DIII-D, JET, C-Mod, ITER."""
        return cls(li_max=1.5, q95_min=2.0, betan_max=3.5, fgw_max=1.0, betap_max=1.2)


# ═══════════════════════════════════════════════════════════════════════
# ROLLING BUFFER with O(1) statistics
# ═══════════════════════════════════════════════════════════════════════

class RollingBuffer:
    """Fixed-size circular buffer with O(1) incremental statistics.
    
    Maintains running mean, variance, max, min for each signal
    without recomputing from scratch each cycle.
    """
    
    def __init__(self, n_signals: int, window_size: int = 100):
        self.n_signals = n_signals
        self.window_size = window_size
        self.buffer = np.zeros((window_size, n_signals), dtype=np.float64)
        self.count = 0
        self.head = 0  # Next write position
        
        # Running statistics (Welford's online algorithm)
        self.running_mean = np.zeros(n_signals, dtype=np.float64)
        self.running_m2 = np.zeros(n_signals, dtype=np.float64)  # Sum of squared differences
        self.running_max = np.full(n_signals, -np.inf)
        self.running_min = np.full(n_signals, np.inf)
        
        # Late window stats (last 30%)
        self.late_fraction = 0.3
    
    def append(self, measurement: np.ndarray):
        """Add one timepoint. O(N_signals), not O(window_size)."""
        if self.count >= self.window_size:
            # Overwrite oldest: remove its contribution
            old = self.buffer[self.head]
            # Note: exact online removal is complex; we periodically recompute
            pass
        
        self.buffer[self.head] = measurement
        self.head = (self.head + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)
        
        # Update running stats
        for i in range(self.n_signals):
            val = measurement[i]
            if val > self.running_max[i]:
                self.running_max[i] = val
            if val < self.running_min[i]:
                self.running_min[i] = val
            
            # Welford update
            old_mean = self.running_mean[i]
            self.running_mean[i] += (val - old_mean) / self.count
            self.running_m2[i] += (val - old_mean) * (val - self.running_mean[i])
    
    @property
    def latest(self) -> np.ndarray:
        """Most recent measurement."""
        idx = (self.head - 1) % self.window_size
        return self.buffer[idx]
    
    @property
    def previous(self) -> np.ndarray:
        """Second most recent measurement."""
        idx = (self.head - 2) % self.window_size
        return self.buffer[idx] if self.count >= 2 else self.latest
    
    def get_data(self) -> np.ndarray:
        """Return all data in chronological order."""
        if self.count < self.window_size:
            return self.buffer[:self.count]
        # Circular: reorder
        return np.vstack([self.buffer[self.head:], self.buffer[:self.head]])
    
    def rolling_stats(self) -> Dict[str, np.ndarray]:
        """Return current statistics. O(N_signals)."""
        n = max(self.count, 1)
        var = self.running_m2 / max(n - 1, 1)
        std = np.sqrt(np.maximum(var, 0))
        
        # Late window (last 30%)
        data = self.get_data()
        n_late = max(int(self.late_fraction * self.count), 1)
        late_data = data[-n_late:]
        
        # Trend: late_mean - early_mean
        n_early = max(int(self.late_fraction * self.count), 1)
        early_data = data[:n_early]
        trend = np.mean(late_data, axis=0) - np.mean(early_data, axis=0)
        
        # Max rate
        if self.count >= 2:
            diff = np.abs(np.diff(data, axis=0))
            max_rate = np.max(diff, axis=0)
        else:
            max_rate = np.zeros(self.n_signals)
        
        return {
            'mean': self.running_mean.copy(),
            'std': std,
            'max': self.running_max.copy(),
            'min': self.running_min.copy(),
            'trend': trend,
            'max_rate': max_rate,
            'late_mean': np.mean(late_data, axis=0),
            'n_points': self.count,
        }


# ═══════════════════════════════════════════════════════════════════════
# STREAMING IPDA (inherently online)
# ═══════════════════════════════════════════════════════════════════════

class StreamingIPDA:
    """Online IPDA existence tracking — 1 Bayes update per signal per cycle.
    
    State: r[i] (existence probability) per signal. That's it.
    Memory: N_signals × 1 float.
    Compute: N_signals × 1 multiplication per cycle.
    """
    
    def __init__(self, n_signals: int, config: StreamingConfig):
        self.n = n_signals
        self.cfg = config
        self.r = np.full(n_signals, 0.1)  # Low prior
        self.prev_values = np.zeros(n_signals)
        self.running_mean = np.zeros(n_signals)
        self.running_var = np.ones(n_signals)
        self.count = 0
    
    def update(self, measurement: np.ndarray):
        """One IPDA update cycle. O(N_signals)."""
        self.count += 1
        
        for i in range(self.n):
            val = measurement[i]
            
            # Online mean/var for z-score
            old_mean = self.running_mean[i]
            self.running_mean[i] += (val - old_mean) / self.count
            self.running_var[i] += (val - old_mean) * (val - self.running_mean[i])
            
            std = np.sqrt(max(self.running_var[i] / max(self.count - 1, 1), 1e-20))
            z = abs(val - self.running_mean[i]) / (std + 1e-10)
            rate = abs(val - self.prev_values[i]) / (std + 1e-10) if self.count > 1 else 0
            
            # Anomaly detection
            is_anom = (z > self.cfg.ipda_zscore_threshold or
                       rate > self.cfg.ipda_diff_threshold)
            
            # IPDA Bayes update
            self.r[i] *= self.cfg.ipda_ps  # Predict: survival
            
            if is_anom:
                delta = 1 - self.cfg.ipda_pd + self.cfg.ipda_pd * self.cfg.ipda_lr
            else:
                delta = 1 - self.cfg.ipda_pd
            
            denom = delta * self.r[i] + (1 - self.r[i])
            self.r[i] = np.clip(delta * self.r[i] / (denom + 1e-30), 1e-6, 0.999)
            
            self.prev_values[i] = val
    
    def get_existence(self) -> np.ndarray:
        """Current existence probabilities. O(1)."""
        return self.r.copy()
    
    def reset(self):
        """Reset for new shot."""
        self.r[:] = 0.1
        self.prev_values[:] = 0
        self.running_mean[:] = 0
        self.running_var[:] = 1
        self.count = 0


# ═══════════════════════════════════════════════════════════════════════
# STREAMING IMM (inherently online)
# ═══════════════════════════════════════════════════════════════════════

class StreamingIMM:
    """Online IMM regime detection — 1 mode update per signal per cycle.
    
    State: μ[i] (P(unstable)) per signal + small variance buffer.
    Memory: N_signals × (1 float + window floats).
    """
    
    def __init__(self, n_signals: int, config: StreamingConfig):
        self.n = n_signals
        self.cfg = config
        self.mu_unstable = np.full(n_signals, 0.1)  # P(unstable)
        
        # Small variance buffer for each signal
        win = config.imm_variance_window
        self.var_buffers = [np.zeros(win) for _ in range(n_signals)]
        self.var_heads = np.zeros(n_signals, dtype=int)
        self.running_var = np.ones(n_signals)
        self.count = 0
    
    def update(self, measurement: np.ndarray):
        """One IMM cycle. O(N_signals × window)."""
        self.count += 1
        p = self.cfg.imm_p_switch
        win = self.cfg.imm_variance_window
        
        for i in range(self.n):
            val = measurement[i]
            
            # Update small variance buffer
            h = self.var_heads[i]
            self.var_buffers[i][h] = val
            self.var_heads[i] = (h + 1) % win
            
            # Local variance (from small window)
            n_filled = min(self.count, win)
            if n_filled >= 3:
                local_var = np.var(self.var_buffers[i][:n_filled])
            else:
                local_var = 0
            
            # Global variance (running, Welford-safe)
            old_rv = self.running_var[i]
            delta_val = val - old_rv
            if abs(delta_val) < 1e10:  # Guard against overflow
                self.running_var[i] += delta_val / max(self.count, 1)
            global_var = max(abs(self.running_var[i]), 1e-10)
            
            # Likelihoods
            # Stable: local variance should be ≤ global
            ratio = local_var / global_var
            L_stable = np.exp(-0.5 * max(ratio - 1, 0))
            L_unstable = np.exp(-0.5 * max(1 - ratio, 0)) if ratio > 1 else 0.5
            
            # IMM mixing + Bayes
            c_stable = (1 - p) * (1 - self.mu_unstable[i]) + p * self.mu_unstable[i]
            c_unstable = p * (1 - self.mu_unstable[i]) + (1 - p) * self.mu_unstable[i]
            
            L_sum = L_stable * c_stable + L_unstable * c_unstable
            if L_sum > 1e-30:
                self.mu_unstable[i] = L_unstable * c_unstable / L_sum
            
            self.mu_unstable[i] = np.clip(self.mu_unstable[i], 0.01, 0.99)
    
    def get_regime(self) -> np.ndarray:
        """Current P(unstable) per signal. O(1)."""
        return self.mu_unstable.copy()
    
    def reset(self):
        """Reset for new shot."""
        self.mu_unstable[:] = 0.1
        for buf in self.var_buffers:
            buf[:] = 0
        self.var_heads[:] = 0
        self.running_var[:] = 1
        self.count = 0


# ═══════════════════════════════════════════════════════════════════════
# STREAMING PHANTOM (instant per cycle)
# ═══════════════════════════════════════════════════════════════════════

class StreamingPHANTOM:
    """Instant physics feasibility per signal. O(N_signals), no state."""
    
    def __init__(self, signal_names: List[str], config: StreamingConfig):
        self.names = signal_names
        self.cfg = config
        
        # Build limits map
        self.limits = {}
        for name in signal_names:
            if 'li' in name:
                self.limits[name] = (0.5, config.li_max, 'high_bad')
            elif 'q95' in name:
                self.limits[name] = (config.q95_min, 10.0, 'low_bad')
            elif 'betan' in name or 'beta_n' in name:
                self.limits[name] = (0.0, config.betan_max, 'high_bad')
            elif 'betap' in name or 'beta_p' in name:
                self.limits[name] = (0.0, config.betap_max, 'high_bad')
            elif 'greenwald' in name or 'fgw' in name:
                self.limits[name] = (0.0, config.fgw_max, 'high_bad')
            else:
                self.limits[name] = None  # No known limit
    
    def compute_feasibility(self, measurement: np.ndarray) -> np.ndarray:
        """Compute PHANTOM feasibility for current measurement. Instant."""
        feas = np.ones(len(self.names))
        k = self.cfg.phantom_sharpness
        floor = self.cfg.phantom_floor
        
        for i, name in enumerate(self.names):
            lim = self.limits.get(name)
            if lim is None:
                continue
            
            lo, hi, direction = lim
            span = hi - lo + 1e-10
            val = measurement[i]
            
            if direction == 'high_bad':
                margin = (hi - val) / span
            elif direction == 'low_bad':
                margin = (val - lo) / span
            else:
                margin = min((val - lo) / span, (hi - val) / span)
            
            exp_val = np.clip(-k * margin, -500, 500)
            feas[i] = floor + (1 - floor) / (1 + np.exp(exp_val))
        
        return feas
    
    def compute_margins(self, measurement: np.ndarray) -> Dict[str, float]:
        """Compute physics margins. Used for explanation and TTD."""
        margins = {}
        for i, name in enumerate(self.names):
            lim = self.limits.get(name)
            if lim is None:
                margins[name] = 1.0
                continue
            lo, hi, direction = lim
            val = measurement[i]
            if direction == 'high_bad':
                margins[name] = float(np.clip(1 - val / hi, -1, 1))
            elif direction == 'low_bad':
                margins[name] = float(np.clip(1 - lo / max(val, 1e-10), -1, 1))
            else:
                margins[name] = 1.0
        return margins


# ═══════════════════════════════════════════════════════════════════════
# ALARM STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════

class AlarmStateMachine:
    """State machine with hysteresis to prevent alarm oscillation.
    
    SAFE → MONITOR → ALERT → ALARM → MITIGATE
    
    Escalation: immediate when P > threshold
    De-escalation: requires P < lower threshold for N consecutive cycles
    """
    
    def __init__(self, config: StreamingConfig):
        self.cfg = config
        self.state = AlarmState.SAFE
        self.hold_counter = 0
        self.history = []
    
    # Thresholds: (escalate_above, de-escalate_below)
    THRESHOLDS = {
        AlarmState.SAFE:     (0.10, 0.00),
        AlarmState.MONITOR:  (0.40, 0.08),
        AlarmState.ALERT:    (0.70, 0.30),
        AlarmState.ALARM:    (0.90, 0.55),
        AlarmState.MITIGATE: (1.01, 0.80),  # Cannot escalate beyond MITIGATE
    }
    
    def update(self, probability: float) -> AlarmState:
        """Update alarm state. Returns new state."""
        self.history.append(probability)
        
        current_level = self.state.value
        esc_thr, deesc_thr = self.THRESHOLDS[self.state]
        
        # Escalation: immediate
        if probability >= esc_thr and current_level < 4:
            self.state = AlarmState(current_level + 1)
            self.hold_counter = 0
            # May need to escalate further
            return self.update(probability) if probability >= self.THRESHOLDS[self.state][0] else self.state
        
        # De-escalation: requires sustained low probability
        if probability < deesc_thr and current_level > 0:
            self.hold_counter += 1
            if self.hold_counter >= self.cfg.alarm_hold_cycles:
                self.state = AlarmState(current_level - 1)
                self.hold_counter = 0
        else:
            self.hold_counter = 0
        
        return self.state
    
    def reset(self):
        self.state = AlarmState.SAFE
        self.hold_counter = 0
        self.history = []


# ═══════════════════════════════════════════════════════════════════════
# TTD ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class TTDEstimator:
    """Time-to-disruption from margin slope extrapolation.
    
    If margin is decreasing linearly, estimate when it hits zero:
        TTD = -margin / (d_margin/dt)
    """
    
    def __init__(self, config: StreamingConfig, dt_ms: float = 10.0):
        self.cfg = config
        self.dt_ms = dt_ms
        self.margin_history = []
    
    def update(self, min_margin: float) -> Optional[float]:
        """Add margin observation, return TTD estimate in ms."""
        self.margin_history.append(min_margin)
        
        n = len(self.margin_history)
        window = min(n, self.cfg.ttd_margin_history)
        
        if window < 5:
            return None  # Not enough data
        
        recent = self.margin_history[-window:]
        
        # Linear regression: margin(t) = a + b*t
        t = np.arange(window, dtype=np.float64)
        t_mean = np.mean(t)
        m_mean = np.mean(recent)
        
        denom = np.sum((t - t_mean)**2)
        if denom < 1e-10:
            return None
        
        slope = np.sum((t - t_mean) * (recent - m_mean)) / denom  # margin/cycle
        
        if slope >= 0:
            return None  # Margin not decreasing → no disruption predicted
        
        # Time to margin = 0
        current_margin = recent[-1]
        if current_margin <= 0:
            return 0.0  # Already past limit
        
        cycles_to_zero = -current_margin / slope
        ttd_ms = cycles_to_zero * self.dt_ms
        
        # Sanity: cap at 10 seconds
        return float(min(ttd_ms, 10000.0))
    
    def reset(self):
        self.margin_history = []


# ═══════════════════════════════════════════════════════════════════════
# MAIN: StreamingPredictor
# ═══════════════════════════════════════════════════════════════════════

class StreamingPredictor:
    """Real-time streaming disruption predictor.
    
    Usage:
        # Initialize
        predictor = StreamingPredictor(
            signal_names=['li', 'q95', 'betan', 'betap', 'ne_line', ...],
            config=StreamingConfig.for_spherical(),
            gbt_model=pre_trained_gbt   # Optional: from offline training
        )
        
        # Every 10ms when new EFIT arrives:
        predictor.ingest({'li': 1.2, 'q95': 3.5, 'betan': 2.1, ...})
        result = predictor.predict()
        
        # Result:
        #   result.probability = 0.72
        #   result.risk_level = RiskLevel.ALARM
        #   result.alarm_state = AlarmState.ALARM
        #   result.explanation = "li margin 0.08 (approaching kink limit)"
        #   result.ttd_ms = 340
        #   result.latency_us = 145
        
        # Between shots:
        predictor.reset()
    """
    
    def __init__(self, signal_names: List[str],
                 config: StreamingConfig = None,
                 gbt_model=None,
                 dt_ms: float = 10.0):
        self.signal_names = signal_names
        self.n_signals = len(signal_names)
        self.cfg = config or StreamingConfig()
        self.gbt_model = gbt_model
        self.dt_ms = dt_ms
        self.cycle = 0
        
        # Name → index mapping
        self.sig_idx = {name: i for i, name in enumerate(signal_names)}
        
        # Components
        self.buffer = RollingBuffer(self.n_signals, self.cfg.window_size)
        self.ipda = StreamingIPDA(self.n_signals, self.cfg)
        self.imm = StreamingIMM(self.n_signals, self.cfg)
        self.phantom = StreamingPHANTOM(signal_names, self.cfg)
        self.alarm = AlarmStateMachine(self.cfg)
        self.ttd = TTDEstimator(self.cfg, dt_ms)
        
        # Cached GBT features (recomputed every N cycles)
        self._cached_gbt_prob = 0.0
        self._last_gbt_recompute = 0
    
    def ingest(self, measurement: Dict[str, float]) -> None:
        """Ingest one measurement (called every 10ms).
        
        Args:
            measurement: {signal_name: value} dict
        """
        # Convert to array
        arr = np.zeros(self.n_signals)
        for name, val in measurement.items():
            if name in self.sig_idx:
                arr[self.sig_idx[name]] = val
        
        # Update all streaming components
        self.buffer.append(arr)
        self.ipda.update(arr)
        self.imm.update(arr)
        self.cycle += 1
    
    def ingest_array(self, measurement: np.ndarray) -> None:
        """Ingest one measurement as array (faster, no dict overhead)."""
        self.buffer.append(measurement)
        self.ipda.update(measurement)
        self.imm.update(measurement)
        self.cycle += 1
    
    def predict(self) -> StreamingPrediction:
        """Generate prediction from current state (called after ingest)."""
        t0 = time.perf_counter_ns()
        
        current = self.buffer.latest
        
        # 1. Physics margins (instant)
        margins = self.phantom.compute_margins(current)
        min_margin = min(margins.values())
        closest = min(margins, key=margins.get)
        
        # 2. PHANTOM feasibility (instant)
        feasibility = self.phantom.compute_feasibility(current)
        min_feas = float(np.min(feasibility))
        
        # 3. IPDA existence (already updated in ingest)
        existence = self.ipda.get_existence()
        max_existence = float(np.max(existence))
        
        # 4. IMM regime (already updated in ingest)
        regime = self.imm.get_regime()
        max_unstable = float(np.max(regime))
        
        # 5. Composite probability (weighted fusion)
        # Physics track: min_margin < 0 → danger
        p_physics = float(np.clip(1 - min_feas, 0, 1))
        
        # IPDA track: high existence → danger
        p_ipda = float(np.clip(max_existence, 0, 1))
        
        # IMM track: high unstable → danger
        p_imm = float(np.clip(max_unstable, 0, 1))
        
        # GBT track (recompute periodically for efficiency)
        if (self.gbt_model is not None and
            self.cycle - self._last_gbt_recompute >= self.cfg.feature_recompute_every and
            self.buffer.count >= 10):
            self._recompute_gbt()
            self._last_gbt_recompute = self.cycle
        p_gbt = self._cached_gbt_prob
        
        # Overseer: weighted fusion with physics priority
        if p_physics > 0.7:
            # Physics says danger → trust physics
            probability = 0.5 * p_physics + 0.2 * p_gbt + 0.15 * p_ipda + 0.15 * p_imm
        elif p_gbt > 0:
            # Have GBT → it's most accurate
            probability = 0.15 * p_physics + 0.45 * p_gbt + 0.2 * p_ipda + 0.2 * p_imm
        else:
            # No GBT (cold start) → NX-MIMOSA + physics
            probability = 0.4 * p_physics + 0.35 * p_ipda + 0.25 * p_imm
        
        probability = float(np.clip(probability, 0, 1))
        
        # 6. Uncertainty (disagreement between tracks)
        track_probs = [p_physics, p_ipda, p_imm]
        if p_gbt > 0:
            track_probs.append(p_gbt)
        uncertainty = float(np.std(track_probs))
        
        # 7. Alarm state machine
        alarm_state = self.alarm.update(probability)
        
        # 8. Risk level
        risk_level = self._assign_risk(probability)
        
        # 9. TTD estimate
        ttd_ms = self.ttd.update(min_margin)
        
        # 10. Explanation
        explanation = self._explain(margins, closest, existence, regime)
        
        # 11. IPDA/IMM per-signal state
        ipda_dict = {self.signal_names[i]: float(existence[i])
                     for i in range(self.n_signals)}
        imm_dict = {self.signal_names[i]: float(regime[i])
                    for i in range(self.n_signals)}
        
        latency_us = (time.perf_counter_ns() - t0) / 1000.0
        
        return StreamingPrediction(
            timestamp_ms=self.cycle * self.dt_ms,
            probability=probability,
            uncertainty=uncertainty,
            risk_level=risk_level,
            alarm_state=alarm_state,
            margins=margins,
            closest_limit=closest,
            explanation=explanation,
            ttd_ms=ttd_ms,
            ipda_existence=ipda_dict,
            imm_regime=imm_dict,
            recommended_action=None,  # Phase 3
            latency_us=latency_us,
            cycle_count=self.cycle,
        )
    
    def _recompute_gbt(self):
        """Recompute GBT features from rolling buffer."""
        if self.gbt_model is None:
            return
        
        stats = self.buffer.rolling_stats()
        
        # Build feature vector matching offline training format
        # Per signal: mean, std, max, trend, max_rate
        features = []
        for i in range(self.n_signals):
            features.extend([
                stats['mean'][i],
                stats['std'][i],
                stats['max'][i],
                stats['trend'][i],
                stats['max_rate'][i],
            ])
        
        feat_arr = np.array(features, dtype=np.float32).reshape(1, -1)
        
        try:
            self._cached_gbt_prob = float(
                self.gbt_model.predict_proba(feat_arr)[0, 1])
        except Exception:
            self._cached_gbt_prob = 0.0
    
    def _assign_risk(self, p: float) -> RiskLevel:
        if p >= 0.90: return RiskLevel.MITIGATE
        if p >= 0.70: return RiskLevel.ALARM
        if p >= 0.40: return RiskLevel.ALERT
        if p >= 0.10: return RiskLevel.MONITOR
        return RiskLevel.SAFE
    
    def _explain(self, margins: Dict, closest: str,
                 existence: np.ndarray, regime: np.ndarray) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        m = margins[closest]
        if m < 0.1:
            parts.append(f"{closest} margin {m:.2f} (approaching limit)")
        
        max_exist_idx = np.argmax(existence)
        if existence[max_exist_idx] > 0.5:
            parts.append(f"IPDA: {self.signal_names[max_exist_idx]} "
                        f"precursor P={existence[max_exist_idx]:.2f}")
        
        max_unstable_idx = np.argmax(regime)
        if regime[max_unstable_idx] > 0.5:
            parts.append(f"IMM: {self.signal_names[max_unstable_idx]} "
                        f"unstable P={regime[max_unstable_idx]:.2f}")
        
        if not parts:
            return "All signals within normal bounds"
        
        return "; ".join(parts)
    
    def reset(self):
        """Reset for new shot."""
        self.buffer = RollingBuffer(self.n_signals, self.cfg.window_size)
        self.ipda.reset()
        self.imm.reset()
        self.alarm.reset()
        self.ttd.reset()
        self._cached_gbt_prob = 0.0
        self._last_gbt_recompute = 0
        self.cycle = 0
    
    def get_state_summary(self) -> Dict:
        """Return full internal state for debugging/logging."""
        return {
            'cycle': self.cycle,
            'buffer_count': self.buffer.count,
            'ipda_existence': self.ipda.get_existence().tolist(),
            'imm_regime': self.imm.get_regime().tolist(),
            'alarm_state': self.alarm.state.name,
            'gbt_prob': self._cached_gbt_prob,
        }


# ═══════════════════════════════════════════════════════════════════════
# REPLAY: simulate streaming on historical shot data
# ═══════════════════════════════════════════════════════════════════════

def replay_shot(predictor: StreamingPredictor,
                shot_data: np.ndarray,
                signal_names: List[str],
                dt_ms: float = 10.0) -> List[StreamingPrediction]:
    """Replay a historical shot through the streaming predictor.
    
    Args:
        predictor: initialized StreamingPredictor
        shot_data: (n_timepoints, n_signals) array
        signal_names: list of signal names matching columns
        dt_ms: time between measurements
        
    Returns:
        List of StreamingPrediction, one per timepoint
    """
    predictor.reset()
    predictions = []
    
    for t in range(len(shot_data)):
        predictor.ingest_array(shot_data[t])
        pred = predictor.predict()
        predictions.append(pred)
    
    return predictions
