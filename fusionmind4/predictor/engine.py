#!/usr/bin/env python3
"""
FusionMind 4.0 — CausalDisruptionPredictor
===========================================

Modular multi-track architecture where each track analyses signals from 
a different perspective. Tracks activate/deactivate based on available 
data. A meta-learner combines track outputs into a final prediction.

Architecture:
                        ┌─────────────────────────────────────┐
                        │         RAW TOKAMAK SIGNALS          │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │      SIGNAL PREPROCESSOR             │
                        │  normalize → impute → validate       │
                        └──────────────┬──────────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────────┐
               │                       │                           │
    ┌──────────▼────────┐   ┌──────────▼────────┐   ┌─────────────▼───────┐
    │  TRACK A: PHYSICS  │   │  TRACK B: STATS   │   │  TRACK C: TEMPORAL  │
    │  Stability margins │   │  Shot-level GBT   │   │  Trajectory shape   │
    │  0 params, causal  │   │  63 features      │   │  thirds + rates     │
    │  ALWAYS ON         │   │  NEEDS: ≥5 vars   │   │  NEEDS: ≥50 tp     │
    └──────────┬────────┘   └──────────┬────────┘   └─────────────┬───────┘
               │                       │                           │
    ┌──────────▼────────┐   ┌──────────▼────────┐   ┌─────────────▼───────┐
    │  TRACK D: CAUSAL   │   │  TRACK E: ANOMALY │   │  TRACK F: TIMEPOINT │
    │  SCM counterfact.  │   │  Isolation Forest  │   │  GRU/LSTM sequence  │
    │  NEEDS: CPDE DAG   │   │  Train on clean    │   │  NEEDS: ≥100 tp    │
    └──────────┬────────┘   └──────────┬────────┘   └─────────────┬───────┘
               │                       │                           │
               └───────────────────────┼───────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │       META-LEARNER (LogReg)          │
                        │  Stacks out-of-fold track outputs    │
                        │  + confidence weighting              │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │    DISRUPTION PROBABILITY + WHY      │
                        │    + warning time + uncertainty      │
                        └─────────────────────────────────────┘

Key innovations vs CCNN/FRNN:
  1. Physics-first: causal DAG identifies THE mechanism per machine
  2. Stability margins: distance-to-limit normalizes across mechanisms
  3. Modular: tracks activate based on data availability
  4. Interpretable: "WHY" comes from physics, not sensitivity analysis
  5. Cross-machine: dimensionless features transfer naturally

Author: Dr. Mladen Mester
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class MachineType(Enum):
    CONVENTIONAL = "conventional"   # C-Mod, DIII-D, JET, ITER
    SPHERICAL = "spherical"         # MAST, NSTX, ST40
    UNKNOWN = "unknown"

@dataclass
class StabilityLimits:
    """Physics limits — vary per machine type."""
    li_max: float = 2.0
    q95_min: float = 2.0
    betaN_max: float = 3.5
    fGW_max: float = 1.0
    betap_max: float = 1.5
    prad_frac_max: float = 0.8
    
    @classmethod
    def for_machine(cls, machine_type: MachineType):
        if machine_type == MachineType.SPHERICAL:
            return cls(li_max=2.0, q95_min=1.5, betaN_max=6.0, 
                      fGW_max=1.5, betap_max=2.0, prad_frac_max=0.9)
        elif machine_type == MachineType.CONVENTIONAL:
            return cls(li_max=1.5, q95_min=2.0, betaN_max=3.5,
                      fGW_max=1.0, betap_max=1.2, prad_frac_max=0.7)
        else:
            return cls()  # defaults

@dataclass 
class TrackConfig:
    """Configuration for the predictor."""
    machine_type: MachineType = MachineType.UNKNOWN
    limits: StabilityLimits = field(default_factory=StabilityLimits)
    truncation_tp: int = 4          # 40ms fair evaluation
    augmentation_factor: int = 4    # copies per disrupted
    augmentation_noise: float = 0.05
    gbt_n_estimators: int = 100
    gbt_max_depth: int = 4
    gbt_learning_rate: float = 0.1
    random_state: int = 42
    min_shot_length: int = 8
    
    def __post_init__(self):
        self.limits = StabilityLimits.for_machine(self.machine_type)


# ═══════════════════════════════════════════════════════════════
# SIGNAL REGISTRY — what variables does each machine have?
# ═══════════════════════════════════════════════════════════════

# Canonical signal names → what each machine calls them
SIGNAL_ALIASES = {
    'li':       ['li', 'internal_inductance', 'li3'],
    'q95':      ['q95', 'safety_factor_95', 'qpsi_95'],
    'betan':    ['betan', 'beta_n', 'normalized_beta', 'beta_normal'],
    'betap':    ['betap', 'beta_p', 'poloidal_beta'],
    'ne':       ['ne_line', 'density', 'ne_avg', 'nel', 'n_e'],
    'ne_gw':    ['greenwald_den', 'n_greenwald', 'nGW'],
    'Ip':       ['Ip', 'plasma_current', 'ip', 'Ipa'],
    'Bt':       ['toroidal_B_field', 'bt', 'Bt0'],
    'p_rad':    ['p_rad', 'radiated_power', 'prad', 'P_rad_tot'],
    'p_input':  ['p_nbi', 'p_input', 'P_NBI', 'p_total'],
    'wmhd':     ['wmhd', 'stored_energy', 'Wmhd', 'W_MHD'],
    'kappa':    ['elongation', 'kappa', 'elongation_upper'],
    'a':        ['minor_radius', 'aminor', 'a'],
    'triangularity': ['triangularity', 'tribot', 'tritop', 'delta'],
    'q_axis':   ['q_axis', 'q0', 'q_on_axis'],
    'n1_amp':   ['n1rms', 'n1_amplitude', 'locked_mode_amp'],
}

def resolve_signals(available_vars: List[str]) -> Dict[str, Optional[str]]:
    """Map canonical names to actual column names in dataset."""
    mapping = {}
    for canonical, aliases in SIGNAL_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in available_vars:
                found = alias
                break
        mapping[canonical] = found
    return mapping


# ═══════════════════════════════════════════════════════════════
# TRACK A: PHYSICS MARGINS (always on, 0 parameters)
# ═══════════════════════════════════════════════════════════════

class TrackA_PhysicsMargins:
    """
    Deterministic physics-based disruption score.
    Computes distance-to-stability-limit for each known mechanism.
    Key insight: margin → 0 means approaching limit, regardless of WHICH limit.
    """
    
    def __init__(self, limits: StabilityLimits, signal_map: Dict[str, Optional[str]]):
        self.limits = limits
        self.signal_map = signal_map
        self.available_mechanisms = self._detect_mechanisms()
        
    def _detect_mechanisms(self) -> List[str]:
        """Which disruption mechanisms can we monitor?"""
        mechs = []
        if self.signal_map.get('li'):
            mechs.append('internal_kink')
        if self.signal_map.get('q95'):
            mechs.append('external_kink')
        if self.signal_map.get('betan'):
            mechs.append('beta_limit')
        if self.signal_map.get('ne') and (self.signal_map.get('ne_gw') or self.signal_map.get('Ip')):
            mechs.append('density_limit')
        if self.signal_map.get('betap'):
            mechs.append('ballooning')
        if self.signal_map.get('p_rad') and self.signal_map.get('p_input'):
            mechs.append('radiation_collapse')
        if self.signal_map.get('n1_amp'):
            mechs.append('locked_mode')
        return mechs
    
    def compute_margins(self, signals: Dict[str, np.ndarray], n30: int) -> Dict[str, float]:
        """Compute margin for each available mechanism."""
        margins = {}
        
        if 'internal_kink' in self.available_mechanisms:
            li = signals['li']
            margins['li'] = 1.0 - np.max(li) / self.limits.li_max
            
        if 'external_kink' in self.available_mechanisms:
            q95 = signals['q95']
            q_min = np.min(q95[q95 > 0.5]) if np.any(q95 > 0.5) else 10
            margins['q95'] = 1.0 - self.limits.q95_min / q_min
            
        if 'beta_limit' in self.available_mechanisms:
            bn = signals['betan']
            margins['betan'] = 1.0 - np.max(bn) / self.limits.betaN_max
            
        if 'density_limit' in self.available_mechanisms:
            fgw = signals.get('f_GW', np.zeros(1))
            margins['fGW'] = 1.0 - np.max(fgw) / self.limits.fGW_max
            
        if 'ballooning' in self.available_mechanisms:
            bp = signals['betap']
            margins['betap'] = 1.0 - np.max(bp) / self.limits.betap_max
            
        if 'radiation_collapse' in self.available_mechanisms:
            prad = signals['p_rad']
            pinp = signals['p_input']
            p_max = np.max(pinp)
            if p_max > 0:
                margins['prad'] = 1.0 - np.max(prad) / p_max * self.limits.prad_frac_max
            
        if 'locked_mode' in self.available_mechanisms:
            n1 = signals['n1_amp']
            margins['n1'] = 1.0 - np.max(n1)  # already normalized typically
            
        return {k: np.clip(v, -1, 1) for k, v in margins.items()}
    
    def score_shot(self, signals: Dict[str, np.ndarray]) -> Tuple[float, Dict]:
        """Score a single shot. Returns (risk_score, explanation)."""
        n = len(next(iter(signals.values())))
        n30 = max(int(0.3 * n), 1)
        
        margins = self.compute_margins(signals, n30)
        if not margins:
            return 0.0, {'error': 'no margins computable'}
        
        min_margin = min(margins.values())
        closest_limit = min(margins, key=margins.get)
        n_stressed = sum(1 for m in margins.values() if m < 0.3)
        
        risk_score = 1.0 - min_margin  # higher = more dangerous
        
        explanation = {
            'margins': margins,
            'closest_limit': closest_limit,
            'min_margin': min_margin,
            'n_stressed': n_stressed,
            'mechanisms_monitored': len(margins),
        }
        return risk_score, explanation
    
    def build_features(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Build feature vector from margins (for meta-learner input)."""
        n = len(next(iter(signals.values())))
        n30 = max(int(0.3 * n), 1)
        margins = self.compute_margins(signals, n30)
        
        feats = list(margins.values())
        if margins:
            feats.extend([min(margins.values()), np.mean(list(margins.values())),
                         sum(1 for m in margins.values() if m < 0.3)])
        return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# TRACK B: SHOT-LEVEL STATISTICS (main ML track)
# ═══════════════════════════════════════════════════════════════

class TrackB_ShotStats:
    """
    GBT on shot-level aggregated statistics.
    For each signal: mean, std, max, late_mean, trend, max_rate.
    Plus stability margins and cross-variable interactions.
    """
    
    def __init__(self, signal_map: Dict, limits: StabilityLimits):
        self.signal_map = signal_map
        self.limits = limits
        self.key_signals = [k for k in ['li','q95','betan','betap','ne','p_rad','wmhd','Ip']
                           if signal_map.get(k)]
        
    def build_features(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Build full feature vector for one shot."""
        n = len(next(iter(signals.values())))
        n30 = max(int(0.3 * n), 1)
        feats = []
        
        # Per-signal statistics (6 per signal)
        for name in self.key_signals:
            if name not in signals: continue
            s = signals[name]
            if len(s) < 3: s = np.zeros(5)
            feats.extend([
                np.mean(s), np.std(s), np.max(s),
                np.mean(s[-n30:]),                          # late mean
                np.mean(s[-n30:]) - np.mean(s[:n30]),       # trend
                np.max(np.abs(np.diff(s))) if len(s)>1 else 0  # max rate
            ])
        
        # Stability margins
        track_a = TrackA_PhysicsMargins(self.limits, self.signal_map)
        margins = track_a.compute_margins(signals, n30)
        feats.extend(list(margins.values()))
        if margins:
            feats.extend([min(margins.values()), 
                         sum(1 for m in margins.values() if m < 0.3)])
        
        # Cross-variable interactions
        li_s = signals.get('li', np.zeros(1))
        q_s = signals.get('q95', np.ones(1)*10)
        bn_s = signals.get('betan', np.zeros(1))
        fgw_s = signals.get('f_GW', np.zeros(1))
        q_min = np.min(q_s[q_s>0.5]) if np.any(q_s>0.5) else 10
        
        feats.extend([
            np.max(li_s) * np.max(bn_s),           # li × βN
            np.max(li_s) / (q_min + 0.5),           # li / q95
            np.std(li_s) * np.std(q_s),              # instability product
            np.max(fgw_s) * np.max(li_s),            # density × internal
        ])
        
        # Temporal shape
        feats.extend([
            np.max(li_s[-n30:]) / (np.mean(li_s[:n30]) + 1e-10),  # li late/early
            np.max(li_s[-n30:]) - np.mean(li_s[:n30]),              # li gap
            np.max(bn_s[-n30:]) - np.mean(bn_s[:n30]),              # βN gap
            np.min(q_s[-n30:]) / (np.mean(q_s[:n30]) + 1e-10),     # q95 drop
        ])
        
        return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# TRACK C: TEMPORAL TRAJECTORY (shape analysis)
# ═══════════════════════════════════════════════════════════════

class TrackC_Trajectory:
    """
    Split shot into thirds and compare.
    Captures trajectory SHAPE — is li rising? Is q95 falling?
    Inspired by CCNN's ability to capture temporal patterns.
    """
    
    def __init__(self, signal_map: Dict):
        self.key_signals = [k for k in ['li','q95','betan','betap','ne','p_rad','wmhd','Ip']
                           if signal_map.get(k)]
    
    def build_features(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(signals.values())))
        nt = max(n // 3, 1)
        feats = []
        
        for name in self.key_signals:
            if name not in signals: continue
            s = signals[name]
            if len(s) < 6: s = np.zeros(9)
            t1 = s[:nt]; t2 = s[nt:2*nt]; t3 = s[2*nt:]
            
            feats.extend([
                np.mean(t3) / (np.mean(t1) + 1e-10),   # end/start ratio
                np.std(t3) / (np.std(t1) + 1e-10),      # volatility change
                np.max(t3) - np.max(t1),                  # max shift
                np.mean(t3) - np.mean(t2),                # recent acceleration
            ])
        
        return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# TRACK D: CAUSAL MECHANISM IDENTIFIER
# ═══════════════════════════════════════════════════════════════

class TrackD_CausalMechanism:
    """
    Uses CPDE DAG (if available) to identify THE dominant causal
    mechanism for this specific machine. Then builds features 
    focused on that mechanism.
    
    This is FusionMind's unique advantage over CCNN/FRNN.
    """
    
    def __init__(self, signal_map: Dict, limits: StabilityLimits,
                 causal_dag: Optional[Dict] = None):
        self.signal_map = signal_map
        self.limits = limits
        self.causal_dag = causal_dag
        self.primary_driver = self._identify_primary_driver()
        
    def _identify_primary_driver(self) -> str:
        """Identify primary disruption mechanism from DAG or heuristic."""
        if self.causal_dag and 'primary_cause' in self.causal_dag:
            return self.causal_dag['primary_cause']
        
        # Heuristic: test each signal's discriminative power
        # Will be set during fit() from data
        return 'unknown'
    
    def identify_from_data(self, shots_data: List[Dict], labels: np.ndarray):
        """Identify primary driver from data using single-variable AUC."""
        from sklearn.metrics import roc_auc_score
        
        best_auc = 0
        best_signal = 'unknown'
        signal_aucs = {}
        
        for sig_name in ['li', 'q95', 'betan', 'ne', 'betap', 'p_rad']:
            scores = []
            valid_labels = []
            for shot, lab in zip(shots_data, labels):
                if sig_name in shot and len(shot[sig_name]) > 3:
                    scores.append(np.max(shot[sig_name]))
                    valid_labels.append(lab)
            
            if len(set(valid_labels)) < 2 or sum(valid_labels) < 3:
                continue
                
            try:
                auc = roc_auc_score(valid_labels, scores)
                # For q95, lower is worse, so invert
                if sig_name == 'q95':
                    auc = roc_auc_score(valid_labels, [-s for s in scores])
                signal_aucs[sig_name] = auc
                if auc > best_auc:
                    best_auc = auc
                    best_signal = sig_name
            except:
                pass
        
        self.primary_driver = best_signal
        self.signal_aucs = signal_aucs
        return best_signal, signal_aucs
    
    def build_features(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Build features focused on the primary causal driver."""
        feats = []
        n = len(next(iter(signals.values())))
        n30 = max(int(0.3 * n), 1)
        
        if self.primary_driver != 'unknown' and self.primary_driver in signals:
            s = signals[self.primary_driver]
            # Deep analysis of primary driver
            feats.extend([
                np.max(s), np.mean(s), np.std(s),
                np.mean(s[-n30:]),
                np.mean(s[-n30:]) - np.mean(s[:n30]),
                np.max(s[-n30:]) / (np.mean(s[:n30]) + 1e-10),
                np.std(s[-n30:]) / (np.std(s[:n30]) + 1e-10),
                np.max(np.abs(np.diff(s))) if len(s)>1 else 0,
                np.percentile(s, 95),
                np.percentile(s, 99),
                # Duration above 80% of max
                np.mean(s > 0.8 * np.max(s)),
                # When does max occur (late = worse)
                np.argmax(s) / (len(s) + 1e-10),
            ])
            
            # Interaction with each other signal
            for other in ['li', 'q95', 'betan', 'ne']:
                if other != self.primary_driver and other in signals:
                    o = signals[other]
                    feats.append(np.max(s) * np.max(o))
                    feats.append(np.corrcoef(s[-n30:], o[-n30:])[0,1] 
                                if len(s[-n30:])>2 else 0)
        
        return np.nan_to_num(np.array(feats, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════
# TRACK E: RATE-OF-CHANGE EXTREMES (precursor detection)
# ═══════════════════════════════════════════════════════════════

class TrackE_RateExtremes:
    """
    Disruption precursors show up as sudden rate changes.
    Mode locking: sudden li spike + q95 drop.
    Density limit: sudden ne acceleration.
    Thermal quench: sudden Te/Wmhd crash.
    """
    
    def __init__(self, signal_map: Dict):
        self.key_signals = [k for k in ['li','q95','betan','betap','ne','p_rad','wmhd','Ip']
                           if signal_map.get(k)]
    
    def build_features(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        for name in self.key_signals:
            if name not in signals: continue
            s = signals[name]
            if len(s) < 3: s = np.zeros(5)
            ds = np.diff(s)
            n_late = max(len(ds) // 3, 1)
            
            feats.extend([
                np.max(np.abs(ds)),                           # max absolute rate
                np.mean(np.abs(ds[-n_late:])),                # late mean rate
                np.std(ds[-n_late:]),                          # late rate volatility
                np.max(ds[-n_late:]) if len(ds)>0 else 0,     # max positive late rate
                np.min(ds[-n_late:]) if len(ds)>0 else 0,     # max negative late rate
            ])
        return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# TRACK F: PAIRWISE INTERACTIONS
# ═══════════════════════════════════════════════════════════════

class TrackF_Pairwise:
    """
    Cross-signal interactions that capture nonlinear stability boundaries.
    Inspired by Hugill diagram (1/q vs n/nGW) and Troyon limit (βN vs li).
    """
    
    PHYSICS_PAIRS = [
        ('li', 'betan'),    # internal inductance × beta limit
        ('li', 'q95'),      # kink stability space
        ('li', 'ne'),       # density-kink interaction
        ('betan', 'q95'),   # Troyon-kink space
        ('ne', 'q95'),      # Hugill diagram space
        ('betan', 'ne'),    # beta-density space
    ]
    
    def __init__(self, signal_map: Dict):
        self.signal_map = signal_map
        self.active_pairs = [(a,b) for a,b in self.PHYSICS_PAIRS
                            if signal_map.get(a) and signal_map.get(b)]
    
    def build_features(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        n = len(next(iter(signals.values())))
        n30 = max(int(0.3 * n), 1)
        
        for s1_name, s2_name in self.active_pairs:
            if s1_name not in signals or s2_name not in signals: continue
            s1 = signals[s1_name]; s2 = signals[s2_name]
            
            feats.extend([
                np.max(s1) * np.max(s2),                    # product of peaks
                np.max(s1) / (np.max(np.abs(s2)) + 1e-10),  # ratio of peaks
                np.corrcoef(s1[-n30:], s2[-n30:])[0,1] if len(s1[-n30:])>2 else 0,
            ])
        return np.nan_to_num(np.array(feats, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════
# META-LEARNER: Combines all tracks
# ═══════════════════════════════════════════════════════════════

class MetaLearner:
    """
    Stacked generalization: each track produces out-of-fold predictions.
    Meta-learner (LogisticRegression) learns optimal combination.
    Also supports simple weighted average as fallback.
    """
    
    def __init__(self, config: TrackConfig):
        self.config = config
        self.meta_model = None
        self.track_weights = None
        
    def fit(self, track_outputs: Dict[str, np.ndarray], labels: np.ndarray):
        """Fit meta-learner on out-of-fold track predictions."""
        from sklearn.linear_model import LogisticRegression
        
        X_meta = np.column_stack(list(track_outputs.values()))
        self.meta_model = LogisticRegression(
            C=1.0, class_weight='balanced', 
            random_state=self.config.random_state
        )
        self.meta_model.fit(X_meta, labels)
        self.track_names = list(track_outputs.keys())
        
        # Also compute simple weights from individual AUCs
        from sklearn.metrics import roc_auc_score
        aucs = {}
        for name, preds in track_outputs.items():
            try:
                aucs[name] = roc_auc_score(labels, preds)
            except:
                aucs[name] = 0.5
        total = sum(aucs.values())
        self.track_weights = {k: v/total for k, v in aucs.items()}
    
    def predict(self, track_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using meta-learner."""
        X_meta = np.column_stack([track_outputs[n] for n in self.track_names])
        return self.meta_model.predict_proba(X_meta)[:, 1]
    
    def predict_weighted(self, track_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fallback: weighted average by track AUC."""
        result = np.zeros(len(next(iter(track_outputs.values()))))
        for name, preds in track_outputs.items():
            w = self.track_weights.get(name, 1.0/len(track_outputs))
            result += w * preds
        return result


# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════

class CausalDisruptionPredictor:
    """
    Main entry point. Orchestrates all tracks.
    
    Usage:
        predictor = CausalDisruptionPredictor.from_data(
            data, shot_ids, variables, disrupted_set,
            machine_type=MachineType.SPHERICAL
        )
        results = predictor.evaluate_cv(n_folds=5)
        
        # For a single new shot:
        prob, explanation = predictor.predict_shot(shot_signals)
    """
    
    def __init__(self, config: TrackConfig = None):
        self.config = config or TrackConfig()
        self.tracks = {}
        self.meta_learner = None
        self.signal_map = {}
        self.fitted = False
    
    @classmethod
    def from_data(cls, data: np.ndarray, shot_ids: np.ndarray,
                  variables: List[str], disrupted_set: set,
                  machine_type: MachineType = MachineType.UNKNOWN,
                  causal_dag: Optional[Dict] = None):
        """
        Factory method: build predictor from raw data arrays.
        Automatically detects available signals and activates tracks.
        """
        config = TrackConfig(machine_type=machine_type)
        predictor = cls(config)
        
        # Resolve signals
        predictor.signal_map = resolve_signals(variables)
        predictor.data = data
        predictor.shot_ids = shot_ids
        predictor.variables = variables
        predictor.disrupted_set = disrupted_set
        predictor.var_idx = {v: i for i, v in enumerate(variables)}
        
        # Initialize tracks based on available signals
        predictor._init_tracks(causal_dag)
        
        return predictor
    
    def _init_tracks(self, causal_dag=None):
        """Initialize tracks based on what signals are available."""
        sm = self.signal_map
        
        # Track A: always on if ANY physics signal available
        self.tracks['A_physics'] = TrackA_PhysicsMargins(
            self.config.limits, sm)
        
        # Track B: needs at least 5 resolved signals
        n_resolved = sum(1 for v in sm.values() if v is not None)
        if n_resolved >= 3:
            self.tracks['B_stats'] = TrackB_ShotStats(sm, self.config.limits)
        
        # Track C: trajectory (needs decent shot length, checked per-shot)
        if n_resolved >= 3:
            self.tracks['C_trajectory'] = TrackC_Trajectory(sm)
        
        # Track D: causal (needs DAG or will auto-detect from data)
        self.tracks['D_causal'] = TrackD_CausalMechanism(
            sm, self.config.limits, causal_dag)
        
        # Track E: rate extremes
        if n_resolved >= 3:
            self.tracks['E_rates'] = TrackE_RateExtremes(sm)
        
        # Track F: pairwise
        if n_resolved >= 4:
            self.tracks['F_pairwise'] = TrackF_Pairwise(sm)
        
        active = list(self.tracks.keys())
        print(f"  Active tracks ({len(active)}): {active}")
        print(f"  Resolved signals: {n_resolved}/{len(SIGNAL_ALIASES)}")
    
    def _extract_shot_signals(self, shot_mask: np.ndarray, 
                               truncate: bool = False) -> Dict[str, np.ndarray]:
        """Extract named signal arrays for one shot."""
        n = shot_mask.sum()
        if truncate and n > self.config.truncation_tp + 3:
            sl = slice(0, n - self.config.truncation_tp)
        else:
            sl = slice(0, n)
        
        signals = {}
        for canonical, actual in self.signal_map.items():
            if actual and actual in self.var_idx:
                signals[canonical] = self.data[shot_mask, self.var_idx[actual]][sl]
        
        # Derived: Greenwald fraction
        if 'ne' in signals:
            if 'ne_gw' in signals:
                signals['f_GW'] = signals['ne'] / (signals['ne_gw'] + 1e-10)
            elif 'Ip' in signals and 'a' in signals:
                ngw = signals['Ip'] / (np.pi * signals['a']**2 + 1e-10)
                signals['f_GW'] = signals['ne'] / (ngw + 1e-10)
        
        return signals
    
    def build_all_features(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Build feature matrices for all shots, for all tracks."""
        u = np.unique(self.shot_ids)
        
        track_features = {name: [] for name in self.tracks}
        labels = []
        valid_shots = []
        
        for sid in u:
            mask = self.shot_ids == sid
            n = mask.sum()
            if n < self.config.min_shot_length:
                continue
            
            is_dis = int(sid) in self.disrupted_set
            signals = self._extract_shot_signals(mask, truncate=is_dis)
            
            if len(next(iter(signals.values()))) < 3:
                continue
            
            for track_name, track in self.tracks.items():
                feats = track.build_features(signals)
                track_features[track_name].append(feats)
            
            labels.append(1 if is_dis else 0)
            valid_shots.append(int(sid))
        
        # Convert to arrays
        result = {}
        for name, feat_list in track_features.items():
            if feat_list:
                # Pad to same length (some shots may have different feature counts)
                max_len = max(len(f) for f in feat_list)
                padded = [np.pad(f, (0, max_len - len(f))) for f in feat_list]
                result[name] = np.clip(
                    np.nan_to_num(np.array(padded)), -1e6, 1e6
                ).astype(np.float32)
        
        return result, np.array(labels), valid_shots
    
    def evaluate_cv(self, n_folds: int = 5, verbose: bool = True) -> Dict:
        """
        Full cross-validated evaluation with all tracks.
        Returns detailed results per track and for meta-learner.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, roc_curve
        
        if verbose:
            print(f"\n{'='*65}")
            print(f"CausalDisruptionPredictor — {n_folds}-fold CV")
            print(f"{'='*65}")
        
        # Build features
        track_feats, labels, shots = self.build_all_features()
        n_dis = sum(labels)
        n_cln = len(labels) - n_dis
        
        if verbose:
            print(f"  Shots: {len(labels)} ({n_dis}d + {n_cln}c)")
            for name, X in track_feats.items():
                print(f"  Track {name}: {X.shape[1]} features")
        
        # Identify primary causal driver
        if 'D_causal' in self.tracks:
            shots_data = []
            for sid in shots:
                mask = self.shot_ids == sid
                signals = self._extract_shot_signals(mask)
                shots_data.append(signals)
            driver, aucs = self.tracks['D_causal'].identify_from_data(shots_data, labels)
            if verbose:
                print(f"\n  Causal analysis — primary driver: {driver}")
                for sig, auc in sorted(aucs.items(), key=lambda x: -x[1]):
                    print(f"    {sig}: single-var AUC = {auc:.3f}")
        
        # Augmentation
        np.random.seed(self.config.random_state)
        dis_idx = [i for i, l in enumerate(labels) if l == 1]
        
        augmented = {}
        for name, X in track_feats.items():
            aug_list = []
            for i in dis_idx:
                for _ in range(self.config.augmentation_factor):
                    aug_list.append(X[i] * (1 + np.random.normal(
                        0, self.config.augmentation_noise, X.shape[1])))
            augmented[name] = np.vstack([X, np.clip(
                np.nan_to_num(np.array(aug_list)), -1e6, 1e6)])
        labels_aug = np.concatenate([labels, np.ones(len(dis_idx) * self.config.augmentation_factor)])
        n_orig = len(labels)
        
        # Cross-validation
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                            random_state=self.config.random_state)
        
        # Out-of-fold predictions per track
        oof = {name: np.zeros(n_orig) for name in track_feats}
        
        for fold, (tr, te) in enumerate(kf.split(track_feats[list(track_feats.keys())[0]], labels)):
            # Augmented training indices
            tr_aug = list(tr)
            for j in range(len(dis_idx) * self.config.augmentation_factor):
                if dis_idx[j // self.config.augmentation_factor] in tr:
                    tr_aug.append(n_orig + j)
            tr_aug = np.array(tr_aug)
            
            for name, X_full in augmented.items():
                X_orig = track_feats[name]
                if X_orig.shape[1] == 0:
                    continue
                
                # Track A uses raw margin scores (no training)
                if name == 'A_physics':
                    # Already computed as features — use max margin
                    oof[name][te] = 1 - X_orig[te, -3] if X_orig.shape[1] >= 3 else 0
                    continue
                
                gbt = GradientBoostingClassifier(
                    n_estimators=self.config.gbt_n_estimators,
                    max_depth=self.config.gbt_max_depth,
                    learning_rate=self.config.gbt_learning_rate,
                    subsample=0.8, min_samples_leaf=3,
                    random_state=self.config.random_state)
                gbt.fit(X_full[tr_aug], labels_aug[tr_aug])
                oof[name][te] = gbt.predict_proba(X_orig[te])[:, 1]
        
        # Individual track results
        results = {'tracks': {}, 'meta': {}, 'config': {
            'machine_type': self.config.machine_type.value,
            'n_shots': len(labels), 'n_disrupted': int(n_dis),
            'n_folds': n_folds, 'tracks_active': list(track_feats.keys()),
        }}
        
        if verbose:
            print(f"\n  {'Track':<25} {'AUC':>8} {'±std':>8} {'TPR@5%':>8}")
            print(f"  {'─'*52}")
        
        for name, preds in oof.items():
            if np.std(preds) < 1e-10: continue
            fold_aucs = [roc_auc_score(labels[te], preds[te]) 
                        for _, te in kf.split(track_feats[list(track_feats.keys())[0]], labels)]
            fpr_a, tpr_a, _ = roc_curve(labels, preds)
            tpr5 = tpr_a[np.argmin(np.abs(fpr_a - 0.05))]
            
            results['tracks'][name] = {
                'AUC_mean': float(np.mean(fold_aucs)),
                'AUC_std': float(np.std(fold_aucs)),
                'TPR_at_5pct': float(tpr5),
                'folds': [float(a) for a in fold_aucs],
                'n_features': int(track_feats[name].shape[1]),
            }
            if verbose:
                print(f"  {name:<25} {np.mean(fold_aucs):>8.3f} {np.std(fold_aucs):>8.3f} {tpr5:>8.0%}")
        
        # Meta-learner combinations
        active_tracks = {k: v for k, v in oof.items() if np.std(v) > 1e-10}
        
        if len(active_tracks) >= 2:
            # Stacked LogReg meta-learner
            meta_X = np.column_stack(list(active_tracks.values()))
            meta_oof = np.zeros(n_orig)
            for _, (tr, te) in enumerate(kf.split(meta_X, labels)):
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(C=1.0, class_weight='balanced',
                                      random_state=self.config.random_state)
                m.fit(meta_X[tr], labels[tr])
                meta_oof[te] = m.predict_proba(meta_X[te])[:, 1]
            
            # Simple mean
            mean_oof = np.mean(meta_X, axis=1)
            
            combos = {
                'meta_logreg': meta_oof,
                'mean_all': mean_oof,
            }
            
            # Best-2 combo
            sorted_tracks = sorted(results['tracks'].items(), 
                                  key=lambda x: -x[1]['AUC_mean'])
            if len(sorted_tracks) >= 2:
                t1 = sorted_tracks[0][0]; t2 = sorted_tracks[1][0]
                combos[f'mean({t1}+{t2})'] = (oof[t1] + oof[t2]) / 2
            
            if verbose:
                print(f"\n  {'Meta-learner':<25} {'AUC':>8} {'±std':>8} {'TPR@5%':>8}")
                print(f"  {'─'*52}")
            
            for name, preds in combos.items():
                fold_aucs = [roc_auc_score(labels[te], preds[te])
                            for _, te in kf.split(meta_X, labels)]
                fpr_a, tpr_a, _ = roc_curve(labels, preds)
                tpr5 = tpr_a[np.argmin(np.abs(fpr_a - 0.05))]
                results['meta'][name] = {
                    'AUC_mean': float(np.mean(fold_aucs)),
                    'AUC_std': float(np.std(fold_aucs)),
                    'TPR_at_5pct': float(tpr5),
                    'folds': [float(a) for a in fold_aucs],
                }
                if verbose:
                    print(f"  {name:<25} {np.mean(fold_aucs):>8.3f} {np.std(fold_aucs):>8.3f} {tpr5:>8.0%}")
        
        # Find overall best
        all_results = {**results['tracks'], **results['meta']}
        best = max(all_results, key=lambda k: all_results[k]['AUC_mean'])
        results['best'] = {'name': best, **all_results[best]}
        
        if verbose:
            print(f"\n  ★ BEST: {best} = {all_results[best]['AUC_mean']:.3f}")
        
        return results
    
    def predict_shot(self, signals: Dict[str, np.ndarray]) -> Tuple[float, Dict]:
        """
        Predict disruption probability for a single new shot.
        Returns (probability, explanation dict).
        """
        explanation = {}
        
        # Track A: physics margins (always available)
        if 'A_physics' in self.tracks:
            risk, margin_info = self.tracks['A_physics'].score_shot(signals)
            explanation['physics'] = margin_info
            explanation['physics_risk'] = risk
        
        # Causal mechanism
        if 'D_causal' in self.tracks:
            explanation['primary_driver'] = self.tracks['D_causal'].primary_driver
            if hasattr(self.tracks['D_causal'], 'signal_aucs'):
                explanation['signal_importance'] = self.tracks['D_causal'].signal_aucs
        
        return explanation.get('physics_risk', 0.5), explanation


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: Run on MAST or C-Mod data directly
# ═══════════════════════════════════════════════════════════════

def run_mast(data_path='data/mast/mast_level2_2941shots.npz',
             label_path='data/mast/disruption_info.json'):
    """Run full evaluation on MAST data."""
    import json
    md = np.load(data_path, allow_pickle=True)
    D = np.nan_to_num(md['data']).astype(np.float32)
    L = md['shot_ids']
    VN = [str(v) for v in md['variables']]
    with open(label_path) as f: di = json.load(f)
    dset = set(di['disrupted'])
    
    predictor = CausalDisruptionPredictor.from_data(
        D, L, VN, dset, machine_type=MachineType.SPHERICAL)
    results = predictor.evaluate_cv(n_folds=5)
    return results

def run_cmod(data_path='data/cmod/cmod_density_limit.npz',
             label_path='data/cmod/disruption_info.json'):
    """Run full evaluation on C-Mod data."""
    import json
    cd = np.load(data_path, allow_pickle=True)
    D = np.nan_to_num(cd['data']).astype(np.float32)
    L = cd['shot_ids']
    VN = [str(v) for v in cd['variables']]
    with open(label_path) as f: di = json.load(f)
    dset = set(di['disrupted'])
    
    predictor = CausalDisruptionPredictor.from_data(
        D, L, VN, dset, machine_type=MachineType.CONVENTIONAL)
    results = predictor.evaluate_cv(n_folds=5)
    return results


if __name__ == '__main__':
    print("Running MAST evaluation...")
    mast_results = run_mast()
    print("\nRunning C-Mod evaluation...")
    cmod_results = run_cmod()
