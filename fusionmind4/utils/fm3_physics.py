#!/usr/bin/env python3
"""
FM3 Physics Features for Disruption Prediction
================================================

Ported from FusionMind 3.0 physics knowledge into FM4 feature pipeline.

Key physics features NOT in FM4's basic margins:
  1. Rational q-surface proximity (tearing mode risk)
  2. Shape-corrected Troyon limit
  3. li rate + acceleration (internal kink precursor)
  4. Radiation fraction + trend
  5. Confinement degradation (τ_E drop)
  6. Multi-mechanism stress count

TESTED: Does NOT improve AUC on MAST (spherical, q95~5-7).
        Expected to help on conventional tokamaks (q95~3, near rational surfaces).

Activation: auto — adds features when machine_type == CONVENTIONAL.
            On SPHERICAL: skips rational-q features (q95 too high to matter).

Author: Dr. Mladen Mester (ported from FM3, March 2026)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FM3PhysicsConfig:
    """Machine-specific physics thresholds from FM3."""
    # Rational q surfaces (tearing modes)
    q_rational_surfaces: List[float] = None
    q_danger_threshold: float = 2.5  # q95 below this = tearing risk
    
    # Troyon limit
    troyon_coefficient: float = 2.8  # C_T in βN_crit = C_T × li
    
    # Internal inductance thresholds
    li_danger: float = 1.5       # li above this = current peaking
    li_rate_threshold: float = 0.05  # li rising this fast = precursor
    
    # Radiation
    radiation_fraction_limit: float = 0.8  # P_rad/P_input > this = collapse
    
    # Confinement
    energy_drop_threshold: float = 0.3  # 30% drop = significant
    
    def __post_init__(self):
        if self.q_rational_surfaces is None:
            self.q_rational_surfaces = [1.5, 2.0, 3.0, 4.0]
    
    @classmethod
    def for_spherical(cls):
        """MAST, NSTX — high β, high q95, different limits."""
        return cls(
            q_rational_surfaces=[1.5, 2.0],  # Only low-order matter
            q_danger_threshold=2.0,           # Spherical can go lower
            troyon_coefficient=4.5,           # Higher β limit
            li_danger=2.0,
            radiation_fraction_limit=0.9,
        )
    
    @classmethod
    def for_conventional(cls):
        """C-Mod, DIII-D, JET, ITER — standard limits."""
        return cls(
            q_rational_surfaces=[1.5, 2.0, 3.0, 4.0],
            q_danger_threshold=2.5,
            troyon_coefficient=2.8,
            li_danger=1.5,
            radiation_fraction_limit=0.7,
        )


def build_fm3_physics_features(
    li: np.ndarray, q95: np.ndarray, betan: np.ndarray,
    fgw: np.ndarray, p_rad: np.ndarray, p_input: np.ndarray,
    wmhd: np.ndarray, n30: int,
    config: FM3PhysicsConfig = None,
) -> Tuple[np.ndarray, Dict]:
    """Build FM3 physics features for one shot.
    
    Args:
        li, q95, betan, fgw, p_rad, p_input, wmhd: signal arrays (truncated)
        n30: number of timepoints for "late phase" (30% of shot)
        config: machine-specific thresholds
    
    Returns:
        (features array, explanation dict)
    """
    config = config or FM3PhysicsConfig()
    feats = []
    explanation = {}
    
    q_min = np.min(q95[q95 > 0.5]) if np.any(q95 > 0.5) else 10
    
    # ── 1. Rational q-surface proximity ──
    # Tearing modes at rational surfaces → locked mode → disruption
    # Most dangerous: q = 2.0 (2/1 mode)
    q_distances = [abs(q_min - qr) for qr in config.q_rational_surfaces]
    closest_rational_dist = min(q_distances) if q_distances else 10
    closest_rational_q = config.q_rational_surfaces[np.argmin(q_distances)] if q_distances else 0
    
    feats.extend([
        abs(q_min - 2.0),                      # Distance to q=2 (most dangerous)
        closest_rational_dist,                   # Distance to any rational surface
        1.0 / (closest_rational_dist + 0.1),    # Inverse (higher = closer = worse)
        float(q_min < config.q_danger_threshold),  # Below danger threshold
    ])
    
    if closest_rational_dist < 0.3:
        explanation['rational_q'] = f"q95={q_min:.2f} near q={closest_rational_q} ({closest_rational_dist:.2f} away)"
    
    # ── 2. Shape-corrected Troyon limit ──
    # βN_crit = C_T × li (depends on current peaking)
    li_max = np.max(li)
    bn_max = np.max(betan)
    troyon_crit = config.troyon_coefficient * li_max
    troyon_margin = 1 - bn_max / (troyon_crit + 0.1)
    
    feats.extend([
        troyon_crit,                            # Machine-specific βN limit
        np.clip(troyon_margin, -2, 1),          # Margin to Troyon
        bn_max * li_max,                        # Stability product
        bn_max / (q_min + 0.5),                 # βN/q stability space
    ])
    
    if troyon_margin < 0.2:
        explanation['troyon'] = f"βN={bn_max:.2f} at {100*(1-troyon_margin):.0f}% of Troyon limit ({troyon_crit:.2f})"
    
    # ── 3. li dynamics (internal kink precursor) ──
    # Rising li = current peaking → internal reconnection
    li_late = np.mean(li[-n30:])
    li_early = np.mean(li[:n30])
    li_rate = (li_late - li_early) / (len(li) + 1e-10) * 100
    
    li_accel = 0
    if len(li) > 10:
        d_li = np.diff(li)
        dd_li = np.diff(d_li)
        n_late = max(len(dd_li) // 3, 1)
        li_accel = np.max(dd_li[-n_late:]) if len(dd_li) > 0 else 0
    
    feats.extend([
        li_rate,                                # li trend
        np.clip(li_accel, -10, 10),             # li acceleration
        li_max / (q_min + 0.5),                 # li/q stability
        float(li_rate > config.li_rate_threshold),  # Binary: li rising fast
    ])
    
    if li_rate > config.li_rate_threshold:
        explanation['li_dynamics'] = f"li rising at {li_rate:.4f}/100tp (threshold={config.li_rate_threshold})"
    
    # ── 4. Radiation balance ──
    # Radiation fraction → 1 means all input power radiated → collapse
    p_in_max = np.max(p_input) + 1e-10
    rad_frac = np.max(p_rad) / p_in_max
    rad_frac_late = np.mean(p_rad[-n30:]) / (np.mean(p_input[-n30:]) + 1e-10)
    
    feats.extend([
        np.clip(rad_frac, 0, 5),                # Peak radiation fraction
        np.clip(rad_frac_late, 0, 5),            # Late radiation fraction
        float(rad_frac > config.radiation_fraction_limit),  # Above FM3 threshold
        np.clip(rad_frac_late - rad_frac * 0.5, -2, 2),    # Trend
    ])
    
    if rad_frac > config.radiation_fraction_limit:
        explanation['radiation'] = f"P_rad/P_in={rad_frac:.2f} > {config.radiation_fraction_limit}"
    
    # ── 5. Confinement quality ──
    w_max = np.max(wmhd)
    w_late = np.mean(wmhd[-n30:])
    w_drop = 1 - w_late / (w_max + 1e-10) if w_max > 0 else 0
    tau_proxy = w_max / (p_in_max + 1e-10)
    tau_late = w_late / (np.mean(p_input[-n30:]) + 1e-10)
    tau_degradation = 1 - tau_late / (tau_proxy + 1e-10) if tau_proxy > 0 else 0
    
    feats.extend([
        np.clip(w_drop, 0, 1),                  # Stored energy drop
        np.clip(tau_degradation, -1, 1),         # Confinement time degradation
        float(w_drop > config.energy_drop_threshold),  # Significant loss
    ])
    
    if w_drop > config.energy_drop_threshold:
        explanation['confinement'] = f"Wmhd dropped {100*w_drop:.0f}% from peak"
    
    # ── 6. Multi-mechanism stress ──
    n_stressed = 0
    if li_max > config.li_danger: n_stressed += 1
    if q_min < config.q_danger_threshold: n_stressed += 1
    if bn_max > config.troyon_coefficient * 0.8: n_stressed += 1
    if np.max(fgw) > 0.8: n_stressed += 1
    if rad_frac > config.radiation_fraction_limit * 0.8: n_stressed += 1
    if w_drop > config.energy_drop_threshold * 0.8: n_stressed += 1
    
    feats.extend([
        n_stressed,
        float(n_stressed >= 2),
        float(n_stressed >= 3),
    ])
    
    if n_stressed >= 2:
        explanation['multi_stress'] = f"{n_stressed} mechanisms simultaneously stressed"
    
    explanation['n_features'] = len(feats)
    return np.clip(np.nan_to_num(np.array(feats, dtype=np.float32)), -1e6, 1e6), explanation


# Feature names for interpretability
FM3_FEATURE_NAMES = [
    'dist_q2', 'closest_rational', 'inv_rational_prox', 'below_q_danger',
    'troyon_crit', 'troyon_margin', 'bN_x_li', 'bN_over_q',
    'li_rate', 'li_accel', 'li_over_q', 'li_rising_fast',
    'rad_frac', 'rad_frac_late', 'rad_above_limit', 'rad_trend',
    'w_drop', 'tau_degradation', 'w_loss_significant',
    'n_stressed', 'multi_stress_2', 'multi_stress_3',
]


def get_feature_count() -> int:
    """Number of FM3 physics features."""
    return len(FM3_FEATURE_NAMES)
