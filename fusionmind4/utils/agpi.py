#!/usr/bin/env python3
"""
AGPI — Adaptive Geometry-aware Physics Injector
=================================================

Soft-gates FM3 physics features based on machine geometry:
  weight = σ(4.2 - q95) × min(3.5/A, 2.0)

where:
  q95 = edge safety factor (mean over shot)
  A = aspect ratio R0/a

This gives:
  MAST  (q95~7, A~1.5): weight ≈ 0.10 → FM3 physics nearly off
  C-Mod (q95~3.5, A~3): weight ≈ 0.78 → FM3 physics mostly on
  DIII-D (q95~3, A~2.8): weight ≈ 0.96 → FM3 physics fully on
  ITER  (q95~3, A~3.1): weight ≈ 0.87 → FM3 physics on

Better than binary on/off because:
  - Edge cases (MAST-U, ST40) get smooth transition
  - No manual threshold tuning per machine
  - Single formula works for any tokamak

TESTED ON MAST: No AUC improvement (weight=10% ≈ off).
Expected improvement on conventional tokamaks when data available.

Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple
from fusionmind4.utils.fm3_physics import (
    build_fm3_physics_features, FM3PhysicsConfig,
    FM3_FEATURE_NAMES, get_feature_count
)


def agpi_weight(q95_mean: float, aspect_ratio: float = 2.5) -> float:
    """Compute soft gate weight for FM3 physics features.
    
    Args:
        q95_mean: Mean edge safety factor over the shot
        aspect_ratio: R0/a (major radius / minor radius)
        
    Returns:
        Weight in [0.05, 1.0]. Higher = more FM3 physics influence.
    """
    # Sigmoid centered at q95 = 4.2
    # Below 4.2: conventional tokamak → high weight
    # Above 4.2: spherical tokamak → low weight
    base = 1.0 / (1.0 + np.exp(q95_mean - 4.2))
    
    # Geometry bonus: conventional (A~3) gets full weight
    # Spherical (A~1.5) gets reduced weight
    geometry_factor = min(3.5 / aspect_ratio, 2.0)
    
    return float(np.clip(base * geometry_factor, 0.05, 1.0))


def compute_aspect_ratio(R0: float = None, a: float = None,
                          minor_radius_signal: np.ndarray = None) -> float:
    """Estimate aspect ratio from available data.
    
    Known values:
      MAST:  R=0.85m, a=0.65m → A=1.3
      C-Mod: R=0.67m, a=0.22m → A=3.0
      DIII-D: R=1.67m, a=0.67m → A=2.5
      JET:   R=2.96m, a=1.25m → A=2.4
      ITER:  R=6.2m, a=2.0m → A=3.1
    """
    if R0 and a:
        return R0 / a
    
    # Estimate from minor_radius signal if available
    if minor_radius_signal is not None:
        a_mean = np.mean(minor_radius_signal[minor_radius_signal > 0.01])
        if a_mean < 0.4:  # Small minor radius → conventional
            return 3.0
        elif a_mean < 0.6:
            return 2.5
        else:  # Large minor radius → spherical
            return 1.5
    
    return 2.5  # Default: mid-range


def build_agpi_features(
    li: np.ndarray, q95: np.ndarray, betan: np.ndarray,
    fgw: np.ndarray, p_rad: np.ndarray, p_input: np.ndarray,
    wmhd: np.ndarray, n30: int,
    aspect_ratio: float = 2.5,
    machine_type: str = 'auto',
) -> Tuple[np.ndarray, float, Dict]:
    """Build AGPI-weighted FM3 physics features.
    
    Returns:
        (weighted_features, agpi_weight_value, explanation)
    """
    # Determine q95 for gating
    q_valid = q95[q95 > 0.5]
    q95_mean = float(np.mean(q_valid)) if len(q_valid) > 0 else 10.0
    
    # Compute AGPI weight
    weight = agpi_weight(q95_mean, aspect_ratio)
    
    # Select physics config based on weight
    if weight > 0.5:
        config = FM3PhysicsConfig.for_conventional()
    else:
        config = FM3PhysicsConfig.for_spherical()
    
    # Build raw FM3 features
    raw_features, explanation = build_fm3_physics_features(
        li, q95, betan, fgw, p_rad, p_input, wmhd, n30, config
    )
    
    # Apply soft weight
    weighted_features = raw_features * weight
    
    # Add AGPI metadata to explanation
    explanation['agpi_weight'] = round(weight, 3)
    explanation['q95_mean'] = round(q95_mean, 2)
    explanation['aspect_ratio'] = round(aspect_ratio, 2)
    explanation['config_used'] = 'conventional' if weight > 0.5 else 'spherical'
    
    return weighted_features, weight, explanation


# ═══════════════════════════════════════════════════
# Δq TEARING PROXIMITY FEATURE (from Grok suggestion)
# ═══════════════════════════════════════════════════

def tearing_proximity_features(q95: np.ndarray) -> np.ndarray:
    """Additional tearing proximity features.
    
    Δq = |q95 - m/n| for dangerous rational surfaces.
    Also: q-shear indicator from q95 trajectory.
    """
    q_valid = q95[q95 > 0.5]
    if len(q_valid) < 3:
        return np.zeros(4, dtype=np.float32)
    
    q_min = np.min(q_valid)
    q_late = np.mean(q_valid[-max(len(q_valid)//3, 1):])
    
    # Δq for each dangerous surface
    dq_2_1 = abs(q_min - 2.0)     # 2/1 tearing (most dangerous)
    dq_3_2 = abs(q_min - 1.5)     # 3/2 NTM
    
    # q-shear: how fast q95 is changing
    q_rate = (q_late - np.mean(q_valid[:max(len(q_valid)//3, 1)])) / (len(q_valid) + 1e-10) * 100
    
    # Combined tearing risk: closer to ANY rational + dropping q = worse
    tearing_risk = 1.0 / (min(dq_2_1, dq_3_2) + 0.1) * max(-q_rate, 0)
    
    return np.array([dq_2_1, dq_3_2, q_rate, np.clip(tearing_risk, 0, 100)], dtype=np.float32)


AGPI_FEATURE_NAMES = FM3_FEATURE_NAMES + ['Δq_2/1', 'Δq_3/2', 'q_rate', 'tearing_risk']

def get_total_feature_count() -> int:
    return get_feature_count() + 4  # FM3 + tearing proximity
