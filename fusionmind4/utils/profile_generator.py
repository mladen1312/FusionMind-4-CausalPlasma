#!/usr/bin/env python3
"""
H-Mode Profile Generator — ported from FusionMind 3.0
=======================================================

Generates realistic radial plasma profiles (Te(r), ne(r), Ti(r))
with proper H-mode pedestal structure for testing PINO and D3R modules.

Ported from FM3 AdvancedPlasmaPhysics.create_h_mode_profile()
with additions: disrupted profile generation, profile evolution.

Activates when: PINO or D3R need synthetic profile test data.

Author: Originally FM3 (Dr. Mladen Mester), ported to FM4 March 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ProfileConfig:
    """Configuration for profile generation."""
    n_radial: int = 64
    # Typical values (SI): Te in keV, ne in 1e19 m^-3
    Te_core: float = 10.0   # keV
    Te_edge: float = 0.1
    ne_core: float = 5.0    # 1e19 m^-3
    ne_edge: float = 0.5
    Ti_core: float = 8.0
    Ti_edge: float = 0.1
    pedestal_width: float = 0.05   # Normalized radius
    pedestal_location: float = 0.90


def generate_h_mode_profile(core_value: float, edge_value: float,
                             n_radial: int = 64,
                             pedestal_width: float = 0.05) -> np.ndarray:
    """Generate H-mode profile with pedestal.

    Structure: flat core → steep pedestal → exponential SOL decay.

    Ported from FM3 AdvancedPlasmaPhysics.create_h_mode_profile()

    Args:
        core_value: Value at magnetic axis (r=0)
        edge_value: Value at separatrix
        n_radial: Number of radial grid points
        pedestal_width: Width of pedestal in normalized radius

    Returns:
        profile: [n_radial] array from core (r=0) to edge (r=1)
    """
    r = np.linspace(0, 1, n_radial)
    profile = np.zeros(n_radial)
    ped_start = 1.0 - 2 * pedestal_width

    # Core region: parabolic-like, slightly peaked
    core_mask = r < ped_start
    profile[core_mask] = core_value * (1 - 0.9 * (r[core_mask] / ped_start)**2)

    # Pedestal region: steep gradient
    ped_mask = (r >= ped_start) & (r < 1 - pedestal_width)
    ped_r = (r[ped_mask] - ped_start) / pedestal_width
    profile[ped_mask] = core_value * 0.1 + (core_value * 0.9 - edge_value) * (1 - ped_r)

    # SOL region: exponential decay
    sol_mask = r >= 1 - pedestal_width
    profile[sol_mask] = edge_value * np.exp(-(r[sol_mask] - (1 - pedestal_width)) / 0.02)

    return profile


def generate_full_profiles(config: ProfileConfig = None,
                            seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate a complete set of H-mode profiles.

    Returns:
        Dictionary with 'Te', 'ne', 'Ti', 'r' arrays
    """
    config = config or ProfileConfig()
    rng = np.random.RandomState(seed)

    # Add small random variation to make it realistic
    noise_factor = 0.02

    Te = generate_h_mode_profile(config.Te_core, config.Te_edge,
                                  config.n_radial, config.pedestal_width)
    Te *= (1 + noise_factor * rng.randn(config.n_radial))
    Te = np.maximum(Te, 0.01)

    ne = generate_h_mode_profile(config.ne_core, config.ne_edge,
                                  config.n_radial, config.pedestal_width)
    ne *= (1 + noise_factor * rng.randn(config.n_radial))
    ne = np.maximum(ne, 0.01)

    Ti = generate_h_mode_profile(config.Ti_core, config.Ti_edge,
                                  config.n_radial, config.pedestal_width)
    Ti *= (1 + noise_factor * rng.randn(config.n_radial))
    Ti = np.maximum(Ti, 0.01)

    r = np.linspace(0, 1, config.n_radial)

    return {'Te': Te, 'ne': ne, 'Ti': Ti, 'r': r}


def generate_profile_evolution(n_time: int = 100, dt_ms: float = 1.0,
                                 disrupt_at: Optional[int] = None,
                                 config: ProfileConfig = None,
                                 seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate time-evolving profiles for PINO testing.

    Normal evolution: profiles slowly diffuse and reach steady state.
    Disruption: at disrupt_at, profiles collapse rapidly.

    Args:
        n_time: Number of time steps
        dt_ms: Time step in milliseconds
        disrupt_at: Time index of disruption onset (None = no disruption)
        config: Profile configuration
        seed: Random seed

    Returns:
        Dictionary with 'Te', 'ne', 'Ti': [n_time, n_radial] arrays
        and 'r': [n_radial], 'time_ms': [n_time], 'disrupted': bool
    """
    config = config or ProfileConfig()
    rng = np.random.RandomState(seed)
    nr = config.n_radial
    r = np.linspace(0, 1, nr)

    # Initial profiles
    base = generate_full_profiles(config, seed)

    Te = np.zeros((n_time, nr))
    ne = np.zeros((n_time, nr))
    Ti = np.zeros((n_time, nr))

    Te[0] = base['Te']
    ne[0] = base['ne']
    Ti[0] = base['Ti']

    # Diffusion coefficient (normalized, small for stability)
    D = 0.0001
    dr = 1.0 / (nr - 1)

    for t in range(1, n_time):
        for profile, base_p in [(Te, base['Te']), (ne, base['ne']), (Ti, base['Ti'])]:
            prev = profile[t - 1].copy()
            # Simple diffusion: d²f/dr² with stability limit
            diff = np.zeros(nr)
            diff[1:-1] = (prev[2:] - 2*prev[1:-1] + prev[:-2]) / (dr**2)
            relaxation = 0.02 * (base_p - prev)  # Relax toward base
            noise = 0.002 * rng.randn(nr) * np.abs(prev)
            profile[t] = prev + (D * diff + relaxation + noise)
            profile[t] = np.clip(profile[t], 0.001, 1e6)

        # Disruption: thermal quench
        if disrupt_at is not None and t >= disrupt_at:
            phase = t - disrupt_at
            # Thermal quench: Te drops to ~10% in ~5 timesteps
            quench = np.exp(-phase / 3.0)
            Te[t] *= quench
            Ti[t] *= quench
            # Current quench (density spike then drop)
            if phase < 5:
                ne[t] *= (1 + 0.3 * np.exp(-((r - 0.3) / 0.2)**2))  # Edge spike
            else:
                ne[t] *= np.exp(-(phase - 5) / 5.0)

    return {
        'Te': Te, 'ne': ne, 'Ti': Ti, 'r': r,
        'time_ms': np.arange(n_time) * dt_ms,
        'disrupted': disrupt_at is not None,
        'disruption_time_ms': disrupt_at * dt_ms if disrupt_at else None,
    }


def generate_pino_test_dataset(n_clean: int = 50, n_disrupted: int = 20,
                                 n_time: int = 100,
                                 config: ProfileConfig = None,
                                 seed: int = 42) -> Tuple:
    """Generate complete test dataset for PINO module.

    Returns:
        profiles: List of [n_time, 3, n_radial] arrays (Te, ne, Ti stacked)
        labels: [n_shots] binary labels
    """
    config = config or ProfileConfig()
    rng = np.random.RandomState(seed)

    profiles = []
    labels = []

    for i in range(n_clean + n_disrupted):
        is_dis = i >= n_clean

        # Vary core values for diversity
        cfg = ProfileConfig(
            n_radial=config.n_radial,
            Te_core=config.Te_core * rng.uniform(0.7, 1.3),
            ne_core=config.ne_core * rng.uniform(0.7, 1.3),
            Ti_core=config.Ti_core * rng.uniform(0.7, 1.3),
        )

        disrupt_at = rng.randint(int(0.5 * n_time), int(0.9 * n_time)) if is_dis else None

        evo = generate_profile_evolution(
            n_time=n_time, disrupt_at=disrupt_at,
            config=cfg, seed=seed + i
        )

        # Stack Te, ne, Ti into [n_time, 3, n_radial]
        stacked = np.stack([evo['Te'], evo['ne'], evo['Ti']], axis=1)
        profiles.append(stacked)
        labels.append(1 if is_dis else 0)

    return profiles, np.array(labels)
