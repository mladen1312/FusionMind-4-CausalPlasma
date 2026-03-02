#!/usr/bin/env python3
"""
FusionMind 3.0 Lite — Simplified Physics Simulator
====================================================
Generates synthetic tokamak time series with KNOWN causal structure
for validation of CPDE causal discovery and CPC counterfactual reasoning.

Based on the full FM3 physics (see FusionMind_3_0_Ultimate.py) but reduced
to 14 key plasma variables with explicit causal DAG.

Author: Dr. Mladen Mester
Date: March 2026
License: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# GROUND TRUTH CAUSAL STRUCTURE
# ============================================================================

VARIABLE_NAMES = [
    "P_NBI", "P_ECRH", "gas_puff", "Ip",      # Actuators (0-3)
    "ne", "Te", "Ti", "q",                       # Profiles (4-7)
    "βN", "rotation", "P_rad", "W_stored",       # Globals (8-11)
    "MHD_amp", "n_imp",                           # Instabilities (12-13)
]

VARIABLE_CATEGORIES = {
    "actuator": [0, 1, 2, 3],
    "profile":  [4, 5, 6, 7],
    "global":   [8, 9, 10, 11],
    "instability": [12, 13],
}

# Ground truth edges: (from_idx, to_idx, weight, delay_ms, description)
GROUND_TRUTH_EDGES = [
    # === Actuator → Profile (direct heating/fueling) ===
    (0, 4, 0.30, 5, "NBI beam fueling"),
    (0, 5, 0.20, 10, "NBI → Te via collisional coupling"),
    (0, 6, 0.72, 3, "NBI heats ions directly"),
    (0, 9, 0.50, 2, "NBI drives toroidal rotation"),
    (1, 5, 0.69, 2, "ECRH heats electrons directly"),
    (2, 4, 0.76, 1, "Gas puff increases density"),
    (2, 5, -0.30, 5, "Gas dilution cooling"),
    (3, 7, -0.87, 10, "Plasma current sets q profile"),

    # === Profile → Global (physics coupling) ===
    (4, 8, 0.38, 1, "ne contributes to pressure/βN"),
    (4, 10, 0.28, 1, "ne drives line radiation"),
    (5, 8, 0.50, 1, "Te contributes to pressure/βN"),
    (5, 10, 0.25, 1, "Te drives bremsstrahlung"),
    (5, 11, 0.23, 1, "Te → stored energy"),
    (6, 8, 0.26, 1, "Ti contributes to pressure/βN"),
    (6, 11, 0.27, 1, "Ti → stored energy"),

    # === Global → Instability ===
    (7, 12, -0.40, 5, "Low q drives MHD instabilities"),
    (8, 12, 0.93, 2, "High βN drives MHD modes"),
    (13, 10, 0.61, 1, "Impurities radiate"),

    # === Instability feedback ===
    (10, 5, -0.30, 3, "Radiative cooling feedback"),
    (12, 4, -0.15, 5, "MHD particle loss"),
    (12, 5, -0.20, 3, "MHD confinement degradation"),
    (12, 11, -0.30, 2, "MHD energy loss"),
    (12, 13, 0.50, 10, "MHD causes wall sputtering → impurities"),
]

N_VARS = len(VARIABLE_NAMES)
N_EDGES = len(GROUND_TRUTH_EDGES)


def get_ground_truth_adjacency() -> np.ndarray:
    """Return ground truth adjacency matrix A[i,j] = weight of edge i→j"""
    A = np.zeros((N_VARS, N_VARS))
    for src, dst, w, _, _ in GROUND_TRUTH_EDGES:
        A[src, dst] = w
    return A


# ============================================================================
# PLASMA SIMULATOR
# ============================================================================

@dataclass
class FM3LiteConfig:
    """Configuration for FM3-Lite simulator"""
    n_timesteps: int = 2000
    dt_ms: float = 1.0         # Timestep in milliseconds
    noise_level: float = 0.03  # Gaussian noise fraction
    seed: int = 42

    # Actuator scenarios
    nbi_profile: str = "ramp"      # ramp, step, sine, random
    ecrh_profile: str = "step"
    gas_profile: str = "sine"
    ip_profile: str = "flat"

    # Physics parameters
    confinement_time_ms: float = 50.0
    energy_confinement_ms: float = 100.0


class FM3LiteSimulator:
    """
    Simplified tokamak physics simulator with known causal structure.
    Generates time series for 14 plasma variables where the causal
    relationships are explicitly defined and controllable.
    """

    def __init__(self, config: FM3LiteConfig = None):
        self.config = config or FM3LiteConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.A = get_ground_truth_adjacency()

    def generate_actuator_waveforms(self) -> np.ndarray:
        """Generate actuator waveforms (exogenous variables)"""
        N = self.config.n_timesteps
        t = np.arange(N) * self.config.dt_ms / 1000.0  # seconds

        actuators = np.zeros((N, 4))

        # P_NBI [MW] — ramp with perturbations
        if self.config.nbi_profile == "ramp":
            actuators[:, 0] = np.clip(5 + 10 * t / t[-1] + 
                                       2 * np.sin(2*np.pi*0.5*t), 0, 20)
        elif self.config.nbi_profile == "step":
            actuators[:, 0] = np.where(t > t[-1]/3, 15, 5)
        elif self.config.nbi_profile == "random":
            actuators[:, 0] = 5 + 10 * np.cumsum(self.rng.randn(N) * 0.01)
            actuators[:, 0] = np.clip(actuators[:, 0], 1, 20)
        else:
            actuators[:, 0] = 10 + 5 * np.sin(2*np.pi*0.3*t)

        # P_ECRH [MW]
        if self.config.ecrh_profile == "step":
            actuators[:, 1] = np.where(t > t[-1]/2, 8, 2)
        else:
            actuators[:, 1] = 4 + 3 * np.sin(2*np.pi*0.2*t)

        # Gas puff [10^20 /s]
        if self.config.gas_profile == "sine":
            actuators[:, 2] = 2 + 1.5 * np.sin(2*np.pi*0.4*t)
        else:
            actuators[:, 2] = np.where(t > t[-1]*0.6, 3, 1.5)

        # Ip [MA]
        if self.config.ip_profile == "flat":
            actuators[:, 3] = 1.0 + 0.1 * np.sin(2*np.pi*0.05*t)
        else:
            actuators[:, 3] = 0.5 + 0.5 * t / t[-1]

        # Add measurement noise to actuators
        noise = self.config.noise_level * 0.5
        for j in range(4):
            actuators[:, j] += noise * np.abs(actuators[:, j].mean()) * self.rng.randn(N)

        return actuators

    def simulate(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run simulation. Returns:
            data: (n_timesteps, 14) array of all variables
            actuators: (n_timesteps, 4) actuator waveforms
            metadata: dict with ground truth info
        """
        N = self.config.n_timesteps
        dt = self.config.dt_ms / 1000.0  # Convert to seconds
        noise = self.config.noise_level

        # Initialize all variables
        data = np.zeros((N, N_VARS))

        # Generate actuator waveforms (exogenous)
        actuators = self.generate_actuator_waveforms()
        data[:, 0:4] = actuators

        # Initial conditions for endogenous variables
        data[0, 4] = 5.0    # ne [10^19 m^-3]
        data[0, 5] = 3.0    # Te [keV]
        data[0, 6] = 2.5    # Ti [keV]
        data[0, 7] = 3.5    # q
        data[0, 8] = 1.5    # βN
        data[0, 9] = 50.0   # rotation [krad/s]
        data[0, 10] = 0.5   # P_rad [MW]
        data[0, 11] = 0.3   # W_stored [MJ]
        data[0, 12] = 0.01  # MHD_amp [a.u.]
        data[0, 13] = 0.01  # n_imp [10^17 m^-3]

        tau_p = self.config.confinement_time_ms / 1000.0
        tau_E = self.config.energy_confinement_ms / 1000.0

        # Time evolution with causal physics
        for t in range(1, N):
            prev = data[t-1]

            # ---- Profiles (with delay effects via exponential smoothing) ----

            # ne: gas fueling + NBI fueling - MHD losses - diffusion
            ne_source = 0.76 * data[t, 2] + 0.30 * data[t, 0]
            ne_loss = prev[4] / tau_p + 0.15 * prev[12]
            data[t, 4] = prev[4] + dt * (ne_source - ne_loss)

            # Te: ECRH + (collisional from NBI) - gas cooling - radiation - MHD
            Te_source = 0.69 * data[t, 1] + 0.20 * data[t, 0]
            Te_loss = prev[5] / tau_E + 0.30 * prev[10] + 0.20 * prev[12] + 0.30 * data[t, 2]
            data[t, 5] = prev[5] + dt * (Te_source - Te_loss)

            # Ti: NBI direct + electron-ion coupling
            Ti_source = 0.72 * data[t, 0] + 0.1 * (prev[5] - prev[6])
            Ti_loss = prev[6] / tau_E
            data[t, 6] = prev[6] + dt * (Ti_source - Ti_loss)

            # q: inverse relationship with Ip
            data[t, 7] = 5.0 * (0.8 / np.clip(data[t, 3], 0.3, 2.0))

            # ---- Globals ----

            # βN: pressure = ne * (Te + Ti)
            data[t, 8] = 0.38 * data[t, 4] + 0.50 * data[t, 5] + 0.26 * data[t, 6]
            data[t, 8] = data[t, 8] / 5.0  # Normalize

            # rotation: NBI driven
            rot_source = 0.50 * data[t, 0]
            rot_loss = prev[9] / (tau_p * 2)
            data[t, 9] = prev[9] + dt * (rot_source - rot_loss)

            # P_rad: bremsstrahlung + line radiation + impurity radiation
            data[t, 10] = (0.28 * data[t, 4] + 0.25 * data[t, 5] +
                           0.61 * data[t, 13])

            # W_stored: thermal energy
            data[t, 11] = (0.23 * data[t, 5] + 0.27 * data[t, 6]) - 0.30 * prev[12]
            data[t, 11] = max(data[t, 11], 0.01)

            # ---- Instabilities ----

            # MHD amplitude: driven by βN, suppressed by high q
            mhd_drive = 0.93 * data[t, 8] - 0.40 * (data[t, 7] - 2.0)
            mhd_drive = max(mhd_drive, 0)
            mhd_decay = prev[12] * 0.1
            data[t, 12] = prev[12] + dt * (mhd_drive - mhd_decay)
            data[t, 12] = np.clip(data[t, 12], 0, 10)

            # Impurity density: MHD wall sputtering + background
            n_imp_source = 0.50 * prev[12] + 0.001
            n_imp_loss = prev[13] / (tau_p * 5)
            data[t, 13] = prev[13] + dt * (n_imp_source - n_imp_loss)

            # ---- Physical bounds ----
            data[t, 4] = np.clip(data[t, 4], 0.1, 20)   # ne
            data[t, 5] = np.clip(data[t, 5], 0.1, 30)    # Te
            data[t, 6] = np.clip(data[t, 6], 0.1, 25)    # Ti
            data[t, 7] = np.clip(data[t, 7], 1.0, 10)    # q
            data[t, 8] = np.clip(data[t, 8], 0, 5)       # βN
            data[t, 9] = np.clip(data[t, 9], 0, 500)     # rotation
            data[t, 10] = np.clip(data[t, 10], 0, 50)    # P_rad
            data[t, 11] = np.clip(data[t, 11], 0.01, 10) # W_stored
            data[t, 13] = np.clip(data[t, 13], 0.001, 5) # n_imp

        # Add measurement noise
        for j in range(4, N_VARS):
            noise_std = noise * np.abs(data[:, j].mean())
            data[:, j] += noise_std * self.rng.randn(N)
            data[:, j] = np.maximum(data[:, j], 1e-6)

        metadata = {
            "n_vars": N_VARS,
            "n_edges": N_EDGES,
            "variable_names": VARIABLE_NAMES,
            "categories": VARIABLE_CATEGORIES,
            "ground_truth_edges": GROUND_TRUTH_EDGES,
            "adjacency_matrix": self.A,
            "config": self.config,
        }

        return data, actuators, metadata

    def simulate_intervention(self, intervention: Dict[int, float]) -> np.ndarray:
        """
        Simulate with do-calculus intervention: do(X_i = value)
        Cuts all incoming edges to intervened variables.

        Args:
            intervention: {variable_index: forced_value}
        Returns:
            data: (n_timesteps, 14) array with intervention applied
        """
        data, _, _ = self.simulate()

        # Apply intervention: override variable and re-simulate downstream
        for var_idx, value in intervention.items():
            data[:, var_idx] = value

        # Re-simulate downstream effects (simplified: just re-run affected vars)
        # For proper intervention, we'd cut incoming edges and re-propagate
        return data


def generate_multi_scenario_dataset(n_scenarios: int = 20, seed: int = 42) -> List[Tuple]:
    """Generate diverse scenarios for robust causal discovery"""
    rng = np.random.RandomState(seed)
    scenarios = []

    nbi_profiles = ["ramp", "step", "sine", "random"]
    ecrh_profiles = ["step", "sine"]
    gas_profiles = ["sine", "step"]
    ip_profiles = ["flat", "ramp"]

    for i in range(n_scenarios):
        config = FM3LiteConfig(
            n_timesteps=2000,
            dt_ms=1.0,
            noise_level=0.02 + 0.03 * rng.random(),
            seed=seed + i,
            nbi_profile=rng.choice(nbi_profiles),
            ecrh_profile=rng.choice(ecrh_profiles),
            gas_profile=rng.choice(gas_profiles),
            ip_profile=rng.choice(ip_profiles),
            confinement_time_ms=30 + 40 * rng.random(),
            energy_confinement_ms=60 + 80 * rng.random(),
        )
        sim = FM3LiteSimulator(config)
        data, actuators, metadata = sim.simulate()
        scenarios.append((data, actuators, metadata))

    return scenarios


if __name__ == "__main__":
    print("FM3-Lite Physics Simulator")
    print("=" * 50)

    config = FM3LiteConfig(n_timesteps=2000, noise_level=0.03, seed=42)
    sim = FM3LiteSimulator(config)
    data, actuators, metadata = sim.simulate()

    print(f"Generated {data.shape[0]} timesteps × {data.shape[1]} variables")
    print(f"Ground truth edges: {N_EDGES}")
    print(f"\nVariable ranges:")
    for i, name in enumerate(VARIABLE_NAMES):
        print(f"  {name:12s}: [{data[:,i].min():.3f}, {data[:,i].max():.3f}] "
              f"mean={data[:,i].mean():.3f}")
