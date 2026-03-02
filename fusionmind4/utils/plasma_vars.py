"""Plasma variable definitions and ground truth causal graph."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class PlasmaVariable:
    """Single plasma variable in the causal graph."""
    id: int
    name: str
    category: str  # actuator, profile, global, instability
    label: str
    is_exogenous: bool = False  # True for actuators (no causes)


# ── 14 Plasma Variables ──────────────────────────────────────────────────────
PLASMA_VARS: List[PlasmaVariable] = [
    PlasmaVariable(0,  "P_NBI",    "actuator",    "NBI Power",         True),
    PlasmaVariable(1,  "P_ECRH",   "actuator",    "ECRH Power",        True),
    PlasmaVariable(2,  "gas_puff", "actuator",    "Gas Puff Rate",     True),
    PlasmaVariable(3,  "Ip",       "actuator",    "Plasma Current",    True),
    PlasmaVariable(4,  "ne",       "profile",     "Electron Density",  False),
    PlasmaVariable(5,  "Te",       "profile",     "Electron Temp",     False),
    PlasmaVariable(6,  "Ti",       "profile",     "Ion Temp",          False),
    PlasmaVariable(7,  "q",        "profile",     "Safety Factor",     False),
    PlasmaVariable(8,  "betaN",    "global",      "Normalized Beta",   False),
    PlasmaVariable(9,  "rotation", "global",      "Toroidal Rotation", False),
    PlasmaVariable(10, "P_rad",    "global",      "Radiated Power",    False),
    PlasmaVariable(11, "W_stored", "global",      "Stored Energy",     False),
    PlasmaVariable(12, "MHD_amp",  "instability", "MHD Amplitude",     False),
    PlasmaVariable(13, "n_imp",    "instability", "Impurity Density",  False),
]

N_VARS = len(PLASMA_VARS)
VAR_NAMES = [v.name for v in PLASMA_VARS]
ACTUATOR_IDS = {v.id for v in PLASMA_VARS if v.is_exogenous}


def get_var_by_name(name: str) -> PlasmaVariable:
    """Look up variable by name."""
    for v in PLASMA_VARS:
        if v.name == name:
            return v
    raise ValueError(f"Unknown variable: {name}")


# ── Ground Truth Causal Graph ────────────────────────────────────────────────
# 28 edges from plasma physics knowledge
# Format: (cause_id, effect_id, weight, description)
GROUND_TRUTH_EDGES: List[Tuple[int, int, float, str]] = [
    # Actuator → Profile
    (0,  6,  +0.72, "NBI heats ions (direct)"),
    (0,  9,  +0.50, "NBI drives toroidal rotation"),
    (0,  4,  +0.30, "NBI beam fueling"),
    (0,  5,  +0.20, "NBI→Te via collisions (weak)"),
    (1,  5,  +0.69, "ECRH heats electrons"),
    (2,  4,  +0.76, "Gas puff fuels density"),
    (2,  5,  -0.30, "Gas dilution cooling"),
    (3,  7,  -0.87, "Plasma current sets q profile"),

    # Profile → Global
    (4,  8,  +0.38, "Density → pressure (βN)"),
    (4,  10, +0.28, "Density → bremsstrahlung radiation"),
    (4,  11, +0.15, "Density → stored energy"),
    (5,  8,  +0.50, "Te → pressure (βN)"),
    (5,  10, +0.25, "Te → bremsstrahlung"),
    (5,  11, +0.23, "Te → stored energy"),
    (5,  6,  +0.10, "Te→Ti equipartition (very weak)"),
    (6,  8,  +0.26, "Ti → pressure (βN)"),
    (6,  11, +0.27, "Ti → stored energy"),
    (6,  9,  +0.20, "Ti gradient drives rotation"),

    # Global → Instability
    (7,  12, -0.40, "Low q → MHD instability"),
    (8,  12, +0.93, "High βN → MHD modes"),
    (9,  12, -0.35, "Rotation stabilizes MHD"),

    # Feedback paths
    (10, 5,  -0.30, "Radiative cooling"),
    (12, 4,  -0.15, "MHD particle loss (weak)"),
    (12, 5,  -0.20, "MHD confinement degradation"),
    (12, 11, -0.30, "MHD energy loss"),
    (12, 13, +0.50, "MHD wall sputtering → impurities"),

    # Impurity chain
    (13, 10, +0.61, "Impurity line radiation"),
    (13, 5,  -0.15, "Impurity dilution cooling (weak)"),
]


def build_ground_truth_adjacency() -> np.ndarray:
    """Build adjacency matrix from ground truth edges."""
    adj = np.zeros((N_VARS, N_VARS))
    for cause, effect, weight, _ in GROUND_TRUTH_EDGES:
        adj[cause, effect] = weight
    return adj


def evaluate_dag(discovered: np.ndarray, threshold: float = 0.0) -> dict:
    """Evaluate a discovered DAG against ground truth.

    Args:
        discovered: NxN adjacency matrix of discovered graph
        threshold: minimum weight to count as an edge

    Returns:
        Dictionary with TP, FP, FN, precision, recall, F1, SHD
    """
    gt = build_ground_truth_adjacency()
    gt_binary = (np.abs(gt) > 0).astype(int)
    disc_binary = (np.abs(discovered) > threshold).astype(int)

    tp = int(np.sum(gt_binary * disc_binary))
    fp = int(np.sum((1 - gt_binary) * disc_binary))
    fn = int(np.sum(gt_binary * (1 - disc_binary)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    shd = fp + fn

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "shd": shd,
        "n_discovered": int(np.sum(disc_binary)),
        "n_true": int(np.sum(gt_binary)),
    }
