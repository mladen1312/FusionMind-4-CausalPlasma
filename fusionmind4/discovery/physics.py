"""Physics priors and validation for causal plasma discovery.

Encodes domain knowledge from plasma physics as:
1. Prior edge weights (known causal relationships)
2. Structural constraints (DAG, actuator exogeneity)
3. Post-hoc validation checks (10 physics tests)
"""
import numpy as np
from typing import Dict, List, Tuple
from ..utils.plasma_vars import PLASMA_VARS, N_VARS, ACTUATOR_IDS


# ── Physics Prior Edges ──────────────────────────────────────────────────────
# Known causal relationships from plasma physics with confidence scores
PHYSICS_PRIORS: List[Tuple[int, int, float]] = [
    # Required edges (high confidence)
    (0, 6, 1.0),   # NBI → Ti (direct ion heating)
    (1, 5, 1.0),   # ECRH → Te (direct electron heating)
    (2, 4, 1.0),   # gas_puff → ne (particle fueling)
    (3, 7, 1.0),   # Ip → q (Ampere's law)

    # Known physics paths (medium confidence)
    (0, 9, 0.6),   # NBI → rotation (momentum injection)
    (0, 4, 0.6),   # NBI → ne (beam fueling)
    (0, 5, 0.6),   # NBI → Te (collisional coupling)
    (2, 5, 0.6),   # gas → Te (dilution cooling, negative)
    (4, 8, 0.6),   # ne → betaN (pressure contribution)
    (5, 8, 0.6),   # Te → betaN
    (6, 8, 0.6),   # Ti → betaN
    (4, 10, 0.6),  # ne → P_rad (bremsstrahlung)
    (5, 10, 0.6),  # Te → P_rad
    (6, 11, 0.6),  # Ti → W_stored
    (5, 11, 0.6),  # Te → W_stored
    (4, 11, 0.6),  # ne → W_stored
    (8, 12, 1.0),  # betaN → MHD_amp (beta limit)
    (7, 12, 0.6),  # q → MHD (rational surfaces)
    (13, 10, 0.6), # n_imp → P_rad (impurity radiation)
    (9, 12, 0.6),  # rotation → MHD (stabilization)
    (6, 9, 0.6),   # Ti → rotation (neoclassical)
]


def get_physics_prior_matrix() -> np.ndarray:
    """Build physics prior matrix.

    Returns:
        (N_VARS, N_VARS) matrix with prior confidence [0, 1]
    """
    prior = np.zeros((N_VARS, N_VARS))
    for i, j, conf in PHYSICS_PRIORS:
        prior[i, j] = conf
    return prior


def validate_physics(dag: np.ndarray) -> Dict[str, bool]:
    """Run 10 physics validation checks on discovered DAG.

    Args:
        dag: (N_VARS, N_VARS) adjacency matrix

    Returns:
        Dictionary of {check_name: passed}
    """
    checks = {}

    # 1. DAG Acyclicity
    checks["dag_acyclic"] = bool(_find_cycle(dag) is None)

    # 2. Actuator Exogeneity (no edges INTO actuators)
    checks["actuator_exogeneity"] = bool(all(
        all(dag[i, a] == 0 for i in range(N_VARS) if i not in ACTUATOR_IDS)
        for a in ACTUATOR_IDS
    ))

    # 3. Energy Conservation (heating → stored energy path)
    checks["energy_conservation"] = bool(_path_exists(dag, [0, 1], 11))

    # 4. NBI → Ion Heating
    checks["nbi_ion_heating"] = bool(dag[0, 6] > 0)

    # 5. ECRH → Electron Heating
    checks["ecrh_electron_heating"] = bool(dag[1, 5] > 0)

    # 6. Gas → Density
    checks["gas_density"] = bool(dag[2, 4] > 0)

    # 7. Current → Safety Factor
    checks["current_q"] = bool(dag[3, 7] > 0)

    # 8. Beta → MHD chain
    checks["beta_mhd_chain"] = bool(dag[8, 12] > 0 or _path_exists(dag, [4, 5, 6], 12))

    # 9. Radiation chain
    checks["radiation_chain"] = bool(dag[4, 10] > 0 or dag[13, 10] > 0 or dag[5, 10] > 0)

    # 10. No actuator crosstalk
    checks["no_actuator_crosstalk"] = bool(all(
        dag[i, j] == 0 for i in ACTUATOR_IDS for j in ACTUATOR_IDS if i != j
    ))

    # Summary
    passed = sum(1 for v in checks.values() if v is True)
    total = len(checks)
    checks["passed"] = passed
    checks["total"] = total

    return checks


def _find_cycle(adj: np.ndarray):
    """Detect cycle in adjacency matrix using DFS."""
    d = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * d

    def dfs(u):
        color[u] = GRAY
        for v in range(d):
            if adj[u, v] > 0:
                if color[v] == GRAY:
                    return (u, v)
                if color[v] == WHITE:
                    result = dfs(v)
                    if result:
                        return result
        color[u] = BLACK
        return None

    for u in range(d):
        if color[u] == WHITE:
            result = dfs(u)
            if result:
                return result
    return None


def _path_exists(dag: np.ndarray, sources: List[int], target: int,
                 max_depth: int = 5) -> bool:
    """Check if a directed path exists from any source to target."""
    for src in sources:
        visited = set()
        queue = [(src, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node == target and depth > 0:
                return True
            if depth >= max_depth or node in visited:
                continue
            visited.add(node)
            for j in range(dag.shape[0]):
                if dag[node, j] > 0 and j not in visited:
                    queue.append((j, depth + 1))
    return False
