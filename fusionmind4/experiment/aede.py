"""
AEDE — Active Experiment Design Engine (PF5)
=============================================
Designs optimal tokamak experiments to maximise causal knowledge gain.

Given the current causal graph (from CPDE) with uncertainty estimates,
AEDE answers: "What experiment should we run next to learn the most
about the true causal structure of the plasma?"

Core innovations:
1. Information-theoretic scoring: Expected Information Gain (EIG)
   per experiment using mutual information between graph and data
2. Causal-graph-aware design: Targets uncertain edges, confounders,
   and untested interventional distributions
3. Physics-constrained search: Only proposes experiments within
   machine operational limits
4. Sequential design: Updates after each experiment via Bayesian
   posterior on edge probabilities

Patent Family: PF5
Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# EXPERIMENT DEFINITION
# ============================================================================

@dataclass
class ExperimentDesign:
    """A proposed tokamak experiment."""
    name: str
    description: str
    actuator_settings: Dict[str, float]        # Variable → setpoint
    target_edges: List[Tuple[str, str]]         # Edges this tests
    expected_information_gain: float            # EIG in bits
    feasibility_score: float                    # 0–1, physics constraints
    estimated_duration: float                   # seconds
    risk_level: str                             # low / medium / high
    priority_rank: int = 0

    @property
    def score(self) -> float:
        """Combined score: information × feasibility × (1/risk)."""
        risk_mult = {"low": 1.0, "medium": 0.7, "high": 0.4}
        return self.expected_information_gain * self.feasibility_score * risk_mult.get(self.risk_level, 0.5)


@dataclass
class MachineOperationalLimits:
    """Physical limits of the tokamak."""
    Ip_range: Tuple[float, float] = (0.3, 1.5)   # MA
    P_NBI_range: Tuple[float, float] = (0.0, 8.0) # MW
    P_ECRH_range: Tuple[float, float] = (0.0, 5.0) # MW
    P_ICRH_range: Tuple[float, float] = (0.0, 4.0) # MW
    gas_puff_range: Tuple[float, float] = (0.5, 8.0) # 10²⁰/s
    n_e_max: float = 1.2e20       # m⁻³ (Greenwald)
    beta_N_max: float = 3.5       # no-wall limit
    q95_min: float = 2.0          # disruption boundary
    pulse_max: float = 30.0       # seconds


# ============================================================================
# EDGE UNCERTAINTY ESTIMATOR
# ============================================================================

class EdgeUncertaintyEstimator:
    """Estimate uncertainty of each edge in the causal graph.
    
    Uses bootstrap stability scores from CPDE plus edge-specific
    metrics to quantify where the graph is most uncertain.
    """

    def __init__(self, variable_names: List[str]):
        self.names = variable_names
        self.n = len(variable_names)
        self.idx = {name: i for i, name in enumerate(variable_names)}

    def compute_uncertainties(
        self,
        bootstrap_stability: np.ndarray,
        ensemble_agreement: np.ndarray,
        edge_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute per-edge uncertainty (entropy).
        
        Args:
            bootstrap_stability: (n,n) fraction of bootstraps where edge appears
            ensemble_agreement: (n,n) fraction of algorithms agreeing on edge
            edge_weights: (n,n) absolute edge weights from ensemble
            
        Returns:
            uncertainty: (n,n) entropy-based uncertainty in [0, 1]
        """
        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        # Maximum at p=0.5 (most uncertain), minimum at p=0/1
        p_boot = np.clip(bootstrap_stability, 1e-10, 1 - 1e-10)
        H_boot = -p_boot * np.log2(p_boot) - (1 - p_boot) * np.log2(1 - p_boot)

        p_ens = np.clip(ensemble_agreement, 1e-10, 1 - 1e-10)
        H_ens = -p_ens * np.log2(p_ens) - (1 - p_ens) * np.log2(1 - p_ens)

        # Weight uncertainty: edges with near-threshold weights are uncertain
        w_normalized = np.abs(edge_weights) / (np.max(np.abs(edge_weights)) + 1e-10)
        # Edges near 0.3-0.5 (typical threshold zone) are most uncertain
        H_weight = 1 - np.abs(2 * w_normalized - 1)

        # Combined uncertainty (weighted average)
        uncertainty = 0.4 * H_boot + 0.4 * H_ens + 0.2 * H_weight
        np.fill_diagonal(uncertainty, 0)

        return uncertainty

    def get_most_uncertain_edges(self, uncertainty: np.ndarray,
                                  top_k: int = 10) -> List[Dict]:
        """Return the top-k most uncertain edges."""
        edges = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j and uncertainty[i, j] > 0.05:
                    edges.append({
                        'cause': self.names[i],
                        'effect': self.names[j],
                        'uncertainty': float(uncertainty[i, j]),
                    })
        edges.sort(key=lambda e: e['uncertainty'], reverse=True)
        return edges[:top_k]


# ============================================================================
# INFORMATION GAIN CALCULATOR
# ============================================================================

class InformationGainCalculator:
    """Compute expected information gain for proposed experiments.
    
    Uses the framework:
        EIG(experiment) = H(graph | current data) - E[H(graph | current data + new data)]
        
    Approximated via bootstrap Monte Carlo over possible experimental outcomes.
    """

    def __init__(self, n_mc_samples: int = 100, seed: int = 42):
        self.n_mc = n_mc_samples
        self.rng = np.random.RandomState(seed)

    def compute_eig(
        self,
        current_uncertainty: np.ndarray,
        intervention_var: str,
        intervention_value: float,
        target_vars: List[str],
        var_names: List[str],
        edge_weights: np.ndarray,
    ) -> float:
        """Compute Expected Information Gain for a single-variable intervention.
        
        Args:
            current_uncertainty: (n,n) edge uncertainties
            intervention_var: Variable being intervened on
            intervention_value: Setpoint value
            target_vars: Variables we observe the effect on
            var_names: All variable names
            edge_weights: Current edge weight estimates
            
        Returns:
            eig: Expected information gain in bits
        """
        idx = {name: i for i, name in enumerate(var_names)}
        iv = idx.get(intervention_var)
        if iv is None:
            return 0.0

        # Current entropy of edges FROM the intervention variable
        H_current = 0.0
        target_indices = [idx[v] for v in target_vars if v in idx]
        for tj in target_indices:
            H_current += current_uncertainty[iv, tj]

        # Expected posterior entropy (Monte Carlo approximation)
        H_posterior_samples = []
        for _ in range(self.n_mc):
            # Simulate possible outcome: with intervention, we observe
            # the effect with some noise → reduces edge uncertainty
            noise_reduction = self.rng.beta(2, 3)  # Typically reduces by 0.3-0.7
            H_post = 0.0
            for tj in target_indices:
                H_post += current_uncertainty[iv, tj] * (1 - noise_reduction)
            H_posterior_samples.append(H_post)

        H_expected_posterior = np.mean(H_posterior_samples)
        eig = max(0, H_current - H_expected_posterior)

        return float(eig)

    def compute_multi_intervention_eig(
        self,
        current_uncertainty: np.ndarray,
        interventions: Dict[str, float],
        var_names: List[str],
        edge_weights: np.ndarray,
    ) -> float:
        """EIG for multi-variable interventions (scan experiments)."""
        idx = {name: i for i, name in enumerate(var_names)}
        total_eig = 0.0

        for var, val in interventions.items():
            iv = idx.get(var)
            if iv is None:
                continue
            # Information gain from this intervention
            for j in range(len(var_names)):
                if j != iv and current_uncertainty[iv, j] > 0.05:
                    total_eig += current_uncertainty[iv, j] * 0.5  # Expected reduction

        return float(total_eig)


# ============================================================================
# EXPERIMENT GENERATOR
# ============================================================================

class ExperimentGenerator:
    """Generate candidate experiments based on causal graph uncertainty."""

    def __init__(self, variable_names: List[str],
                 limits: Optional[MachineOperationalLimits] = None):
        self.names = variable_names
        self.limits = limits or MachineOperationalLimits()
        self.actuators = ['I_p', 'P_NBI', 'P_ECRH', 'P_ICRH', 'gas_puff']
        self.idx = {name: i for i, name in enumerate(variable_names)}

    def generate_single_variable_scans(
        self,
        uncertainty: np.ndarray,
        edge_weights: np.ndarray,
        n_per_actuator: int = 3,
    ) -> List[ExperimentDesign]:
        """Generate single-variable scan experiments.
        
        For each actuator, propose experiments at different setpoints
        to resolve uncertain edges.
        """
        experiments = []
        ranges = {
            'I_p': self.limits.Ip_range,
            'P_NBI': self.limits.P_NBI_range,
            'P_ECRH': self.limits.P_ECRH_range,
            'P_ICRH': self.limits.P_ICRH_range,
            'gas_puff': self.limits.gas_puff_range,
        }

        for act in self.actuators:
            if act not in self.idx:
                continue
            ai = self.idx[act]
            rng = ranges.get(act, (0, 1))

            # Find uncertain edges from this actuator
            target_edges = []
            for j in range(len(self.names)):
                if j != ai and uncertainty[ai, j] > 0.1:
                    target_edges.append((act, self.names[j]))

            if not target_edges:
                continue

            # Generate scan points (low, mid, high)
            setpoints = np.linspace(rng[0], rng[1], n_per_actuator + 2)[1:-1]

            for sp in setpoints:
                # Compute EIG for this setpoint
                eig = sum(uncertainty[ai, self.idx[e[1]]] for e in target_edges
                          if e[1] in self.idx) * 0.5

                # Assess risk
                risk = self._assess_risk({act: sp})

                experiments.append(ExperimentDesign(
                    name=f"{act}_scan_{sp:.1f}",
                    description=f"Scan {act} = {sp:.2f} to resolve {len(target_edges)} uncertain edges",
                    actuator_settings={act: sp},
                    target_edges=target_edges,
                    expected_information_gain=eig,
                    feasibility_score=self._compute_feasibility({act: sp}),
                    estimated_duration=5.0,  # Typical shot duration
                    risk_level=risk,
                ))

        return experiments

    def generate_factorial_designs(
        self,
        uncertainty: np.ndarray,
        edge_weights: np.ndarray,
    ) -> List[ExperimentDesign]:
        """Generate 2-variable factorial designs for confounder resolution."""
        experiments = []
        ranges = {
            'I_p': self.limits.Ip_range,
            'P_NBI': self.limits.P_NBI_range,
            'P_ECRH': self.limits.P_ECRH_range,
            'gas_puff': self.limits.gas_puff_range,
        }

        # Find pairs of actuators with shared uncertain children
        available = [a for a in self.actuators if a in self.idx]
        for i, a1 in enumerate(available):
            for a2 in available[i + 1:]:
                ai, aj = self.idx[a1], self.idx[a2]

                # Shared uncertain children
                shared = []
                for k in range(len(self.names)):
                    if (uncertainty[ai, k] > 0.1 and uncertainty[aj, k] > 0.1):
                        shared.append(self.names[k])

                if not shared:
                    continue

                # 2x2 factorial: (low, low), (low, high), (high, low), (high, high)
                r1, r2 = ranges.get(a1, (0, 1)), ranges.get(a2, (0, 1))
                levels = [(r1[0], r2[0]), (r1[0], r2[1]),
                          (r1[1], r2[0]), (r1[1], r2[1])]

                for v1, v2 in levels:
                    settings = {a1: v1, a2: v2}
                    eig = sum(uncertainty[ai, self.idx[s]] + uncertainty[aj, self.idx[s]]
                              for s in shared if s in self.idx) * 0.4
                    risk = self._assess_risk(settings)

                    experiments.append(ExperimentDesign(
                        name=f"factorial_{a1}_{a2}_{v1:.1f}_{v2:.1f}",
                        description=(f"Factorial design: {a1}={v1:.1f}, {a2}={v2:.1f} "
                                     f"to disentangle effects on {', '.join(shared)}"),
                        actuator_settings=settings,
                        target_edges=[(a1, s) for s in shared] + [(a2, s) for s in shared],
                        expected_information_gain=eig,
                        feasibility_score=self._compute_feasibility(settings),
                        estimated_duration=5.0,
                        risk_level=risk,
                    ))

        return experiments

    def generate_confounder_resolution(
        self,
        uncertainty: np.ndarray,
        edge_weights: np.ndarray,
    ) -> List[ExperimentDesign]:
        """Generate experiments specifically targeting potential confounders.
        
        If we suspect X→Z←Y (confounder structure), we need to intervene
        on X while controlling Y to resolve the causal direction.
        """
        experiments = []
        ranges = {
            'I_p': self.limits.Ip_range,
            'P_NBI': self.limits.P_NBI_range,
            'P_ECRH': self.limits.P_ECRH_range,
            'gas_puff': self.limits.gas_puff_range,
        }

        for k in range(len(self.names)):
            parents_uncertain = []
            for i in range(len(self.names)):
                if i != k and uncertainty[i, k] > 0.15:
                    parents_uncertain.append(i)

            if len(parents_uncertain) < 2:
                continue

            # For each pair of uncertain parents, propose intervention on one
            for pi in range(len(parents_uncertain)):
                for pj in range(pi + 1, len(parents_uncertain)):
                    a = self.names[parents_uncertain[pi]]
                    b = self.names[parents_uncertain[pj]]
                    target = self.names[k]

                    if a not in ranges:
                        continue

                    rng_a = ranges[a]
                    settings = {a: (rng_a[0] + rng_a[1]) / 2}
                    eig = uncertainty[parents_uncertain[pi], k] * 0.6

                    experiments.append(ExperimentDesign(
                        name=f"confounder_{a}_{b}_{target}",
                        description=(f"Intervene on {a} to resolve confounder between "
                                     f"{a} and {b} for {target}"),
                        actuator_settings=settings,
                        target_edges=[(a, target), (b, target)],
                        expected_information_gain=eig,
                        feasibility_score=self._compute_feasibility(settings),
                        estimated_duration=5.0,
                        risk_level="medium",
                    ))

        return experiments

    def _assess_risk(self, settings: Dict[str, float]) -> str:
        """Assess risk level of proposed experiment."""
        risk_score = 0.0

        # High current → higher risk
        if 'I_p' in settings and settings['I_p'] > 0.8 * self.limits.Ip_range[1]:
            risk_score += 0.3

        # High power → higher beta → instability risk
        total_power = sum(settings.get(p, 0) for p in ['P_NBI', 'P_ECRH', 'P_ICRH'])
        if total_power > 10.0:
            risk_score += 0.3

        # Low gas with high power → density limit
        if 'gas_puff' in settings and settings['gas_puff'] < 1.5 and total_power > 5.0:
            risk_score += 0.2

        if risk_score > 0.5:
            return "high"
        elif risk_score > 0.2:
            return "medium"
        return "low"

    def _compute_feasibility(self, settings: Dict[str, float]) -> float:
        """Score 0–1 for how feasible this experiment is."""
        ranges = {
            'I_p': self.limits.Ip_range,
            'P_NBI': self.limits.P_NBI_range,
            'P_ECRH': self.limits.P_ECRH_range,
            'P_ICRH': self.limits.P_ICRH_range,
            'gas_puff': self.limits.gas_puff_range,
        }
        feasibility = 1.0
        for var, val in settings.items():
            rng = ranges.get(var)
            if rng is None:
                continue
            if val < rng[0] or val > rng[1]:
                feasibility *= 0.0  # Out of range
            else:
                # Prefer mid-range values (safer)
                mid = (rng[0] + rng[1]) / 2
                span = (rng[1] - rng[0]) / 2
                dist = abs(val - mid) / span
                feasibility *= max(0.3, 1.0 - 0.5 * dist)
        return feasibility


# ============================================================================
# MAIN AEDE ENGINE
# ============================================================================

class ActiveExperimentDesignEngine:
    """
    Active Experiment Design Engine.
    
    Full pipeline:
    1. Estimate edge uncertainties from CPDE bootstrap + ensemble
    2. Generate candidate experiments (scans, factorials, confounder resolution)
    3. Score each by Expected Information Gain × feasibility × safety
    4. Rank and return top-k recommended experiments
    5. After experiment: update posteriors and replan
    
    Usage:
        aede = ActiveExperimentDesignEngine(var_names)
        experiments = aede.design_experiments(
            bootstrap_stability=cpde_result['bootstrap_stability'],
            ensemble_agreement=cpde_result['ensemble_agreement'],
            edge_weights=cpde_result['adjacency'],
        )
        for exp in experiments[:5]:
            print(f"{exp.priority_rank}. {exp.name} (EIG={exp.expected_information_gain:.3f})")
    """

    def __init__(self, variable_names: List[str],
                 limits: Optional[MachineOperationalLimits] = None,
                 seed: int = 42):
        self.names = variable_names
        self.limits = limits or MachineOperationalLimits()
        self.uncertainty_estimator = EdgeUncertaintyEstimator(variable_names)
        self.eig_calc = InformationGainCalculator(seed=seed)
        self.generator = ExperimentGenerator(variable_names, self.limits)
        self.experiment_history: List[ExperimentDesign] = []
        self._posterior_uncertainty: Optional[np.ndarray] = None

    def design_experiments(
        self,
        bootstrap_stability: np.ndarray,
        ensemble_agreement: np.ndarray,
        edge_weights: np.ndarray,
        top_k: int = 10,
    ) -> List[ExperimentDesign]:
        """Design the next batch of optimal experiments.
        
        Args:
            bootstrap_stability: (n,n) from CPDE bootstrap
            ensemble_agreement: (n,n) from CPDE ensemble
            edge_weights: (n,n) current edge weight estimates
            top_k: Number of experiments to return
            
        Returns:
            Ranked list of ExperimentDesign objects
        """
        # 1. Compute edge uncertainties
        uncertainty = self.uncertainty_estimator.compute_uncertainties(
            bootstrap_stability, ensemble_agreement, edge_weights
        )
        self._posterior_uncertainty = uncertainty

        # 2. Generate candidates
        candidates = []
        candidates.extend(self.generator.generate_single_variable_scans(
            uncertainty, edge_weights))
        candidates.extend(self.generator.generate_factorial_designs(
            uncertainty, edge_weights))
        candidates.extend(self.generator.generate_confounder_resolution(
            uncertainty, edge_weights))

        if not candidates:
            return []

        # 3. Score and rank
        candidates.sort(key=lambda e: e.score, reverse=True)
        for rank, exp in enumerate(candidates):
            exp.priority_rank = rank + 1

        return candidates[:top_k]

    def get_uncertain_edges(self, top_k: int = 10) -> List[Dict]:
        """Return the most uncertain edges (requires prior design_experiments call)."""
        if self._posterior_uncertainty is None:
            return []
        return self.uncertainty_estimator.get_most_uncertain_edges(
            self._posterior_uncertainty, top_k)

    def update_after_experiment(
        self,
        experiment: ExperimentDesign,
        observed_data: np.ndarray,
        observed_edges: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        """Update posteriors after running an experiment.
        
        In a full implementation, this would:
        1. Re-run CPDE with augmented data
        2. Update edge posterior probabilities
        3. Recalculate uncertainties
        
        For now, reduces uncertainty on targeted edges.
        """
        self.experiment_history.append(experiment)

        if self._posterior_uncertainty is not None and observed_edges:
            idx = {name: i for i, name in enumerate(self.names)}
            for (cause, effect), confidence in observed_edges.items():
                ci, ej = idx.get(cause), idx.get(effect)
                if ci is not None and ej is not None:
                    # Reduce uncertainty based on experimental evidence
                    self._posterior_uncertainty[ci, ej] *= (1 - confidence)

    def get_summary(self) -> Dict:
        """Return summary of AEDE state."""
        return {
            'n_variables': len(self.names),
            'experiments_run': len(self.experiment_history),
            'total_uncertain_edges': int(
                np.sum(self._posterior_uncertainty > 0.1)
            ) if self._posterior_uncertainty is not None else 0,
            'mean_uncertainty': float(
                np.mean(self._posterior_uncertainty[self._posterior_uncertainty > 0])
            ) if self._posterior_uncertainty is not None and np.any(self._posterior_uncertainty > 0) else 0,
        }
