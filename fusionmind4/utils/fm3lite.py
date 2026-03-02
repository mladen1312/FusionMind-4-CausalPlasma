"""FM3-Lite Physics Engine — Causally-faithful plasma data generation.

Generates synthetic multi-variate time series where causal relationships
are known by construction, matching the ground truth graph in plasma_vars.py.
Supports interventional data generation for do-calculus validation.
"""
import numpy as np
from typing import Tuple, Optional, Dict
from ..utils.plasma_vars import PLASMA_VARS, N_VARS, GROUND_TRUTH_EDGES, ACTUATOR_IDS


class FM3LitePhysicsEngine:
    """Physics-based synthetic data generator with known causal structure.

    Generates data where:
    - Actuators are exogenous (independent random drives)
    - Downstream variables are computed from structural equations
    - Noise is added at each stage to simulate measurement uncertainty
    - Causal effects match the ground truth graph

    Args:
        n_samples: Number of time samples to generate
        noise_scale: Relative noise amplitude (default 0.08)
        seed: Random seed for reproducibility
    """

    def __init__(self, n_samples: int = 20000, noise_scale: float = 0.08, seed: int = 42):
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

        # Build causal weight matrix from ground truth
        self.W = np.zeros((N_VARS, N_VARS))
        for cause, effect, weight, _ in GROUND_TRUTH_EDGES:
            self.W[cause, effect] = weight

    def generate(self) -> Tuple[np.ndarray, Optional[Dict]]:
        """Generate synthetic plasma data and optional interventional data.

        Returns:
            data: (n_samples, N_VARS) array of observations
            interventions: dict of {actuator_id: (data_base, data_intervened)}
        """
        data = self._generate_observational()
        interventions = self._generate_interventional()
        return data, interventions

    def _generate_observational(self) -> np.ndarray:
        """Generate observational data following the causal graph."""
        N = self.n_samples
        data = np.zeros((N, N_VARS))
        noise = lambda: self.rng.randn(N) * self.noise_scale

        # Layer 0: Actuators (exogenous)
        data[:, 0] = self.rng.uniform(0.5, 2.0, N)   # P_NBI
        data[:, 1] = self.rng.uniform(0.3, 1.5, N)   # P_ECRH
        data[:, 2] = self.rng.uniform(0.1, 1.0, N)   # gas_puff
        data[:, 3] = self.rng.uniform(0.8, 1.5, N)   # Ip

        # Layer 1: Profiles (from actuators)
        data[:, 4] = (  # ne
            self.W[2, 4] * data[:, 2] +
            self.W[0, 4] * data[:, 0] +
            noise()
        )
        data[:, 5] = (  # Te
            self.W[1, 5] * data[:, 1] +
            self.W[0, 5] * data[:, 0] +
            self.W[2, 5] * data[:, 2] +
            noise()
        )
        data[:, 6] = (  # Ti
            self.W[0, 6] * data[:, 0] +
            self.W[5, 6] * data[:, 5] +
            noise()
        )
        data[:, 7] = (  # q
            self.W[3, 7] * data[:, 3] +
            noise()
        )

        # Layer 2: Global (from profiles)
        data[:, 8] = (  # betaN
            self.W[4, 8] * data[:, 4] +
            self.W[5, 8] * data[:, 5] +
            self.W[6, 8] * data[:, 6] +
            noise()
        )
        data[:, 9] = (  # rotation
            self.W[0, 9] * data[:, 0] +
            self.W[6, 9] * data[:, 6] +
            noise()
        )
        data[:, 10] = (  # P_rad
            self.W[4, 10] * data[:, 4] +
            self.W[5, 10] * data[:, 5] +
            noise()
        )
        data[:, 11] = (  # W_stored
            self.W[4, 11] * data[:, 4] +
            self.W[5, 11] * data[:, 5] +
            self.W[6, 11] * data[:, 6] +
            noise()
        )

        # Layer 3: Instability (from global + profiles)
        data[:, 12] = (  # MHD_amp
            self.W[7, 12] * data[:, 7] +
            self.W[8, 12] * data[:, 8] +
            self.W[9, 12] * data[:, 9] +
            noise()
        )
        data[:, 13] = (  # n_imp
            self.W[12, 13] * data[:, 12] +
            noise()
        )

        # Layer 4: Feedback (add to existing)
        data[:, 5] += self.W[10, 5] * data[:, 10] + self.W[12, 5] * data[:, 12]
        data[:, 10] += self.W[13, 10] * data[:, 13]
        data[:, 11] += self.W[12, 11] * data[:, 12]
        data[:, 4] += self.W[12, 4] * data[:, 12]

        return data

    def _generate_interventional(self) -> Dict:
        """Generate interventional data for each actuator.

        For each actuator, generates paired data:
        - Base: actuator at low value
        - Intervened: actuator at high value, everything else regenerated

        Returns:
            dict: {actuator_id: (data_low, data_high)}
        """
        interventions = {}
        n_int = min(2000, self.n_samples // 5)

        for act_id in ACTUATOR_IDS:
            # Generate with low actuator value
            data_low = self._generate_with_intervention(act_id, 0.3, n_int)
            # Generate with high actuator value
            data_high = self._generate_with_intervention(act_id, 1.8, n_int)
            interventions[act_id] = (data_low, data_high)

        return interventions

    def _generate_with_intervention(self, act_id: int, value: float, n: int) -> np.ndarray:
        """Generate data with a specific actuator fixed to a value."""
        saved_n = self.n_samples
        self.n_samples = n
        data = self._generate_observational()
        data[:, act_id] = value
        # Recompute downstream
        self.n_samples = saved_n
        return data

    def add_noise(self, data: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise for OOD robustness testing."""
        noise = np.random.randn(*data.shape) * noise_level
        return data + noise * np.std(data, axis=0, keepdims=True)
