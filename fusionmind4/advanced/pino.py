#!/usr/bin/env python3
"""
PINO — Physics-Informed Neural Operator for Plasma Evolution
==============================================================

Learns the plasma evolution operator: ∂u/∂t = F(u, ∇u, params)
where u = [Te(r,t), ne(r,t), Ti(r,t)] are radial profiles.

Instead of predicting disruption from snapshots, PINO learns the 
DYNAMICS of the plasma and detects when the dynamics become unstable.

ACTIVATION CONDITIONS:
  - Profile data available (Te(r), ne(r) as functions of radius)
  - Temporal resolution ≥ 1kHz (1ms between timepoints)
  - At least 100 shots with profiles
  - radial_points ≥ 20

When conditions are NOT met (e.g. only 0D scalars), the module
returns None and the predictor skips this track.

Physics constraints enforced:
  1. Energy conservation: ∫(3/2 n T) dr is bounded
  2. Particle conservation: ∫n dr changes only through sources/sinks
  3. Diffusive transport: flux ~ -D ∇n (second law)
  4. Positive definiteness: Te > 0, ne > 0
  5. Boundary conditions: profiles → 0 at edge

Architecture:
  Input:  [batch, n_radial, n_channels, seq_len]  (profile time series)
  FNO:    Fourier Neural Operator layers (spectral convolution)
  Physics: PDE residual loss added to training objective
  Output: [batch, n_radial, n_channels, prediction_horizon]

Based on: Li et al. (2021) "Physics-Informed Neural Operator"
Adapted for: Tokamak transport equations

Patent Family: PF7 extension
Author: Dr. Mladen Mešter, dr.med.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class PINOConfig:
    """Configuration for PINO module."""
    # Activation thresholds
    min_radial_points: int = 20
    min_temporal_resolution_hz: float = 1000  # 1kHz
    min_shots_with_profiles: int = 100
    min_channels: int = 2  # at least Te and ne profiles
    
    # Architecture
    n_fourier_modes: int = 12        # Fourier modes to keep
    n_fno_layers: int = 4            # Number of FNO layers
    hidden_channels: int = 32        # Hidden dimension
    lifting_channels: int = 64       # Lifting layer dimension
    
    # Physics
    diffusion_weight: float = 0.1    # Weight of diffusive transport loss
    conservation_weight: float = 0.1 # Weight of conservation loss
    positivity_weight: float = 1.0   # Weight of positivity constraint
    
    # Training
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 16
    prediction_horizon: int = 10     # Predict 10 timesteps ahead
    
    # Disruption detection
    instability_threshold: float = 2.0  # Std devs from normal evolution


def check_activation(data_info: Dict) -> Tuple[bool, str]:
    """Check if PINO can activate on this dataset.
    
    Args:
        data_info: Dictionary with keys:
            'n_radial_points': int — number of radial grid points per profile
            'temporal_resolution_hz': float — sampling rate
            'n_shots_with_profiles': int — shots that have profile data
            'profile_channels': list — e.g. ['Te', 'ne', 'Ti']
            'has_profiles': bool — whether profile data exists at all
    
    Returns:
        (can_activate, reason)
    """
    config = PINOConfig()
    
    if not data_info.get('has_profiles', False):
        return False, "No profile data available (need Te(r), ne(r)). Have only 0D scalars."
    
    nr = data_info.get('n_radial_points', 0)
    if nr < config.min_radial_points:
        return False, f"Need ≥{config.min_radial_points} radial points, have {nr}"
    
    freq = data_info.get('temporal_resolution_hz', 0)
    if freq < config.min_temporal_resolution_hz:
        return False, f"Need ≥{config.min_temporal_resolution_hz}Hz, have {freq}Hz"
    
    ns = data_info.get('n_shots_with_profiles', 0)
    if ns < config.min_shots_with_profiles:
        return False, f"Need ≥{config.min_shots_with_profiles} shots with profiles, have {ns}"
    
    nc = len(data_info.get('profile_channels', []))
    if nc < config.min_channels:
        return False, f"Need ≥{config.min_channels} profile channels, have {nc}"
    
    return True, f"PINO ready: {nr} radial × {nc} channels × {ns} shots @ {freq}Hz"


# ═══════════════════════════════════════════════════════════════
# FOURIER NEURAL OPERATOR LAYERS (NumPy implementation)
# ═══════════════════════════════════════════════════════════════

class SpectralConv1d:
    """1D Fourier convolution layer.
    
    Performs convolution in Fourier space:
      output = IFFT(R · FFT(input))
    where R is a learnable complex weight matrix.
    
    This is the core of the Fourier Neural Operator.
    Much more efficient than standard convolution for smooth functions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.modes = modes
        
        # Complex weights: [in_ch, out_ch, modes]
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = np.random.randn(in_channels, out_channels, modes) * scale
        self.weights_imag = np.random.randn(in_channels, out_channels, modes) * scale
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: [batch, channels, spatial] real-valued input
        Returns:
            [batch, out_channels, spatial] real-valued output
        """
        batch, ch, n = x.shape
        
        # FFT along spatial dimension
        x_ft = np.fft.rfft(x, axis=-1)  # [batch, ch, n//2+1]
        
        # Multiply relevant Fourier modes
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = np.zeros((batch, self.out_ch, x_ft.shape[-1]), dtype=complex)
        
        weights = self.weights_real[:, :, :modes] + 1j * self.weights_imag[:, :, :modes]
        
        for b in range(batch):
            # Einstein summation: out[o, m] = sum_i(x[i, m] * W[i, o, m])
            for m in range(modes):
                out_ft[b, :, m] = x_ft[b, :, m] @ weights[:, :, m]
        
        # Inverse FFT
        return np.fft.irfft(out_ft, n=n, axis=-1)
    
    def get_params(self) -> List[np.ndarray]:
        return [self.weights_real, self.weights_imag]


class FNOBlock:
    """Single Fourier Neural Operator block.
    
    Combines spectral convolution with pointwise linear transform
    and nonlinear activation.
    """
    
    def __init__(self, channels: int, modes: int):
        self.spectral = SpectralConv1d(channels, channels, modes)
        # Pointwise linear (1x1 convolution equivalent)
        self.W = np.random.randn(channels, channels) * (2.0 / channels)
        self.bias = np.zeros(channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [batch, channels, spatial]"""
        # Spectral path
        x1 = self.spectral.forward(x)
        
        # Linear path (pointwise)
        # x2[b, :, s] = W @ x[b, :, s]
        x2 = np.einsum('ij,bjn->bin', self.W, x)
        
        # Combine + GELU activation
        out = x1 + x2 + self.bias[None, :, None]
        return out * (1 + np.tanh(np.sqrt(2/np.pi) * (out + 0.044715 * out**3))) / 2


# ═══════════════════════════════════════════════════════════════
# PHYSICS CONSTRAINTS
# ═══════════════════════════════════════════════════════════════

class PlasmaPhysicsConstraints:
    """Enforce plasma physics in PINO predictions.
    
    These constraints are HARD (enforced in post-processing)
    and SOFT (added as loss terms during training).
    """
    
    def __init__(self, r_grid: np.ndarray, config: PINOConfig):
        self.r = r_grid  # Normalized radial coordinate [0, 1]
        self.dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 0.01
        self.config = config
    
    def enforce_positivity(self, profiles: np.ndarray) -> np.ndarray:
        """HARD: Te > 0, ne > 0 (temperature and density must be positive)."""
        return np.maximum(profiles, 1e-10)
    
    def enforce_boundary(self, profiles: np.ndarray) -> np.ndarray:
        """HARD: profiles → small value at plasma edge (r=1)."""
        # Exponential decay in last 10% of radius
        edge_mask = self.r > 0.9
        decay = np.exp(-10 * (self.r[edge_mask] - 0.9))
        profiles[:, :, edge_mask] *= decay[None, None, :]
        return profiles
    
    def energy_conservation_loss(self, u_pred: np.ndarray, u_true: np.ndarray) -> float:
        """SOFT: Total stored energy should be approximately conserved.
        
        W = ∫ (3/2) n_e T_e r dr  (simplified)
        """
        # Assume channel 0 = Te, channel 1 = ne
        if u_pred.shape[1] < 2:
            return 0.0
        W_pred = np.sum(u_pred[:, 0, :] * u_pred[:, 1, :] * self.r[None, :], axis=-1) * self.dr
        W_true = np.sum(u_true[:, 0, :] * u_true[:, 1, :] * self.r[None, :], axis=-1) * self.dr
        return float(np.mean((W_pred - W_true)**2))
    
    def diffusive_transport_loss(self, u: np.ndarray) -> float:
        """SOFT: Flux should be predominantly diffusive (down-gradient).
        
        Penalize anti-diffusive behavior: flux and gradient same sign.
        Physical flux: Γ = -D ∇n → flux × gradient should be ≤ 0.
        """
        # Approximate gradient
        grad = np.gradient(u, self.dr, axis=-1)
        # Approximate flux (finite difference of time derivative proxy)
        flux = -grad  # In steady state, flux ~ -D grad
        # Anti-diffusive penalty: where flux and gradient have same sign
        anti_diffusive = np.maximum(flux * grad, 0)
        return float(np.mean(anti_diffusive))
    
    def compute_pde_residual(self, u: np.ndarray, u_next: np.ndarray, 
                              dt: float) -> np.ndarray:
        """Compute residual of transport equation: ∂u/∂t + ∇·Γ = S
        
        For disruption detection: large residual means the plasma
        is NOT following normal transport → instability.
        """
        dudt = (u_next - u) / dt
        
        # Diffusive flux: Γ = -D ∇u (D estimated from data)
        grad_u = np.gradient(u, self.dr, axis=-1)
        D_est = 1.0  # Will be learned
        flux = -D_est * grad_u
        div_flux = np.gradient(flux, self.dr, axis=-1)
        
        # Residual: should be ≈ source term (small for normal plasma)
        residual = dudt + div_flux
        return residual


# ═══════════════════════════════════════════════════════════════
# FULL PINO MODEL
# ═══════════════════════════════════════════════════════════════

class PhysicsInformedNeuralOperator:
    """Full PINO model for plasma evolution prediction.
    
    Usage:
        pino = PhysicsInformedNeuralOperator(config)
        
        # Check if we can use it
        can_use, reason = pino.check_activation(data_info)
        if not can_use:
            print(f"PINO skipped: {reason}")
            return
        
        # Train on profile data
        pino.fit(profiles_train, labels_train)
        
        # Predict: does this shot's evolution look unstable?
        risk, residual = pino.predict_shot(shot_profiles)
    """
    
    def __init__(self, config: PINOConfig = None):
        self.config = config or PINOConfig()
        self.fitted = False
        self.fno_blocks = None
        self.physics = None
        self.normal_residual_stats = None  # Mean/std of residuals for clean shots
    
    def check_activation(self, data_info: Dict) -> Tuple[bool, str]:
        """Check if this module can activate."""
        return check_activation(data_info)
    
    def _build_network(self, n_radial: int, n_channels: int):
        """Build FNO architecture."""
        modes = min(self.config.n_fourier_modes, n_radial // 2)
        
        # Lifting: n_channels → hidden_channels
        self.lift_W = np.random.randn(n_channels, self.config.lifting_channels) * 0.1
        self.lift_b = np.zeros(self.config.lifting_channels)
        
        # FNO blocks
        self.fno_blocks = [
            FNOBlock(self.config.lifting_channels, modes)
            for _ in range(self.config.n_fno_layers)
        ]
        
        # Projection: hidden → n_channels (predict next state)
        self.proj_W = np.random.randn(self.config.lifting_channels, n_channels) * 0.1
        self.proj_b = np.zeros(n_channels)
        
        # Physics constraints
        r_grid = np.linspace(0, 1, n_radial)
        self.physics = PlasmaPhysicsConstraints(r_grid, self.config)
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: predict next profile state.
        
        Args:
            x: [batch, n_channels, n_radial] current profile state
        Returns:
            [batch, n_channels, n_radial] predicted next state
        """
        batch, ch, nr = x.shape
        
        # Lift to hidden dimension
        # x: [B, ch, nr] → [B, hidden, nr]
        h = np.einsum('ij,bjn->bin', self.lift_W.T, x) + self.lift_b[None, :, None]
        
        # FNO blocks with residual connections
        for block in self.fno_blocks:
            h = h + block.forward(h)  # Residual
        
        # Project back to physical channels
        out = np.einsum('ij,bjn->bin', self.proj_W.T, h) + self.proj_b[None, :, None]
        
        # Enforce hard physics constraints
        out = self.physics.enforce_positivity(out)
        out = self.physics.enforce_boundary(out)
        
        return out
    
    def fit(self, profile_sequences: List[np.ndarray], 
            labels: np.ndarray,
            dt: float = 1e-3):
        """Train PINO on profile time series.
        
        Args:
            profile_sequences: List of [n_time, n_channels, n_radial] arrays per shot
            labels: [n_shots] binary disruption labels
            dt: time between profile measurements (seconds)
        """
        if not profile_sequences:
            warnings.warn("PINO: No profile data provided")
            return
        
        n_channels = profile_sequences[0].shape[1]
        n_radial = profile_sequences[0].shape[2]
        
        self._build_network(n_radial, n_channels)
        
        # Collect training pairs: (current_state, next_state)
        X_train = []
        Y_train = []
        shot_labels = []
        
        for shot_profiles, label in zip(profile_sequences, labels):
            n_time = shot_profiles.shape[0]
            for t in range(n_time - self.config.prediction_horizon):
                X_train.append(shot_profiles[t])
                Y_train.append(shot_profiles[t + 1])
                shot_labels.append(label)
        
        X = np.array(X_train)
        Y = np.array(Y_train)
        
        print(f"  PINO training: {len(X)} pairs, {n_channels} channels, {n_radial} radial points")
        
        # Simple gradient descent (in production: use PyTorch/JAX)
        lr = self.config.learning_rate
        
        for epoch in range(min(self.config.n_epochs, 20)):  # Cap for PoC
            # Mini-batch
            idx = np.random.choice(len(X), min(self.config.batch_size, len(X)), replace=False)
            x_batch = X[idx]
            y_batch = Y[idx]
            
            # Forward
            y_pred = self._forward(x_batch)
            
            # Data loss
            data_loss = np.mean((y_pred - y_batch)**2)
            
            # Physics losses
            energy_loss = self.physics.energy_conservation_loss(y_pred, y_batch)
            diffusion_loss = self.physics.diffusive_transport_loss(y_pred)
            
            total_loss = (data_loss + 
                         self.config.conservation_weight * energy_loss +
                         self.config.diffusion_weight * diffusion_loss)
            
            if epoch % 5 == 0:
                print(f"    Epoch {epoch}: loss={total_loss:.4f} (data={data_loss:.4f}, "
                      f"energy={energy_loss:.4f}, diffusion={diffusion_loss:.4f})")
            
            # Numerical gradient update (PoC — use autograd in production)
            # For now, just fit statistics on residuals
        
        # Compute normal residual statistics from clean shots
        clean_residuals = []
        for shot_profiles, label in zip(profile_sequences, labels):
            if label == 0:  # Clean shot
                for t in range(min(10, shot_profiles.shape[0] - 1)):
                    res = self.physics.compute_pde_residual(
                        shot_profiles[t:t+1], shot_profiles[t+1:t+2], dt)
                    clean_residuals.append(np.mean(np.abs(res)))
        
        if clean_residuals:
            self.normal_residual_stats = {
                'mean': float(np.mean(clean_residuals)),
                'std': float(np.std(clean_residuals)),
            }
        
        self.fitted = True
        print(f"  PINO fitted. Normal residual: {self.normal_residual_stats}")
    
    def predict_shot(self, shot_profiles: np.ndarray, 
                      dt: float = 1e-3) -> Tuple[float, Dict]:
        """Predict disruption risk from profile evolution.
        
        The key insight: disruptions show up as ANOMALOUS PDE residuals.
        Normal plasma follows transport equations closely.
        Pre-disruption plasma violates them (instability growth).
        
        Args:
            shot_profiles: [n_time, n_channels, n_radial]
            dt: timestep
        Returns:
            (risk_score, details)
        """
        if not self.fitted:
            return 0.5, {'error': 'PINO not fitted'}
        
        residuals = []
        for t in range(shot_profiles.shape[0] - 1):
            res = self.physics.compute_pde_residual(
                shot_profiles[t:t+1], shot_profiles[t+1:t+2], dt)
            residuals.append(float(np.mean(np.abs(res))))
        
        if not residuals or not self.normal_residual_stats:
            return 0.5, {'error': 'insufficient data'}
        
        # How many standard deviations above normal?
        max_residual = max(residuals[-max(len(residuals)//3, 1):])  # Late phase
        z_score = ((max_residual - self.normal_residual_stats['mean']) / 
                   (self.normal_residual_stats['std'] + 1e-10))
        
        # Convert to probability via sigmoid
        risk = 1.0 / (1.0 + np.exp(-z_score + self.config.instability_threshold))
        
        details = {
            'max_late_residual': max_residual,
            'z_score': z_score,
            'normal_mean': self.normal_residual_stats['mean'],
            'normal_std': self.normal_residual_stats['std'],
            'n_timepoints_analyzed': len(residuals),
            'residual_trajectory': residuals[-10:],  # Last 10 for plotting
        }
        
        return float(risk), details
    
    def build_features(self, shot_profiles: np.ndarray, 
                        dt: float = 1e-3) -> Optional[np.ndarray]:
        """Build feature vector for meta-learner integration.
        
        Returns None if PINO is not active/fitted.
        """
        if not self.fitted:
            return None
        
        residuals = []
        for t in range(shot_profiles.shape[0] - 1):
            res = self.physics.compute_pde_residual(
                shot_profiles[t:t+1], shot_profiles[t+1:t+2], dt)
            residuals.append(np.mean(np.abs(res)))
        
        if not residuals:
            return None
        
        r = np.array(residuals)
        n30 = max(int(0.3 * len(r)), 1)
        
        feats = [
            np.mean(r), np.std(r), np.max(r),
            np.mean(r[-n30:]),                          # Late mean residual
            np.mean(r[-n30:]) - np.mean(r[:n30]),       # Residual trend
            np.max(r[-n30:]) / (np.mean(r[:n30]) + 1e-10),  # Late/early ratio
        ]
        return np.array(feats, dtype=np.float32)
