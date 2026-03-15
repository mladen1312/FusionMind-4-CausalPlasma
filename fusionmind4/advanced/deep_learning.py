#!/usr/bin/env python3
"""
Deep Learning Track — GRU + Temporal CNN + Transformer
=======================================================

Activates ONLY when ALL conditions met:
  1. PyTorch importable
  2. GPU available (CUDA or MPS) — CPU deep learning too slow for production
  3. ≥ 200 disrupted shots (otherwise GBT wins, proven empirically)
  4. ≥ 50 timepoints per shot average

When conditions NOT met, returns None and predictor uses GBT tracks.

Why these thresholds:
  - 83 disrupted + GBT = AUC 0.979. Deep learning can't beat this on 83 samples.
  - At ~200+ disrupted, temporal patterns become learnable by RNN/CNN.
  - At ~1000+ disrupted, Transformer starts outperforming GRU.
  - GPU requirement: GRU training on CPU takes 10-30min vs 30s on GPU.

Three sub-models (activated progressively):
  DL-A: GRU (≥200 disrupted) — temporal sequence prediction
  DL-B: Temporal CNN (≥500 disrupted) — multi-scale convolution 
  DL-C: Transformer (≥1000 disrupted) — self-attention, longest context

Each produces shot-level P(disruption) for meta-learner integration.

Author: Dr. Mladen Mešter, dr.med.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
import time


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning track."""
    # Activation thresholds
    min_disrupted_gru: int = 200
    min_disrupted_cnn: int = 500
    min_disrupted_transformer: int = 1000
    min_timepoints_per_shot: int = 50
    require_gpu: bool = True
    
    # GRU
    gru_hidden: int = 64
    gru_layers: int = 2
    gru_dropout: float = 0.3
    gru_seq_len: int = 30
    
    # Temporal CNN
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # Transformer
    tfm_d_model: int = 64
    tfm_nhead: int = 4
    tfm_layers: int = 2
    tfm_seq_len: int = 50
    
    # Training
    epochs_gru: int = 15
    epochs_cnn: int = 20
    epochs_transformer: int = 25
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5, 7]


# ═══════════════════════════════════════════════════════════════
# HARDWARE + DATA CHECKS
# ═══════════════════════════════════════════════════════════════

def check_pytorch() -> Tuple[bool, str]:
    """Check if PyTorch is available and usable."""
    try:
        import torch
        version = torch.__version__
        return True, f"PyTorch {version}"
    except ImportError:
        return False, "PyTorch not installed (pip install torch)"


def check_gpu() -> Tuple[bool, str]:
    """Check for GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            return True, f"CUDA: {name} ({mem:.1f}GB)"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, "Apple MPS (Metal)"
        else:
            return False, "No GPU found (CUDA/MPS unavailable)"
    except:
        return False, "PyTorch GPU check failed"


def check_activation(data_info: Dict, config: DeepLearningConfig = None) -> Tuple[bool, str, List[str]]:
    """Full activation check. Returns (can_activate, reason, active_models).
    
    Args:
        data_info: {
            'n_disrupted': int,
            'n_shots': int,
            'mean_timepoints_per_shot': float,
            'n_variables': int,
        }
    """
    config = config or DeepLearningConfig()
    
    # Check PyTorch
    has_torch, torch_reason = check_pytorch()
    if not has_torch:
        return False, f"Deep learning OFF: {torch_reason}", []
    
    # Check GPU (unless overridden)
    if config.require_gpu:
        has_gpu, gpu_reason = check_gpu()
        if not has_gpu:
            return False, f"Deep learning OFF: {gpu_reason}. Set require_gpu=False to use CPU.", []
    
    # Check data size
    n_dis = data_info.get('n_disrupted', 0)
    mean_tp = data_info.get('mean_timepoints_per_shot', 0)
    
    if mean_tp < config.min_timepoints_per_shot:
        return False, (f"Deep learning OFF: shots too short "
                      f"(mean={mean_tp:.0f}tp, need ≥{config.min_timepoints_per_shot})"), []
    
    # Determine which models activate
    active = []
    if n_dis >= config.min_disrupted_gru:
        active.append('GRU')
    if n_dis >= config.min_disrupted_cnn:
        active.append('TemporalCNN')
    if n_dis >= config.min_disrupted_transformer:
        active.append('Transformer')
    
    if not active:
        return False, (f"Deep learning OFF: need ≥{config.min_disrupted_gru} disrupted shots, "
                      f"have {n_dis}. GBT (AUC=0.979) is optimal for small datasets."), []
    
    return True, f"Deep learning ON: {active} ({n_dis} disrupted)", active


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (PyTorch — only imported when activated)
# ═══════════════════════════════════════════════════════════════

def _build_gru_model(n_features, config):
    """Build GRU model. Only called if PyTorch available."""
    import torch
    import torch.nn as nn
    
    class PlasmaGRU(nn.Module):
        def __init__(self, n_feat, hidden, n_layers, dropout):
            super().__init__()
            self.gru = nn.GRU(n_feat, hidden, n_layers, 
                             batch_first=True, dropout=dropout if n_layers > 1 else 0)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1)
            )
        
        def forward(self, x):
            # x: [batch, seq_len, features]
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)
    
    return PlasmaGRU(n_features, config.gru_hidden, config.gru_layers, config.gru_dropout)


def _build_temporal_cnn(n_features, config):
    """Build multi-scale temporal CNN."""
    import torch
    import torch.nn as nn
    
    class TemporalCNN(nn.Module):
        def __init__(self, n_feat, channels, kernel_sizes):
            super().__init__()
            # Multi-scale: parallel convolutions with different kernel sizes
            self.branches = nn.ModuleList()
            for ch, ks in zip(channels, kernel_sizes):
                self.branches.append(nn.Sequential(
                    nn.Conv1d(n_feat, ch, ks, padding=ks//2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.Conv1d(ch, ch, ks, padding=ks//2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)  # Global pool
                ))
            total_ch = sum(channels)
            self.head = nn.Sequential(
                nn.Linear(total_ch, total_ch // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(total_ch // 2, 1)
            )
        
        def forward(self, x):
            # x: [batch, seq_len, features] → transpose to [batch, features, seq_len]
            x = x.transpose(1, 2)
            branch_outs = [b(x).squeeze(-1) for b in self.branches]
            combined = torch.cat(branch_outs, dim=-1)
            return self.head(combined).squeeze(-1)
    
    return TemporalCNN(n_features, config.cnn_channels, config.cnn_kernel_sizes)


def _build_transformer(n_features, config):
    """Build lightweight Transformer encoder for disruption prediction."""
    import torch
    import torch.nn as nn
    
    class PlasmaTransformer(nn.Module):
        def __init__(self, n_feat, d_model, nhead, n_layers, seq_len):
            super().__init__()
            self.input_proj = nn.Linear(n_feat, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=0.1, activation='gelu', batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, 1)
            )
        
        def forward(self, x):
            # x: [batch, seq_len, features]
            h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
            h = self.encoder(h)
            # Use CLS-like: mean pool over sequence
            h = h.mean(dim=1)
            return self.head(h).squeeze(-1)
    
    return PlasmaTransformer(n_features, config.tfm_d_model, config.tfm_nhead,
                             config.tfm_layers, config.tfm_seq_len)


# ═══════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def _train_model(model, train_sequences, train_labels, config, model_name, device):
    """Generic training loop for any PyTorch model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    
    # Positive weight for class imbalance
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([min(n_neg / (n_pos + 1), 10.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    X_tensor = torch.from_numpy(train_sequences).float().to(device)
    y_tensor = torch.from_numpy(train_labels).float().to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    epochs = getattr(config, f'epochs_{model_name.lower()}', config.epochs_gru)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        
        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break
    
    return model


# ═══════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════

class DeepLearningTrack:
    """Deep learning track for disruption prediction.
    
    Activates only when hardware and data conditions are met.
    Otherwise returns None and the predictor uses GBT.
    
    Usage:
        dl = DeepLearningTrack()
        
        can_use, reason, models = dl.check_activation(data_info)
        if not can_use:
            print(f"DL skipped: {reason}")
            return  # Predictor uses GBT (AUC=0.979)
        
        dl.fit(data, shot_ids, labels, seq_len=30)
        features = dl.predict_shot(shot_data)  # For meta-learner
    """
    
    def __init__(self, config: DeepLearningConfig = None):
        self.config = config or DeepLearningConfig()
        self.models = {}
        self.fitted = False
        self.device = None
        self.active_models = []
        self.feature_mean = None
        self.feature_std = None
    
    def check_activation(self, data_info: Dict) -> Tuple[bool, str, List[str]]:
        return check_activation(data_info, self.config)
    
    def _get_device(self):
        """Get best available device."""
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def _build_sequences(self, data, shot_ids, labels, disrupted_set, seq_len):
        """Build training sequences from shot data.
        
        For each shot, extract rolling windows of seq_len.
        Label: 1 if from disrupted shot (last 30%), 0 otherwise.
        """
        unique_shots = np.unique(shot_ids)
        sequences = []
        seq_labels = []
        
        for sid in unique_shots:
            mask = shot_ids == sid
            shot_data = data[mask]
            n = len(shot_data)
            if n < seq_len + 3:
                continue
            
            is_dis = int(sid) in disrupted_set
            
            # Normalize per-shot
            shot_norm = (shot_data - self.feature_mean) / (self.feature_std + 1e-10)
            shot_norm = np.clip(shot_norm, -10, 10)
            
            # Extract windows
            step = max(seq_len // 4, 1)
            for start in range(0, n - seq_len, step):
                seq = shot_norm[start:start + seq_len]
                
                # Label: disrupted shot, in last 30%?
                if is_dis and start > 0.6 * n:
                    seq_labels.append(1.0)
                else:
                    seq_labels.append(0.0)
                sequences.append(seq)
        
        return np.array(sequences, dtype=np.float32), np.array(seq_labels, dtype=np.float32)
    
    def fit(self, data: np.ndarray, shot_ids: np.ndarray,
            disrupted_set: set, n_features: int = None):
        """Train deep learning models on plasma time series.
        
        Args:
            data: [n_timepoints, n_features]
            shot_ids: [n_timepoints]
            disrupted_set: set of disrupted shot IDs
        """
        import torch
        
        self.device = self._get_device()
        n_features = n_features or data.shape[1]
        
        # Compute normalization stats
        self.feature_mean = np.nanmean(data, axis=0)
        self.feature_std = np.nanstd(data, axis=0) + 1e-10
        
        # Count disrupted
        unique_shots = np.unique(shot_ids)
        n_dis = sum(1 for s in unique_shots if s in disrupted_set)
        mean_tp = len(data) / len(unique_shots)
        
        print(f"  DL Track: {n_dis} disrupted, {len(unique_shots)} total, "
              f"mean {mean_tp:.0f} tp/shot, device={self.device}")
        
        # Build sequences for each active model
        for model_name in self.active_models:
            if model_name == 'GRU':
                sl = self.config.gru_seq_len
            elif model_name == 'TemporalCNN':
                sl = self.config.gru_seq_len
            else:  # Transformer
                sl = self.config.tfm_seq_len
            
            sequences, labels = self._build_sequences(
                data, shot_ids, None, disrupted_set, sl)
            
            if len(sequences) < 100:
                print(f"    {model_name}: too few sequences ({len(sequences)}), skipping")
                continue
            
            n_pos = labels.sum()
            print(f"    {model_name}: {len(sequences)} sequences "
                  f"({int(n_pos)} pos, {int(len(labels)-n_pos)} neg), sl={sl}")
            
            # Build model
            if model_name == 'GRU':
                model = _build_gru_model(n_features, self.config)
            elif model_name == 'TemporalCNN':
                model = _build_temporal_cnn(n_features, self.config)
            else:
                model = _build_transformer(n_features, self.config)
            
            # Train
            t0 = time.time()
            model = _train_model(model, sequences, labels, self.config, 
                               model_name, self.device)
            elapsed = time.time() - t0
            print(f"    {model_name}: trained in {elapsed:.1f}s")
            
            self.models[model_name] = model
        
        self.fitted = len(self.models) > 0
        if self.fitted:
            print(f"  DL Track fitted: {list(self.models.keys())}")
    
    def predict_shot(self, shot_data: np.ndarray) -> Optional[Dict[str, float]]:
        """Predict disruption for a single shot using DL models.
        
        Returns dict of {model_name: probability} or None if not fitted.
        """
        if not self.fitted:
            return None
        
        import torch
        
        # Normalize
        shot_norm = (shot_data - self.feature_mean) / (self.feature_std + 1e-10)
        shot_norm = np.clip(shot_norm, -10, 10).astype(np.float32)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            model.eval()
            
            # Determine sequence length
            if model_name == 'Transformer':
                sl = self.config.tfm_seq_len
            else:
                sl = self.config.gru_seq_len
            
            n = len(shot_norm)
            if n < sl:
                continue
            
            # Extract sequences covering the shot
            probs = []
            with torch.no_grad():
                for start in range(0, n - sl, max(sl // 4, 1)):
                    seq = torch.from_numpy(shot_norm[start:start+sl]).unsqueeze(0).to(self.device)
                    logit = model(seq)
                    prob = torch.sigmoid(logit).item()
                    probs.append(prob)
            
            if probs:
                # Shot-level: max probability (any window triggers alarm)
                predictions[model_name] = {
                    'max_prob': float(max(probs)),
                    'mean_prob': float(np.mean(probs)),
                    'late_prob': float(np.mean(probs[-max(len(probs)//3, 1):])),
                    'n_windows': len(probs),
                }
        
        return predictions if predictions else None
    
    def build_features(self, shot_data: np.ndarray) -> Optional[np.ndarray]:
        """Build feature vector for meta-learner integration.
        
        Returns [n_features] array or None.
        """
        preds = self.predict_shot(shot_data)
        if preds is None:
            return None
        
        feats = []
        for model_name in ['GRU', 'TemporalCNN', 'Transformer']:
            if model_name in preds:
                p = preds[model_name]
                feats.extend([p['max_prob'], p['mean_prob'], p['late_prob']])
            else:
                feats.extend([0, 0, 0])
        
        return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """Test activation conditions."""
    print("Deep Learning Track — Activation Check")
    print("=" * 50)
    
    # Current MAST data
    info_mast = {
        'n_disrupted': 83,
        'n_shots': 2941,
        'mean_timepoints_per_shot': 91,
        'n_variables': 16,
    }
    
    # Future: more disrupted shots from ops-log
    info_future = {
        'n_disrupted': 448,
        'n_shots': 5000,
        'mean_timepoints_per_shot': 91,
        'n_variables': 16,
    }
    
    # Large dataset (DIII-D scale)
    info_large = {
        'n_disrupted': 2000,
        'n_shots': 20000,
        'mean_timepoints_per_shot': 200,
        'n_variables': 14,
    }
    
    # Test without GPU requirement for demonstration
    config_no_gpu = DeepLearningConfig(require_gpu=False)
    
    for name, info in [("MAST current (83d)", info_mast),
                        ("MAST +ops-log (448d)", info_future),
                        ("DIII-D scale (2000d)", info_large)]:
        ok, reason, models = check_activation(info, config_no_gpu)
        status = f"✓ ON: {models}" if ok else "✗ OFF"
        print(f"\n  {name}:")
        print(f"    {status}")
        print(f"    {reason}")
    
    # Also check hardware
    print(f"\n  Hardware:")
    ok, r = check_pytorch(); print(f"    PyTorch: {'✓' if ok else '✗'} {r}")
    ok, r = check_gpu(); print(f"    GPU: {'✓' if ok else '✗'} {r}")
