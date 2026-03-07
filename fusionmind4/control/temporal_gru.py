"""
Temporal GRU Disruption Predictor
==================================

The key insight: single-timepoint models can't distinguish
"stressed but stable" from "pre-disruption" because both look the same
at any given moment. The difference is in the TRAJECTORY:

  Clean:     spike → recovery (signal drops back to baseline)
  Disrupted: spike → sustained → crash (signal stays elevated or worsens)

A GRU learns this temporal pattern from rolling windows of 20-40 timepoints
(200-400ms of plasma history), enabling it to suppress false alarms from
transient spikes while catching genuine precursors.

Architecture:
  Input: [seq_len, n_features] — rolling window of plasma state
  GRU: 2 layers, hidden=64 — learns temporal dynamics
  Output: disruption probability at current timepoint

Training: Per-shot sequences with anomaly-onset labels.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


class PlasmaSequenceDataset(Dataset):
    """Rolling window sequences from plasma shots."""
    
    def __init__(self, data, labels, shot_ids, disruption_set, 
                 seq_len=30, sigma_threshold=2.5, var_indices=None):
        """
        Args:
            data: (N, n_vars) array of plasma measurements
            labels: (N,) shot ID per timepoint
            shot_ids: which shots to include
            disruption_set: set of disrupted shot IDs
            seq_len: lookback window in timepoints
            sigma_threshold: for anomaly-onset label generation
        """
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        
        for sid in shot_ids:
            mask = labels == sid
            X = data[mask]
            if len(X) < seq_len + 5:
                continue
            
            # Compute rates
            rates = np.abs(np.diff(X, axis=0, prepend=X[:1]))
            features = np.concatenate([X, rates], axis=1)
            
            # Generate anomaly-onset label for this shot
            is_dis = sid in disruption_set
            y = np.zeros(len(X))
            
            if is_dis:
                n40 = max(int(0.4 * len(X)), 5)
                earliest = len(X)
                vi_check = var_indices or list(range(X.shape[1]))
                for vi in vi_check:
                    sig = rates[:, vi] if vi < X.shape[1] else X[:, vi]
                    mu = np.mean(sig[:n40])
                    sd = np.std(sig[:n40]) + 1e-10
                    for i in range(n40, len(sig)):
                        if sig[i] > mu + sigma_threshold * sd and i < earliest:
                            earliest = i
                            break
                if earliest < len(X):
                    y[earliest:] = 1
                else:
                    n30 = max(int(0.3 * len(X)), 2)
                    y[-n30:] = 1
            
            # Create rolling window sequences
            for i in range(seq_len, len(X)):
                seq = features[i - seq_len:i]  # (seq_len, n_features*2)
                self.sequences.append(seq.astype(np.float32))
                self.targets.append(np.float32(y[i]))
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        # Balance classes (oversample minority)
        pos = np.where(self.targets == 1)[0]
        neg = np.where(self.targets == 0)[0]
        if len(pos) > 0 and len(neg) > len(pos) * 3:
            # Oversample positive to 1:3 ratio
            n_oversample = min(len(neg) // 3, len(pos) * 5)
            extra_pos = np.random.choice(pos, n_oversample - len(pos), replace=True)
            keep = np.concatenate([neg, pos, extra_pos])
            np.random.shuffle(keep)
            self.sequences = self.sequences[keep]
            self.targets = self.targets[keep]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])


class GRUDisruptionPredictor(nn.Module):
    """2-layer GRU with attention for disruption prediction."""
    
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        # Simple attention over sequence
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden)
        
        # Attention: weight recent timesteps more
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)  # (batch, seq_len, 1)
        context = (gru_out * attn_weights).sum(dim=1)  # (batch, hidden)
        
        logit = self.classifier(context).squeeze(-1)  # (batch,)
        return logit


def train_gru(train_dataset, val_dataset=None, epochs=30, lr=1e-3, batch_size=64):
    """Train GRU with early stopping."""
    device = torch.device('cpu')
    
    n_features = train_dataset.sequences.shape[2]
    model = GRUDisruptionPredictor(input_dim=n_features, hidden_dim=64, n_layers=2)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Weighted BCE for imbalanced classes
    pos_weight = torch.tensor([3.0])  # Weight positive class more
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    best_val_auc = 0
    best_state = None
    patience = 8
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        losses = []
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        
        # Validation
        if val_dataset is not None and len(val_dataset) > 0:
            model.eval()
            with torch.no_grad():
                X_val = torch.from_numpy(val_dataset.sequences).to(device)
                y_val = val_dataset.targets
                logits_val = model(X_val)
                probs_val = torch.sigmoid(logits_val).cpu().numpy()
                if len(np.unique(y_val)) > 1:
                    val_auc = roc_auc_score(y_val, probs_val)
                else:
                    val_auc = 0.5
            
            scheduler.step(-val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def predict_shot(model, shot_data, seq_len=30):
    """Run GRU on a single shot, return per-timepoint probabilities."""
    model.eval()
    rates = np.abs(np.diff(shot_data, axis=0, prepend=shot_data[:1]))
    features = np.concatenate([shot_data, rates], axis=1).astype(np.float32)
    
    probs = np.zeros(len(shot_data))
    
    if len(features) <= seq_len:
        return probs
    
    # Build sequences
    seqs = []
    for i in range(seq_len, len(features)):
        seqs.append(features[i - seq_len:i])
    
    seqs = np.array(seqs)
    
    with torch.no_grad():
        X = torch.from_numpy(seqs)
        logits = model(X)
        p = torch.sigmoid(logits).numpy()
    
    probs[seq_len:] = p
    return probs
