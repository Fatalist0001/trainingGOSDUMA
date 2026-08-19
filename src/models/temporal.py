"""Temporal models: GRU, LSTM, Transformer for per-region election sequences.

Each training/test sample is a sequence of past elections for one region:
  X = [features(year_1), features(year_2), ..., features(year_{k})]  (k time steps)
  y = targets at the next election (year_{k+1}).
The models are many-to-one: read the whole sequence and predict the next
election's party shares. Features are scaled internally (fit on train only);
targets are standardized internally and inverse-transformed at prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_sequences(X_list, device):
    """Pad a list of (k_i, F) arrays to (B, Lmax, F) and return lengths."""
    lengths = [int(x.shape[0]) for x in X_list]
    max_len = max(lengths)
    feat_dim = X_list[0].shape[1]
    padded = np.zeros((len(X_list), max_len, feat_dim), dtype=np.float32)
    for i, x in enumerate(X_list):
        padded[i, : x.shape[0]] = x
    return torch.from_numpy(padded).to(device), torch.tensor(lengths, dtype=torch.long).to(device)


class _RNNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, dropout, cell):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(
            input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x_padded, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, h_n = self.rnn(packed)
        # GRU: h_n is a tensor (num_layers, B, H). LSTM: h_n is (h_n, c_n) tuple.
        if isinstance(h_n, tuple):
            h_n = h_n[0]
        # h_n shape (num_layers, B, H); take last layer's hidden state.
        h_last = h_n[-1]
        return self.head(h_last)


class _TransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, num_layers, dim_feedforward, dropout, max_len=16):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x_padded, lengths):
        B, L, _ = x_padded.shape
        x = self.proj(x_padded) + self.pos[:L].unsqueeze(0)
        # Padding mask: True for padded positions (ignored by attention).
        mask = torch.arange(L, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        enc = self.encoder(x, src_key_padding_mask=mask)
        last_idx = (lengths - 1).unsqueeze(1)  # (B,1)
        gathered = torch.gather(enc, 1, last_idx.unsqueeze(2).expand(-1, -1, enc.size(2))).squeeze(1)
        return self.head(gathered)


class _TemporalModel(BaseEstimator, RegressorMixin):
    """Shared base for GRU/LSTM/Transformer temporal models."""

    _cell = "gru"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 20,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None
        self.x_scaler_ = None
        self.y_scaler_ = None
        self.x_imputer_ = None
        self.y_imputer_ = None
        self.device_ = None

    def fit(self, X: list[np.ndarray], y: np.ndarray):
        _set_seed(self.random_state)
        y = np.asarray(y, dtype=np.float32)
        self.target_columns_ = list(range(y.shape[1]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        # Impute (median, fit on train only -> no leakage) then scale.
        self.x_imputer_ = SimpleImputer(strategy="median")
        all_x = np.concatenate([x.astype(np.float32) for x in X], axis=0)
        self.x_imputer_.fit(all_x)
        X_imp = [self.x_imputer_.transform(x.astype(np.float32)) for x in X]
        all_x_imp = np.concatenate(X_imp, axis=0)
        self.x_scaler_ = StandardScaler().fit(all_x_imp)
        self.y_imputer_ = SimpleImputer(strategy="median")
        self.y_imputer_.fit(y)
        y_imp = self.y_imputer_.transform(y)
        self.y_scaler_ = StandardScaler().fit(y_imp)
        X_scaled = [self.x_scaler_.transform(x) for x in X_imp]
        y_scaled = self.y_scaler_.transform(y_imp)

        # all_x_imp may have fewer columns than all_x if SimpleImputer dropped
        # all-NaN features; derive dims from the imputed data.
        input_dim = all_x_imp.shape[1]
        output_dim = y_imp.shape[1]
        if self._cell == "transformer":
            net = _TransformerNet(
                input_dim, output_dim, self.hidden_size, 4, self.num_layers, 128, self.dropout
            )
        else:
            net = _RNNNet(
                input_dim, output_dim, self.hidden_size, self.num_layers, self.dropout, self._cell
            )
        net.to(device)

        X_pad, lengths = _pad_sequences(X_scaled, device)
        y_t = torch.from_numpy(y_scaled).to(device)

        # Internal train/val split for early stopping (no leakage: each sample
        # is an independent region-year target).
        n = len(X)
        if self.patience and n >= 10:
            idx = np.arange(n)
            np.random.shuffle(idx)
            n_val = max(1, int(0.15 * n))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
        else:
            train_idx = np.arange(n)
            val_idx = train_idx

        X_train_t, y_train_t, L_train = X_pad[train_idx], y_t[train_idx], lengths[train_idx]
        X_val, y_val, L_val = X_pad[val_idx], y_t[val_idx], lengths[val_idx]

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for _ in range(self.epochs):
            net.train()
            optimizer.zero_grad()
            pred = net(X_train_t, L_train)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            net.eval()
            with torch.no_grad():
                val_pred = net(X_val, L_val)
                val_loss = loss_fn(val_pred, y_val).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.patience and epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model_ = net
        return self

    def predict(self, X: list[np.ndarray]) -> np.ndarray:
        check_is_fitted(self, "model_")
        self.model_.eval()
        X_imp = [self.x_imputer_.transform(x.astype(np.float32)) for x in X]
        X_scaled = [self.x_scaler_.transform(x) for x in X_imp]
        X_pad, lengths = _pad_sequences(X_scaled, self.device_)
        with torch.no_grad():
            y_scaled_pred = self.model_(X_pad, lengths).cpu().numpy()
        return self.y_scaler_.inverse_transform(y_scaled_pred)


class GRUModel(_TemporalModel):
    """GRU over per-region election sequences (many-to-one)."""

    _cell = "gru"


class LSTMModel(_TemporalModel):
    """LSTM over per-region election sequences (many-to-one)."""

    _cell = "lstm"


class TransformerModel(_TemporalModel):
    """Transformer encoder over per-region election sequences (many-to-one)."""

    _cell = "transformer"
