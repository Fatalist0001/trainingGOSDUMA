"""Neural models: sklearn MLP and a PyTorch MLP with multi-output support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _set_seed(seed: int) -> None:
    """Set random seeds for numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPSklearn(BaseEstimator, RegressorMixin):
    """Wrapper for sklearn MLPRegressor with multi-output support.

    A feed-forward neural network trained with backpropagation. Features must be
    scaled before fitting (handled by backtest preprocessing). Targets are passed
    as-is (multi-output); MLPRegressor handles several outputs natively.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
        activation: str = "relu",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = False,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        **kwargs: Any,
    ):
        # NOTE: sklearn MLP cannot consume an external validation set, so
        # ``early_stopping`` is disabled by default: its internal split shuffles
        # region-year rows randomly, which violates the temporal-split principle.
        # Early stopping is delegated to external temporal-validated tuning.
        # Coerce hidden_layer_sizes: yaml grids may pass a list of lists.
        if (
            isinstance(hidden_layer_sizes, list)
            and hidden_layer_sizes
            and isinstance(hidden_layer_sizes[0], (list, tuple))
        ):
            hidden_layer_sizes = tuple(hidden_layer_sizes[0])
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class _TorchMLP(nn.Module):
    """Simple feed-forward MLP used by MLPTorch."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...], dropout: float
    ):
        super().__init__()
        dims = [input_dim, *hidden_sizes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTorch(BaseEstimator, RegressorMixin):
    """PyTorch feed-forward MLP with multi-output support and early stopping.

    Targets are standardized internally (scaler fit on train only, no leakage) and
    inverse-transformed at prediction time. Training uses Adam with early stopping
    on a validation split. Seeds are controlled via `random_state` for stability.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 200,
        patience: int = 20,
        random_state: int = 42,
        **kwargs: Any,
    ):
        if (
            isinstance(hidden_sizes, list)
            and hidden_sizes
            and isinstance(hidden_sizes[0], (list, tuple))
        ):
            hidden_sizes = tuple(hidden_sizes[0])
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None
        self.y_scaler_ = None
        self.device_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        _set_seed(self.random_state)
        self.target_columns_ = y.columns.tolist()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        # Standardize targets (fit on train only).
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y_np)

        input_dim = X_np.shape[1]
        output_dim = y_scaled.shape[1]
        net = _TorchMLP(input_dim, output_dim, self.hidden_sizes, self.dropout).to(device)

        X_t = torch.from_numpy(X_np).to(device)
        y_t = torch.from_numpy(y_scaled).to(device)

        # Early stopping set: caller-provided validation year (temporal split,
        # no random shuffle). Without one (final forecast) track train loss only.
        use_val = False
        if X_val is not None and y_val is not None and self.patience:
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = self.y_scaler_.transform(
                y_val.values.astype(np.float32)
            )
            X_val_t = torch.from_numpy(X_val_np).to(device)
            y_val_t = torch.from_numpy(y_val_np.astype(np.float32)).to(device)
            use_val = True

        train_ds = torch.utils.data.TensorDataset(X_t, y_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True
        )

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for _ in range(self.epochs):
            net.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                if use_val:
                    val_loss = loss_fn(net(X_val_t), y_val_t).item()
                else:
                    val_loss = loss_fn(net(X_t), y_t).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.patience and use_val and epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

        self.model_ = net
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        self.model_.eval()
        X_np = X.values.astype(np.float32)
        X_t = torch.from_numpy(X_np).to(self.device_)
        with torch.no_grad():
            y_scaled_pred = self.model_(X_t).cpu().numpy()
        y_pred = self.y_scaler_.inverse_transform(y_scaled_pred)
        return y_pred
