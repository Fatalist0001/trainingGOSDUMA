"""Ensemble models: WeightedEnsemble and StackingEnsemble.

Both wrap a set of *feature-based* base models (linear / trees / KNN / MLP) and
follow the same flat interface as the other models:
    fit(X: DataFrame[features], y: DataFrame[targets])
    predict(X: DataFrame[features]) -> ndarray (n_regions, n_targets)

Baselines (NaivePreviousElection, etc.) are intentionally excluded: they ignore
features and require region/year identifier columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..utils.reproducibility import set_seed

DEFAULT_BASES = [
    "XGBoost",
    "CatBoost",
    "RandomForest",
    "HistGradientBoosting",
    "MLPSklearn",
    "LinearRegression",
]


def _to_array(pred: np.ndarray | pd.DataFrame, n_targets: int) -> np.ndarray:
    """Coerce a model prediction to a 2D ndarray of shape (n, n_targets)."""
    arr = pred.values if isinstance(pred, pd.DataFrame) else np.asarray(pred)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[:, :n_targets]


class WeightedEnsemble:
    """Average of base model predictions, weighted by inverse OOF MAE."""

    def __init__(
        self,
        base_models: list[str] | None = None,
        weights: list[float] | None = None,
        weight_strategy: str = "inverse_mae",
        random_state: int = 42,
        n_splits: int = 3,
    ):
        self.base_models = base_models or list(DEFAULT_BASES)
        self.weights = weights
        self.weight_strategy = weight_strategy
        self.random_state = random_state
        self.n_splits = n_splits
        self.fitted_models_: list = []
        self.weights_: np.ndarray | None = None
        self.target_columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        from .registry import get_model

        set_seed(self.random_state)
        self.target_columns_ = list(y.columns)
        t = y.shape[1]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.fitted_models_ = []
        n, b = len(X), len(self.base_models)
        oof = np.zeros((n, t, b))
        for j, name in enumerate(self.base_models):
            # Final model on all data.
            final = get_model(name)
            final.fit(X, y)
            self.fitted_models_.append(final)

            # Out-of-fold predictions to estimate this base's per-party MAE.
            for tr, te in kf.split(X):
                mi = get_model(name)
                mi.fit(X.iloc[tr], y.iloc[tr])
                oof[te, :, j] = _to_array(mi.predict(X.iloc[te]), t)

        # Per-party inverse-MAE weights (each target weighted independently).
        yv = y.values
        self.weights_ = np.zeros((t, b))
        for j in range(t):
            maes = np.array(
                [np.mean(np.abs(oof[:, j, bi] - yv[:, j])) for bi in range(b)]
            )
            maes = np.where(maes > 0, maes, 1e-9)
            w = 1.0 / maes
            self.weights_[j] = w / w.sum()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_models_:
            raise RuntimeError("WeightedEnsemble must be fitted before predict.")
        t = len(self.target_columns_)
        preds = np.stack(
            [_to_array(m.predict(X), t) for m in self.fitted_models_], axis=-1
        )  # (n, T, B)
        out = np.zeros((len(X), t))
        for j in range(t):
            out[:, j] = (preds[:, j, :] * self.weights_[j]).sum(axis=1)
        return out


class StackingEnsemble:
    """Stacking: meta-model (default LinearRegression) on base OOF predictions."""

    def __init__(
        self,
        base_models: list[str] | None = None,
        meta_model: str = "LinearRegression",
        random_state: int = 42,
        n_splits: int = 3,
    ):
        self.base_models = base_models or list(DEFAULT_BASES)
        self.meta_model = meta_model
        self.random_state = random_state
        self.n_splits = n_splits
        self.fitted_models_: list = []
        self.meta_: Any | None = None
        self.target_columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        from .registry import get_model

        set_seed(self.random_state)
        self.target_columns_ = list(y.columns)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        n, t, b = len(X), y.shape[1], len(self.base_models)
        oof = np.zeros((n, t, b))
        for j, name in enumerate(self.base_models):
            for tr, te in kf.split(X):
                mi = get_model(name)
                mi.fit(X.iloc[tr], y.iloc[tr])
                oof[te, :, j] = _to_array(mi.predict(X.iloc[te]), t)

        meta_X = oof.reshape(n, -1)
        self.meta_ = get_model(self.meta_model)
        self.meta_.fit(meta_X, y)

        # Refit base models on full data for inference.
        self.fitted_models_ = []
        for name in self.base_models:
            m = get_model(name)
            m.fit(X, y)
            self.fitted_models_.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_models_ or self.meta_ is None:
            raise RuntimeError("StackingEnsemble must be fitted before predict.")
        t = len(self.target_columns_)
        base_preds = np.stack(
            [_to_array(m.predict(X), t) for m in self.fitted_models_], axis=-1
        )  # (n, T, B)
        meta_X = base_preds.reshape(len(X), -1)
        return _to_array(self.meta_.predict(meta_X), t)
