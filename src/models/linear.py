"""Linear models: LinearRegression, Ridge, ElasticNet.

Ridge and ElasticNet hyperparameters (alpha / l1_ratio) are selected externally
via temporal-validated tuning (``src/evaluation/tuning.py``) on a strictly past
validation year. sklearn's built-in RidgeCV/ElasticNetCV are intentionally NOT
used: their internal KFold shuffles region-year rows randomly, which violates
the "past -> future" temporal-split principle.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class LinearModel(BaseEstimator, RegressorMixin):
    """Wrapper for sklearn LinearRegression with multi-output support."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()
        self.model_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients as DataFrame."""
        check_is_fitted(self, "model_")
        if self.model_.coef_.ndim == 1:
            coef = self.model_.coef_.reshape(1, -1)
        else:
            coef = self.model_.coef_

        return pd.DataFrame(
            coef, columns=self.target_columns_, index=[f"feature_{i}" for i in range(coef.shape[1])]
        ).T


class RidgeModel(BaseEstimator, RegressorMixin):
    """Ridge regression with a fixed (externally tuned) alpha."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        if isinstance(self.model_, dict):
            preds = np.column_stack([self.model_[col].predict(X) for col in self.target_columns_])
        else:
            preds = self.model_.predict(X)

        return preds


class ElasticNetModel(BaseEstimator, RegressorMixin):
    """ElasticNet with fixed (externally tuned) alpha and l1_ratio."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 5000,
        selection: Literal["cyclic", "random"] = "cyclic",
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.selection = selection
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()
        self.model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            selection=self.selection,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        if isinstance(self.model_, dict):
            preds = np.column_stack([self.model_[col].predict(X) for col in self.target_columns_])
        else:
            preds = self.model_.predict(X)

        return preds

    def get_selected_features(self) -> dict[str, list[str]]:
        """Get non-zero coefficient features per target."""
        check_is_fitted(self, "model_")

        selected = {}
        for col in self.target_columns_:
            model = self.model_[col] if isinstance(self.model_, dict) else self.model_
            coef = model.coef_
            selected[col] = [f"feature_{i}" for i, c in enumerate(coef) if abs(c) > 1e-10]
        return selected


def create_linear_pipeline(
    model_type: Literal["linear", "ridge", "elasticnet"],
    **kwargs,
) -> Pipeline:
    """
    Create a pipeline with StandardScaler + Linear Model.

    Args:
        model_type: "linear", "ridge", or "elasticnet"
        **kwargs: Model-specific parameters

    Returns:
        sklearn Pipeline
    """
    models = {
        "linear": LinearModel,
        "ridge": RidgeModel,
        "elasticnet": ElasticNetModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", models[model_type](**kwargs)),
        ]
    )
