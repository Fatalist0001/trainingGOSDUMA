"""Linear models: LinearRegression, Ridge, ElasticNet with CV."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
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
    """Ridge regression with cross-validation for alpha selection."""

    def __init__(
        self,
        alphas: list[float] | None = None,
        fit_intercept: bool = True,
        cv: int = 5,
        scoring: str = "neg_mean_absolute_error",
    ):
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.scoring = scoring
        self.model_ = None
        self.target_columns_ = None
        self.best_alpha_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()

        if y.shape[1] == 1:
            # Single target - use RidgeCV
            self.model_ = RidgeCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                cv=self.cv,
                scoring=self.scoring,
            )
            self.model_.fit(X, y.iloc[:, 0])
            self.best_alpha_ = self.model_.alpha_
        else:
            # Multi-target - use RidgeCV for each target
            models = {}
            alphas = {}
            for col in self.target_columns_:
                model = RidgeCV(
                    alphas=self.alphas,
                    fit_intercept=self.fit_intercept,
                    cv=self.cv,
                    scoring=self.scoring,
                )
                model.fit(X, y[col])
                models[col] = model
                alphas[col] = model.alpha_
            self.model_ = models
            self.best_alpha_ = alphas

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")

        if isinstance(self.model_, dict):
            # Multi-target
            preds = np.column_stack([self.model_[col].predict(X) for col in self.target_columns_])
        else:
            # Single target
            preds = self.model_.predict(X).reshape(-1, 1)

        return preds

    def get_best_alphas(self) -> dict[str, float]:
        """Get best alpha per target."""
        check_is_fitted(self, "model_")
        return (
            self.best_alpha_
            if isinstance(self.best_alpha_, dict)
            else {self.target_columns_[0]: self.best_alpha_}
        )


class ElasticNetModel(BaseEstimator, RegressorMixin):
    """ElasticNet with cross-validation for alpha and l1_ratio selection."""

    def __init__(
        self,
        alphas: list[float] | None = None,
        l1_ratio: list[float] | None = None,
        fit_intercept: bool = True,
        cv: int = 5,
        max_iter: int = 5000,
        scoring: str = "neg_mean_absolute_error",
        selection: Literal["cyclic", "random"] = "cyclic",
        random_state: int = 42,
    ):
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.l1_ratio = l1_ratio or [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.max_iter = max_iter
        self.scoring = scoring
        self.selection = selection
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None
        self.best_params_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()

        if y.shape[1] == 1:
            # Single target - use ElasticNetCV
            self.model_ = ElasticNetCV(
                alphas=self.alphas,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                cv=self.cv,
                max_iter=self.max_iter,
                selection=self.selection,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model_.fit(X, y.iloc[:, 0])
            self.best_params_ = {"alpha": self.model_.alpha_, "l1_ratio": self.model_.l1_ratio_}
        else:
            # Multi-target - fit separately
            models = {}
            params = {}
            for col in self.target_columns_:
                model = ElasticNetCV(
                    alphas=self.alphas,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=self.fit_intercept,
                    cv=self.cv,
                    max_iter=self.max_iter,
                    selection=self.selection,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                model.fit(X, y[col])
                models[col] = model
                params[col] = {"alpha": model.alpha_, "l1_ratio": model.l1_ratio_}
            self.model_ = models
            self.best_params_ = params

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")

        if isinstance(self.model_, dict):
            preds = np.column_stack([self.model_[col].predict(X) for col in self.target_columns_])
        else:
            preds = self.model_.predict(X).reshape(-1, 1)

        return preds

    def get_best_params(self) -> dict[str, dict]:
        """Get best alpha and l1_ratio per target."""
        check_is_fitted(self, "model_")
        return self.best_params_

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
