"""Tree-based models: RandomForest, HistGradientBoosting, XGBoost, CatBoost."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.validation import check_is_fitted

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class RandomForestModel(BaseEstimator, RegressorMixin):
    """Random Forest with multi-output support."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Literal["sqrt", "log2", None] = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()

        base_rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        if y.shape[1] > 1:
            self.model_ = MultiOutputRegressor(base_rf, n_jobs=self.n_jobs)
        else:
            self.model_ = base_rf

        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def feature_importances(self) -> pd.DataFrame:
        """Get feature importances."""
        check_is_fitted(self, "model_")

        if hasattr(self.model_, "estimators_"):
            # MultiOutputRegressor
            importances = np.mean(
                [est.feature_importances_ for est in self.model_.estimators_], axis=0
            )
        else:
            importances = self.model_.feature_importances_

        feature_names = getattr(self.model_, "feature_names_in_", None)
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(len(importances))]

        return pd.DataFrame(
            {
                "feature": list(feature_names),
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)


class HistGBModel(BaseEstimator, RegressorMixin):
    """HistGradientBoosting with multi-output support."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()

        base_hgb = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

        if y.shape[1] > 1:
            self.model_ = MultiOutputRegressor(base_hgb, n_jobs=-1)
        else:
            self.model_ = base_hgb

        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class XGBoostModel(BaseEstimator, RegressorMixin):
    """XGBoost Regressor with multi-output support."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "reg:squarederror",
        tree_method: str = "hist",
        early_stopping_rounds: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Run: uv add xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.tree_method = tree_method
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None
        self.target_columns_ = None

    def _create_model(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            tree_method=self.tree_method,
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        eval_set: list[tuple] | None = None,
        verbose: bool = False,
    ):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()

        if y.shape[1] > 1:
            # Multi-target: fit separate models
            self.model_ = {}
            for col in self.target_columns_:
                model = self._create_model()
                if eval_set:
                    eval_set_col = [(X_val, y_val[col]) for X_val, y_val in eval_set]
                    model.fit(X, y[col], eval_set=eval_set_col, verbose=verbose)
                else:
                    model.fit(X, y[col], verbose=verbose)
                self.model_[col] = model
        else:
            model = self._create_model()
            if eval_set:
                eval_set_col = [(X_val, y_val.iloc[:, 0]) for X_val, y_val in eval_set]
                model.fit(X, y.iloc[:, 0], eval_set=eval_set_col, verbose=verbose)
            else:
                model.fit(X, y.iloc[:, 0], verbose=verbose)
            self.model_ = model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")

        if isinstance(self.model_, dict):
            preds = np.column_stack([self.model_[col].predict(X) for col in self.target_columns_])
        else:
            preds = self.model_.predict(X).reshape(-1, 1)

        return preds

    def feature_importances(self, importance_type: str = "gain") -> pd.DataFrame:
        """Get feature importances."""
        check_is_fitted(self, "model_")

        if isinstance(self.model_, dict):
            importances = {}
            for col, model in self.model_.items():
                imp = model.get_score(importance_type=importance_type)
                importances[col] = imp
            return pd.DataFrame(importances).fillna(0)
        else:
            imp = self.model_.get_score(importance_type=importance_type)
            return pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})


class CatBoostModel(BaseEstimator, RegressorMixin):
    """CatBoost Regressor with multi-output support."""

    def __init__(
        self,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "RMSE",
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        random_seed: int = 42,
        cat_features: list[str] | None = None,
        **kwargs,
    ):
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost not installed. Run: uv add catboost")

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_seed = random_seed
        self.cat_features = cat_features
        self.kwargs = kwargs
        self.model_ = None
        self.target_columns_ = None

    def _create_model(self) -> CatBoostRegressor:
        return CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            random_seed=self.random_seed,
            **self.kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        eval_set: list[tuple] | None = None,
        cat_features: list[str] | None = None,
        verbose: bool = False,
    ):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()
        self.verbose = verbose
        cat_features = cat_features or self.cat_features or []

        if y.shape[1] > 1:
            self.model_ = {}
            for col in self.target_columns_:
                model = self._create_model()
                if eval_set:
                    eval_set_col = [(X_val, y_val[col]) for X_val, y_val in eval_set]
                    model.fit(
                        X,
                        y[col],
                        eval_set=eval_set_col,
                        cat_features=cat_features,
                        verbose=self.verbose,
                    )
                else:
                    model.fit(X, y[col], cat_features=cat_features, verbose=self.verbose)
                self.model_[col] = model
        else:
            model = self._create_model()
            if eval_set:
                eval_set_col = [(X_val, y_val.iloc[:, 0]) for X_val, y_val in eval_set]
                model.fit(
                    X,
                    y.iloc[:, 0],
                    eval_set=eval_set_col,
                    cat_features=cat_features,
                    verbose=self.verbose,
                )
            else:
                model.fit(X, y.iloc[:, 0], cat_features=cat_features, verbose=self.verbose)
            self.model_ = model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")

        if isinstance(self.model_, dict):
            preds = np.column_stack([self.model_[col].predict(X) for col in self.target_columns_])
        else:
            preds = self.model_.predict(X).reshape(-1, 1)

        return preds

    def feature_importances(self) -> pd.DataFrame:
        """Get feature importances."""
        check_is_fitted(self, "model_")

        if isinstance(self.model_, dict):
            importances = {}
            for col, model in self.model_.items():
                imp = model.get_feature_importance()
                importances[col] = imp
            return pd.DataFrame(importances, index=model.feature_names_)
        else:
            imp = self.model_.get_feature_importance()
            return pd.DataFrame({"feature": self.model_.feature_names_, "importance": imp})
