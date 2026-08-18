"""Preprocessing utilities: scaling, encoding, feature engineering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    scaler_type: Literal["standard", "robust", "minmax"] = "standard"
    handle_missing: Literal["drop", "mean", "median", "zero"] = "mean"
    clip_outliers: bool = False
    clip_quantile: float = 0.99


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from DataFrame."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)


class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper around sklearn scalers that preserves DataFrame structure.

    Important: fit() only on training data, then transform() on both train and test.
    """

    def __init__(
        self,
        scaler_type: Literal["standard", "robust", "minmax"] = "standard",
        columns: list[str] | None = None,
    ):
        self.scaler_type = scaler_type
        self.columns = columns
        self.scaler_ = None
        self.feature_names_ = None

    def _get_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        self.scaler_ = self._get_scaler()
        self.scaler_.fit(X[self.columns])
        self.feature_names_ = self.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X_out = X.copy()
        X_out[self.columns] = self.scaler_.transform(X[self.columns])
        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns) if self.columns else input_features


def fit_transform_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    scaler_type: Literal["standard", "robust", "minmax"] = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScalerWrapper]:
    """
    Fit scaler on train, transform both train and test.

    This is the correct way to avoid data leakage.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_columns: Columns to scale
        scaler_type: Type of scaler

    Returns:
        Tuple of (scaled_train, scaled_test, fitted_scaler)
    """
    scaler = StandardScalerWrapper(scaler_type=scaler_type, columns=feature_columns)
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, scaler


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal["drop", "mean", "median", "zero"] = "mean",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Handle missing values in DataFrame."""
    df_out = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    if strategy == "drop":
        df_out = df_out.dropna(subset=cols)
    elif strategy == "mean":
        df_out[cols] = df_out[cols].fillna(df_out[cols].mean())
    elif strategy == "median":
        df_out[cols] = df_out[cols].fillna(df_out[cols].median())
    elif strategy == "zero":
        df_out[cols] = df_out[cols].fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_out


def clip_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip outliers at specified quantile."""
    df_out = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if df_out[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            lower = df_out[col].quantile(1 - quantile)
            upper = df_out[col].quantile(quantile)
            df_out[col] = df_out[col].clip(lower, upper)

    return df_out


def create_preprocessing_pipeline(
    config: PreprocessingConfig | None = None,
) -> list[tuple[str, BaseEstimator]]:
    """
    Create a list of preprocessing steps for sklearn Pipeline.

    Returns:
        List of (name, transformer) tuples
    """
    if config is None:
        config = PreprocessingConfig()

    steps = []

    if config.handle_missing != "drop":
        steps.append(("imputer", MissingValueHandler(strategy=config.handle_missing)))

    if config.clip_outliers:
        steps.append(("clipper", OutlierClipper(quantile=config.clip_quantile)))

    steps.append(("scaler", StandardScalerWrapper(scaler_type=config.scaler_type)))

    return steps


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values in numeric columns."""

    def __init__(self, strategy: Literal["mean", "median", "zero"] = "mean"):
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if self.strategy == "mean":
            self.fill_values_ = X[numeric_cols].mean().to_dict()
        elif self.strategy == "median":
            self.fill_values_ = X[numeric_cols].median().to_dict()
        elif self.strategy == "zero":
            self.fill_values_ = {c: 0 for c in numeric_cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, val in self.fill_values_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(val)
        return X_out


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers at specified quantiles."""

    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            lower = X[col].quantile(1 - self.quantile)
            upper = X[col].quantile(self.quantile)
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(lower, upper)
        return X_out