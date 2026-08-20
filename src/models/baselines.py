"""Baseline models for election forecasting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class BaseBaseline(BaseEstimator, RegressorMixin, ABC):
    """Abstract base class for baseline models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass


class NaivePreviousElection(BaseBaseline):
    """
    Naive baseline: predict the previous election result for each region.

    For each region, uses the most recent historical election result
    as the prediction for the next election.
    """

    def __init__(self, party_column: str = "UR_share"):
        self.party_column = party_column
        self.last_results_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame):
        """
        Store the last known result for each region.

        Args:
            X: Features DataFrame with 'region_id' and 'year' columns
            y: Target values (not used, but required by sklearn API)
        """
        if "region_id" not in X.columns or "year" not in X.columns:
            raise ValueError("X must contain 'region_id' and 'year' columns")

        if self.party_column not in X.columns:
            raise ValueError(f"Party column '{self.party_column}' not found in X")

        # For each region, get the most recent election result
        X_sorted = X.sort_values(["region_id", "year"])
        self.last_results_ = X_sorted.groupby("region_id")[self.party_column].last().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using last known result for each region."""
        check_is_fitted(self, "last_results_")

        if "region_id" not in X.columns:
            raise ValueError("X must contain 'region_id' column")

        predictions = X["region_id"].map(self.last_results_)

        # Fill missing with global mean
        global_mean = np.mean(list(self.last_results_.values()))
        predictions = predictions.fillna(global_mean)

        return predictions.values


class HistoricalMean(BaseBaseline):
    """
    Baseline: predict the historical mean for each region.

    Computes the average of all previous election results for each region.
    """

    def __init__(self, party_column: str = "UR_share", min_periods: int = 1):
        self.party_column = party_column
        self.min_periods = min_periods
        self.region_means_ = {}
        self.global_mean_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame):
        """
        Compute historical mean for each region.

        Args:
            X: Features DataFrame with 'region_id', 'year', and party_column
            y: Target values (not used)
        """
        required_cols = ["region_id", "year", self.party_column]
        for col in required_cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X")

        X_sorted = X.sort_values(["region_id", "year"])

        # Compute expanding mean for each region (excluding current year)
        def expanding_mean(group):
            return group.expanding(min_periods=self.min_periods).mean().shift(1)

        X_sorted["hist_mean"] = X_sorted.groupby("region_id")[self.party_column].transform(
            expanding_mean
        )

        # Get the last available historical mean for each region
        last_means = X_sorted.dropna(subset=["hist_mean"]).groupby("region_id")["hist_mean"].last()
        self.region_means_ = last_means.to_dict()
        self.global_mean_ = X[self.party_column].mean()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using historical mean for each region."""
        check_is_fitted(self, "region_means_")

        if "region_id" not in X.columns:
            raise ValueError("X must contain 'region_id' column")

        predictions = X["region_id"].map(self.region_means_)
        predictions = predictions.fillna(self.global_mean_)

        return predictions.values


class WeightedHistoricalMean(BaseBaseline):
    """
    Baseline: weighted historical mean with decay.

    More recent elections get higher weight. Supports exponential and linear decay.
    """

    def __init__(
        self,
        party_column: str = "UR_share",
        decay: Literal["exponential", "linear"] = "exponential",
        decay_rate: float = 0.5,
        max_history: int = 5,
    ):
        self.party_column = party_column
        self.decay = decay
        self.decay_rate = decay_rate
        self.max_history = max_history
        self.region_predictions_ = {}
        self.global_mean_ = 0.0

    def _compute_weights(self, n: int) -> np.ndarray:
        """Compute decay weights for n historical points.

        Index 0 is the oldest election, index ``n - 1`` the most recent one.
        The most recent election receives the highest weight (``decay_rate^0``),
        older ones fade geometrically/linearly.
        """
        if self.decay == "exponential":
            # Exponential decay: w_i = decay_rate^(n-1-i) -> newest gets decay_rate^0.
            weights = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
        elif self.decay == "linear":
            # Linear decay: newest gets ~1, oldest gets decay_rate.
            weights = np.array(
                [1 - ((n - 1 - i) / max(1, n - 1)) * (1 - self.decay_rate) for i in range(n)]
            )
        else:
            raise ValueError(f"Unknown decay type: {self.decay}")

        # Normalize
        return weights / weights.sum()

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame):
        """
        Compute weighted historical mean for each region.

        Args:
            X: Features DataFrame with 'region_id', 'year', and party_column
            y: Target values (not used)
        """
        required_cols = ["region_id", "year", self.party_column]
        for col in required_cols:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X")

        X_sorted = X.sort_values(["region_id", "year"])
        self.global_mean_ = X[self.party_column].mean()

        predictions = {}

        for region_id, group in X_sorted.groupby("region_id"):
            values = group[self.party_column].values
            n = len(values)

            if n == 0:
                predictions[region_id] = self.global_mean_
                continue

            # Use only last max_history values
            if n > self.max_history:
                values = values[-self.max_history :]
                n = self.max_history

            weights = self._compute_weights(n)
            # Apply weights (most recent = last element)
            weighted_mean = np.sum(values * weights)
            predictions[region_id] = weighted_mean

        self.region_predictions_ = predictions
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using weighted historical mean for each region."""
        check_is_fitted(self, "region_predictions_")

        if "region_id" not in X.columns:
            raise ValueError("X must contain 'region_id' column")

        predictions = X["region_id"].map(self.region_predictions_)
        predictions = predictions.fillna(self.global_mean_)

        return predictions.values


def get_baseline_model(name: str, **kwargs) -> BaseBaseline:
    """Factory function to get baseline model by name."""
    models = {
        "NaivePreviousElection": NaivePreviousElection,
        "HistoricalMean": HistoricalMean,
        "WeightedHistoricalMean": WeightedHistoricalMean,
    }

    if name not in models:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(models.keys())}")

    return models[name](**kwargs)
