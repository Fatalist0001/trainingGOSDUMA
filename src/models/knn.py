"""KNN model: KNeighborsRegressor wrapper with multi-output support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted


class KNNModel(BaseEstimator, RegressorMixin):
    """K-Nearest Neighbors regressor wrapper with multi-output support.

    Predicts a region's party shares as the (optionally weighted) average of the
    K most similar regions in the (scaled) feature space. KNN is distance-based,
    so features must be scaled before fitting — this is handled by the backtest
    preprocessing (StandardScalerWrapper fits on train only, no leakage).
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: int = 2,
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: int = -1,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        # KNN is deterministic; random_state accepted for API parity but unused.
        self.random_state = random_state
        self.model_ = None
        self.target_columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.target_columns_ = y.columns.tolist()
        self.model_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X)
