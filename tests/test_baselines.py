"""Tests for baseline models (weight direction, equal-weight equivalence)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baselines import WeightedHistoricalMean


def test_weighted_historical_mean_recent_year_wins():
    model = WeightedHistoricalMean(
        party_column="UR_share", decay="exponential", decay_rate=0.5, max_history=5
    )
    X = pd.DataFrame(
        {
            "region_id": [1, 1, 1, 1, 1],
            "year": [2003, 2007, 2011, 2016, 2021],
            "UR_share": [30.0, 30.0, 30.0, 30.0, 60.0],
        }
    )
    model.fit(X, X["UR_share"])
    pred = model.predict(pd.DataFrame({"region_id": [1], "year": [2026]}))[0]
    assert pred > 30.0, "recent election must receive the highest decay weight"


def test_weighted_historical_mean_rate1_is_plain_mean():
    X = pd.DataFrame(
        {
            "region_id": [1, 1, 1, 1, 1],
            "year": [2003, 2007, 2011, 2016, 2021],
            "UR_share": [30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )
    whm = WeightedHistoricalMean(
        party_column="UR_share", decay="exponential", decay_rate=1.0, max_history=5
    )
    whm.fit(X, X["UR_share"])
    X_pred = pd.DataFrame({"region_id": [1], "year": [2026]})
    assert np.isclose(whm.predict(X_pred)[0], np.mean([30.0, 40.0, 50.0, 60.0, 70.0]))
