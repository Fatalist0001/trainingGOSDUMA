"""Smoke tests for 2026 forecast functions (C/D experiments, no ground truth)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtest import forecast_baseline, forecast_temporal


def test_forecast_baseline_returns_regions():
    res = forecast_baseline("WeightedHistoricalMean", "C")
    assert res["note"] == "forecast"
    preds = res["predictions"]
    assert len(preds) == 83
    assert res["forecast_year"] == 2026
    pred_cols = [f"{c}_pred" for c in res["target_columns"]]
    assert list(preds.columns[:2]) == ["region_id", "year"]
    assert all(np.isfinite(preds[p].values).all() for p in pred_cols)
    assert set(preds[p].dtype.name for p in pred_cols) == {"float64"}


def test_forecast_baseline_deterministic():
    a = forecast_baseline("WeightedHistoricalMean", "C")["predictions"]
    b = forecast_baseline("WeightedHistoricalMean", "D")["predictions"]
    pd.testing.assert_frame_equal(a, b)


def test_forecast_temporal_returns_regions():
    res = forecast_temporal("GRU", "C", model_kwargs={"random_state": 42})
    assert res["note"] == "forecast"
    assert len(res["predictions"]) == 83
