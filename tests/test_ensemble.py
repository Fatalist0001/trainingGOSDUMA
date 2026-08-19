"""Smoke tests for ensemble models (WeightedEnsemble, StackingEnsemble)."""

from __future__ import annotations

import numpy as np

from src.evaluation.backtest import run_single_model_backtest


def test_weighted_ensemble_runs():
    r = run_single_model_backtest("WeightedEnsemble", "A", "ALL_FEATURES")
    assert r.get("test_metrics") is not None
    maes = [m["mae"] for m in r["test_metrics"]]
    assert len(maes) == 3
    assert all(np.isfinite(m) for m in maes)


def test_stacking_ensemble_runs():
    r = run_single_model_backtest("StackingEnsemble", "A", "ALL_FEATURES")
    assert r.get("test_metrics") is not None
    maes = [m["mae"] for m in r["test_metrics"]]
    assert len(maes) == 3
    assert all(np.isfinite(m) for m in maes)
