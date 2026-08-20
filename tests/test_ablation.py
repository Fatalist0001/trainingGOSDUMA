"""Smoke tests for ablation studies (feature groups and history depth)."""

from __future__ import annotations

import numpy as np

from src.evaluation.ablation import feature_ablation, history_depth_ablation


def test_feature_ablation_returns_groups():
    rows = feature_ablation("NaivePreviousElection", "A")
    assert len(rows) == 3
    for r in rows:
        assert r["mae"] is not None
        assert r["mae"] > 0


def test_history_depth_ablation_returns_depths():
    rows = history_depth_ablation("XGBoost", "A")
    # depths 1, 2 for experiment A (train 2003-2007)
    assert [r["depth"] for r in rows] == [1, 2]
    maes = [r["mae"] for r in rows if r["mae"] is not None]
    assert len(maes) >= 2  # depth 1 is valid for flat models
    assert all(np.isfinite(m) for m in maes)
