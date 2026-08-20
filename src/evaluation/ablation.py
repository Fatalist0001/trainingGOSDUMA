"""Ablation studies for the election forecasting experiment.

Two ablations answer the core research questions:
- ``feature_ablation`` (Q2): does Rosstat add predictive power? Compares
  ELECTORAL_ONLY vs ROSSTAT_ONLY vs ALL_FEATURES.
- ``history_depth_ablation`` (Q3): does a longer electoral history help? Varies
  the number of past elections used as training context (depth 1..N).
"""

from __future__ import annotations

import numpy as np

from ..data.splits import get_experiment_splits
from .backtest import run_single_model_backtest, run_temporal_backtest

# Temporal (sequence) models use the dedicated sequence backtest.
TEMPORAL_MODELS = {"GRU", "LSTM", "Transformer"}


def _avg_mae(result: dict) -> float | None:
    """Average per-party MAE from a backtest result dict, or None if absent."""
    tm = result.get("test_metrics")
    if not tm:
        return None
    return float(np.mean([m["mae"] for m in tm]))


def feature_ablation(
    model_name: str,
    experiment: str,
    feature_groups: list[str] | None = None,
    level: str = "region",
    model_kwargs: dict | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Run a model across feature groups; returns MAE per group.

    Args:
        model_name: Model key from the registry.
        experiment: Experiment name ("A", "B").
        feature_groups: Groups to compare. Defaults to
            [ELECTORAL_ONLY, ROSSTAT_ONLY, ALL_FEATURES].
        seeds: If provided and the model is temporal, average MAE over seeds.

    Returns:
        List of dicts: {model, experiment, feature_group, mae}.
    """
    feature_groups = feature_groups or ["ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"]
    is_temporal = model_name in TEMPORAL_MODELS
    rows: list[dict] = []

    for fg in feature_groups:
        mae_vals: list[float] = []
        if is_temporal and seeds:
            for s in seeds:
                r = run_temporal_backtest(model_name, experiment, fg, level, {"random_state": s})
                a = _avg_mae(r)
                if a is not None:
                    mae_vals.append(a)
        elif is_temporal:
            r = run_temporal_backtest(model_name, experiment, fg, level, model_kwargs)
            a = _avg_mae(r)
            if a is not None:
                mae_vals.append(a)
        else:
            r = run_single_model_backtest(model_name, experiment, fg, level, model_kwargs)
            a = _avg_mae(r)
            if a is not None:
                mae_vals.append(a)

        rows.append(
            {
                "model": model_name,
                "experiment": experiment,
                "feature_group": fg,
                "mae": float(np.mean(mae_vals)) if mae_vals else None,
            }
        )
    return rows


def history_depth_ablation(
    model_name: str,
    experiment: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
    seeds: list[int] | None = None,
    max_depth: int | None = None,
) -> list[dict]:
    """Vary the number of past elections used as training context.

    Args:
        model_name: Model key from the registry.
        experiment: Experiment name ("A", "B").
        feature_group: Feature group to use.
        seeds: If provided and the model is temporal, average MAE over seeds.
        max_depth: Cap on history depth (number of elections).

    Returns:
        List of dicts: {model, experiment, depth, n_train_years, train_years, mae}.
    """
    split = get_experiment_splits(experiment)
    all_train = sorted(split.train_years)
    if not split.test_years:
        return []
    depths = list(range(1, len(all_train) + 1))
    if max_depth is not None:
        depths = [d for d in depths if d <= max_depth]

    is_temporal = model_name in TEMPORAL_MODELS
    rows: list[dict] = []

    for d in depths:
        train_years = all_train[-d:]
        mae_vals: list[float] = []
        if is_temporal and seeds:
            for s in seeds:
                r = run_temporal_backtest(
                    model_name,
                    experiment,
                    feature_group,
                    level,
                    {"random_state": s},
                    train_years_override=train_years,
                )
                a = _avg_mae(r)
                if a is not None:
                    mae_vals.append(a)
        elif is_temporal:
            r = run_temporal_backtest(
                model_name,
                experiment,
                feature_group,
                level,
                model_kwargs,
                train_years_override=train_years,
            )
            a = _avg_mae(r)
            if a is not None:
                mae_vals.append(a)
        else:
            r = run_single_model_backtest(
                model_name,
                experiment,
                feature_group,
                level,
                model_kwargs,
                train_years_override=train_years,
            )
            a = _avg_mae(r)
            if a is not None:
                mae_vals.append(a)

        rows.append(
            {
                "model": model_name,
                "experiment": experiment,
                "depth": d,
                "n_train_years": len(train_years),
                "train_years": train_years,
                "mae": float(np.mean(mae_vals)) if mae_vals else None,
            }
        )
    return rows
