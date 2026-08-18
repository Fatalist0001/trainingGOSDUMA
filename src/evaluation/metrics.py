"""Evaluation metrics for election forecasting."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias (prediction - actual). Positive = overprediction."""
    return np.mean(y_pred - y_true)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(y_true) < 2:
        return np.nan
    r, _ = stats.pearsonr(y_true, y_pred)
    return r


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    if len(y_true) < 2:
        return np.nan
    r, _ = stats.spearmanr(y_true, y_pred)
    return r


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all standard metrics at once."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
    }


def party_metrics(
    df: pd.DataFrame,
    party_columns: list[str],
    pred_suffix: str = "_pred",
    actual_suffix: str = "",
) -> pd.DataFrame:
    """
    Compute metrics per party.

    Args:
        df: DataFrame with actual and predicted columns
        party_columns: List of party column names (actual)
        pred_suffix: Suffix for prediction columns
        actual_suffix: Suffix for actual columns (default none)

    Returns:
        DataFrame with metrics per party
    """
    results = []

    for party in party_columns:
        actual_col = party + actual_suffix
        pred_col = party + pred_suffix

        if actual_col not in df.columns or pred_col not in df.columns:
            continue

        y_true = df[actual_col].values
        y_pred = df[pred_col].values

        # Remove NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            continue

        metrics = compute_all_metrics(y_true, y_pred)
        metrics["party"] = party
        metrics["n_samples"] = len(y_true)
        results.append(metrics)

    return pd.DataFrame(results)


def federal_aggregation(
    df: pd.DataFrame,
    party_columns: list[str],
    weight_column: str = "electorate",
    pred_suffix: str = "_pred",
    actual_suffix: str = "",
) -> dict[str, dict[str, float]]:
    """
    Aggregate regional predictions to federal level using weighted average.

    Args:
        df: DataFrame with regional predictions
        party_columns: List of party column names
        weight_column: Column to use as weight (e.g., electorate, turnout)
        pred_suffix: Suffix for prediction columns
        actual_suffix: Suffix for actual columns

    Returns:
        Dict with federal-level metrics per party
    """
    results = {}

    if weight_column not in df.columns:
        raise ValueError(f"Weight column '{weight_column}' not found")

    weights = df[weight_column].values

    for party in party_columns:
        actual_col = party + actual_suffix
        pred_col = party + pred_suffix

        if actual_col not in df.columns or pred_col not in df.columns:
            continue

        y_true = df[actual_col].values
        y_pred = df[pred_col].values

        # Weighted average
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        if mask.sum() == 0:
            continue

        y_true_w = y_true[mask]
        y_pred_w = y_pred[mask]
        w = weights[mask]

        federal_true = np.average(y_true_w, weights=w)
        federal_pred = np.average(y_pred_w, weights=w)

        results[party] = {
            "federal_actual": federal_true,
            "federal_pred": federal_pred,
            "federal_error": federal_pred - federal_true,
            "federal_abs_error": abs(federal_pred - federal_true),
        }

    return results


def error_distribution(
    df: pd.DataFrame,
    party_columns: list[str],
    pred_suffix: str = "_pred",
    actual_suffix: str = "",
) -> pd.DataFrame:
    """
    Compute error distribution statistics per party.

    Returns:
        DataFrame with quantiles, mean, std of errors
    """
    results = []

    for party in party_columns:
        actual_col = party + actual_suffix
        pred_col = party + pred_suffix

        if actual_col not in df.columns or pred_col not in df.columns:
            continue

        errors = df[pred_col] - df[actual_col]
        errors = errors.dropna()

        if len(errors) == 0:
            continue

        results.append({
            "party": party,
            "mean_error": errors.mean(),
            "std_error": errors.std(),
            "median_error": errors.median(),
            "q25": errors.quantile(0.25),
            "q75": errors.quantile(0.75),
            "min_error": errors.min(),
            "max_error": errors.max(),
            "mae": errors.abs().mean(),
            "rmse": np.sqrt((errors ** 2).mean()),
        })

    return pd.DataFrame(results)


def regional_breakdown(
    df: pd.DataFrame,
    party_columns: list[str],
    region_column: str = "region_id",
    pred_suffix: str = "_pred",
    actual_suffix: str = "",
) -> pd.DataFrame:
    """
    Compute metrics per region.

    Returns:
        DataFrame with metrics per region per party
    """
    results = []

    for party in party_columns:
        actual_col = party + actual_suffix
        pred_col = party + pred_suffix

        if actual_col not in df.columns or pred_col not in df.columns:
            continue

        for region_id, group in df.groupby(region_column):
            y_true = group[actual_col].values
            y_pred = group[pred_col].values

            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() < 2:
                continue

            y_true = y_true[mask]
            y_pred = y_pred[mask]

            metrics = compute_all_metrics(y_true, y_pred)
            metrics["party"] = party
            metrics["region_id"] = region_id
            metrics["n_samples"] = len(y_true)
            results.append(metrics)

    return pd.DataFrame(results)


def worst_predictions(
    df: pd.DataFrame,
    party_columns: list[str],
    n: int = 20,
    pred_suffix: str = "_pred",
    actual_suffix: str = "",
) -> pd.DataFrame:
    """
    Find worst predictions by absolute error.

    Returns:
        DataFrame with worst predictions
    """
    all_errors = []

    for party in party_columns:
        actual_col = party + actual_suffix
        pred_col = party + pred_suffix

        if actual_col not in df.columns or pred_col not in df.columns:
            continue

        temp = df[[actual_col, pred_col]].copy()
        temp["party"] = party
        temp["error"] = temp[pred_col] - temp[actual_col]
        temp["abs_error"] = temp["error"].abs()
        all_errors.append(temp)

    if not all_errors:
        return pd.DataFrame()

    combined = pd.concat(all_errors, ignore_index=True)
    return combined.nlargest(n, "abs_error")