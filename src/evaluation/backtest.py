"""Rolling temporal backtest runner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.features import get_feature_columns, get_target_columns_from_df
from ..data.splits import (
    create_temporal_splits,
)
from ..evaluation.metrics import (
    error_distribution,
    federal_aggregation,
    party_metrics,
)
from ..models.registry import get_model
from ..utils.io import save_predictions


def prepare_data_for_experiment(
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    train_years_override: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Prepare train/val/test DataFrames for an experiment.

    Returns:
        train_df, val_df, test_df, feature_columns, target_columns
    """
    # Load raw data (with year/type columns for temporal models and baselines)
    from ..data.splits import load_raw_region

    df = load_raw_region()

    # Create temporal splits
    train_df, val_df, test_df = create_temporal_splits(
        df, experiment_name, train_years_override=train_years_override
    )

    # Get feature and target columns from processed data
    # Apply feature selection to get feature columns
    from ..data.features import select_features

    train_processed = select_features(train_df, feature_group)
    target_columns = get_target_columns_from_df(train_processed, level)
    feature_columns = get_feature_columns(train_processed, feature_group)

    # Drop constant (zero-variance) features: they break StandardScaler
    # (0/0 -> NaN) and HistGradientBoosting binning (needs >=2 distinct values).
    non_constant = [c for c in feature_columns if train_df[c].nunique(dropna=True) > 1]
    dropped = sorted(set(feature_columns) - set(non_constant))
    if dropped:
        print(f"[preprocess] Dropping {len(dropped)} constant feature(s): {dropped}")
    feature_columns = non_constant

    return train_df, val_df, test_df, feature_columns, target_columns


def run_single_model_backtest(
    model_name: str,
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
    normalize_predictions: bool = False,
    train_years_override: list[int] | None = None,
) -> dict[str, Any]:
    """
    Run backtest for a single model on a single experiment.

    Predictions are NOT post-hoc normalized to 100% by default: targets are real
    party shares (summing to ~78%) and every party is forecast independently.
    Set ``normalize_predictions=True`` to scale rows to 100% (Approach A).

    Returns:
        Dictionary with metrics, predictions, and metadata
    """
    model_kwargs = model_kwargs or {}

    # Prepare data
    train_df, val_df, test_df, feature_columns, target_columns = prepare_data_for_experiment(
        experiment_name, feature_group, level, train_years_override
    )

    # Handle empty val set
    has_val = len(val_df) > 0

    # Handle experiments without a test set (e.g. final 2026 prediction):
    # there is nothing to evaluate, so skip prediction/metrics.
    if len(test_df) == 0:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "note": "no_test_data",
            "n_train": len(train_df),
            "n_test": 0,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
        }

    # Preprocessing: fit on train, transform all
    from ..data.preprocessing import StandardScalerWrapper

    scaler = StandardScalerWrapper(scaler_type="standard", columns=feature_columns)
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df) if has_val else val_df
    test_scaled = scaler.transform(test_df)

    # Get model
    model = get_model(model_name, **model_kwargs)

    # Prepare targets
    y_train = train_scaled[target_columns]
    y_val = val_scaled[target_columns] if has_val else None

    # Fit model - baselines need year/region_id and target columns (per party), other models use feature_columns
    is_baseline = model_name in [
        "NaivePreviousElection",
        "HistoricalMean",
        "WeightedHistoricalMean",
    ]

    if is_baseline:
        # Baselines are single-party - run for each target column separately
        all_train_preds = []
        all_test_preds = []

        for i, target_col in enumerate(target_columns):
            # Create single-party model
            single_model = get_model(model_name, party_column=target_col, **model_kwargs)

            # Prepare single-party targets
            y_train_single = train_df[target_col]
            y_val_single = val_df[target_col] if has_val else None

            train_cols = ["region_id", "year"] + feature_columns + [target_col]
            val_cols = ["region_id", "year"] + feature_columns + [target_col]

            if has_val:
                if (
                    hasattr(single_model, "fit")
                    and "eval_set" in single_model.fit.__code__.co_varnames
                ):
                    single_model.fit(
                        train_df[train_cols],
                        y_train_single,
                        eval_set=[(val_df[val_cols], y_val_single)],
                        verbose=False,
                    )
                else:
                    single_model.fit(train_df[train_cols], y_train_single)
            else:
                single_model.fit(train_df[train_cols], y_train_single)

            # Predict
            train_pred_single = single_model.predict(train_df[train_cols])
            test_pred_single = single_model.predict(test_df[train_cols])

            all_train_preds.append(train_pred_single)
            all_test_preds.append(test_pred_single)

        # Combine predictions
        train_pred = np.column_stack(all_train_preds)
        test_pred = np.column_stack(all_test_preds)
    else:
        # Regular models use feature_columns (support multi-output)
        if has_val:
            if hasattr(model, "fit") and "eval_set" in model.fit.__code__.co_varnames:
                model.fit(
                    train_scaled[feature_columns],
                    y_train,
                    eval_set=[(val_scaled[feature_columns], y_val)],
                    verbose=False,
                )
            else:
                model.fit(train_scaled[feature_columns], y_train)
        else:
            model.fit(train_scaled[feature_columns], y_train)

        train_pred = model.predict(train_scaled[feature_columns])
        test_pred = model.predict(test_scaled[feature_columns])

    # Create prediction DataFrames
    train_pred_df = pd.DataFrame(
        train_pred, columns=[f"{c}_pred" for c in target_columns], index=train_df.index
    )
    test_pred_df = pd.DataFrame(
        test_pred, columns=[f"{c}_pred" for c in target_columns], index=test_df.index
    )

    # Normalize predictions if needed (compositional constraint)
    if normalize_predictions:
        train_pred_df = normalize_compositional(train_pred_df, target_columns)
        test_pred_df = normalize_compositional(test_pred_df, target_columns)

    # Combine actuals and predictions
    train_results = pd.concat([train_df[target_columns], train_pred_df], axis=1)
    test_results = pd.concat([test_df[target_columns], test_pred_df], axis=1)

    # Compute metrics
    train_metrics = party_metrics(
        train_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )
    test_metrics = party_metrics(
        test_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )

    # Federal aggregation
    weight_col = "electorate" if "electorate" in train_df.columns else None
    if weight_col is None and "population" in train_df.columns:
        weight_col = "population"

    train_federal = (
        federal_aggregation(train_results, target_columns, weight_column=weight_col or "electorate")
        if weight_col
        else {}
    )
    test_federal = (
        federal_aggregation(test_results, target_columns, weight_column=weight_col or "electorate")
        if weight_col
        else {}
    )

    # Error distributions
    test_error_dist = error_distribution(
        test_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )

    # Save predictions
    save_predictions(
        test_results,
        model_name,
        experiment_name,
        target_columns[0].replace("_share", ""),  # Use first party as reference
    )

    return {
        "model": model_name,
        "experiment": experiment_name,
        "feature_group": feature_group,
        "level": level,
        "train_metrics": train_metrics.to_dict("records"),
        "test_metrics": test_metrics.to_dict("records"),
        "train_federal": train_federal,
        "test_federal": test_federal,
        "test_error_distribution": test_error_dist.to_dict("records"),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
    }


def normalize_compositional(
    pred_df: pd.DataFrame,
    target_columns: list[str],
    pred_suffix: str = "_pred",
) -> pd.DataFrame:
    """
    Normalize predictions so they sum to 100% per row (Approach A).

    Args:
        pred_df: DataFrame with prediction columns
        target_columns: Target column names
        pred_suffix: Suffix for prediction columns

    Returns:
        DataFrame with normalized predictions
    """
    pred_cols = [f"{c}{pred_suffix}" for c in target_columns]
    pred_df = pred_df.copy()

    # Clip negative values
    for col in pred_cols:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].clip(lower=0)

    # Sum and normalize
    row_sum = pred_df[pred_cols].sum(axis=1)
    row_sum = row_sum.replace(0, 1)  # Avoid division by zero

    for col in pred_cols:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col] / row_sum * 100

    return pred_df


def run_multioutput_backtest(
    model_name: str,
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Run backtest using multi-output model (Approach B).

    Returns:
        Dictionary with metrics and metadata
    """
    # For now, use the same logic but with multi-output model
    # The model should handle multiple targets internally
    return run_single_model_backtest(
        model_name,
        experiment_name,
        feature_group,
        level,
        model_kwargs,
        normalize_predictions=False,  # Multi-output handles this internally
    )


def _build_region_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    context_years: list[int],
    target_year: int,
    region_col: str = "region_id",
    year_col: str = "year",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build (X_seq, y) samples for one target year from per-region context years.

    Returns lists aligned by sample; skips regions missing any context/target row.
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for region, grp in df.groupby(region_col):
        grp = grp.set_index(year_col)
        if target_year not in grp.index:
            continue
        ctx = [y for y in context_years if y < target_year and y in grp.index]
        if not ctx:
            continue
        ctx = sorted(ctx)
        seq = grp.loc[ctx, feature_columns].values.astype(np.float32)  # (k, F)
        if seq.shape[0] == 0:
            continue
        target = grp.loc[target_year, target_columns].values.astype(np.float32)  # (T,)
        X_list.append(seq)
        y_list.append(target)
    return X_list, y_list


def run_temporal_backtest(
    model_name: str,
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
    normalize_predictions: bool = False,
    train_years_override: list[int] | None = None,
) -> dict[str, Any]:
    """Run a temporal (sequence) backtest for GRU/LSTM/Transformer models.

    Builds per-region election sequences (past years -> next election) and
    evaluates on the experiment's test year(s). Returns the same dict shape as
    run_single_model_backtest so it integrates with the benchmark aggregator.
    """
    from ..data.features import (
        get_feature_columns,
        get_target_columns_from_df,
        select_features,
    )
    from ..data.splits import get_experiment_splits, load_raw_region
    from ..models.registry import get_model

    model_kwargs = model_kwargs or {}

    split = get_experiment_splits(experiment_name)
    if train_years_override is not None:
        split.train_years = list(train_years_override)
    train_years = sorted(split.train_years)
    test_years = split.test_years
    if not test_years:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "note": "no_test_data",
            "n_train": 0,
            "n_test": 0,
        }

    # Raw df keeps identifiers (region_id, year) needed for sequence building;
    # feature/target columns are taken from the processed (feature-selected) view.
    df = load_raw_region()
    processed = select_features(df, feature_group)
    target_columns = get_target_columns_from_df(processed, level)
    feature_columns = get_feature_columns(processed, feature_group)
    non_constant = [c for c in feature_columns if df[c].nunique(dropna=True) > 1]
    dropped = sorted(set(feature_columns) - set(non_constant))
    if dropped:
        print(f"[preprocess] Dropping {len(dropped)} constant feature(s): {dropped}")
    feature_columns = non_constant

    # Training samples: each train year (with prior context) is a target.
    X_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    for target_year in train_years:
        if target_year == min(train_years):
            continue
        Xs, ys = _build_region_sequences(
            df, feature_columns, target_columns, train_years, target_year
        )
        X_train.extend(Xs)
        y_train.extend(ys)

    if not X_train:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "note": "no_train_data",
            "n_train": 0,
            "n_test": 0,
        }

    # Test samples: predict the test year from all train years as context.
    test_year = test_years[0]
    X_test, y_test = _build_region_sequences(
        df, feature_columns, target_columns, train_years, test_year
    )
    if not X_test:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "note": "no_test_data",
            "n_train": len(X_train),
            "n_test": 0,
        }

    y_train_arr = np.stack(y_train).astype(np.float32)  # (n_train, T)
    y_test_arr = np.stack(y_test).astype(np.float32)  # (n_test, T)

    model = get_model(model_name, **model_kwargs)
    model.fit(X_train, y_train_arr)
    pred = model.predict(X_test)  # (n_test, T)

    # Per-party MAE (ignore NaN).
    test_metrics = []
    for j, party in enumerate(target_columns):
        yt = y_test_arr[:, j]
        pt = pred[:, j]
        mask = ~(np.isnan(yt) | np.isnan(pt))
        if mask.sum() == 0:
            continue
        mae_val = float(np.mean(np.abs(yt[mask] - pt[mask])))
        test_metrics.append({"party": party, "mae": mae_val, "n_samples": int(mask.sum())})

    return {
        "model": model_name,
        "experiment": experiment_name,
        "feature_group": feature_group,
        "level": level,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
    }


def forecast_temporal(
    model_name: str,
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
    normalize_predictions: bool = True,
) -> dict[str, Any]:
    """Train on all history and forecast a future election year (no ground truth).

    Used for experiments C/D (target 2026). Builds training sequences from
    ``train_years`` and predicts the future year for each region from its full
    history context. Returns a predictions DataFrame (no metrics, since there is
    no ground truth).
    """
    from ..data.features import (
        get_feature_columns,
        get_target_columns_from_df,
        select_features,
    )
    from ..data.splits import get_experiment_splits, load_raw_region
    from ..models.registry import get_model

    model_kwargs = model_kwargs or {}
    split = get_experiment_splits(experiment_name)
    train_years = sorted(split.train_years)
    test_year = split.test_years[0] if split.test_years else None

    df = load_raw_region()
    processed = select_features(df, feature_group)
    target_columns = get_target_columns_from_df(processed, level)
    feature_columns = get_feature_columns(processed, feature_group)
    non_constant = [c for c in feature_columns if df[c].nunique(dropna=True) > 1]
    dropped = sorted(set(feature_columns) - set(non_constant))
    if dropped:
        print(f"[preprocess] Dropping {len(dropped)} constant feature(s): {dropped}")
    feature_columns = non_constant

    # Training sequences (context -> next historical election).
    X_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    for target_year in train_years:
        if target_year == min(train_years):
            continue
        Xs, ys = _build_region_sequences(
            df, feature_columns, target_columns, train_years, target_year
        )
        X_train.extend(Xs)
        y_train.extend(ys)
    if not X_train:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "note": "no_train_data",
            "n_train": 0,
            "n_test": 0,
        }

    # Forecast sequences: each region's full history context -> future year.
    region_col, year_col = "region_id", "year"
    X_test: list[np.ndarray] = []
    region_ids: list = []
    for region, grp in df.groupby(region_col):
        grp = grp.set_index(year_col)
        ctx = [y for y in train_years if y in grp.index]
        if not ctx:
            continue
        seq = grp.loc[sorted(ctx), feature_columns].values.astype(np.float32)
        if seq.shape[0] == 0:
            continue
        X_test.append(seq)
        region_ids.append(region)

    y_train_arr = np.stack(y_train).astype(np.float32)
    model = get_model(model_name, **model_kwargs)
    model.fit(X_train, y_train_arr)

    if not X_test:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "note": "no_test_data",
            "n_train": len(X_train),
            "n_test": 0,
            "forecast_year": test_year,
        }

    pred = model.predict(X_test)  # (n_regions, T)

    if normalize_predictions:
        pred = np.clip(pred, 0, None)
        row_sum = pred.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        pred = pred / row_sum * 100

    pred_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in target_columns])
    pred_df.insert(0, region_col, region_ids)
    pred_df.insert(1, year_col, test_year)

    return {
        "model": model_name,
        "experiment": experiment_name,
        "feature_group": feature_group,
        "level": level,
        "forecast_year": test_year,
        "predictions": pred_df,
        "target_columns": target_columns,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "note": "forecast",
    }


BASELINE_NAMES = (
    "NaivePreviousElection",
    "HistoricalMean",
    "WeightedHistoricalMean",
)


def forecast_baseline(
    model_name: str,
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Train a flat baseline on all history and forecast a future election year.

    Used for experiments C/D (target 2026) with models from ``BASELINE_NAMES``.
    Baselines are single-party: each party is predicted independently from the
    region's own history (features are not used). Deterministic, so no seeds.
    Returns a predictions DataFrame (no metrics, since there is no ground truth).
    """
    from ..data.features import get_target_columns_from_df
    from ..data.splits import get_experiment_splits, load_raw_region
    from ..models.registry import get_model

    model_kwargs = model_kwargs or {}
    split = get_experiment_splits(experiment_name)
    train_years = sorted(split.train_years)
    test_year = split.test_years[0] if split.test_years else None

    df = load_raw_region()
    train_df = df[df["year"].isin(train_years)]
    target_columns = get_target_columns_from_df(train_df, level)

    region_ids = sorted(df["region_id"].unique())
    X_pred = pd.DataFrame({"region_id": region_ids, "year": test_year})
    pred = np.empty((len(region_ids), len(target_columns)))

    for i, target_col in enumerate(target_columns):
        model = get_model(model_name, party_column=target_col, **model_kwargs)
        model.fit(train_df, train_df[target_col])
        pred[:, i] = model.predict(X_pred)

    pred_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in target_columns])
    pred_df.insert(0, "region_id", region_ids)
    pred_df.insert(1, "year", test_year)

    return {
        "model": model_name,
        "experiment": experiment_name,
        "feature_group": feature_group,
        "level": level,
        "forecast_year": test_year,
        "predictions": pred_df,
        "target_columns": target_columns,
        "n_train": len(train_df),
        "n_test": len(region_ids),
        "note": "forecast",
    }


def run_experiment(
    experiment_name: str,
    models: list[str],
    feature_groups: list[str] = ["ALL_FEATURES"],
    level: str = "region",
) -> list[dict[str, Any]]:
    """
    Run full experiment for multiple models and feature groups.

    Returns:
        List of result dictionaries
    """
    results = []

    for feature_group in feature_groups:
        for model_name in models:
            print(f"Running {model_name} on {experiment_name} with {feature_group}...")
            try:
                result = run_single_model_backtest(
                    model_name,
                    experiment_name,
                    feature_group,
                    level,
                )
                results.append(result)
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                results.append(
                    {
                        "model": model_name,
                        "experiment": experiment_name,
                        "feature_group": feature_group,
                        "error": str(e),
                    }
                )

    return results
