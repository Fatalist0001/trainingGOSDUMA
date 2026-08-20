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
from ..evaluation.tuning import (
    _fit_with_val,
    refit_kwargs,
    tune_flat_model,
    tune_temporal_model,
    tune_weighted_historical_mean,
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

    # Prepare targets
    y_train = train_scaled[target_columns]
    y_val = val_scaled[target_columns] if has_val else None

    # Baselines need year/region_id and target columns (per party), other models use feature_columns.
    is_baseline = model_name in [
        "NaivePreviousElection",
        "HistoricalMean",
        "WeightedHistoricalMean",
    ]

    # Two-stage protocol (see AGENTS.md):
    #   1) hyperparameters are selected ONLY on the validation year (past -> future);
    #   2) the final model is refit on train+val (all data strictly before test)
    #      and only then scored on the test year.
    tuned_params: dict[str, Any] = {}
    tuning_mae: float | None = None

    if is_baseline:
        if model_name == "WeightedHistoricalMean" and has_val:
            tune_res = tune_weighted_historical_mean(train_df, val_df, target_columns)
            tuned_params = tune_res["params"] or {}
            tuning_mae = tune_res["score"]

        # Refit on train+val with the chosen decay.
        fit_df = pd.concat([train_df, val_df], axis=0) if has_val else train_df
        all_train_preds = []
        all_test_preds = []

        for target_col in target_columns:
            kwargs = dict(model_kwargs)
            kwargs.update(tuned_params)
            single_model = get_model(model_name, party_column=target_col, **kwargs)

            train_cols = ["region_id", "year"] + feature_columns + [target_col]
            single_model.fit(fit_df[train_cols], fit_df[target_col])

            all_train_preds.append(single_model.predict(train_df[train_cols]))
            all_test_preds.append(single_model.predict(test_df[train_cols]))

        train_pred = np.column_stack(all_train_preds)
        test_pred = np.column_stack(all_test_preds)
    else:
        # Temporal-validated tuning: candidates are fit on train and scored on val.
        best_candidate = None
        if has_val:
            tune_res = tune_flat_model(
                model_name,
                train_scaled[feature_columns],
                y_train,
                val_scaled[feature_columns],
                y_val,
            )
            tuned_params = tune_res["params"] or {}
            tuning_mae = tune_res["score"]
            best_candidate = tune_res["model"]

        # Refit the best config on train+val, then predict train/test.
        combined_scaled = pd.concat([train_scaled, val_scaled], axis=0) if has_val else train_scaled
        combined_y = pd.concat([y_train, y_val], axis=0) if has_val else y_train

        final_kwargs = refit_kwargs(model_name, tuned_params, best_candidate)
        final_kwargs.update(model_kwargs)
        final_model = get_model(model_name, **final_kwargs)
        _fit_with_val(
            final_model,
            combined_scaled[feature_columns],
            combined_y,
            None,
            None,
            years=combined_scaled["year"],
        )

        train_pred = final_model.predict(train_scaled[feature_columns])
        test_pred = final_model.predict(test_scaled[feature_columns])

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
    train_results = pd.concat(
        [train_df[["region_id", "year", "type"] + target_columns], train_pred_df], axis=1
    )
    test_results = pd.concat(
        [test_df[["region_id", "year", "type"] + target_columns], test_pred_df], axis=1
    )

    # Compute metrics
    train_metrics = party_metrics(
        train_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )
    test_metrics = party_metrics(
        test_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )

    # Federal aggregation: weight regional predictions by the actual electorate /
    # turnout / valid-vote totals aggregated from RED (no uniform fallback).
    from ..data.loader import load_electoral_weights

    weight_meta = ["electorate", "turnout", "valid", "invalid"]
    weights = load_electoral_weights()[["region_id", "year", "type"] + weight_meta]

    def _federal(results: pd.DataFrame) -> dict[str, dict[str, float]]:
        merged = results.merge(weights, on=["region_id", "year", "type"], how="left")
        return federal_aggregation(merged, target_columns)

    # Train may pool several years; report federal aggregation per year.
    train_federal: dict[str, dict[str, dict[str, float]]] = {}
    for year in sorted(train_df["year"].unique()):
        train_federal[int(year)] = _federal(train_results[train_results["year"] == year])
    test_federal = _federal(test_results)

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
        "tuned_params": tuned_params,
        "tuning_mae": tuning_mae,
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
    train_years_override: list[int] | None = None,
) -> dict[str, Any]:
    """Run an *independent per-party* backtest (compositional approach).

    Unlike ``run_single_model_backtest`` (one shared model trained on all
    targets jointly), this fits one regression per party with its own
    temporal-validated hyperparameters. Predictions are NOT post-hoc normalized:
    targets are real shares summing to ~78%.

    Returns the same dict shape as ``run_single_model_backtest`` so it
    integrates with the benchmark aggregator.
    """
    from ..data.preprocessing import StandardScalerWrapper

    model_kwargs = model_kwargs or {}

    train_df, val_df, test_df, feature_columns, target_columns = prepare_data_for_experiment(
        experiment_name, feature_group, level, train_years_override
    )
    has_val = len(val_df) > 0
    if len(test_df) == 0:
        return {
            "model": model_name,
            "experiment": experiment_name,
            "feature_group": feature_group,
            "level": level,
            "n_train": len(train_df),
            "n_test": 0,
        }

    scaler = StandardScalerWrapper(scaler_type="standard", columns=feature_columns)
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df) if has_val else val_df
    test_scaled = scaler.transform(test_df)

    y_train = train_scaled[target_columns]
    y_val = val_scaled[target_columns] if has_val else None

    # Two-stage protocol per party: tune on val -> refit on train+val -> predict.
    tuned_params: dict[str, dict[str, Any]] = {}
    tuning_mae: float | None = None

    all_train_preds: list[np.ndarray] = []
    all_test_preds: list[np.ndarray] = []

    for party in target_columns:
        best_candidate = None
        params: dict[str, Any] = {}
        if has_val:
            tune_res = tune_flat_model(
                model_name,
                train_scaled[feature_columns],
                y_train[[party]],
                val_scaled[feature_columns],
                y_val[[party]],
            )
            params = tune_res["params"] or {}
            tuning_mae = tune_res["score"]
            best_candidate = tune_res["model"]
        tuned_params[party] = params

        combined_scaled = pd.concat([train_scaled, val_scaled], axis=0) if has_val else train_scaled
        combined_y = (
            pd.concat([y_train[[party]], y_val[[party]]], axis=0) if has_val else y_train[[party]]
        )

        final_kwargs = refit_kwargs(model_name, params, best_candidate)
        final_kwargs.update(model_kwargs)
        final_model = get_model(model_name, **final_kwargs)
        _fit_with_val(
            final_model,
            combined_scaled[feature_columns],
            combined_y,
            None,
            None,
            years=combined_scaled["year"],
        )
        all_train_preds.append(final_model.predict(train_scaled[feature_columns]).reshape(-1))
        all_test_preds.append(final_model.predict(test_scaled[feature_columns]).reshape(-1))

    train_pred = np.column_stack(all_train_preds)
    test_pred = np.column_stack(all_test_preds)

    train_pred_df = pd.DataFrame(
        train_pred, columns=[f"{c}_pred" for c in target_columns], index=train_df.index
    )
    test_pred_df = pd.DataFrame(
        test_pred, columns=[f"{c}_pred" for c in target_columns], index=test_df.index
    )

    train_results = pd.concat(
        [train_df[["region_id", "year", "type"] + target_columns], train_pred_df], axis=1
    )
    test_results = pd.concat(
        [test_df[["region_id", "year", "type"] + target_columns], test_pred_df], axis=1
    )

    train_metrics = party_metrics(
        train_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )
    test_metrics = party_metrics(
        test_results, target_columns, pred_suffix="_pred", actual_suffix=""
    )

    from ..data.loader import load_electoral_weights

    weight_meta = ["electorate", "turnout", "valid", "invalid"]
    weights = load_electoral_weights()[["region_id", "year", "type"] + weight_meta]

    def _federal(results: pd.DataFrame) -> dict[str, dict[str, float]]:
        merged = results.merge(weights, on=["region_id", "year", "type"], how="left")
        return federal_aggregation(merged, target_columns)

    train_federal: dict[str, dict[str, dict[str, float]]] = {}
    for year in sorted(train_df["year"].unique()):
        train_federal[int(year)] = _federal(train_results[train_results["year"] == year])
    test_federal = _federal(test_results)

    return {
        "model": model_name,
        "experiment": experiment_name,
        "feature_group": feature_group,
        "level": level,
        "approach": "independent_per_party",
        "train_metrics": train_metrics.to_dict("records"),
        "test_metrics": test_metrics.to_dict("records"),
        "train_federal": train_federal,
        "test_federal": test_federal,
        "tuned_params": tuned_params,
        "tuning_mae": tuning_mae,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
    }


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


def _parl_feature_columns(df: pd.DataFrame, feature_group: str, level: str) -> list[str]:
    """Feature columns for temporal models, dropping columns constant in parl rows.

    The full raw df also contains presidential rows, so some columns populated
    only there (e.g. ``leading_candidate_share``) would otherwise survive a
    full-frame constant check and inject all-NaN features into the training
    sequences. Constantness is therefore evaluated on parliamentary rows only.
    """
    from ..data.features import (
        get_feature_columns,
        select_features,
    )

    processed = select_features(df, feature_group)
    feature_columns = get_feature_columns(processed, feature_group)
    parl_mask = df["type"] == "parl"
    non_constant = [c for c in feature_columns if df.loc[parl_mask, c].nunique(dropna=True) > 1]
    dropped = sorted(set(feature_columns) - set(non_constant))
    if dropped:
        print(f"[preprocess] Dropping {len(dropped)} constant feature(s): {dropped}")
    return non_constant


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
    val_years = list(split.val_years) if split.val_years else []
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
    target_columns = get_target_columns_from_df(select_features(df, feature_group), level)
    feature_columns = _parl_feature_columns(df, feature_group, level)

    def _samples(target_years: list[int], context_years: list[int]):
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for target_year in target_years:
            if target_year == min(train_years):
                continue
            x, y = _build_region_sequences(
                df, feature_columns, target_columns, context_years, target_year
            )
            Xs.extend(x)
            ys.extend(y)
        return Xs, ys

    # Training samples: each train year (with prior context) is a target.
    X_train, y_train = _samples(train_years, train_years)

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

    # Validation samples: the val year predicted from train context (past -> future).
    has_val = bool(val_years)
    X_val, y_val = [], []
    if has_val:
        X_val, y_val = _samples([val_years[0]], train_years)

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

    # Two-stage protocol: tune on val -> refit on train+val -> predict test.
    tuned_params: dict[str, Any] = {}
    tuning_mae: float | None = None
    if has_val:
        tune_res = tune_temporal_model(model_name, X_train, y_train_arr, X_val, y_val)
        tuned_params = tune_res["params"] or {}
        tuning_mae = tune_res["score"]

    final_kwargs = refit_kwargs(model_name, tuned_params, None)
    final_kwargs.update(model_kwargs)
    model = get_model(model_name, **final_kwargs)

    if has_val and X_val:
        # Refit on train+val samples (all data strictly before test).
        X_refit = list(X_train) + list(X_val)
        y_refit = np.concatenate([y_train_arr, np.stack(y_val).astype(np.float32)], axis=0)
        model.fit(X_refit, y_refit)
    else:
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
        "tuned_params": tuned_params,
        "tuning_mae": tuning_mae,
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
    normalize_predictions: bool = False,
) -> dict[str, Any]:
    """Train on all history and forecast a future election year (no ground truth).

    Used for experiments C/D (target 2026). Builds training sequences from
    ``train_years`` and predicts the future year for each region from its full
    history context. Returns a predictions DataFrame (no metrics, since there is
    no ground truth).
    """
    from ..data.features import (
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
    target_columns = get_target_columns_from_df(select_features(df, feature_group), level)
    feature_columns = _parl_feature_columns(df, feature_group, level)

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
    # The most recent presidential election (e.g. 2024 for 2026) is appended as
    # a synthetic final context element so fresh presidential results inform the
    # forecast. Its signal is folded into the lag columns (``pres_turnout_lag``,
    # ``pres_leading_candidate_share_lag``) to stay consistent with the parl-row
    # feature distribution the model was trained on.
    from ..data.presidential_features import _most_recent_pres_year

    pres_years = sorted(df.loc[df["type"] == "pres", "year"].unique())
    recent_pres = _most_recent_pres_year(pres_years, int(test_year)) if test_year else None

    region_col, year_col = "region_id", "year"
    X_test: list[np.ndarray] = []
    region_ids: list = []
    for region, grp in df.groupby(region_col):
        grp = grp.set_index(year_col)
        ctx = [y for y in train_years if y in grp.index]
        if not ctx:
            continue
        rows = grp.loc[sorted(ctx), feature_columns].copy()
        if recent_pres is not None and recent_pres in grp.index:
            pres_row = grp.loc[recent_pres].copy()
            pres_row["pres_turnout_lag"] = pres_row["turnout_rate"]
            pres_row["pres_leading_candidate_share_lag"] = pres_row["leading_candidate_share"]
            pres_row["leading_candidate_share"] = np.nan
            rows = pd.concat([rows, pres_row.to_frame().T[feature_columns]])
        seq = rows.values.astype(np.float32)
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

    # The WeightedHistoricalMean decay for the 2026 forecast is tuned on
    # experiment B (train 2003-2011, val 2016), then refit on all history
    # (see AGENTS.md "Допущения"). Explicit model_kwargs win over tuning.
    if model_name == "WeightedHistoricalMean" and not model_kwargs:
        b_split = get_experiment_splits("B")
        b_train = df[df["year"].isin(sorted(b_split.train_years))]
        b_val = df[df["year"].isin(sorted(b_split.val_years))]
        tune_res = tune_weighted_historical_mean(b_train, b_val, target_columns)
        model_kwargs = dict(model_kwargs)
        model_kwargs.update(tune_res["params"] or {})

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
