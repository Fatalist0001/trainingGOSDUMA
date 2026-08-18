"""Rolling temporal backtest runner."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..data.loader import get_party_list, load_region
from ..data.splits import create_temporal_splits, get_experiment_splits, get_internal_validation_years
from ..data.features import get_target_columns_from_df, get_feature_columns
from ..data.preprocessing import fit_transform_train_test
from ..evaluation.metrics import compute_all_metrics, party_metrics, federal_aggregation, error_distribution
from ..utils.io import save_predictions, save_results
from ..models.registry import get_model


def prepare_data_for_experiment(
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
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
    train_df, val_df, test_df = create_temporal_splits(df, experiment_name)

    # Get feature and target columns from processed data
    # Apply feature selection to get feature columns
    from ..data.features import select_features
    train_processed = select_features(train_df, feature_group)
    target_columns = get_target_columns_from_df(train_processed, level)
    feature_columns = get_feature_columns(train_processed, feature_group)

    return train_df, val_df, test_df, feature_columns, target_columns


def run_single_model_backtest(
    model_name: str,
    experiment_name: str,
    feature_group: str = "ALL_FEATURES",
    level: str = "region",
    model_kwargs: dict | None = None,
    normalize_predictions: bool = True,
) -> dict[str, Any]:
    """
    Run backtest for a single model on a single experiment.

    Returns:
        Dictionary with metrics, predictions, and metadata
    """
    model_kwargs = model_kwargs or {}

    # Prepare data
    train_df, val_df, test_df, feature_columns, target_columns = prepare_data_for_experiment(
        experiment_name, feature_group, level
    )

    # Handle empty val set
    has_val = len(val_df) > 0

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
    y_test = test_scaled[target_columns]

    # Fit model - baselines need year/region_id and target columns (per party), other models use feature_columns
    is_baseline = model_name in ["NaivePreviousElection", "HistoricalMean", "WeightedHistoricalMean"]
    
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
                if hasattr(single_model, "fit") and "eval_set" in single_model.fit.__code__.co_varnames:
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
    train_pred_df = pd.DataFrame(train_pred, columns=[f"{c}_pred" for c in target_columns], index=train_df.index)
    test_pred_df = pd.DataFrame(test_pred, columns=[f"{c}_pred" for c in target_columns], index=test_df.index)

    # Normalize predictions if needed (compositional constraint)
    if normalize_predictions:
        train_pred_df = normalize_compositional(train_pred_df, target_columns)
        test_pred_df = normalize_compositional(test_pred_df, target_columns)

    # Combine actuals and predictions
    train_results = pd.concat([train_df[target_columns], train_pred_df], axis=1)
    test_results = pd.concat([test_df[target_columns], test_pred_df], axis=1)

    # Compute metrics
    train_metrics = party_metrics(train_results, target_columns, pred_suffix="_pred", actual_suffix="")
    test_metrics = party_metrics(test_results, target_columns, pred_suffix="_pred", actual_suffix="")

    # Federal aggregation
    weight_col = "electorate" if "electorate" in train_df.columns else None
    if weight_col is None and "population" in train_df.columns:
        weight_col = "population"

    train_federal = federal_aggregation(train_results, target_columns, weight_column=weight_col or "electorate") if weight_col else {}
    test_federal = federal_aggregation(test_results, target_columns, weight_column=weight_col or "electorate") if weight_col else {}

    # Error distributions
    test_error_dist = error_distribution(test_results, target_columns, pred_suffix="_pred", actual_suffix="")

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
                results.append({
                    "model": model_name,
                    "experiment": experiment_name,
                    "feature_group": feature_group,
                    "error": str(e),
                })

    return results