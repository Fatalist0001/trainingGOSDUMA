"""Feature selection utilities based on feature group configuration."""
from __future__ import annotations

import fnmatch
import yaml
from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.io import load_yaml_config


def _load_feature_config() -> dict:
    """Load feature group configuration."""
    return load_yaml_config("config/features.yaml")


def get_feature_groups() -> dict[str, dict]:
    """Get all feature group definitions."""
    config = _load_feature_config()
    return config["feature_groups"]


def get_forbidden_features() -> list[str]:
    """Get list of forbidden features (leakage risk)."""
    config = _load_feature_config()
    return config.get("forbidden_features", [])


def get_target_columns(level: Literal["region", "precinct"] = "region") -> list[str]:
    """Get target column names for a given level."""
    config = _load_feature_config()
    key = f"{level}_level"
    return config.get("targets", {}).get(key, [])


def _match_patterns(columns: list[str], patterns: list[str]) -> list[str]:
    """Match columns against glob patterns."""
    matched = set()
    for pattern in patterns:
        matched.update(fnmatch.filter(columns, pattern))
    return sorted(matched)


def _exclude_patterns(columns: list[str], patterns: list[str]) -> list[str]:
    """Exclude columns matching glob patterns."""
    excluded = set()
    for pattern in patterns:
        excluded.update(fnmatch.filter(columns, pattern))
    return sorted(set(columns) - excluded)


def select_features(
    df: pd.DataFrame,
    feature_group: str = "ALL_FEATURES",
) -> pd.DataFrame:
    """
    Select features from DataFrame based on feature group.

    Args:
        df: Input DataFrame
        feature_group: One of "ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"

    Returns:
        DataFrame with selected features (including targets)
    """
    config = _load_feature_config()
    groups = config["feature_groups"]

    if feature_group not in groups:
        raise ValueError(
            f"Unknown feature group: {feature_group}. "
            f"Available: {list(groups.keys())}"
        )

    group_config = groups[feature_group]
    include_patterns = group_config.get("include_patterns", ["*"])
    exclude_patterns = group_config.get("exclude_patterns", [])

    all_columns = df.columns.tolist()

    # Start with all columns matching include patterns
    included = _match_patterns(all_columns, include_patterns)

    # Remove excluded patterns
    included = _exclude_patterns(included, exclude_patterns)

    # Always keep target columns
    target_cols = get_target_columns("region") + get_target_columns("precinct")
    for tc in target_cols:
        if tc in all_columns and tc not in included:
            included.append(tc)

    # Ensure we have the columns
    missing = [c for c in included if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    return df[included].copy()


def get_feature_columns(
    df: pd.DataFrame,
    feature_group: str = "ALL_FEATURES",
) -> list[str]:
    """
    Get feature column names (excluding targets).

    Args:
        df: Input DataFrame
        feature_group: Feature group name

    Returns:
        List of feature column names
    """
    target_cols = get_target_columns("region") + get_target_columns("precinct")
    selected = select_features(df, feature_group)
    return [c for c in selected.columns if c not in target_cols]


def get_target_columns_from_df(
    df: pd.DataFrame,
    level: Literal["region", "precinct"] = "region",
) -> list[str]:
    """Get target columns that exist in the given DataFrame."""
    target_cols = get_target_columns(level)
    return [c for c in target_cols if c in df.columns]