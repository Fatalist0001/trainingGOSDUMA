"""Data loading utilities for region and precinct level datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.io import load_yaml_config


def _get_dataset_root() -> Path:
    """Get dataset root from config or environment."""
    import os

    root = os.environ.get("DATASET_ROOT")
    if root:
        return Path(root)
    # Fallback to config
    config = load_yaml_config("config/paths.yaml")
    return Path(config["dataset_root"])


def load_region(
    feature_group: str = "ALL_FEATURES",
    dataset_root: str | Path | None = None,
    apply_feature_selection: bool = True,
) -> pd.DataFrame:
    """
    Load region-level master dataset.

    Args:
        feature_group: One of "ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"
        dataset_root: Override dataset root path
        apply_feature_selection: Whether to apply feature group selection (default True)

    Returns:
        DataFrame with region-level features and targets
    """
    root = Path(dataset_root) if dataset_root else _get_dataset_root()
    file_path = root / "data" / "processed" / "master_region_election.parquet"
    df = pd.read_parquet(file_path)
    if apply_feature_selection:
        return select_features(df, feature_group)
    return df


def load_precinct(
    feature_group: str = "ALL_FEATURES",
    dataset_root: str | Path | None = None,
    apply_feature_selection: bool = True,
) -> pd.DataFrame:
    """
    Load precinct-level dataset.

    Args:
        feature_group: One of "ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"
        dataset_root: Override dataset root path
        apply_feature_selection: Whether to apply feature group selection (default True)

    Returns:
        DataFrame with precinct-level features and targets
    """
    root = Path(dataset_root) if dataset_root else _get_dataset_root()
    file_path = root / "data" / "processed" / "elections_precinct.parquet"
    df = pd.read_parquet(file_path)
    if apply_feature_selection:
        return select_features(df, feature_group)
    return df


def load_elections_metadata(dataset_root: str | Path | None = None) -> pd.DataFrame:
    """Load elections metadata (years, types)."""
    root = Path(dataset_root) if dataset_root else _get_dataset_root()
    file_path = root / "metadata" / "elections.csv"
    return pd.read_csv(file_path)


def load_electoral_weights(dataset_root: str | Path | None = None) -> pd.DataFrame:
    """Load federal aggregation weights (electorate/turnout/valid) by region-event.

    Built from RED precinct data by ``scripts/data/build_electoral_weights.py``.
    Columns: region_id, year, type, electorate, turnout, valid, invalid.
    """
    root = Path(dataset_root) if dataset_root else _get_dataset_root()
    file_path = root / "data" / "processed" / "electoral_weights.parquet"
    return pd.read_parquet(file_path)


def get_party_list(level: Literal["region", "precinct"] = "region") -> list[str]:
    """
    Get list of parties dynamically from data.

    Args:
        level: "region" or "precinct"

    Returns:
        List of party column names (vote share columns)
    """
    if level == "region":
        df = load_region("ALL_FEATURES")
        # Region level has UR_share, KPRF_share, LDPR_share
        party_cols = [c for c in df.columns if c.endswith("_share") and not c.startswith("target_")]
    else:
        df = load_precinct("ALL_FEATURES")
        # Precinct level has more parties
        party_cols = [c for c in df.columns if c.endswith("_share") and not c.startswith("target_")]

    return sorted(party_cols)


def get_available_years(
    election_type: str = "parliamentary",
    dataset_root: str | Path | None = None,
) -> list[int]:
    """Get available election years from metadata."""
    meta = load_elections_metadata(dataset_root)
    years = meta[meta["election_type"] == election_type]["year"].tolist()
    return sorted([int(y) for y in years])


def select_features(df: pd.DataFrame, feature_group: str) -> pd.DataFrame:
    """
    Select features based on feature group configuration.

    Args:
        df: Input DataFrame
        feature_group: One of "ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"

    Returns:
        DataFrame with selected features
    """
    from .features import select_features as _select_features

    return _select_features(df, feature_group)
