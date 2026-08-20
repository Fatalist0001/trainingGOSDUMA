"""Temporal splitting utilities for rolling backtest experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..utils.io import load_yaml_config
from .loader import load_precinct, load_region


def load_raw_region() -> pd.DataFrame:
    """Load region data without feature selection (for splitting).

    Includes presidential lag features (``pres_turnout_lag``,
    ``pres_leading_candidate_share_lag``) computed from past presidential rows.
    """
    from .presidential_features import add_presidential_features

    return add_presidential_features(load_region(apply_feature_selection=False))


def load_raw_precinct() -> pd.DataFrame:
    """Load precinct data without feature selection (for splitting)."""
    return load_precinct(apply_feature_selection=False)


@dataclass
class TemporalSplit:
    """Represents a single temporal train/val/test split."""

    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    experiment_name: str
    election_type: str = "parliamentary"
    is_final: bool = False


def load_experiment_config() -> dict:
    """Load experiment configuration from YAML."""
    return load_yaml_config("config/experiments.yaml")


def get_experiment_splits(experiment_name: str) -> TemporalSplit:
    """
    Get temporal split for a named experiment.

    Args:
        experiment_name: One of "A", "B", "C", "D"

    Returns:
        TemporalSplit with train/val/test years
    """
    config = load_experiment_config()
    experiments = config["experiments"]

    if experiment_name not in experiments:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. Available: {list(experiments.keys())}"
        )

    exp = experiments[experiment_name]
    return TemporalSplit(
        train_years=exp["train_years"],
        val_years=exp.get("val_years", []),
        test_years=exp["test_years"],
        experiment_name=exp["name"],
        election_type=exp.get("election_type", "parliamentary"),
        is_final=exp.get("is_final", False),
    )


def get_all_experiment_splits() -> dict[str, TemporalSplit]:
    """Get all experiment splits as a dictionary."""
    config = load_experiment_config()
    return {
        name: TemporalSplit(
            train_years=exp["train_years"],
            val_years=exp.get("val_years", []),
            test_years=exp["test_years"],
            experiment_name=exp["name"],
            election_type=exp.get("election_type", "parliamentary"),
        )
        for name, exp in config["experiments"].items()
    }


def filter_by_years(
    df: pd.DataFrame,
    years: list[int],
    year_column: str = "year",
    election_type_column: str = "type",
    election_type: str = "parl",
) -> pd.DataFrame:
    """
    Filter DataFrame to specific years and election type.

    Args:
        df: Input DataFrame
        years: List of years to keep
        year_column: Name of year column
        election_type_column: Name of election type column
        election_type: Election type to filter ("parliamentary" or "pres")

    Returns:
        Filtered DataFrame
    """
    mask = df[year_column].isin(years)
    if election_type_column in df.columns:
        mask = mask & (df[election_type_column] == election_type)
    return df[mask].copy()


def create_temporal_splits(
    df: pd.DataFrame,
    experiment_name: str,
    year_column: str = "year",
    election_type_column: str = "type",
    train_years_override: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test DataFrames for a given experiment.

    Args:
        df: Full DataFrame
        experiment_name: Experiment name ("A", "B", "C", "D")
        year_column: Name of year column
        election_type_column: Name of election type column
        train_years_override: Optional explicit list of train years (for
            history-depth ablation). Overrides the experiment's config.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    split = get_experiment_splits(experiment_name)
    train_years = train_years_override or split.train_years

    train_df = filter_by_years(
        df, train_years, year_column, election_type_column, split.election_type
    )
    val_df = (
        filter_by_years(df, split.val_years, year_column, election_type_column, split.election_type)
        if split.val_years
        else pd.DataFrame(columns=df.columns)
    )
    test_df = filter_by_years(
        df, split.test_years, year_column, election_type_column, split.election_type
    )

    return train_df, val_df, test_df


class TemporalSplitter:
    """
    Iterator for rolling temporal splits.

    Generates splits where each subsequent split adds more training data
    and tests on the next election year.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        year_column: str = "year",
        election_type_column: str = "type",
        election_type: str = "parl",
        min_train_years: int = 3,
    ):
        self.df = df
        self.year_column = year_column
        self.election_type_column = election_type_column
        self.election_type = election_type
        self.min_train_years = min_train_years

        # Get all available parliamentary election years
        elections = load_yaml_config("config/experiments.yaml")
        self.all_years = sorted(
            set(
                y
                for exp in elections["experiments"].values()
                for y in exp["train_years"] + exp.get("val_years", []) + exp["test_years"]
            )
        )

    def split(self):
        """Generate (train_idx, test_idx) pairs for each test year."""
        parliamentary_years = [y for y in self.all_years if self._is_parliamentary_year(y)]

        for i, test_year in enumerate(parliamentary_years):
            # Train on all previous parliamentary elections
            train_years = parliamentary_years[:i]

            if len(train_years) < self.min_train_years:
                continue

            train_mask = self.df[self.year_column].isin(train_years)
            test_mask = self.df[self.year_column] == test_year

            if self.election_type_column in self.df.columns:
                train_mask &= self.df[self.election_type_column] == self.election_type
                test_mask &= self.df[self.election_type_column] == self.election_type

            train_idx = self.df.index[train_mask].tolist()
            test_idx = self.df.index[test_mask].tolist()

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def _is_parliamentary_year(self, year: int) -> bool:
        """Check if year has parliamentary election."""
        meta = load_yaml_config("config/experiments.yaml")
        for exp in meta["experiments"].values():
            if year in exp.get("train_years", []) + exp.get("val_years", []) + exp.get(
                "test_years", []
            ):
                if exp.get("election_type") == "parl":
                    return True
        return False

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return sum(1 for _ in self.split())


def get_internal_validation_years(experiment_name: str) -> list[int]:
    """Get validation years for internal hyperparameter tuning."""
    config = load_experiment_config()
    temporal_val = config.get("internal_validation", {}).get("temporal_val_years", {})
    return temporal_val.get(experiment_name, [])
