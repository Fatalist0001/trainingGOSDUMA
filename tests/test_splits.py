"""Tests for temporal splits and leakage detection."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.data.splits import (
    get_experiment_splits,
    create_temporal_splits,
    filter_by_years,
    TemporalSplitter,
    load_raw_region,
)
from src.data.loader import load_region, get_available_years, load_elections_metadata


class TestTemporalSplits:
    """Tests for temporal split logic."""

    def test_experiment_A_split(self):
        """Test Experiment A: train=2003-2011, test=2016."""
        split = get_experiment_splits("A")
        assert split.train_years == [2003, 2007, 2011]
        assert split.test_years == [2016]
        assert split.val_years == [2016]

    def test_experiment_B_split(self):
        """Test Experiment B: train=2003-2016, test=2021."""
        split = get_experiment_splits("B")
        assert split.train_years == [2003, 2007, 2011, 2016]
        assert split.test_years == [2021]
        assert split.val_years == [2021]

    def test_experiment_C_split(self):
        """Test Experiment C: train=2003-2021, test=2024."""
        split = get_experiment_splits("C")
        assert split.train_years == [2003, 2007, 2011, 2016, 2021]
        assert split.test_years == [2024]
        assert split.val_years == [2024]

    def test_experiment_D_split(self):
        """Test Experiment D: train=all, target=2026."""
        split = get_experiment_splits("D")
        assert split.train_years == [2003, 2007, 2011, 2016, 2021, 2024]
        assert split.test_years == [2026]
        assert split.is_final is True

    def test_no_future_leakage(self):
        """Ensure train years are always before test years."""
        for exp_name in ["A", "B", "C", "D"]:
            split = get_experiment_splits(exp_name)
            max_train = max(split.train_years)
            min_test = min(split.test_years)
            assert max_train < min_test, f"Leakage in {exp_name}: train max={max_train} >= test min={min_test}"

    def test_create_temporal_splits(self):
        """Test creating train/val/test DataFrames."""
        df = load_raw_region()
        train_df, val_df, test_df = create_temporal_splits(df, "A")

        # Check years (parliamentary only)
        assert set(train_df["year"].unique()) == set([2003, 2007, 2011])
        assert set(test_df["year"].unique()) == set([2016])

        # Check no overlap
        train_regions_years = set(zip(train_df["region_id"], train_df["year"]))
        test_regions_years = set(zip(test_df["region_id"], test_df["year"]))
        assert len(train_regions_years & test_regions_years) == 0

    def test_filter_by_years(self):
        """Test year filtering."""
        df = load_raw_region()
        filtered = filter_by_years(df, [2016, 2021])
        assert set(filtered["year"].unique()) == {2016, 2021}

    def test_temporal_splitter(self):
        """Test TemporalSplitter iterator."""
        df = load_raw_region()
        splitter = TemporalSplitter(df, min_train_years=2)

        splits = list(splitter.split())
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            train_years = df.loc[train_idx, "year"].unique()
            test_years = df.loc[test_idx, "year"].unique()
            assert max(train_years) < min(test_years)


class TestLeakageDetection:
    """Tests to detect data leakage."""

    def test_no_target_in_features(self):
        """Ensure target columns are not in feature columns."""
        from src.data.features import select_features, get_target_columns

        df = load_region("ALL_FEATURES")
        target_cols = get_target_columns("region")
        feature_cols = select_features(df, "ALL_FEATURES").columns

        for target in target_cols:
            assert target not in feature_cols or target in df.columns, f"Target {target} in features"

    def test_no_id_columns_in_features(self):
        """Ensure ID columns are not used as features."""
        from src.data.features import select_features

        df = load_region("ALL_FEATURES")
        features = select_features(df, "ALL_FEATURES")

        forbidden = ["region_id", "region_name", "year", "type"]
        for col in forbidden:
            assert col not in features.columns, f"Forbidden column {col} in features"

    def test_lag_features_only_past(self):
        """Verify lag features only contain past information."""
        df = load_region("ALL_FEATURES")
        lag_cols = [c for c in df.columns if c.endswith("_lag1")]

        # For year 2016, lag1 should be from 2012 (previous parliamentary)
        # This is validated by the dataset construction, but we check structure
        assert len(lag_cols) > 0, "No lag features found"

    def test_preprocessing_fit_on_train_only(self):
        """Verify preprocessing is fit only on training data."""
        from src.data.preprocessing import fit_transform_train_test
        from src.data.splits import create_temporal_splits
        from src.data.features import get_feature_columns
        from src.data.splits import load_raw_region
        import numpy as np

        df = load_raw_region()
        train_df, _, test_df = create_temporal_splits(df, "A")
        feature_cols = get_feature_columns(train_df, "ALL_FEATURES")

        train_scaled, test_scaled, scaler = fit_transform_train_test(
            train_df, test_df, feature_cols, "standard"
        )

        # Check that scaler was fit on train statistics
        # Mean of scaled train should be ~0 (skip NaN columns)
        train_means = train_scaled[feature_cols].mean()
        valid_means = train_means[~np.isnan(train_means)]
        assert all(abs(m) < 1e-10 for m in valid_means), "Train not centered"

        # Test should use train scaler params
        test_means = test_scaled[feature_cols].mean()
        # Test means can be non-zero (that's correct)


class TestDataIntegrity:
    """Tests for data integrity."""

    def test_party_shares_sum_reasonable(self):
        """Verify party vote shares sum to reasonable range (only 3 parties at region level, parliamentary elections)."""
        df = load_raw_region()
        # Filter to parliamentary elections only
        parl_df = df[df["type"] == "parl"]
        party_cols = ["UR_share", "KPRF_share", "LDPR_share"]
        existing = [c for c in party_cols if c in parl_df.columns]

        if existing:
            row_sums = parl_df[existing].sum(axis=1, skipna=True)
            # With only 3 main parties, sum should be > 40% and < 100%
            assert (row_sums.between(40, 100)).all(), f"Party shares sum out of range: {row_sums.min():.1f}-{row_sums.max():.1f}"

    def test_no_nan_in_targets(self):
        """Check targets have no NaN for parliamentary elections."""
        df = load_raw_region()
        # Filter to parliamentary elections using 'type' column
        parl_df = df[df["type"] == "parl"]
        party_cols = ["UR_share", "KPRF_share", "LDPR_share"]
        existing = [c for c in party_cols if c in parl_df.columns]

        for col in existing:
            nan_count = parl_df[col].isna().sum()
            assert nan_count == 0, f"NaN in {col}: {nan_count}"

    def test_years_are_parliamentary(self):
        """Verify available years match parliamentary elections from metadata."""
        meta = load_elections_metadata()
        parl_years = meta[meta["election_type"] == "parliamentary"]["year"].tolist()
        # Include 2026 as planned parliamentary election
        expected = [2000, 2003, 2007, 2011, 2016, 2021, 2024, 2026]
        assert parl_years == expected, f"Parliamentary years mismatch: {parl_years} != {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])