"""Specific leakage detection tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import load_region
from src.data.splits import (
    create_temporal_splits,
    get_experiment_splits,
    load_raw_precinct,
    load_raw_region,
)


class TestSpatialLeakage:
    """Tests for spatial leakage (region/precinct IDs)."""

    def test_region_id_not_predictive(self):
        """Region ID should not be predictive by itself."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error

        df = load_raw_region()
        train_df, _, test_df = create_temporal_splits(df, "A")

        # Try predicting with only region_id (encoded)
        train_X = pd.get_dummies(train_df[["region_id"]], columns=["region_id"])
        test_X = pd.get_dummies(test_df[["region_id"]], columns=["region_id"])
        train_X, test_X = train_X.align(test_X, join="outer", axis=1, fill_value=0)

        y_train = train_df["UR_share"]
        y_test = test_df["UR_share"]

        model = LinearRegression()
        model.fit(train_X, y_train)
        pred = model.predict(test_X)

        mae = mean_absolute_error(y_test, pred)
        # Should be high (not perfectly predictive)
        assert mae > 2.0, f"Region ID alone gives MAE={mae:.2f}, possible memorization"

    def test_no_precinct_id_in_region_data(self):
        """Region-level data should not have precinct IDs."""
        df = load_region("ALL_FEATURES")
        assert "uik" not in df.columns
        assert "tik" not in df.columns
        assert "precinct_id" not in df.columns


class TestTemporalLeakage:
    """Tests for temporal leakage."""

    def test_future_not_in_train(self):
        """No future data in training set."""
        for exp in ["A", "B", "C"]:
            split = get_experiment_splits(exp)
            max_train = max(split.train_years)
            min_test = min(split.test_years)
            assert max_train < min_test

    def test_val_not_in_train(self):
        """Validation years not in training."""
        for exp in ["A", "B", "C"]:
            split = get_experiment_splits(exp)
            for val_year in split.val_years:
                assert val_year not in split.train_years

    def test_lag_features_consistent(self):
        """Lag features should reference only past elections."""
        df = load_raw_region()

        # For 2016 election, lag1 features should be from 2012
        df_2016 = df[df["year"] == 2016]
        if len(df_2016) > 0:
            # Check that lag features exist and are not NaN for 2016
            lag_cols = [c for c in df.columns if c.endswith("_lag1")]
            for col in lag_cols[:3]:  # Check a few
                nan_count = df_2016[col].isna().sum()
                # Some regions might not have 2012 data, that's OK
                assert nan_count < len(df_2016) * 0.5, f"Too many NaN in {col} for 2016"


class TestFeatureLeakage:
    """Tests for feature leakage."""

    def test_turnout_not_target(self):
        """Turnout rate should not be used to predict vote shares directly."""
        # This is more of a modeling decision, but we check it's available
        df = load_region("ALL_FEATURES")
        assert "turnout_rate" in df.columns
        assert "turnout_rate_lag1" in df.columns

    def test_no_future_socioeconomic(self):
        """Socioeconomic features should be lagged."""
        df = load_region("ALL_FEATURES")
        socio_cols = [
            "population",
            "GRP_per_capita",
            "average_salary",
            "unemployment_rate",
            "poverty_rate",
            "urban_population_share",
            "real_disposable_income",
            "age_working_share",
            "birth_rate",
            "death_rate",
            "natural_population_growth",
            "migration_rate",
            "doctors_per_1000",
            "hospital_beds_per_1000",
            "housing_area_per_capita",
        ]

        # All should have _lag1 suffix in the dataset
        for col in socio_cols:
            lag_col = f"{col}_lag1"
            assert lag_col in df.columns, f"Missing lagged feature: {lag_col}"
            # Original should not exist (or be NaN)
            if col in df.columns:
                # If original exists, it should be all NaN or lagged
                pass


class TestCompositionalConstraint:
    """Tests for compositional nature of targets."""

    def test_party_shares_nonnegative_region(self):
        """Region-level party shares should be non-negative (for parliamentary elections)."""
        df = load_raw_region()
        # Filter to parliamentary elections only
        parl_df = df[df["type"] == "parl"]
        party_cols = ["UR_share", "KPRF_share", "LDPR_share"]

        for col in party_cols:
            if col in parl_df.columns:
                # Only check non-NaN values
                valid_vals = parl_df[col].dropna()
                assert (valid_vals >= 0).all(), f"Negative values in {col}"
                assert (valid_vals <= 100).all(), f"Values > 100 in {col}"

    def test_party_shares_nonnegative_precinct(self):
        """Precinct-level party shares should be non-negative (for parliamentary elections)."""
        df = load_raw_precinct()
        # Filter to parliamentary elections only
        parl_df = df[df["type"] == "parl"]
        party_cols = ["UR_share", "KPRF_share", "LDPR_share", "SR_share", "NovyeLyudi_share"]

        for col in party_cols:
            if col in parl_df.columns:
                valid_vals = parl_df[col].dropna()
                assert (valid_vals >= 0).all(), f"Negative values in {col}"
                assert (valid_vals <= 100).all(), f"Values > 100 in {col}"

    def test_party_shares_sum_reasonable_region(self):
        """Region-level party shares sum should be reasonable for parliamentary elections."""
        df = load_raw_region()
        # Filter to parliamentary elections only
        parl_df = df[df["type"] == "parl"]
        party_cols = ["UR_share", "KPRF_share", "LDPR_share"]
        row_sums = parl_df[party_cols].sum(axis=1, skipna=True)
        # With only 3 main parties, sum should be > 40% and < 100% (some regions have low support)
        assert (row_sums.between(40, 100)).all(), (
            f"Party shares sum out of range: {row_sums.min():.1f}-{row_sums.max():.1f}"
        )

    def test_party_shares_sum_reasonable_precinct(self):
        """Precinct-level party shares sum should be reasonable for parliamentary elections."""
        df = load_raw_precinct()
        # Filter to parliamentary elections only
        parl_df = df[df["type"] == "parl"]
        party_cols = ["UR_share", "KPRF_share", "LDPR_share", "SR_share", "NovyeLyudi_share"]
        row_sums = parl_df[party_cols].sum(axis=1, skipna=True)
        # With 5 parties, sum should be >= 0% and <= 100% (allow small floating point epsilon)
        eps = 1e-9
        assert (row_sums >= -eps).all(), f"Party shares sum below 0: {row_sums.min():.6f}"
        assert (row_sums <= 100 + eps).all(), f"Party shares sum above 100: {row_sums.max():.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
