"""Tests for federal aggregation weights (built from RED precinct data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_electoral_weights, load_region
from src.evaluation.metrics import federal_aggregation


class TestElectoralWeights:
    """Tests for electoral weights used in federal aggregation."""

    def test_weights_load_and_shape(self):
        """Weights cover every parliamentary region-event (83 regions x 5 years)."""
        w = load_electoral_weights()
        assert {"region_id", "year", "type", "electorate", "turnout", "valid"}.issubset(w.columns)
        parl = w[w["type"] == "parl"]
        # 2003, 2007, 2011, 2016, 2021 each have 83 regions.
        counts = parl.groupby("year")["region_id"].nunique()
        for year in [2003, 2007, 2011, 2016, 2021]:
            assert counts.get(year) == 83, f"year {year}: {counts.get(year)} regions"

    def test_weights_match_master_region_ids(self):
        """Weight region_ids must align with the master dataset region_ids."""
        master = load_region("ALL_FEATURES", apply_feature_selection=False)
        w = load_electoral_weights()
        w_regions = set(w["region_id"])
        master_regions = set(master["region_id"])
        assert w_regions == master_regions, f"region_id mismatch: {w_regions ^ master_regions}"

    def test_federal_aggregation_weighted_by_valid(self):
        """Federal aggregation uses valid-vote weights, not uniform."""
        w = load_electoral_weights()
        weights = w[(w["type"] == "parl") & (w["year"] == 2016)][
            ["region_id", "year", "type", "valid"]
        ]

        master = load_region("ALL_FEATURES", apply_feature_selection=False)
        df = master[master["year"] == 2016].merge(weights, on=["region_id", "year", "type"])
        df["UR_share_pred"] = df["UR_share"]

        fed = federal_aggregation(df, ["UR_share"], weight_column="valid")
        assert "UR_share" in fed
        # Prediction equals actual, so weighted federal error is ~0.
        assert abs(fed["UR_share"]["federal_error"]) < 1e-9

        uniform = df["UR_share"].mean()
        weighted = fed["UR_share"]["federal_actual"]
        assert not np.isclose(uniform, weighted), (
            "uniform and weighted federal shares should differ for 2016"
        )

    def test_federal_aggregation_raises_without_weights(self):
        """Raises when no weight column is available."""
        df = pd.DataFrame({"UR_share": [50.0], "UR_share_pred": [50.0]})
        with pytest.raises(ValueError):
            federal_aggregation(df, ["UR_share"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
