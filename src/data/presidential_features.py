"""Presidential-election lag features for parliamentary forecasting.

Each parliamentary election row gets the turnout and the leading-candidate share
of the *most recent* presidential election that happened strictly before it:

    2003 -> 2000, 2007 -> 2004, 2011 -> 2008,
    2016 -> 2012, 2021 -> 2018, 2026 -> 2024

These are genuine lag features (presidential results are known before the target
parliamentary election), so they carry no leakage. They also let the 2024
presidential election inform the 2026 parliamentary forecast.
"""

from __future__ import annotations

import pandas as pd


def _most_recent_pres_year(pres_years: list[int], parl_year: int) -> int | None:
    """Return the largest presidential year strictly before ``parl_year``."""
    prior = [y for y in pres_years if y < parl_year]
    return max(prior) if prior else None


def add_presidential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``pres_turnout_lag`` and ``pres_leading_candidate_share_lag``.

    Adds the turnout rate and the leading-candidate vote share of the most
    recent past presidential election to each parliamentary row of ``df``.
    Presidential rows themselves and rows without a prior presidential election
    get NaN for the two new columns. The original index and row order are kept.

    Args:
        df: Master DataFrame containing ``region_id``, ``year``, ``type`` and
            presidential rows with ``turnout_rate`` and
            ``leading_candidate_share`` populated.

    Returns:
        A copy of ``df`` with the two new columns added.
    """
    if "pres_turnout_lag" in df.columns and "pres_leading_candidate_share_lag" in df.columns:
        return df

    df = df.copy()
    df["pres_turnout_lag"] = pd.NA
    df["pres_leading_candidate_share_lag"] = pd.NA

    pres = df[df["type"] == "pres"]
    pres_years = sorted(pres["year"].unique())

    pres_view = pres[["region_id", "year", "turnout_rate", "leading_candidate_share"]]
    pres_view = pres_view.rename(
        columns={
            "turnout_rate": "pres_turnout_lag",
            "leading_candidate_share": "pres_leading_candidate_share_lag",
        }
    )
    pres_view = pres_view.set_index(["region_id", "year"])

    parl = df[df["type"] == "parl"]
    if parl.empty:
        return df

    year_map = {
        py: _most_recent_pres_year(pres_years, int(py)) for py in sorted(parl["year"].unique())
    }

    merged = parl[["region_id", "year"]].copy()
    merged["pres_year"] = merged["year"].map(year_map)
    merged = merged.merge(
        pres_view.reset_index().rename(columns={"year": "pres_year"}),
        on=["region_id", "pres_year"],
        how="left",
    ).drop(columns=["pres_year"])

    parl_idx = parl.index
    df.loc[parl_idx, "pres_turnout_lag"] = merged["pres_turnout_lag"].values
    df.loc[parl_idx, "pres_leading_candidate_share_lag"] = merged[
        "pres_leading_candidate_share_lag"
    ].values
    return df
