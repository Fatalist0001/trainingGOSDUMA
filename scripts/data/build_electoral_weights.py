#!/usr/bin/env python3
"""Build federal aggregation weights from RED precinct data.

RED (precinct-level) is aggregated to region x election-event, summing the
electorate, turnout and valid-votes totals used to weight regional predictions
to the federal level (see ``src/evaluation/metrics.federal_aggregation``).

The result is saved as a sidecar parquet so the runtime pipeline never needs to
read the heavy .rds file again:

    data/processed/electoral_weights.parquet

Columns:
    region_id, year, type, electorate, turnout, valid, invalid

Region ids match the master dataset (``region_mapping.csv``'s ``red_region_id``,
which is the same integer the master uses as ``region_id``).

Run: python scripts/data/build_electoral_weights.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml_config

RED_RDS = os.path.join("data", "raw", "red", "01_elec_offline.rds")
OUTPUT = os.path.join("data", "processed", "electoral_weights.parquet")


def main() -> None:
    cfg = load_yaml_config("config/paths.yaml")
    dataset_root = Path(cfg["dataset_root"])

    mapping = pd.read_csv(dataset_root / "metadata" / "region_mapping.csv")
    name_to_id = dict(zip(mapping["red_region_name"], mapping["red_region_id"]))
    print(f"[weights] region mapping: {len(mapping)} regions")

    import pyreadr

    res = pyreadr.read_r(dataset_root / RED_RDS)
    df = next(iter(res.values()))
    print(f"[weights] RED rows: {len(df):,}")

    df["region_id"] = df["region"].map(name_to_id)
    df = df[df["region_id"].notna()].copy()
    df["year"] = df["year"].astype(int)

    agg = (
        df.groupby(["region_id", "year", "type"], as_index=False)
        .agg(
            electorate=("electorate", "sum"),
            turnout=("turnout", "sum"),
            valid=("valid", "sum"),
            invalid=("invalid", "sum"),
        )
        .sort_values(["region_id", "year", "type"])
    )

    out_path = dataset_root / OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    print(f"[weights] saved {len(agg)} region-events -> {out_path}")
    print(agg.groupby(["year", "type"]).size().to_string())


if __name__ == "__main__":
    main()
