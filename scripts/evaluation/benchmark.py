#!/usr/bin/env python
"""Aggregate benchmark results into summary tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import get_output_dirs, load_results, save_benchmark_table


def aggregate_benchmark(
    experiment: str | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate results from all experiment runs into a benchmark table.

    Returns:
        DataFrame with models as rows, years as columns (MAE values)
    """
    dirs = get_output_dirs()
    results_dir = dirs["results_dir"]

    # Find all result files
    result_files = list(results_dir.glob("*.json"))

    all_results = []
    for f in result_files:
        try:
            data = load_results(f.stem)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict):
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if not all_results:
        print("No results found!")
        return pd.DataFrame()

    # Filter by experiment if specified
    if experiment:
        all_results = [r for r in all_results if r.get("experiment") == experiment]

    # Filter by models if specified
    if models:
        all_results = [r for r in all_results if r.get("model") in models]

    # Build benchmark table
    rows = []
    for r in all_results:
        if "error" in r:
            continue

        model = r.get("model", "unknown")
        exp = r.get("experiment", "unknown")
        feat_group = r.get("feature_group", "unknown")
        test_metrics = r.get("test_metrics", [])

        # Get test years from experiment config
        from src.data.splits import get_experiment_splits

        try:
            split = get_experiment_splits(exp)
            test_years = split.test_years
        except Exception:
            test_years = ["test"]

        # Average MAE across parties for each test year
        row = {"model": model, "experiment": exp, "feature_group": feat_group}
        for i, year in enumerate(test_years):
            # For now, use average across parties
            party_maes = [m["mae"] for m in test_metrics]
            row[f"year_{year}"] = sum(party_maes) / len(party_maes) if party_maes else None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Pivot to have years as columns
    if not df.empty:
        # Create a combined model key
        df["model_key"] = df["model"] + " (" + df["feature_group"] + ")"

        # Pivot
        pivot = df.pivot_table(
            index="model_key",
            columns="experiment",
            values=[c for c in df.columns if c.startswith("year_")],
            aggfunc="first",
        )
        pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
        pivot = pivot.reset_index()
        pivot = pivot.rename(columns={"model_key": "Model"})
    else:
        pivot = df

    return pivot


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print benchmark table in nice format."""
    if df.empty:
        print("No data to display")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK TABLE (MAE in percentage points)")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark table")
    parser.add_argument("--experiment", default=None, help="Filter by experiment")
    parser.add_argument("--models", nargs="+", default=None, help="Filter by models")
    parser.add_argument("--save", action="store_true", help="Save to CSV")
    args = parser.parse_args()

    df = aggregate_benchmark(experiment=args.experiment, models=args.models)
    print_benchmark_table(df)

    if args.save and not df.empty:
        exp_name = args.experiment or "all"
        path = save_benchmark_table(df, exp_name)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
