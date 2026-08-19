#!/usr/bin/env python
"""Final 2026 forecast for experiments C and D.

Trains the best model (by default WeightedHistoricalMean) on all available
history and forecasts the 2026 State Duma election shares per region/party.
There is no ground truth for 2026, so this produces forecasts only (no MAE).
Pass ``--model Transformer`` to forecast with the best temporal model instead.

Quick start:
    uv run python scripts/predict_2026.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.backtest import BASELINE_NAMES, forecast_baseline, forecast_temporal
from src.utils.io import get_output_dirs


def federal_forecast(pred_df: pd.DataFrame, target_columns: list[str], weight_col: str) -> dict:
    """Weighted federal-level forecast (percent of vote, by party)."""
    pred_cols = [f"{c}_pred" for c in target_columns]
    weights = pred_df[weight_col] if weight_col in pred_df.columns else None
    if weights is None:
        weights = pd.Series(1.0, index=pred_df.index)
    w = weights.fillna(weights.mean()) if weights.notna().any() else weights
    total = w.sum()
    return {
        c: float((pred_df[p] * w).sum() / total) for c, p in zip(target_columns, pred_cols)
    }


def main():
    parser = argparse.ArgumentParser(description="Forecast 2026 election")
    parser.add_argument("--experiments", nargs="+", default=["C", "D"])
    parser.add_argument("--model", default="WeightedHistoricalMean")
    parser.add_argument("--feature-group", default="ALL_FEATURES")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    args = parser.parse_args()

    dirs = get_output_dirs()
    pred_dir = dirs["predictions_dir"]
    results_dir = dirs["results_dir"]

    for exp in args.experiments:
        print(f"\n=== Experiment {exp}: 2026 forecast ({args.model}) ===")

        if args.model in BASELINE_NAMES:
            res = forecast_baseline(args.model, exp, args.feature_group)
            if res.get("note") != "forecast" or "predictions" not in res:
                print(f"  note={res.get('note')}")
                continue
            target_columns = res["target_columns"]
            avg = res["predictions"]
            seed_preds = None
        else:
            seed_preds = []
            for seed in args.seeds:
                res = forecast_temporal(
                    args.model,
                    exp,
                    args.feature_group,
                    model_kwargs={"random_state": seed},
                )
                if res.get("note") != "forecast" or "predictions" not in res:
                    print(f"  seed {seed}: note={res.get('note')}")
                    continue
                seed_preds.append(res["predictions"])

            if not seed_preds:
                print("  No forecasts produced.")
                continue

            # Average over seeds.
            target_columns = res["target_columns"]
            pred_cols = [f"{c}_pred" for c in target_columns]
            avg = seed_preds[0][["region_id", "year"]].copy()
            for p in pred_cols:
                stacked = pd.concat([sp[p] for sp in seed_preds], axis=1)
                avg[p] = stacked.mean(axis=1)

        # Save per-region forecast.
        out_path = pred_dir / args.model / exp / "2026_forecast.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        avg.to_csv(out_path, index=False)
        print(f"  Saved per-region forecast: {out_path} ({len(avg)} regions)")

        # Federal-level (weighted by electorate if available).
        weight_col = "electorate" if "electorate" in avg.columns else (
            "population" if "population" in avg.columns else None
        )
        fed = federal_forecast(avg, target_columns, weight_col or "region_id")
        print("  Federal forecast (avg %):")
        for party, val in fed.items():
            print(f"    {party}: {val:.2f}")

        # Save federal summary.
        fed_path = results_dir / f"forecast_{exp}_{args.model}_federal.csv"
        pd.DataFrame([fed]).to_csv(fed_path, index=False)
        print(f"  Saved federal forecast: {fed_path}")


if __name__ == "__main__":
    main()
