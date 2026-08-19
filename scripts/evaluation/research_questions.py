#!/usr/bin/env python
"""Answer remaining research questions (Q4, Q7, Q8, Q9) and save artifacts.

Q4: linear vs nonlinear model classes        -> results/q4_model_class_mae.csv
Q7: per-party MAE across models              -> results/q7_party_mae.csv
Q8: regional error analysis (worst regions)  -> results/q8_regional_errors.csv
Q9: seed stability (mean/std MAE per model)  -> results/q9_seed_stability.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.splits import load_raw_region
from src.evaluation.backtest import run_single_model_backtest, run_temporal_backtest
from src.utils.io import save_results
from src.utils.reproducibility import get_seeds

RESULTS = project_root / "results"

MODEL_CLASSES = {
    "baseline": ["NaivePreviousElection", "HistoricalMean", "WeightedHistoricalMean"],
    "linear": ["LinearRegression", "Ridge", "ElasticNet"],
    "tree": ["RandomForest", "HistGradientBoosting", "XGBoost", "CatBoost", "KNN"],
    "neural": ["MLPSklearn", "MLPTorch"],
    "temporal": ["GRU", "LSTM", "Transformer"],
}

PARTIES = ["UR_share", "KPRF_share", "LDPR_share"]


def mean_mae(test_metrics: list[dict]) -> float:
    """Average per-party MAE (mean across parties)."""
    maes = [m["mae"] for m in test_metrics if "mae" in m]
    return float(np.mean(maes)) if maes else float("nan")


def q4_linear_vs_nonlinear() -> None:
    """Read benchmark_all CSV and aggregate MAE by model class (ALL_FEATURES)."""
    bm_path = sorted(RESULTS.glob("benchmark_all_*.csv"))
    if not bm_path:
        print("[Q4] benchmark_all CSV not found, skipping")
        return
    bm = pd.read_csv(bm_path[-1])
    rows = []
    for _, r in bm.iterrows():
        model_name = str(r["Model"]).split(" (")[0]
        group = str(r["Model"]).split("(")[1].rstrip(")") if "(" in str(r["Model"]) else ""
        if group != "ALL_FEATURES":
            continue
        cls = next((c for c, names in MODEL_CLASSES.items() if model_name in names), "other")
        rows.append(
            {
                "Model": model_name,
                "Class": cls,
                "A_MAE": r["A_year_2016"],
                "B_MAE": r["B_year_2021"],
                "Mean_MAE": (r["A_year_2016"] + r["B_year_2021"]) / 2,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("[Q4] no ALL_FEATURES rows, skipping")
        return
    summary = (
        df.groupby("Class")[["A_MAE", "B_MAE", "Mean_MAE"]]
        .mean()
        .round(3)
        .reset_index()
    )
    save_results(summary.to_dict("records"), "q4_model_class_mae")
    save_results(df.to_dict("records"), "q4_model_class_mae_detail")
    print("[Q4] done:\n", summary.to_string(index=False))


def _collect_json_metrics() -> list[dict]:
    """Parse all results/*.json into per-model/per-experiment party MAE rows."""
    rows: list[dict] = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith(("q4_", "q7_", "q8_", "q9_")):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        entries = data if isinstance(data, list) else [data]
        for e in entries:
            if not isinstance(e, dict) or e.get("feature_group") != "ALL_FEATURES":
                continue
            tm = e.get("test_metrics")
            if not tm:
                continue
            row = {
                "Model": e.get("model"),
                "Experiment": e.get("experiment"),
                "n_train": e.get("n_train"),
                "n_test": e.get("n_test"),
            }
            for party in PARTIES:
                val = next((m["mae"] for m in tm if m.get("party") == party), np.nan)
                row[party] = val
            row["Mean_MAE"] = np.nanmean(
                [row[p] for p in PARTIES if not np.isnan(row[p])]
            ) if not all(np.isnan(row[p]) for p in PARTIES) else np.nan
            rows.append(row)
    return rows


def q7_party_mae() -> None:
    """Per-party MAE across models (mean over experiments A and B)."""
    rows = _collect_json_metrics()
    if not rows:
        print("[Q7] no JSON metrics found, skipping")
        return
    df = pd.DataFrame(rows)
    df = df[df["Experiment"].isin(["A", "B"])].copy()
    df["Model"] = df["Model"].str.replace("_share", "", regex=False)
    party_cols = [p.replace("_share", "") for p in PARTIES]
    df.columns = [
        c.replace("_share", "") if c in PARTIES else c for c in df.columns
    ]
    for p in party_cols:
        df[p] = pd.to_numeric(df[p], errors="coerce")
    df["Mean_MAE"] = df[party_cols].mean(axis=1)

    pivot = (
        df.groupby("Model")[party_cols + ["Mean_MAE"]]
        .mean()
        .round(3)
        .sort_values("Mean_MAE")
        .reset_index()
    )
    save_results(pivot.to_dict("records"), "q7_party_mae")
    save_results(
        df[["Model", "Experiment"] + party_cols + ["Mean_MAE"]].to_dict("records"),
        "q7_party_mae_detail",
    )
    print("[Q7] done (mean MAE over A/B, per party):\n", pivot.to_string(index=False))


def q8_regional_errors() -> None:
    """Per-region error analysis from saved predictions (worst/best regions)."""
    models = [
        "NaivePreviousElection",
        "LinearRegression",
        "RandomForest",
        "HistGradientBoosting",
        "XGBoost",
        "CatBoost",
        "MLPSklearn",
    ]
    df_raw = load_raw_region()
    name_by_idx = df_raw["region_name"].to_dict()

    rows: list[dict] = []
    for model in models:
        for exp in ["A", "B"]:
            path = RESULTS.parent / "predictions" / model / exp / "UR" / "predictions.parquet"
            if not path.exists():
                continue
            pred = pd.read_parquet(path)
            actual_cols = PARTIES
            pred_cols = [f"{p}_pred" for p in PARTIES]
            if not all(c in pred.columns for c in actual_cols + pred_cols):
                continue
            for idx, row in pred.iterrows():
                errors = [abs(row[a] - row[p]) for a, p in zip(actual_cols, pred_cols)]
                biases = [row[p] - row[a] for a, p in zip(actual_cols, pred_cols)]
                rows.append(
                    {
                        "region_name": name_by_idx.get(idx, idx),
                        "Experiment": exp,
                        "Model": model,
                        "region_mae": float(np.mean(errors)),
                        "region_bias": float(np.mean(biases)),
                    }
                )
    if not rows:
        print("[Q8] no predictions found, skipping")
        return
    df = pd.DataFrame(rows)

    # Aggregate per region (mean across models, per experiment).
    agg = (
        df.groupby(["region_name", "Experiment"])[["region_mae", "region_bias"]]
        .mean()
        .round(3)
        .reset_index()
    )
    save_results(agg.to_dict("records"), "q8_regional_errors")
    save_results(df.to_dict("records"), "q8_regional_errors_detail")

    print("[Q8] done. Worst 10 regions by mean MAE across models:")
    for exp in ["A", "B"]:
        sub = agg[agg["Experiment"] == exp].sort_values("region_mae", ascending=False)
        print(f"  Experiment {exp} (test {2016 if exp=='A' else 2021}):")
        print(sub.head(10).to_string(index=False))
    print("\n[Q8] best 5 regions:")
    for exp in ["A", "B"]:
        sub = agg[agg["Experiment"] == exp].sort_values("region_mae")
        print(f"  Experiment {exp}: {list(sub.head(5)['region_name'])}")


def q9_seed_stability(experiments: list[str], seeds: list[int]) -> None:
    """Re-run neural/temporal models per seed and compute mean/std MAE."""
    neural_models = ["MLPSklearn", "MLPTorch"]
    temporal_models = ["GRU", "LSTM", "Transformer"]
    rows: list[dict] = []

    for exp in experiments:
        for model in neural_models:
            per_seed: list[float] = []
            for seed in seeds:
                res = run_single_model_backtest(
                    model, exp, "ALL_FEATURES", "region", model_kwargs={"random_state": seed}
                )
                if "error" in res:
                    continue
                per_seed.append(mean_mae(res.get("test_metrics", [])))
            if per_seed:
                rows.append(_seed_row(model, exp, seeds, per_seed))

        for model in temporal_models:
            per_seed = []
            for seed in seeds:
                res = run_temporal_backtest(
                    model, exp, "ALL_FEATURES", "region", model_kwargs={"random_state": seed}
                )
                if "error" in res:
                    continue
                per_seed.append(mean_mae(res.get("test_metrics", [])))
            if per_seed:
                rows.append(_seed_row(model, exp, seeds, per_seed))

    if not rows:
        print("[Q9] no seed results, skipping")
        return
    df = pd.DataFrame(rows)
    save_results(df.to_dict("records"), "q9_seed_stability")
    print("[Q9] done:\n", df.to_string(index=False))


def _seed_row(model: str, exp: str, seeds: list[int], per_seed: list[float]) -> dict:
    return {
        "Model": model,
        "Experiment": exp,
        "seeds_used": len(per_seed),
        "seed_mae_list": json.dumps([round(v, 3) for v in per_seed]),
        "mean_mae": round(float(np.mean(per_seed)), 3),
        "std_mae": round(float(np.std(per_seed)), 3),
        "min_mae": round(float(np.min(per_seed)), 3),
        "max_mae": round(float(np.max(per_seed)), 3),
        "range_pp": round(float(np.max(per_seed) - np.min(per_seed)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer remaining research questions Q4/Q7/Q8/Q9")
    parser.add_argument("--experiments", nargs="+", default=["A", "B"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-q9", action="store_true", help="Skip the expensive per-seed re-run")
    args = parser.parse_args()

    seeds = args.seeds or get_seeds()

    q4_linear_vs_nonlinear()
    q7_party_mae()
    q8_regional_errors()
    if not args.skip_q9:
        q9_seed_stability(args.experiments, seeds)
    else:
        print("[Q9] skipped (--skip-q9)")


if __name__ == "__main__":
    main()
