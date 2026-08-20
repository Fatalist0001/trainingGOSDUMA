#!/usr/bin/env python
"""Run feature-group and history-depth ablation studies.

Writes results to results/ablation_feature.csv and results/ablation_history.csv.

Quick start:
    make ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.ablation import feature_ablation, history_depth_ablation
from src.utils.io import get_output_dirs
from src.utils.reproducibility import get_seeds

TEMPORAL = ["GRU", "LSTM", "Transformer"]
FLAT = ["NaivePreviousElection", "XGBoost", "CatBoost", "RandomForest", "LinearRegression"]


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--experiments", nargs="+", default=["A", "B"], help="Experiments to run")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Models (default: flat + temporal set)"
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="Seeds for temporal models"
    )
    args = parser.parse_args()

    models = args.models or (FLAT + TEMPORAL)
    seeds = args.seeds or get_seeds()

    feature_rows: list[dict] = []
    history_rows: list[dict] = []

    for exp in args.experiments:
        for model in models:
            is_temporal = model in TEMPORAL
            model_seeds = seeds if is_temporal else None
            print(f"\n[Ablation] {model} | exp {exp}")

            fa = feature_ablation(
                model,
                exp,
                feature_groups=args.feature_groups,
                seeds=model_seeds,
            )
            for r in fa:
                print(f"  feature {r['feature_group']}: MAE={r['mae']}")
            feature_rows.extend(fa)

            hd = history_depth_ablation(model, exp, seeds=model_seeds)
            for r in hd:
                print(f"  depth {r['depth']} ({r['train_years']}): MAE={r['mae']}")
            history_rows.extend(hd)

    dirs = get_output_dirs()
    results_dir = dirs["results_dir"]
    feature_df = pd.DataFrame(feature_rows)
    history_df = pd.DataFrame(history_rows)
    feature_path = results_dir / "ablation_feature.csv"
    history_path = results_dir / "ablation_history.csv"
    feature_df.to_csv(feature_path, index=False)
    history_df.to_csv(history_path, index=False)
    print(f"\nSaved feature ablation: {feature_path}")
    print(f"Saved history-depth ablation: {history_path}")


if __name__ == "__main__":
    main()
