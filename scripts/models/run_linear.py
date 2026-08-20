#!/usr/bin/env python
"""Run linear models (Linear, Ridge, ElasticNet) for all experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.backtest import run_experiment
from src.tracking.json_tracker import create_tracker
from src.utils.io import save_results_per_group


def main():
    parser = argparse.ArgumentParser(description="Run linear models")
    parser.add_argument("--experiment", default="A", help="Experiment name (A, B, C, D)")
    parser.add_argument("--feature-group", default="ALL_FEATURES", help="Feature group")
    parser.add_argument("--models", nargs="+", default=None, help="Models to run")
    parser.add_argument("--level", default="region", help="Data level")
    parser.add_argument(
        "--feature-groups", nargs="+", default=None, help="Feature groups for ablation"
    )
    args = parser.parse_args()

    if args.models is None:
        models = ["LinearRegression", "Ridge", "ElasticNet"]
    else:
        models = args.models

    if args.feature_groups is None:
        feature_groups = [args.feature_group]
    else:
        feature_groups = args.feature_groups

    tracker = create_tracker(f"linear_{args.experiment}")

    print(f"Running linear models on Experiment {args.experiment}")
    results = run_experiment(
        experiment_name=args.experiment,
        models=models,
        feature_groups=feature_groups,
        level=args.level,
    )

    for r in results:
        if "error" not in r:
            test_metrics = r.get("test_metrics", [])
            if test_metrics:
                avg_mae = sum(m["mae"] for m in test_metrics) / len(test_metrics)
                tracker.log(
                    model=r["model"],
                    feature_group=r["feature_group"],
                    split=args.experiment,
                    hyperparameters={},
                    metrics={
                        "mae": avg_mae,
                        **{f"{m['party']}_mae": m["mae"] for m in test_metrics},
                    },
                    tags={"level": args.level},
                )

    save_results_per_group(results, "linear", args.experiment)

    print("\n=== Summary ===")
    for r in results:
        if "error" in r:
            print(f"{r['model']}: ERROR - {r['error']}")
        else:
            test_metrics = r.get("test_metrics", [])
            avg_mae = (
                sum(m["mae"] for m in test_metrics) / len(test_metrics) if test_metrics else "N/A"
            )
            print(f"{r['model']}: MAE = {avg_mae:.4f}")

    return results


if __name__ == "__main__":
    main()
