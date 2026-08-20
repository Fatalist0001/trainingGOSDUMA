#!/usr/bin/env python
"""Run baseline models for all experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src.*
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import using src.* namespace
import src.evaluation.backtest
import src.models.registry
import src.tracking.json_tracker
import src.utils.io

run_experiment = src.evaluation.backtest.run_experiment
save_results_per_group = src.utils.io.save_results_per_group
get_output_dirs = src.utils.io.get_output_dirs
create_tracker = src.tracking.json_tracker.create_tracker
get_p0_models = src.models.registry.get_p0_models


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--experiment", default="A", help="Experiment name (A, B, C, D)")
    parser.add_argument("--feature-group", default="ALL_FEATURES", help="Feature group")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Models to run (default: all baselines)"
    )
    parser.add_argument("--level", default="region", help="Data level (region/precinct)")
    args = parser.parse_args()

    # Get baseline models
    if args.models is None:
        models = ["NaivePreviousElection", "HistoricalMean", "WeightedHistoricalMean"]
    else:
        models = args.models

    # Create tracker
    tracker = create_tracker(f"baselines_{args.experiment}")

    # Run experiment
    print(f"Running baselines on Experiment {args.experiment} with {args.feature_group}")
    results = run_experiment(
        experiment_name=args.experiment,
        models=models,
        feature_groups=[args.feature_group],
        level=args.level,
    )

    # Log results
    for r in results:
        if "error" not in r:
            # Aggregate test metrics
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

    # Save results
    save_results_per_group(results, "baselines", args.experiment)

    # Print summary
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
