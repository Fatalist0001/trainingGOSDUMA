#!/usr/bin/env python
"""Run neural models (MLP sklearn, MLP PyTorch) for all experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tracking.json_tracker import create_tracker
from src.utils.io import save_results
from src.utils.reproducibility import get_seeds


def main():
    parser = argparse.ArgumentParser(description="Run neural models")
    parser.add_argument("--experiment", default="A", help="Experiment name (A, B, C, D)")
    parser.add_argument("--feature-group", default="ALL_FEATURES", help="Feature group")
    parser.add_argument("--models", nargs="+", default=None, help="Models to run")
    parser.add_argument("--level", default="region", help="Data level")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="Random seeds for stability"
    )
    args = parser.parse_args()

    if args.models is None:
        models = ["MLPSklearn"]  # MLPTorch requires more setup
    else:
        models = args.models

    seeds = args.seeds or get_seeds()

    tracker = create_tracker(f"neural_{args.experiment}")

    print(f"Running neural models on Experiment {args.experiment} with seeds {seeds}")
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for model_name in models:
            try:
                # For now, run single seed
                from src.evaluation.backtest import run_single_model_backtest

                result = run_single_model_backtest(
                    model_name,
                    args.experiment,
                    args.feature_group,
                    args.level,
                    model_kwargs={"random_state": seed},
                )
                result["seed"] = seed
                all_results.append(result)

                if "error" not in result:
                    test_metrics = result.get("test_metrics", [])
                    if test_metrics:
                        avg_mae = sum(m["mae"] for m in test_metrics) / len(test_metrics)
                        tracker.log(
                            model=f"{model_name}_seed{seed}",
                            feature_group=args.feature_group,
                            split=args.experiment,
                            hyperparameters={"random_state": seed},
                            metrics={"mae": avg_mae},
                            tags={"level": args.level, "seed": seed},
                        )
            except Exception as e:
                print(f"Error with {model_name} seed {seed}: {e}")
                all_results.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "experiment": args.experiment,
                        "feature_group": args.feature_group,
                        "error": str(e),
                    }
                )

    save_results(all_results, f"neural_{args.experiment}_{args.feature_group}")

    # Print seed stability summary
    print("\n=== Seed Stability Summary ===")
    from src.utils.reproducibility import check_reproducibility

    for model_name in models:
        model_results = [
            r for r in all_results if r.get("model") == model_name and "error" not in r
        ]
        if model_results:
            # Extract MAE
            for r in model_results:
                test_metrics = r.get("test_metrics", [])
                if test_metrics:
                    r["mae"] = sum(m["mae"] for m in test_metrics) / len(test_metrics)

            check = check_reproducibility(model_results, metric="mae", tolerance=0.1)
            print(
                f"{model_name}: MAE mean={check['mean']:.4f}, std={check['std']:.4f}, "
                f"reproducible={check['reproducible']}"
            )

    return all_results


if __name__ == "__main__":
    main()
