#!/usr/bin/env python
"""Run neural models (MLP sklearn, MLP PyTorch) for all experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.backtest import run_single_model_backtest
from src.tracking.json_tracker import create_tracker
from src.utils.io import save_results_per_group
from src.utils.reproducibility import get_seeds


def average_test_metrics(seed_results: list[dict]) -> list[dict]:
    """Average per-party MAE across seed runs into a single test_metrics list."""
    parties: dict[str, list[float]] = {}
    for res in seed_results:
        for m in res.get("test_metrics", []):
            parties.setdefault(m["party"], []).append(m["mae"])
    return [
        {"party": party, "mae": sum(vals) / len(vals), "n_samples": None}
        for party, vals in parties.items()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run neural models")
    parser.add_argument("--experiment", default="A", help="Experiment name (A, B, C, D)")
    parser.add_argument("--feature-group", default="ALL_FEATURES", help="Feature group")
    parser.add_argument("--models", nargs="+", default=None, help="Models to run")
    parser.add_argument("--level", default="region", help="Data level")
    parser.add_argument(
        "--feature-groups", nargs="+", default=None, help="Feature groups for ablation"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="Random seeds for stability"
    )
    args = parser.parse_args()

    models = args.models or ["MLPSklearn", "MLPTorch"]
    feature_groups = args.feature_groups or [args.feature_group]
    seeds = args.seeds or get_seeds()

    tracker = create_tracker(f"neural_{args.experiment}")

    print(f"Running neural models on Experiment {args.experiment} with seeds {seeds}")
    all_results = []

    for feature_group in feature_groups:
        for model_name in models:
            print(f"\n--- {model_name} | {feature_group} ---")
            seed_results = []
            for seed in seeds:
                try:
                    result = run_single_model_backtest(
                        model_name,
                        args.experiment,
                        feature_group,
                        args.level,
                        model_kwargs={"random_state": seed},
                    )
                    if "error" not in result:
                        seed_results.append(result)
                    else:
                        print(f"  seed {seed}: ERROR - {result['error']}")
                except Exception as e:
                    print(f"  seed {seed}: EXCEPTION - {e}")

            if not seed_results:
                continue

            # Aggregate: average MAE across seeds (one row per model/feature group).
            avg_test_metrics = average_test_metrics(seed_results)
            avg_mae = sum(m["mae"] for m in avg_test_metrics) / len(avg_test_metrics)
            aggregated = {
                "model": model_name,
                "experiment": args.experiment,
                "feature_group": feature_group,
                "level": args.level,
                "test_metrics": avg_test_metrics,
                "n_train": seed_results[0].get("n_train"),
                "n_test": seed_results[0].get("n_test"),
                "seeds": seeds,
                "n_seeds": len(seed_results),
            }
            all_results.append(aggregated)

            tracker.log(
                model=model_name,
                feature_group=feature_group,
                split=args.experiment,
                hyperparameters={"seeds": seeds},
                metrics={
                    "mae": avg_mae,
                    **{f"{m['party']}_mae": m["mae"] for m in avg_test_metrics},
                },
                tags={"level": args.level, "n_seeds": len(seed_results)},
            )
            print(f"  averaged MAE over {len(seed_results)} seeds = {avg_mae:.4f}")

    save_results_per_group(all_results, "neural", args.experiment)

    print("\n=== Summary ===")
    for r in all_results:
        test_metrics = r.get("test_metrics", [])
        avg_mae = sum(m["mae"] for m in test_metrics) / len(test_metrics) if test_metrics else "N/A"
        print(f"{r['model']} ({r.get('feature_group', 'N/A')}): MAE = {avg_mae:.4f}")

    return all_results


if __name__ == "__main__":
    main()
