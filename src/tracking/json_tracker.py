"""Simple JSON/CSV experiment tracker."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class ExperimentTracker:
    """
    Simple experiment tracker using JSONL + CSV summary.

    Each experiment run is logged as a JSON line.
    A summary CSV is maintained for easy analysis.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_dir: str | Path | None = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir) if tracking_dir else Path("logs") / experiment_name
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.tracking_dir / "experiments.jsonl"
        self.csv_path = self.tracking_dir / "experiments.csv"

    def log(
        self,
        model: str,
        feature_group: str,
        split: str,
        hyperparameters: dict,
        metrics: dict,
        seed: int | None = None,
        tags: dict | None = None,
        **extra,
    ) -> str:
        """
        Log an experiment run.

        Args:
            model: Model name
            feature_group: Feature group used
            split: Experiment split (A, B, C, D)
            hyperparameters: Model hyperparameters
            metrics: Evaluation metrics
            seed: Random seed
            tags: Additional tags
            **extra: Extra fields to log

        Returns:
            Experiment ID
        """
        experiment_id = f"{self.experiment_name}_{model}_{feature_group}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = {
            "experiment_id": experiment_id,
            "experiment_name": self.experiment_name,
            "model": model,
            "feature_group": feature_group,
            "split": split,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "seed": seed,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat(),
            **extra,
        }

        # Append to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update CSV summary
        self._update_csv(record)

        return experiment_id

    def _update_csv(self, record: dict):
        """Update CSV summary file."""
        # Flatten metrics for CSV
        flat_record = {
            "experiment_id": record["experiment_id"],
            "experiment_name": record["experiment_name"],
            "model": record["model"],
            "feature_group": record["feature_group"],
            "split": record["split"],
            "seed": record.get("seed"),
            "timestamp": record["timestamp"],
        }

        # Add hyperparameters (flattened)
        for k, v in record.get("hyperparameters", {}).items():
            flat_record[f"hp_{k}"] = v

        # Add metrics (flattened)
        for k, v in record.get("metrics", {}).items():
            flat_record[f"metric_{k}"] = v

        # Add tags
        for k, v in record.get("tags", {}).items():
            flat_record[f"tag_{k}"] = v

        # Add extra fields
        for k, v in record.items():
            if k not in [
                "experiment_id",
                "experiment_name",
                "model",
                "feature_group",
                "split",
                "hyperparameters",
                "metrics",
                "seed",
                "tags",
                "timestamp",
            ]:
                flat_record[k] = v

        # Append to CSV
        df_new = pd.DataFrame([flat_record])
        if self.csv_path.exists():
            df_new.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.csv_path, index=False)

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame from CSV."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()

    def get_best(self, metric: str = "metric_mae", ascending: bool = True) -> pd.Series | None:
        """Get best experiment by metric."""
        df = self.get_summary()
        if df.empty or metric not in df.columns:
            return None
        return df.loc[df[metric].idxmin() if ascending else df[metric].idxmax()]


def create_tracker(experiment_name: str) -> ExperimentTracker:
    """Factory function to create tracker."""
    return ExperimentTracker(experiment_name)


def load_experiment_log(experiment_name: str) -> pd.DataFrame:
    """Load experiment log from CSV."""
    tracking_dir = Path("logs") / experiment_name
    csv_path = tracking_dir / "experiments.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()
