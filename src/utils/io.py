"""I/O utilities for saving/loading predictions, results, and configs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to YAML file."""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_output_dirs() -> dict[str, Path]:
    """Get output directories from config."""
    config = load_yaml_config("config/paths.yaml")
    output = config["output"]
    dirs = {}
    for key, path in output.items():
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        dirs[key] = p
    return dirs


def save_predictions(
    predictions_df: pd.DataFrame,
    model_name: str,
    experiment_name: str,
    year_or_party: str,
    base_dir: str | Path | None = None,
) -> Path:
    """
    Save predictions to parquet file.

    Structure: predictions/{model}/{experiment}/{year_or_party}/predictions.parquet
    """
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["predictions_dir"]

    base_path = Path(base_dir)
    save_path = base_path / model_name / experiment_name / str(year_or_party)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / "predictions.parquet"
    predictions_df.to_parquet(file_path, index=True)

    return file_path


def load_predictions(
    model_name: str,
    experiment_name: str,
    year_or_party: str,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load predictions from parquet file."""
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["predictions_dir"]

    base_path = Path(base_dir)
    file_path = (
        base_path / model_name / experiment_name / str(year_or_party) / "predictions.parquet"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"Predictions not found: {file_path}")

    return pd.read_parquet(file_path)


def save_results(
    results: dict[str, Any] | list[dict[str, Any]],
    filename: str,
    base_dir: str | Path | None = None,
) -> Path:
    """
    Save results to JSON and CSV.

    Args:
        results: Results dictionary or list
        filename: Base filename (without extension)
        base_dir: Base directory (default: results/)

    Returns:
        Path to saved JSON file
    """
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["results_dir"]

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = base_path / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Save CSV if it's a list of dicts
    if isinstance(results, list) and results:
        csv_path = base_path / f"{filename}.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)

    return json_path


def load_results(
    filename: str,
    base_dir: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Load results from JSON file."""
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["results_dir"]

    file_path = Path(base_dir) / f"{filename}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Results not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_benchmark_table(
    df: pd.DataFrame,
    experiment_name: str,
    base_dir: str | Path | None = None,
) -> Path:
    """
    Save benchmark table (model vs years MAE).

    Args:
        df: DataFrame with models as rows, years as columns
        experiment_name: Experiment name
        base_dir: Base directory

    Returns:
        Path to saved CSV
    """
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["results_dir"]

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    filename = f"benchmark_{experiment_name}_{datetime.now().strftime('%Y%m%d')}.csv"
    file_path = base_path / filename
    df.to_csv(file_path)

    return file_path


def load_benchmark_table(
    experiment_name: str,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load latest benchmark table for experiment."""
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["results_dir"]

    base_path = Path(base_dir)
    files = list(base_path.glob(f"benchmark_{experiment_name}_*.csv"))

    if not files:
        raise FileNotFoundError(f"No benchmark table found for {experiment_name}")

    # Get latest
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return pd.read_csv(latest, index_col=0)


def list_saved_predictions(
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List all saved prediction files."""
    if base_dir is None:
        dirs = get_output_dirs()
        base_dir = dirs["predictions_dir"]

    base_path = Path(base_dir)
    results = []

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            for year_dir in exp_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                pred_file = year_dir / "predictions.parquet"
                if pred_file.exists():
                    results.append(
                        {
                            "model": model_dir.name,
                            "experiment": exp_dir.name,
                            "year": year_dir.name,
                            "path": str(pred_file),
                            "size_mb": pred_file.stat().st_size / (1024 * 1024),
                        }
                    )

    return results
