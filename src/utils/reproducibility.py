"""Reproducibility utilities: seed setting, environment capture."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow (if used)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_seeds() -> list[int]:
    """Get standard seeds for neural model stability testing."""
    return [42, 123, 456, 789, 2026]


def capture_env() -> dict[str, Any]:
    """
    Capture environment information for reproducibility.

    Returns:
        Dictionary with environment details
    """
    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "env_vars": {
            k: v for k, v in os.environ.items()
            if k.startswith(("DATASET_", "MLFLOW_", "RANDOM_", "CUDA_", "PYTHON"))
        },
    }

    # Package versions
    packages = [
        "numpy", "pandas", "scipy", "sklearn", "xgboost", "catboost",
        "torch", "shap", "optuna", "matplotlib", "seaborn", "jupyter",
        "pyarrow", "yaml", "tqdm", "joblib",
    ]

    env["packages"] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            env["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["packages"][pkg] = "not installed"

    # Git info
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        env["git"] = {
            "commit": git_commit,
            "branch": git_branch,
            "dirty": bool(git_status),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        env["git"] = {"commit": "unknown", "branch": "unknown", "dirty": False}

    # CUDA info
    try:
        import torch
        env["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    except ImportError:
        env["cuda"] = {"available": False}

    return env


def save_env(env: dict[str, Any], path: str | Path) -> None:
    """Save environment capture to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)


def load_env(path: str | Path) -> dict[str, Any]:
    """Load environment capture from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_reproducibility(
    results: list[dict],
    metric: str = "mae",
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """
    Check if results are reproducible across seeds.

    Args:
        results: List of result dicts from different seeds
        metric: Metric to check
        tolerance: Maximum allowed difference

    Returns:
        Dictionary with reproducibility statistics
    """
    values = [r.get(metric) for r in results if metric in r]
    values = [v for v in values if v is not None]

    if len(values) < 2:
        return {"reproducible": True, "reason": "Not enough values to compare"}

    mean_val = np.mean(values)
    std_val = np.std(values)
    max_diff = np.max(values) - np.min(values)

    return {
        "reproducible": max_diff <= tolerance,
        "mean": mean_val,
        "std": std_val,
        "min": np.min(values),
        "max": np.max(values),
        "max_diff": max_diff,
        "tolerance": tolerance,
        "n_seeds": len(values),
    }