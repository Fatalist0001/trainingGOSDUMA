#!/usr/bin/env python
"""Generate key figures for the final report (PLAN §55).

Reads results/*.csv and writes PNGs to reports/figures/.

Quick start:
    uv run python scripts/visualization/generate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.io import get_output_dirs

FIG_DIR = project_root / "reports" / "figures"


def plot_model_comparison(bench: pd.DataFrame) -> None:
    """Bar chart of MAE by model (ALL_FEATURES), experiments A and B."""
    df = bench[bench["Model"].str.endswith("(ALL_FEATURES)")].copy()
    df["model"] = df["Model"].str.replace(" (ALL_FEATURES)", "", regex=False)
    df["model"] = df["model"].str.replace(r"^(Baseline|Tabular|Sequential) \| ", "", regex=True)
    df = df.dropna(subset=["A_year_2016", "B_year_2021"])
    df = df.sort_values("A_year_2016")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    w = 0.4
    ax.bar([i - w / 2 for i in x], df["A_year_2016"], w, label="A (2016)", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], df["B_year_2021"], w, label="B (2021)", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("Model comparison (ALL_FEATURES)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)


def plot_feature_ablation(feat: pd.DataFrame) -> None:
    """Grouped bars: MAE across feature groups for selected models (experiment A)."""
    models = ["XGBoost", "CatBoost", "GRU", "Transformer", "LinearRegression"]
    groups = ["ELECTORAL_ONLY", "ROSSTAT_ONLY", "ALL_FEATURES"]
    sub = feat[(feat["experiment"] == "A") & (feat["model"].isin(models))].copy()
    pivot = sub.pivot(index="model", columns="feature_group", values="mae").reindex(
        index=models, columns=groups
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("Feature-group ablation (Experiment A, Q2)")
    ax.set_xlabel("")
    ax.legend(title="feature group")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_ablation.png", dpi=120)
    plt.close(fig)


def plot_history_depth(hist: pd.DataFrame) -> None:
    """Line: MAE vs history depth for temporal models (experiments A and B)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, exp in zip(axes, ["A", "B"]):
        sub = hist[(hist["experiment"] == exp) & hist["model"].isin(["GRU", "LSTM", "Transformer"])]
        for model, color in zip(["GRU", "LSTM", "Transformer"], ["#4C72B0", "#DD8452", "#55A868"]):
            m = sub[sub["model"] == model].sort_values("depth")
            m = m.dropna(subset=["mae"])
            if not m.empty:
                ax.plot(m["depth"], m["mae"], marker="o", label=model, color=color)
        ax.set_title(f"History-depth ablation (Exp {exp}, Q3)")
        ax.set_xlabel("history depth (# elections)")
        ax.set_ylabel("MAE (pp)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "history_depth.png", dpi=120)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dirs = get_output_dirs()
    results_dir = dirs["results_dir"]

    bench_files = sorted(results_dir.glob("benchmark_all_*.csv"))
    if not bench_files:
        raise FileNotFoundError("No benchmark_all_*.csv found in results/")
    bench = pd.read_csv(bench_files[-1])
    feat = pd.read_csv(results_dir / "ablation_feature.csv")
    hist = pd.read_csv(results_dir / "ablation_history.csv")

    plot_model_comparison(bench)
    plot_feature_ablation(feat)
    plot_history_depth(hist)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
