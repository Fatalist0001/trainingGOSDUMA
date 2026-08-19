# Makefile for trainingGOSDUMA
# Usage: make <target>

.PHONY: help install test lint clean benchmark baselines linear trees knn neural temporal ablation report

# Default target
help:
	@echo "trainingGOSDUMA - ML experiment for Russian election forecasting"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies with uv"
	@echo "  test        - Run tests with pytest"
	@echo "  lint        - Run ruff linter and formatter"
	@echo "  clean       - Remove cache and output directories"
	@echo ""
	@echo "Model training:"
	@echo "  baselines   - Run baseline models (Naive, HistoricalMean, WeightedMean)"
	@echo "  linear      - Run linear models (Linear, Ridge, ElasticNet)"
	@echo "  trees       - Run tree models (RF, HistGB, XGBoost, CatBoost)"
	@echo "  knn         - Run KNN model"
	@echo "  neural      - Run neural models (MLP)"
	@echo "  temporal    - Run temporal models (GRU, LSTM, Transformer)"
	@echo ""
	@echo "Evaluation:"
	@echo "  benchmark   - Generate benchmark table from all results"
	@echo "  ablation    - Run feature/history ablation experiments"
	@echo "  report      - Generate final report with plots"
	@echo ""
	@echo "Full pipeline:"
	@echo "  make benchmark  # Runs all models + generates benchmark table"

# Install dependencies
install:
	uv sync --dev

# Run tests
test:
	uv run pytest tests/ -v

# Run linter
lint:
	uv run ruff check src/ scripts/ tests/
	uv run ruff format --check src/ scripts/ tests/

# Format code
format:
	uv run ruff format src/ scripts/ tests/

# Clean up
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf src/__pycache__ src/*/__pycache__ src/*/*/__pycache__
	rm -rf scripts/__pycache__ scripts/*/__pycache__
	rm -rf tests/__pycache__
	rm -rf predictions/ results/ logs/ reports/
	rm -rf .coverage htmlcov/

# Model training targets
baselines:
	uv run python scripts/models/run_baselines.py --experiment A --feature-group ALL_FEATURES
	uv run python scripts/models/run_baselines.py --experiment B --feature-group ALL_FEATURES

linear:
	uv run python scripts/models/run_linear.py --experiment A --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY
	uv run python scripts/models/run_linear.py --experiment B --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY

trees:
	uv run python scripts/models/run_trees.py --experiment A --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY
	uv run python scripts/models/run_trees.py --experiment B --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY

knn:
	uv run python scripts/models/run_knn.py --experiment A --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY
	uv run python scripts/models/run_knn.py --experiment B --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY

neural:
	uv run python scripts/models/run_neural.py --experiment A --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY
	uv run python scripts/models/run_neural.py --experiment B --feature-groups ALL_FEATURES ELECTORAL_ONLY ROSSTAT_ONLY

temporal:
	@echo "Temporal models not yet implemented"
	# uv run python scripts/models/run_temporal.py --experiment A

# Evaluation targets
benchmark:
	uv run python scripts/evaluation/benchmark.py --save

ablation:
	uv run python scripts/evaluation/ablation.py

report:
	uv run python scripts/visualization/generate_report.py

# Full pipeline (runs all P0 models + benchmark)
full-benchmark: baselines linear trees benchmark
	@echo "Full benchmark complete!"

# Quick test run (single experiment)
quick-test:
	uv run python scripts/models/run_baselines.py --experiment A --feature-group ALL_FEATURES
	uv run python scripts/models/run_trees.py --experiment A --feature-group ALL_FEATURES
	uv run python scripts/evaluation/benchmark.py --experiment A --save

# Development helpers
jupyter:
	uv run jupyter lab --no-browser --port=8888

check-env:
	uv run python -c "from src.utils.reproducibility import capture_env; import json; print(json.dumps(capture_env(), indent=2))"

# Verify no leakage in splits
check-leakage:
	uv run pytest tests/test_leakage.py -v