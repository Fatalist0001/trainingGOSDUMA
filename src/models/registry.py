"""Model registry for dynamic model loading."""
from __future__ import annotations

from typing import Any

# Import all model classes
from .baselines import (
    NaivePreviousElection,
    HistoricalMean,
    WeightedHistoricalMean,
    get_baseline_model,
)
from .linear import LinearModel, RidgeModel, ElasticNetModel, create_linear_pipeline
from .trees import RandomForestModel, HistGBModel, XGBoostModel, CatBoostModel

# Optional imports
try:
    from .neural import MLPSklearn, MLPTorch
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    MLPSklearn = None
    MLPTorch = None

try:
    from .temporal import GRUModel, LSTMModel, TransformerModel
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    GRUModel = None
    LSTMModel = None
    TransformerModel = None

try:
    from .ensemble import WeightedEnsemble, StackingEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    WeightedEnsemble = None
    StackingEnsemble = None


# Model registry mapping name -> class
MODEL_REGISTRY = {
    # Baselines
    "NaivePreviousElection": NaivePreviousElection,
    "HistoricalMean": HistoricalMean,
    "WeightedHistoricalMean": WeightedHistoricalMean,
    # Linear
    "LinearRegression": LinearModel,
    "Ridge": RidgeModel,
    "ElasticNet": ElasticNetModel,
    # Trees
    "RandomForest": RandomForestModel,
    "HistGradientBoosting": HistGBModel,
    "XGBoost": XGBoostModel,
    "CatBoost": CatBoostModel,
}

# Add optional models if available
if NEURAL_AVAILABLE:
    MODEL_REGISTRY["MLPSklearn"] = MLPSklearn
    MODEL_REGISTRY["MLPTorch"] = MLPTorch

if TEMPORAL_AVAILABLE:
    MODEL_REGISTRY["GRU"] = GRUModel
    MODEL_REGISTRY["LSTM"] = LSTMModel
    MODEL_REGISTRY["Transformer"] = TransformerModel

if ENSEMBLE_AVAILABLE:
    MODEL_REGISTRY["WeightedEnsemble"] = WeightedEnsemble
    MODEL_REGISTRY["StackingEnsemble"] = StackingEnsemble


# Model categories for P0/P1/P2/P3 prioritization
MODEL_CATEGORIES = {
    "P0": [
        "NaivePreviousElection",
        "HistoricalMean",
        "WeightedHistoricalMean",
        "LinearRegression",
        "Ridge",
        "ElasticNet",
        "RandomForest",
        "HistGradientBoosting",
        "XGBoost",
        "CatBoost",
    ],
    "P1": [
        "MLPSklearn",
        "MLPTorch",
    ],
    "P2": [
        "GRU",
        "LSTM",
    ],
    "P3": [
        "Transformer",
    ],
}


def get_model(name: str, **kwargs) -> Any:
    """
    Get model instance by name.

    Args:
        name: Model name from MODEL_REGISTRY
        **kwargs: Model constructor arguments

    Returns:
        Instantiated model
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)


def list_models(category: str | None = None) -> list[str]:
    """
    List available models.

    Args:
        category: Optional category filter ("P0", "P1", "P2", "P3")

    Returns:
        List of model names
    """
    if category is None:
        return list(MODEL_REGISTRY.keys())

    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Available: {list(MODEL_CATEGORIES.keys())}")

    return [m for m in MODEL_CATEGORIES[category] if m in MODEL_REGISTRY]


def get_p0_models() -> list[str]:
    """Get all P0 (priority) models."""
    return list_models("P0")


def is_model_available(name: str) -> bool:
    """Check if a model is available."""
    return name in MODEL_REGISTRY