"""Temporal-validated hyperparameter tuning.

All hyperparameters are selected ONLY on the validation set (a strictly past
year relative to test). No random KFold across years is used anywhere in the
pipeline: validation is always the most recent historical election year.

After selection the best configuration is refit on train+val (see
``evaluation/backtest.py``) and only then scored on the test year.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from ..models.registry import get_model
from ..utils.io import load_yaml_config
from ..utils.reproducibility import set_seed

# Model keys whose fit() accepts an ``eval_set`` for early stopping on a
# user-provided validation set (checked dynamically at runtime too).
EVAL_SET_MODELS = {"XGBoost", "CatBoost"}


def get_tuning_config() -> dict[str, Any]:
    """Load the ``tuning`` block from models.yaml."""
    cfg = load_yaml_config("config/models.yaml")
    return cfg.get("tuning", {})


def get_param_grid(model_name: str) -> dict[str, list]:
    """Return the hyperparameter grid for a model from models.yaml.

    Only list-valued entries are treated as grids; scalar entries (metadata
    like ``optimization_method``, ``meta_model``, ...) are ignored.
    """
    cfg = load_yaml_config("config/models.yaml")
    raw = cfg.get("models", {}).get(model_name, {})
    return {k: v for k, v in raw.items() if isinstance(v, (list, tuple))}


def _apply_override(grid: dict[str, list], model_name: str) -> dict[str, list]:
    """Force specific grid entries from the ``tuning.override`` block.

    Used e.g. to disable random internal early stopping for models that cannot
    consume an external validation set (sklearn MLP, HistGradientBoosting).
    """
    over = get_tuning_config().get("override", {}).get(model_name, {})
    grid = dict(grid)
    for k, v in over.items():
        grid[k] = v
    return grid


def _filter_grid_to_constructor(model_name: str, grid: dict[str, list]) -> dict[str, list]:
    """Keep only grid keys the model's constructor accepts.

    Models with ``**kwargs`` still drop scheduler/metadata keys (``seeds``,
    ``seq_len``) that would be forwarded to the underlying regressor and raise.
    """
    import inspect

    from ..models.registry import MODEL_REGISTRY

    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        return grid
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return grid
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        banned = {"seeds", "seq_len"}
        return {k: v for k, v in grid.items() if k not in banned}
    names = set(sig.parameters)
    return {k: v for k, v in grid.items() if k in names}


def _avg_mae(y_true: Any, y_pred: Any) -> float:
    """Average per-target MAE (nan-safe)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    maes = []
    for j in range(yt.shape[1]):
        mask = ~(np.isnan(yt[:, j]) | np.isnan(yp[:, j]))
        if mask.sum() == 0:
            continue
        maes.append(float(np.mean(np.abs(yt[mask, j] - yp[mask, j]))))
    return float(np.mean(maes)) if maes else float("inf")


def _fit_with_val(model: Any, X_train, y_train, X_val, y_val, years=None) -> None:
    """Fit a flat model, passing a validation set when the model supports it.

    Supports ``eval_set=(X_val, y_val)`` (XGBoost/CatBoost), explicit
    ``X_val=.../y_val=...`` keyword arguments (MLPTorch), and ``years=...``
    (ensembles needing temporal out-of-fold estimation).
    """
    fit = model.fit
    co_args = fit.__code__.co_argcount
    co_varnames = set(fit.__code__.co_varnames[:co_args])
    if "eval_set" in co_varnames and X_val is not None:
        fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif "years" in co_varnames:
        fit(X_train, y_train, years=years)
    elif "X_val" in co_varnames and X_val is not None:
        fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        fit(X_train, y_train)


def best_iteration(model: Any) -> int | None:
    """Extract best early-stopping iteration from a fitted candidate model."""
    inner = getattr(model, "model_", None)

    def _bi(mm: Any) -> int | None:
        if mm is None:
            return None
        v = getattr(mm, "best_iteration", None)
        if v is None and hasattr(mm, "get_best_iteration"):
            try:
                v = mm.get_best_iteration()
            except Exception:
                v = None
        return v

    if isinstance(inner, dict):
        its = [_bi(m) for m in inner.values()]
        its = [i for i in its if i is not None]
        return round(float(np.mean(its))) if its else None
    return _bi(inner)


def refit_kwargs(
    model_name: str, params: dict[str, Any] | None, candidate_model: Any | None
) -> dict[str, Any]:
    """Build kwargs for the train+val refit, capping ES capacity when known."""
    kwargs = dict(params or {})
    bi = best_iteration(candidate_model) if candidate_model is not None else None
    if bi is not None:
        if model_name == "XGBoost":
            kwargs["n_estimators"] = int(bi) + 1
        elif model_name == "CatBoost":
            kwargs["iterations"] = int(bi) + 1
    return kwargs


def tune_flat_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    n_iter: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Randomized search over the models.yaml grid; best by val MAE.

    Returns:
        {"params": best hyperparameters, "score": val MAE, "model": fitted best
        candidate (for early-stopping iteration extraction)}. ``params`` is None
        if no candidate could be fitted.
    """
    cfg = get_tuning_config()
    n_iter = n_iter or cfg.get("n_iter", 15)
    random_state = random_state or cfg.get("random_state", 42)
    grid = _filter_grid_to_constructor(
        model_name, _apply_override(get_param_grid(model_name), model_name)
    )
    if not grid:
        return {"params": None, "score": float("inf"), "model": None}

    set_seed(random_state)
    candidates = list(ParameterSampler(grid, n_iter=n_iter, random_state=random_state))

    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_model: Any | None = None
    for params in candidates:
        try:
            model = get_model(model_name, **params)
            _fit_with_val(model, X_train, y_train, X_val, y_val)
        except Exception:
            continue
        score = _avg_mae(y_val, model.predict(X_val))
        if score < best_score:
            best_score = score
            best_params = params
            best_model = model

    return {"params": best_params, "score": best_score, "model": best_model}


def tune_weighted_historical_mean(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    party_columns: list[str],
    grid: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Sweep ``decay``/``decay_rate`` for WeightedHistoricalMean on val.

    The baseline is fit on train (past) rows only and scored on the val year;
    the best (decay, decay_rate) is selected purely on the validation MAE.

    Returns:
        {"params": {"decay", "decay_rate"}, "score": val MAE,
        "model": {party: fitted WeightedHistoricalMean}}
    """
    from ..models.baselines import WeightedHistoricalMean

    grid = grid or get_param_grid("WeightedHistoricalMean")
    decays = grid.get("decay", ["exponential", "linear"])
    rates = grid.get("decay_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_models: dict[str, WeightedHistoricalMean] | None = None

    for decay in decays:
        for rate in rates:
            per_party_mae: list[float] = []
            per_party_models: dict[str, WeightedHistoricalMean] = {}
            for party in party_columns:
                model = WeightedHistoricalMean(party_column=party, decay=decay, decay_rate=rate)
                model.fit(train_df, train_df[party])
                pred = model.predict(val_df)
                per_party_mae.append(float(np.mean(np.abs(val_df[party].values - pred))))
                per_party_models[party] = model
            score = float(np.mean(per_party_mae))
            if score < best_score:
                best_score = score
                best_params = {"decay": decay, "decay_rate": rate}
                best_models = per_party_models

    return {"params": best_params, "score": best_score, "model": best_models}


def tune_temporal_model(
    model_name: str,
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    n_iter: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Randomized search for temporal (sequence) models; early stop on val.

    Returns:
        {"params": best hyperparameters, "score": val MAE, "model": fitted best
        candidate}
    """
    cfg = get_tuning_config()
    n_iter = n_iter or cfg.get("n_iter", 10)
    random_state = random_state or cfg.get("random_state", 42)
    grid = _filter_grid_to_constructor(
        model_name, _apply_override(get_param_grid(model_name), model_name)
    )
    if not grid:
        return {"params": None, "score": float("inf"), "model": None}

    set_seed(random_state)
    candidates = list(ParameterSampler(grid, n_iter=n_iter, random_state=random_state))

    y_val_arr = np.stack(y_val).astype(np.float32) if y_val else None
    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_model: Any | None = None
    for params in candidates:
        try:
            model = get_model(model_name, **params)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val_arr)
        except Exception:
            continue
        pred = model.predict(X_val)
        score = _avg_mae(y_val_arr, pred)
        if score < best_score:
            best_score = score
            best_params = params
            best_model = model

    return {"params": best_params, "score": best_score, "model": best_model}
