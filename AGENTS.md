# AGENTS.md — Архитектура и правила проекта trainingGOSDUMA

## Обзор
Проект: воспроизводимый ML-эксперимент по прогнозированию результатов выборов в ГД РФ.
Датасет: `/home/alexsander/datasetGOSDUMA` (уже подготовлен, содержит region-level и precinct-level данные).

## Структура репозитория
```
trainingGOSDUMA/
├── config/                 # YAML конфиги (не трогать вручную без нужды)
├── src/                    # Основной код (импортируемый пакет)
│   ├── data/               # Загрузка, сплиты, препроцессинг, фичи
│   ├── models/             # Все модели (baselines, linear, trees, neural, temporal, ensemble)
│   ├── evaluation/         # Метрики, бэктест, абляция, анализ ошибок
│   ├── tracking/           # Логирование экспериментов (JSON + опционально MLflow)
│   ├── visualization/      # Графики для отчёта
│   └── utils/              # Воспроизводимость, I/O
├── scripts/                # Точки входа (CLI)
├── tests/                  # Пytest тесты (leakage, splits, metrics)
├── notebooks/              # Исследовательские ноутбуки
├── reports/                # Сгенерированные отчёты
├── results/                # CSV/JSON результатов бенчмарка
├── predictions/            # Parquet предсказаний по моделям/годам
├── logs/                   # Логи запусков
├── pyproject.toml          # Конфиг проекта, инструментов, зависимостей
├── Makefile                # Команды: make benchmark, make test, make lint
├── PROGRESS.md             # Отслеживание выполнения PLAN.md
└── README.md
```

## Ключевые принципы (из PLAN.md)
1. **Только временные сплиты** — прошлое → train, ближайшее будущее → val (для отбора гиперпараметров), далее → test. Никаких `random_state`/перемешивания.
2. **Нет leakage** — препроцессинг fit только на train. Лаговые фичи уже подготовлены в датасете; президентские лаги (`pres_turnout_lag`, `pres_leading_candidate_share_lag`) добавляются кодом и проверяются тестами на соответствие предыдущим президентским выборам.
3. **Динамические таргеты** — список партий читается из данных, не хардкодится.
4. **Композиционность** — предсказываем доли ключевых партий независимыми регрессиями (по одной на партию). В датасете (region-level) только 3 партии: UR, KPRF, LDPR; в парламентских годах они покрывают ~78% голосов (остальное — прочие партии вне данных). Таргеты — реальные доли, **не нормализуются** (normalize_predictions везде по умолчанию `False`; `normalize_compositional` существует только как opt-in).
5. **Базовые бейзлайны обязательны** — NaivePreviousElection, HistoricalMean, WeightedHistoricalMean.
6. **Гиперпараметры только на val** — tuning (Optuna/RandomizedSearch) на валидационном годе (не случайный сплит). `WeightedHistoricalMean` decay тоже тюнится на val.
7. **Двухстадийный протокол** — отбор гиперпараметров на val → финальная модель переобучается на train+val → оценивается на test.
8. **Seed stability** — нейросети: seeds [42, 123, 456, 789, 2026].
9. **Федеральные веса — из RED** — суммирование по УИК (electorate/turnout/valid), не Росстат. Приоритет весовой колонки: `valid` → `turnout` → `electorate`; если ни одной нет — `ValueError`.
10. **Сравнение по лигам** — бенчмарк делится на лиги: Baseline / Tabular / Sequential (см. `LEAGUES` в `scripts/evaluation/benchmark.py`). Модель сравнивается только внутри своей лиги.
11. **Сохранять всё** — predictions/, results/, логи экспериментов.
12. **Порядок выполнения** — строго по §66 PLAN.md (P0 → P1 → P2 → P3).

## Допущения (зафиксированы в документации)
- **Федеральный прогноз 2026** — федеральные веса = последние известные **valid по 2021** как прокси для 2026 (весов 2026 нет в данных).
- **Decay для финального прогноза 2026** (WeightedHistoricalMean) — берётся из tuning на эксперименте B (val=2016), т.е. лучший по 2021.
- **Президентские признаки** — для каждого парламентского года берётся ближайший прошедший президентский год (2003→2000, 2007→2004, 2011→2008, 2016→2012, 2021→2018). Для прогноза 2026 — синтетический контекстный ряд из данных 2024 (pres): сигнал сворачивается в `pres_turnout_lag`/`pres_leading_candidate_share_lag`, а `leading_candidate_share` остаётся NaN (это не признак прогноза, а описание прошлого).
- **Anomalies в данных** — в 2003 `turnout_rate_lag1`/`*_share_lag1` полностью NaN (нет предыдущих парл. выборов) — pre-existing, не чинить.

## Конфигурация (config/)
| Файл | Назначение |
|------|------------|
| `paths.yaml` | Пути к данным датасета, выходным директориям |
| `experiments.yaml` | Определение временных сплитов (Experiments A/B/C/D) |
| `features.yaml` | Группы фич: ELECTORAL_ONLY, ROSSTAT_ONLY, ALL_FEATURES |
| `models.yaml` | Сетки гиперпараметров для каждого класса модели |

## Основные модули (src/)

### data/
- `loader.py` — `load_region()`, `load_precinct()`, `get_party_list()`, `get_available_years()`, `load_electoral_weights()`
- `splits.py` — `TemporalSplitter`, `get_experiment_splits(experiment_name)` — возвращает списки train/val/test лет; `load_raw_region()` (с президентскими фичами)
- `presidential_features.py` — `add_presidential_features()`, `_most_recent_pres_year()` — президентские лаги для парламентских лет
- `preprocessing.py` — `StandardScalerWrapper`, `fit_transform_train_test()`, `ColumnSelector`
- `features.py` — `select_features(df, group_name)`, `get_feature_groups()`

### models/
- `baselines.py` — `NaivePreviousElection`, `HistoricalMean`, `WeightedHistoricalMean`
- `linear.py` — `LinearModel`, `RidgeModel`, `ElasticNetModel` (sklearn wrapper + CV по alpha)
- `trees.py` — `RandomForestModel`, `HistGBModel`, `XGBoostModel`, `CatBoostModel`
- `knn.py` — `KNNModel`
- `neural.py` — `MLPSklearn`, `MLPTorch` (PyTorch)
- `temporal.py` — `GRUModel`, `LSTMModel`, `TransformerModel`
- `ensemble.py` — `WeightedEnsemble`, `StackingEnsemble`
- `registry.py` — `MODEL_REGISTRY`, `get_model(name)`, `list_models()`

### evaluation/
- `metrics.py` — `mae()`, `rmse()`, `r2()`, `bias()`, `pearson_r()`, `spearman_r()`, `party_metrics()`, `federal_aggregation()` (веса из RED, приоритет valid→turnout→electorate)
- `backtest.py` — `run_single_model_backtest`/`run_temporal_backtest` (двухстадийный протокол), `run_multioutput_backtest` (независимые per-party регрессии), `run_rolling_backtest` → dict с метриками по годам; `forecast_temporal` (синтетический pres-контекст для 2026)
- `tuning.py` — `tune_flat_model`/`tune_temporal_model`/`tune_weighted_historical_mean` (на val), `refit_kwargs`, `_filter_grid_to_constructor` (сетка только из сигнатуры конструктора)
- `ablation.py` — `feature_ablation()`, `history_depth_ablation()`
- `analysis.py` — `error_distribution()`, `regional_breakdown()`, `shap_analysis()`

### tracking/
- `json_tracker.py` — `ExperimentTracker` (JSONL + CSV summary)
- `mlflow_tracker.py` — опционально, если нужен UI

### visualization/
- `plots.py` — все обязательные графики из §55
- `maps.py` — опционально, если есть гео-данные

### utils/
- `reproducibility.py` — `set_seed()`, `capture_env()`
- `io.py` — `save_predictions()`, `load_predictions()`, `save_results()`

## Скрипты (scripts/)
| Скрипт | Запускает |
|--------|-----------|
| `scripts/data/build_electoral_weights.py` | Сборка федеральных весов из RED (по УИК) → `electoral_weights.parquet` |
| `scripts/models/run_baselines.py` | P0 baselines |
| `scripts/models/run_linear.py` | Linear, Ridge, ElasticNet |
| `scripts/models/run_trees.py` | RF, HistGB, XGBoost, CatBoost |
| `scripts/models/run_neural.py` | MLP (sklearn + PyTorch) |
| `scripts/models/run_temporal.py` | GRU, LSTM, Transformer |
| `scripts/evaluation/benchmark.py` | Агрегация → benchmark.csv/table (с лигами Baseline/Tabular/Sequential) |
| `scripts/evaluation/ablation.py` | Feature/history ablation |
| `scripts/visualization/generate_report.py` | Графики → reports/ |
| `scripts/predict_2026.py` | Финальный прогноз 2026 (после benchmark) |

## Makefile targets
- `make install` — uv sync
- `make test` — pytest
- `make lint` — ruff check + format
- `make weights` — сборка федеральных весов из RED (`scripts/data/build_electoral_weights.py`)
- `make benchmark` — полный исторический бэктест (запускает все model scripts + benchmark.py)
- `make clean` — удаляет __pycache__, .pytest_cache, predictions/, results/, logs/

## Правила разработки
1. **Никаких жёстких путей** — всё через `config/paths.yaml` и `src/utils/io.py`.
2. **Типизация** — type hints везде, `mypy` в CI.
3. **Документация** — docstrings для публичных функций (Google style).
4. **Тесты** — каждый новый модуль должен иметь хотя бы smoke-test в `tests/`.
5. **Leakage checks** — `tests/test_leakage.py` запускается перед каждым бэктестом.
6. **Прогресс** — обновлять `PROGRESS.md` после каждого выполненного пункта PLAN.md.

## Переменные окружения
- `DATASET_ROOT` — путь к datasetGOSDUMA (по умолчанию `/home/alexsander/datasetGOSDUMA`)
- `MLFLOW_TRACKING_URI` — если используется MLflow
- `RANDOM_SEED` — базовый сид (default 42)

## Запуск эксперимента (пример)
```bash
source .venv/bin/activate
make benchmark
# или поэтапно:
python scripts/models/run_baselines.py --experiment A --feature-group ALL
python scripts/models/run_trees.py --experiment A --feature-group ALL
python scripts/evaluation/benchmark.py --experiment A
```

## Отчёты
- `reports/FINAL_MODEL_REPORT.md` — итоговый отчёт по структуре §64
- `results/benchmark.csv` — сводная таблица MAE по моделям/годам (§48)
- `predictions/<model>/<year>/predictions.parquet` — детальные предсказания (§52)