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
1. **Только временные сплиты** — прошлое → train, будущее → test. Никаких `random_state`.
2. **Нет leakage** — препроцессинг fit только на train. Лаговые фичи уже подготовлены в датасете.
3. **Динамические таргеты** — список партий читается из данных, не хардкодится.
4. **Композиционность** — предсказываем доли ключевых партий. В датасете (region-level) только 3 партии: UR, KPRF, LDPR; в парламентских годах они покрывают ~78% голосов (остальное — прочие партии вне данных), поэтому **нормализация к 100% не применяется**. Используется подход (Б) multi-output / независимые регрессоры на каждую партию.
5. **Базовые бейзлайны обязательны** — NaivePreviousElection, HistoricalMean, WeightedHistoricalMean.
6. **Гиперпараметры только на train** — Optuna/RandomizedSearch внутри internal validation.
7. **Seed stability** — нейросети: seeds [42, 123, 456, 789, 2026].
8. **Сохранять всё** — predictions/, results/, логи экспериментов.
9. **Порядок выполнения** — строго по §66 PLAN.md (P0 → P1 → P2 → P3).

## Конфигурация (config/)
| Файл | Назначение |
|------|------------|
| `paths.yaml` | Пути к данным датасета, выходным директориям |
| `experiments.yaml` | Определение временных сплитов (Experiments A/B/C/D) |
| `features.yaml` | Группы фич: ELECTORAL_ONLY, ROSSTAT_ONLY, ALL_FEATURES |
| `models.yaml` | Сетки гиперпараметров для каждого класса модели |

## Основные модули (src/)

### data/
- `loader.py` — `load_region()`, `load_precinct()`, `get_party_list()`, `get_available_years()`
- `splits.py` — `TemporalSplitter`, `get_experiment_splits(experiment_name)` — возвращает списки train/val/test лет
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
- `metrics.py` — `mae()`, `rmse()`, `r2()`, `bias()`, `pearson_r()`, `spearman_r()`, `party_metrics()`, `federal_aggregation()`
- `backtest.py` — `run_rolling_backtest(model, splits, feature_group)` → возвращает dict с метриками по годам
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
| `scripts/models/run_baselines.py` | P0 baselines |
| `scripts/models/run_linear.py` | Linear, Ridge, ElasticNet |
| `scripts/models/run_trees.py` | RF, HistGB, XGBoost, CatBoost |
| `scripts/models/run_neural.py` | MLP (sklearn + PyTorch) |
| `scripts/models/run_temporal.py` | GRU, LSTM, Transformer |
| `scripts/evaluation/benchmark.py` | Агрегация → benchmark.csv/table |
| `scripts/evaluation/ablation.py` | Feature/history ablation |
| `scripts/visualization/generate_report.py` | Графики → reports/ |
| `scripts/predict_2026.py` | Финальный прогноз 2026 (после benchmark) |

## Makefile targets
- `make install` — uv sync
- `make test` — pytest
- `make lint` — ruff check + format
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