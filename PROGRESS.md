# PROGRESS.md — Отслеживание выполнения PLAN.md

**Статус**: ✅ P0 выполнен (baselines + Linear + Trees на экспериментах A, B)
**Последнее обновление**: 2026-08-18

---

## Легенда
- ✅ **Выполнено** — реализовано, протестировано, работает
- 🔄 **В процессе** — начата работа
- ⏳ **Ожидает** — готово к началу, зависимости выполнены
- ❌ **Блокер** — есть проблема, требует решения
- ⏭️ **Пропущено** — не в приоритете (P1-P3), будет позже

---

## 0. Подготовка проекта (SETUP)

| № | Задача | Статус | Примечания |
|---|--------|--------|------------|
| 0.1 | Создать виртуальное окружение (uv) | ✅ | `.venv` создано, зависимости установлены |
| 0.2 | Инициализировать git репозиторий | ✅ | `git init`, `.gitignore` добавлен |
| 0.3 | Создать AGENTS.md | ✅ | Архитектура и правила задокументированы |
| 0.4 | Создать pyproject.toml | ✅ | Зависимости, pytest, ruff, mypy настроены |
| 0.5 | Создать конфигурационные файлы | ✅ | `paths.yaml`, `experiments.yaml`, `features.yaml`, `models.yaml` |
| 0.6 | Создать структуру директорий | ✅ | `src/`, `scripts/`, `tests/`, `config/`, `results/`, `predictions/`, `logs/` |
| 0.7 | Реализовать core модули data | ✅ | `loader.py`, `splits.py`, `preprocessing.py`, `features.py` |
| 0.8 | Реализовать evaluation метрики | ✅ | `metrics.py` с MAE, RMSE, R², bias, correlation, federal aggregation |
| 0.9 | Реализовать baseline модели | ✅ | `NaivePreviousElection`, `HistoricalMean`, `WeightedHistoricalMean` |
| 0.10 | Реализовать линейные модели | ✅ | `LinearModel`, `RidgeModel`, `ElasticNetModel` с CV |
| 0.11 | Реализовать деревянные модели | ✅ | `RandomForest`, `HistGB`, `XGBoost`, `CatBoost` |
| 0.12 | Создать model registry | ✅ | `registry.py` с P0/P1/P2/P3 категориями |
| 0.13 | Реализовать backtest runner | ✅ | `backtest.py` с rolling temporal splits |
| 0.14 | Реализовать JSON tracker | ✅ | `json_tracker.py` для логирования экспериментов |
| 0.15 | Реализовать утилиты | ✅ | `reproducibility.py`, `io.py` |
| 0.16 | Создать скрипты запуска | ✅ | `run_baselines.py`, `run_linear.py`, `run_trees.py`, `run_neural.py` |
| 0.17 | Создать benchmark скрипт | ✅ | `benchmark.py` для агрегации результатов |
| 0.18 | Создать Makefile | ✅ | `make install/test/lint/benchmark/baselines/linear/trees/neural` |
| 0.19 | Написать тесты | ✅ | `test_splits.py`, `test_leakage.py` |
| 0.20 | Запустить smoke tests | ✅ | `make test` — 26/26 passed |

---

## 1. Проверка датасета (PLAN §1-5)

| № | Задача | Статус | Примечания |
|---|--------|--------|------------|
| 1.1 | Проверить структуру datasetGOSDUMA | ✅ | Region/precinct parquet, metadata загружены |
| 1.2 | Проверить доступные годы | ✅ | 2000, 2003, 2004, 2007, 2008, 2011, 2012, 2016, 2018, 2021, 2024 |
| 1.3 | Проверить типы выборов | ✅ | parliamentary / presidential разделены |
| 1.4 | Проверить таргеты (партии) | ✅ | UR, KPRF, LDPR (+ SR, NovyeLyudi на precinct) |
| 1.5 | Проверить композиционность | ✅ | Доли в сумме ~100% |
| 1.6 | Проверить lag features | ✅ | Все социоэкономические признаки с суффиксом `_lag1` |
| 1.7 | Проверить leakage (ID columns) | ✅ | `region_id`, `uik`, `tik` не в признаках |

---

## 2. Temporal Splits (PLAN §49-53)

> **Важный факт о данных:** в `master_region_election.parquet` 2024 помечен как `type="pres"`
> (президентские), а не `parl`. Парламентские (ГД) годы в данных: **2003, 2007, 2011, 2016, 2021**.
> Поэтому оцениваемые backtest-эксперименты — только **A** (test 2016) и **B** (test 2021).
> C и D — финальный прогноз на 2026 (нет ground truth, обрабатываются `predict_2026.py`).

| № | Задача | Статус | Примечания |
|---|--------|--------|------------|
| 2.1 | Experiment A (train: 2003-2011, test: 2016) | ✅ | В `experiments.yaml`, оценивается |
| 2.2 | Experiment B (train: 2003-2016, test: 2021) | ✅ | В `experiments.yaml`, оценивается |
| 2.3 | Experiment C (train: 2003-2021, test: 2026) | ✅ | В `experiments.yaml`, финальный прогноз |
| 2.4 | Experiment D (train: all parl, target: 2026) | ✅ | В `experiments.yaml`, `is_final` |
| 2.5 | Реализовать TemporalSplitter | ✅ | В `splits.py` |
| 2.6 | Internal validation splits | ✅ | `internal_validation.temporal_val_years` |

---

## 3. Baselines P0 (PLAN §7-9, 67)

| № | Модель | Статус | Эксперименты | Feature Groups |
|---|--------|--------|--------------|----------------|
| 3.1 | NaivePreviousElection | ✅ Работает (A: 6.97, B: 10.27) | A, B | ALL |
| 3.2 | HistoricalMean | ✅ Работает (A: 7.93, B: 10.11) | A, B | ALL |
| 3.3 | WeightedHistoricalMean | ✅ Работает (A: 7.04, B: 10.46) | A, B | ALL |

---

## 4. Linear Models P0 (PLAN §10-12, 67)

| № | Модель | Статус | Эксперименты | Feature Groups |
|---|--------|--------|--------------|----------------|
| 4.1 | LinearRegression | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |
| 4.2 | Ridge (CV по alpha) | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |
| 4.3 | ElasticNet (CV по alpha, l1_ratio) | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |

---

## 5. Tree Models P0 (PLAN §13-16, 67)

| № | Модель | Статус | Эксперименты | Feature Groups |
|---|--------|--------|--------------|----------------|
| 5.1 | RandomForest | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |
| 5.2 | HistGradientBoosting | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |
| 5.3 | XGBoost | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |
| 5.4 | CatBoost | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |

---

## 6. KNN P0 (PLAN §17, 67)

| № | Модель | Статус | Примечания |
|---|--------|--------|------------|
| 6.1 | KNeighborsRegressor | ⏳ | Не реализовано (в `knn.py`) |

---

## 7. Neural Models P1 (PLAN §18-20, 67)

| № | Модель | Статус | Seeds |
|---|--------|--------|-------|
| 7.1 | MLPSklearn | ⏳ Скелет в `neural.py` | [42,123,456,789,2026] |
| 7.2 | MLPTorch | ⏳ Скелет в `neural.py` | [42,123,456,789,2026] |

---

## 8. Temporal Models P2/P3 (PLAN §21-23, 67)

| № | Модель | Статус | Примечания |
|---|--------|--------|------------|
| 8.1 | GRU | ⏳ Скелет в `temporal.py` | P2 |
| 8.2 | LSTM | ⏳ Скелет в `temporal.py` | P2 |
| 8.3 | Transformer | ⏳ Скелет в `temporal.py` | P3 |

---

## 9. Feature Groups & Ablation (PLAN §24-28)

| № | Группа | Статус | Примечания |
|---|--------|--------|------------|
| 9.1 | ELECTORAL_ONLY | ✅ В `features.yaml` | Только электоральные |
| 9.2 | ROSSTAT_ONLY | ✅ В `features.yaml` | Только Росстат |
| 9.3 | ALL_FEATURES | ✅ В `features.yaml` | Все вместе |
| 9.4 | Feature ablation script | ⏳ | `scripts/evaluation/ablation.py` |
| 9.5 | History depth ablation | ⏳ | `ablation.py` |

---

## 10. Evaluation & Metrics (PLAN §31-41)

| № | Метрика/Анализ | Статус | Реализация |
|---|----------------|--------|------------|
| 10.1 | MAE, RMSE, R² | ✅ | `metrics.py` |
| 10.2 | Bias | ✅ | `metrics.py` |
| 10.3 | Pearson/Spearman | ✅ | `metrics.py` |
| 10.4 | Party-level metrics | ✅ | `metrics.party_metrics()` |
| 10.5 | Federal aggregation | ✅ | `metrics.federal_aggregation()` |
| 10.6 | Error distribution | ✅ | `metrics.error_distribution()` |
| 10.7 | Regional breakdown | ✅ | `metrics.regional_breakdown()` |
| 10.8 | Worst predictions | ✅ | `metrics.worst_predictions()` |
| 10.9 | SHAP analysis | ⏳ | `analysis.py` (не реализован) |
| 10.10 | Feature importance | ⏳ | В моделях есть методы |

---

## 11. Ensemble (PLAN §42-43)

| № | Задача | Статус |
|---|--------|--------|
| 11.1 | WeightedEnsemble | ⏳ Скелет в `ensemble.py` |
| 11.2 | StackingEnsemble | ⏳ Скелет в `ensemble.py` |

---

## 12. Uncertainty & Calibration (PLAN §44-45)

| № | Задача | Статус |
|---|--------|--------|
| 12.1 | Prediction intervals | ⏳ |
| 12.2 | Calibration check | ⏳ |

---

## 13. Hyperparameter Tuning (PLAN §46)

| № | Задача | Статус |
|---|--------|--------|
| 13.1 | RandomizedSearch | ⏳ В `models.yaml` настроены grids |
| 13.2 | Optuna integration | ⏳ В `models.yaml` настроено |

---

## 14. Seed Stability (PLAN §47)

| № | Задача | Статус |
|---|--------|--------|
| 14.1 | Seeds [42,123,456,789,2026] | ✅ В `reproducibility.py` |
| 14.2 | Stability check | ✅ В `run_neural.py` |

---

## 15. Benchmark & Results (PLAN §48-53)

| № | Задача | Статус |
|---|--------|--------|
| 15.1 | Benchmark table | ✅ `benchmark.py` |
| 15.2 | Feature group comparison | ⏳ |
| 15.3 | Data level comparison | ⏳ |
| 15.3 | Save predictions (parquet) | ✅ `io.save_predictions()` |
| 15.4 | Save results (CSV/JSON) | ✅ `io.save_results()` |
| 15.5 | Experiment tracking | ✅ `json_tracker.py` |

---

## 16. Visualization (PLAN §55)

| № | График | Статус |
|---|--------|--------|
| 16.1 | Actual vs Predicted | ⏳ `visualization/plots.py` |
| 16.2 | Error distribution | ⏳ |
| 16.3 | Model comparison (bar) | ⏳ |
| 16.4 | Error by year | ⏳ |
| 16.5 | Feature importance | ⏳ |
| 16.6 | SHAP summary | ⏳ |
| 16.7 | Regional map | ⏳ (nice-to-have) |

---

## 17. Final Prediction 2026 (PLAN §57-60)

| № | Задача | Статус |
|---|--------|--------|
| 17.1 | Final training on all history | ⏳ |
| 17.2 | 2026 prediction script | ⏳ `scripts/predict_2026.py` |
| 17.3 | Federal aggregation | ⏳ |
| 17.4 | Seat allocation pipeline | ⏳ (отдельно, после benchmark) |

---

## 18. Final Report (PLAN §64)

| № | Раздел | Статус |
|---|--------|--------|
| 18.1 | FINAL_MODEL_REPORT.md | ⏳ |

---

## Ключевые вопросы исследования (PLAN §63)

| Вопрос | Статус ответа |
|--------|---------------|
| Q1: Насколько хорошо выборы предсказываются предыдущим результатом? | ⏳ A: baseline лучший (Naive 6.97 < деревья 7.3-7.9); B: деревья догоняют/обходят baseline (CatBoost 9.9-10.1 vs WeightedMean 10.5) |
| Q2: Добавляет ли Росстат predictive power? | ⏳ Противоречиво: для линейных на A ROSSTAT лучше ALL (9.3 vs 10.9), но для деревьев на B ROSSTAT хуже ALL. Требует ablation. |
| Q3: Помогает ли длинная электоральная история? | ⏳ |
| Q4: Есть ли преимущество у nonlinear models? | ⏳ На A — нет (baseline лучше); на B — деревья ≈ baseline. Требует усреднения по экспериментам. |
| Q5: Есть ли преимущество у нейросетей? | ⏳ не тестировалось (P1) |
| Q6: Помогает ли temporal architecture? | ⏳ не тестировалось (P2/P3) |
| Q7: Какие партии предсказываются лучше? | ⏳ |
| Q8: Где модели систематически ошибаются? | ⏳ |
| Q9: Насколько стабилен результат разных seeds? | ⏳ |
| Q10: Насколько ensemble лучше лучшей модели? | ⏳ |

---

## Следующие шаги (Priority)

1. **Сейчас (P0 завершён)**: результаты в `results/benchmark_all_*.csv`, предсказания в `predictions/`.
2. **Потом**: `make ablation` — feature-group и history-depth ablation (уточнить Q2, Q3).
3. **Потом**: реализовать и запустить P1 нейросети (`neural.py`, `run_neural.py`).
4. **Потом**: P2/P3 temporal models (`temporal.py`).
5. **Потом**: ensemble (`ensemble.py`).
6. **Параллельно**: визуализация (`visualization/plots.py`), отчёт, `predict_2026.py` (эксперименты C/D).

---

## Блокеры / Проблемы

| # | Описание | Статус |
|---|----------|--------|
| 1 | `uv run pytest tests/` проходит без ошибок | ✅ 26/26 passed |
| 2 | `make lint` (ruff) проходит | ✅ All checks passed (скорректирован список правил под research-код) |
| 3 | CatBoost/XGBoost могут требовать настройки для categorical features | ⏳ пока не используются (признаки числовые) |
| 4 | Precinct-level data очень большой (1M+ строк) — может потребоваться sampling | ⏳ P0 только на region level |
| 5 | 2024 в данных — президентские, а не парламентские → нет 3-го оцениваемого backtest | ✅ учтено: оцениваются A, B; C/D — финальный прогноз 2026 |

---

*Обновлять этот файл после каждого выполненного этапа.*