# PROGRESS.md — Отслеживание выполнения PLAN.md

**Статус**: ✅ P0, P1, P2, P3 выполнены (baselines + Linear + Trees + KNN + Neural + Temporal на экспериментах A, B)
**Последнее обновление**: 2026-08-19

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
| 0.11 | Реализовать модели на основе деревьев | ✅ | `RandomForest`, `HistGB`, `XGBoost`, `CatBoost` |
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

| № | Модель | Статус | Эксп. | Фич-группы |
|---|--------|--------|-------|------------|
| 6.1 | KNN (KNeighborsRegressor) | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT |

MAE (средний по партиям, п.п.):
- Эксп. A (2016): ALL **7.76**, ELECTORAL 9.90, ROSSTAT 8.14
- Эксп. B (2021): ALL **10.82**, ELECTORAL 10.77, ROSSTAT 11.45
- Поведение: непараметрическая модель, близка к деревьям на A (ALL 7.76 ≈ RF 7.91), но слабее на B (ALL 10.82 > CatBoost 10.12). ROSSTAT помогает KNN на A (8.14 против ELECTORAL 9.90) — как и деревьям.

---

## 7. Neural Models P1 (PLAN §18-20, 67)

| № | Модель | Статус | Эксп. | Фич-группы | Seeds |
|---|--------|--------|-------|------------|-------|
| 7.1 | MLPSklearn | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |
| 7.2 | MLPTorch | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |

MAE (усреднён по 5 seeds, средний по партиям, п.п.):
- MLPSklearn: A ALL **8.42** / ELECTORAL **7.55** / ROSSTAT 9.77; B ALL 12.15 / ELECTORAL 10.39 / ROSSTAT 11.99
- MLPTorch: A ALL 12.81 / ELECTORAL 8.66 / ROSSTAT 12.00; B ALL 12.42 / ELECTORAL 11.14 / ROSSTAT 12.40
- Поведение: MLPSklearn конкурентоспособен (на A ELECTORAL 7.55 — лучше деревьев 8.70), но на B проигрывает деревьям и бейзлайну. MLPTorch заметно слабее — дефолтные гиперпараметры (большие слои 256-128-64) переобучаются на малом табличном датасете (~сотни регионов). Требуется tuning (меньшие сети / регуляризация) или отдельный этап P1-tuning.

---

## 8. Temporal Models P2/P3 (PLAN §21-23, 67)

Реализованы sequence-based временные модели в `src/models/temporal.py` (GRU/LSTM/Transformer
на PyTorch). Каждая выборочная единица — последовательность истории региона
(контекстные годы → таргет-год); импутация (median) и скейлинг fit только на train
(без leakage); ранние годы с NaN-lag отбрасываются как контекст, но НЕ как выборка.
Backtest: `run_temporal_backtest` в `backtest.py`; запуск: `make temporal`, `scripts/models/run_temporal.py`.
Усреднение по 5 seeds [42,123,456,789,2026]. MAE — средний по партиям (п.п.).

| № | Модель | Статус | Эксп. | Фич-группы | Seeds |
|---|--------|--------|-------|------------|-------|
| 8.1 | GRU | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |
| 8.2 | LSTM | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |
| 8.3 | Transformer | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |

MAE (усреднён по 5 seeds, средний по партиям, п.п.):
- GRU: A ALL **6.88**, ELECTORAL 11.66, ROSSTAT 10.61; B ALL 7.60, ELECTORAL 6.52, ROSSTAT 10.08
- LSTM: A ALL 11.26, ELECTORAL 10.68, ROSSTAT 14.21; B ALL 8.10, ELECTORAL 7.03, ROSSTAT 9.68
- Transformer: A ALL **6.20**, ELECTORAL 10.86, ROSSTAT 7.03; B ALL **6.70**, ELECTORAL 7.93, ROSSTAT 8.99
- Поведение: Transformer (ALL) — лучшая модель во всём бенчмарке (A 6.20, B 6.70), заметно
  обходит бейзлайны (Naive 6.97/10.27) и деревья (XGBoost 7.30/10.46). GRU стабильно хорош на ALL.
  LSTM нестабилен (на A ROSSTAT 14.21). На B электоральные признаки важнее (GRU ELECTORAL 6.52 —
  лучший single-строка результат). ROSSTAT полезен для Transformer на A (7.03 vs ELECTORAL 10.86).

---

## 9. Feature Groups & Ablation (PLAN §24-28)

| № | Группа | Статус | Примечания |
|---|--------|--------|------------|
| 9.1 | ELECTORAL_ONLY | ✅ В `features.yaml` | Только электоральные |
| 9.2 | ROSSTAT_ONLY | ✅ В `features.yaml` | Только Росстат |
| 9.3 | ALL_FEATURES | ✅ В `features.yaml` | Все вместе |
| 9.4 | Feature ablation script | ✅ | `scripts/evaluation/ablation.py` → `results/ablation_feature.csv` |
| 9.5 | History depth ablation | ✅ | `ablation.py` → `results/ablation_history.csv` |

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

Реализованы в `src/models/ensemble.py` (регистрируются в `registry.py`):
`WeightedEnsemble` (взвешенное среднее базовых предсказаний, веса = 1/OOF-MAE
**по каждой партии отдельно**) и `StackingEnsemble` (мета-модель LinearRegression
на out-of-fold предсказаниях баз). Базы: XGBoost, CatBoost, RandomForest,
HistGradientBoosting, MLPSklearn, LinearRegression (бейзлайны исключены — не
используют признаки). Запуск через `run_single_model_backtest`.

| № | Задача | Статус | Результат (MAE, A / B, ALL) |
|---|--------|--------|------------|
| 11.1 | WeightedEnsemble | ✅ Работает | 7.91 / 10.83 |
| 11.2 | StackingEnsemble | ✅ Работает | 8.57 / 10.94 |

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
| Q2: Добавляет ли Росстат predictive power? | ✅ Да, но преимущественно при короткой истории (A). Деревья на A: ROSSTAT_ONLY ≪ ELECTORAL_ONLY (XGBoost 7.80 vs 8.96, CatBoost 7.76 vs 8.58, RF 7.98 vs 9.08). На B (длинная история) ROSSTAT нейтрален/мешает деревьям (CatBoost ELECTORAL 9.92 < ALL 10.12 < ROSSTAT 10.54; XGBoost ≈ паритет ~10.46). Временные: ALL лучший, ROSSTAT помогает на A (Transformer ROSSTAT 7.03 < ELECTORAL 10.86). |
| Q3: Помогает ли длинная электоральная история? | ✅ Да — для временных архитектур. Transformer (A): depth2 7.03 → depth3 6.20; (B) depth4 6.70 — лучший. GRU аналогично. Плоские модели — шумная/убывающая отдача (CatBoost A depth2 7.53 < depth3 7.76; Linear нестабилен: A depth2 20.5 > depth1 14.2). Naive — константа (только последний год). |
| Q4: Есть ли преимущество у nonlinear models? | ⏳ На A — нет (baseline лучше); на B — деревья ≈ baseline. Требует усреднения по экспериментам. |
| Q5: Есть ли преимущество у нейросетей? | ⏳ P1 запущен: MLPSklearn конкурентоспособен на A (ELECTORAL 7.55 < деревьев 8.70), но на B проигрывает (ALL 12.15 > baseline 10.27). MLPTorch заметно слабее (переобучение на малом табличном датасете). Явного преимущества нейросетей не выявлено — требует tuning. |
| Q6: Помогает ли temporal architecture? | ✅ Да: Transformer (ALL) — лучшая модель (A 6.20, B 6.70), обходит бейзлайны и деревья. GRU также сильна (A 6.88, B 7.60). LSTM нестабильна. |
| Q7: Какие партии предсказываются лучше? | ⏳ |
| Q8: Где модели систематически ошибаются? | ⏳ |
| Q9: Насколько стабилен результат разных seeds? | ⏳ |
| Q10: Насколько ensemble лучше лучшей модели? | ✅ Не лучше. WeightedEnsemble (A 7.91, B 10.83) и StackingEnsemble (A 8.57, B 10.94) уступают как лучшей одиночной модели — Transformer (ALL 6.20/6.70), так и лучшему «плоскому» дереву (XGBoost 7.30/10.46). Базовые модели слабы на KPRF (LinearRegression 23.7, MLPSklearn 13.5 на A), и их усреднение не компенсирует провал по этой партии. Вывод: ансамбль плоских моделей не даёт преимущества над temporal-архитектурами. |

---

## Следующие шаги (Priority)

1. ✅ **P0/P1/P2/P3 завершены**: P0 (baselines+Linear+Trees+KNN), P1 (MLPSklearn+MLPTorch), P2/P3 (GRU+LSTM+Transformer) запущены на A/B × 3 группы; результаты в `results/benchmark_all_*.csv`.
2. ✅ **Ablation выполнен**: `make ablation` → `results/ablation_feature.csv` (Q2) и `results/ablation_history.csv` (Q3). ROSSTAT помогает при короткой истории (A); длинная история помогает временным моделям.
3. ✅ **Ensemble выполнен** (Q10): `WeightedEnsemble`/`StackingEnsemble` в `ensemble.py`. Не превосходят лучшую одиночную модель (Transformer) и даже XGBoost.
4. **Параллельно**: визуализация (`visualization/plots.py`), отчёт (`reports/FINAL_MODEL_REPORT.md`), `predict_2026.py` (эксперименты C/D).

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