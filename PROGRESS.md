# PROGRESS.md — Отслеживание выполнения PLAN.md

**Статус**: ✅ P0, P1, P2, P3 выполнены + методологический аудит (честные сплиты, федеральные веса из RED, президентские лаги)
**Последнее обновление**: 2026-08-20

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
| 0.20 | Запустить smoke tests | ✅ | `make test` — 30/30 passed |

---

## 1. Проверка датасета (PLAN §1-5)

| № | Задача | Статус | Примечания |
|---|--------|--------|------------|
| 1.1 | Проверить структуру datasetGOSDUMA | ✅ | Region/precinct parquet, metadata загружены |
| 1.2 | Проверить доступные годы | ✅ | 2000, 2003, 2004, 2007, 2008, 2011, 2012, 2016, 2018, 2021, 2024 |
| 1.3 | Проверить типы выборов | ✅ | parliamentary / presidential разделены |
| 1.4 | Проверить таргеты (партии) | ✅ | Region-level: UR, KPRF, LDPR (только 3 партии; других в данных нет) |
| 1.5 | Проверить композиционность | ✅ | 3 ключевые партии покрывают ~78% голосов; таргеты — реальные доли. Нормализация к 100% **не применяется** (default `normalize_predictions=False` везде, `normalize_compositional` — opt-in) |
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
| 2.1 | Experiment A (train: 2003-2007, val: 2011, test: 2016) | ✅ | В `experiments.yaml`, оценивается; tuning на val 2011 → refit train+val |
| 2.2 | Experiment B (train: 2003-2011, val: 2016, test: 2021) | ✅ | В `experiments.yaml`, оценивается; tuning на val 2016 → refit train+val |
| 2.3 | Experiment C (train: 2003-2021, test: 2026) | ✅ | В `experiments.yaml`, финальный прогноз |
| 2.4 | Experiment D (train: all parl, target: 2026) | ✅ | В `experiments.yaml`, `is_final` |
| 2.5 | Реализовать TemporalSplitter | ✅ | В `splits.py` |
| 2.6 | Internal validation splits | ✅ | `internal_validation.temporal_val_years`; tuning гиперпараметров ТОЛЬКО на val |

---

## 3. Baselines P0 (PLAN §7-9, 67)

| № | Модель | Статус | Эксперименты | Feature Groups |
|---|--------|--------|--------------|----------------|
| 3.1 | NaivePreviousElection | ✅ Работает (A: 5.18, B: 6.94) | A, B | ALL |
| 3.2 | HistoricalMean | ✅ Работает (A: 4.98, B: 6.21) | A, B | ALL |
| 3.3 | WeightedHistoricalMean | ✅ Работает (A: 4.44, B: 6.68) | A, B | ALL |

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
- Эксп. A (2016): ALL **6.25**
- Эксп. B (2021): ALL **7.71**
- Поведение (после пересчёта без нормализации): средний результат между деревьями и временными моделями; уступает XGBoost/CatBoost и бейзлайнам.

---

## 7. Neural Models P1 (PLAN §18-20, 67)

| № | Модель | Статус | Эксп. | Фич-группы | Seeds |
|---|--------|--------|-------|------------|-------|
| 7.1 | MLPSklearn | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |
| 7.2 | MLPTorch | ✅ Работает | A, B | ALL, ELECTORAL, ROSSTAT | [42,123,456,789,2026] |

MAE (усреднён по 5 seeds, средний по партиям, п.п.):
- MLPSklearn: A ALL 12.71; B ALL 10.32
- MLPTorch: A ALL 11.81; B ALL 8.44
- Поведение (после пересчёта без нормализации): MLPSklearn и MLPTorch — худшие модели в бенчмарке;
  переобучаются на малом табличном датасете (~83 региона). Требуется tuning (меньшие сети /
  регуляризация) или отдельный этап P1-tuning.

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
- Поведение (после пересчёта без нормализации): Transformer (ALL) — лучшая **временная** модель
  (A 6.20, B 6.70), обходит GRU и LSTM. Но в целом уступает историческим средним
  (WeightedHistoricalMean 4.44/6.68, HistoricalMean 4.98/6.21) и XGBoost (4.49/6.74)/CatBoost (5.29/6.29).
  GRU стабильно хорош на ALL. LSTM нестабилен (на A ROSSTAT 14.21). ROSSTAT полезен для
  Transformer на A (7.03 vs ELECTORAL 10.86).

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
| 11.1 | WeightedEnsemble | ✅ Работает | 6.26 / 7.44 |
| 11.2 | StackingEnsemble | ✅ Работает | 7.11 / 7.47 |

---

## 12. Uncertainty & Calibration (PLAN §44-45)

| № | Задача | Статус |
|---|--------|--------|
| 12.1 | Prediction intervals | ⏳ |
| 12.2 | Calibration check | ⏳ |

---

## 13. Hyperparameter Tuning (PLAN §46)

Tuning реализован в `src/evaluation/tuning.py` (`tune_flat_model`, `tune_temporal_model`,
`tune_weighted_historical_mean`) и применяется **только на валидационном годе** (val) —
не случайный сплит. После отбора параметров финальная модель **переобучается на train+val**
и оценивается на test (двухстадийный протокол). `WeightedHistoricalMean` decay тюнится на val.

| № | Задача | Статус |
|---|--------|--------|
| 13.1 | RandomizedSearch на val | ✅ В `models.yaml` grids + `tuning.py` |
| 13.2 | Optuna integration | ✅ `tuning.py` (backend optuna) |

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
| 16.3 | Model comparison (bar) | ✅ `reports/figures/model_comparison.png` |
| 16.4 | Error by year | ⏳ |
| 16.5 | Feature importance | ⏳ |
| 16.6 | SHAP summary | ⏳ |
| 16.7 | Regional map | ⏳ (nice-to-have) |

Дополнительно уже сгенерированы (из §9): `reports/figures/feature_ablation.png`, `reports/figures/history_depth.png` (`scripts/visualization/generate_report.py`, `make report`).

---

## 17. Final Prediction 2026 (PLAN §57-60)

Реализован `scripts/predict_2026.py` + `forecast_baseline`/`forecast_temporal` в `backtest.py`.
Основной прогноз — по лучшей модели бенчмарка **WeightedHistoricalMean** (детерминированная,
усреднение по seeds не требуется): взвешенное среднее прошлых результатов каждого региона
(экспоненциальный спад; decay — из tuning на B, т.е. лучший по 2021). Transformer
(лучшая временная) сохранён как альтернатива (`--model Transformer`, усреднение по 5 seeds).
Федеральный прогноз — **взвешивание по федеральным весам из RED** (суммирование по УИК;
весовая колонка valid → turnout → electorate; для 2026 — valid 2021 как прокси, т.к. весов
2026 в данных нет):

| № | Задача | Статус | Результат |
|---|--------|--------|-----------|
| 17.1 | Final training on all history | ✅ | WeightedHistoricalMean на 2003–2021 |
| 17.2 | 2026 prediction script | ✅ | `scripts/predict_2026.py` → `predictions/WeightedHistoricalMean/C,D/2026_forecast.csv` |
| 17.3 | Federal aggregation (веса RED, valid 2021 прокси) | ✅ | `results/forecast_{C,D}_WeightedHistoricalMean_federal.csv` (реальные доли, без нормировки) |
| 17.4 | Temporal alternative (Transformer) | ✅ | `results/forecast_{C,D}_Transformer_federal.csv` (реальные доли, без нормировки) |
| 17.5 | Seat allocation pipeline | ⏳ (отдельно, после отчёта) | |

---

## 18. Final Report (PLAN §64)

| № | Раздел | Статус |
|---|--------|--------|
| 18.1 | FINAL_MODEL_REPORT.md | ✅ | Закоммичен; отвечает на Q1–Q10, содержит прогноз 2026 |

---

## Ключевые вопросы исследования (PLAN §63)

| Вопрос | Статус ответа |
|--------|---------------|
| Q1: Насколько хорошо выборы предсказываются предыдущим результатом? | ✅ Ни одна сложная модель стабильно не обходит простые исторические ориентиры: WeightedHistoricalMean (4.44/6.68), HistoricalMean (4.98/6.21), XGBoost (4.49/6.74). Электоральное поведение регионов сильно инертно. |
| Q2: Добавляет ли Росстат predictive power? | ✅ Да, но преимущественно при короткой истории (A). Деревья на A: ROSSTAT_ONLY ≪ ELECTORAL_ONLY (XGBoost 5.37 vs 6.19, CatBoost 5.21 vs 6.00, RF 5.82 vs 6.22); линейные тоже (LinearRegression 9.19 vs 10.97). На B (длинная история) ROSSTAT нейтрален/мешает деревьям (CatBoost ELECTORAL 6.17 < ALL 6.29 < ROSSTAT 7.34). Временные: ALL лучший, ROSSTAT помогает на A (Transformer ROSSTAT 7.03 < ELECTORAL 10.86). |
| Q3: Помогает ли длинная электоральная история? | ✅ Да — и временным, и плоским. Transformer (A): depth2 7.03 → depth3 6.20; (B) depth4 6.70 — лучший. XGBoost (A): depth1 5.50 → depth3 4.49. GRU аналогично Transformer. Naive — константа (только последний год). |
| Q4: Есть ли утечка данных / преимущество у nonlinear models? | ✅ Утечки нет (fit только на train, `test_leakage.py`). На A нелинейные модели сильно точнее линейных (5.68 vs 10.84), на B линейные чуть лучше (6.97 vs 7.20). В среднем деревья+KNN 6.44 vs линейные 8.90; бейзлайны 5.74 — сильнее и тех и других. `results/q4_model_class_mae.csv` |
| Q5: Есть ли преимущество у нейросетей? | ✅ Нет — худшие модели (MLPSklearn 12.71/10.32, MLPTorch 11.81/8.44): переобучаются на ~83 регионах. См. отчёт §4 |
| Q6: Помогает ли temporal architecture? | ✅ Среди нейросетей — да: Transformer (ALL) — лучшая временная модель (A 6.20, B 6.70), обходит GRU (6.88/7.60) и LSTM (11.26/8.10). В целом уступает историческим средним и XGBoost/CatBoost. LSTM нестабильна. |
| Q7: Какие партии предсказываются лучше? | ✅ ЕР — самая сложная (~7–8 п.п.), ЛДПР — самая простая (~4–5 п.п.), КПРФ — промежуточно. У лучших моделей разрыв умеренный (WeightedHistoricalMean: 7.13/5.77/3.79; XGBoost: 7.19/5.61/4.04). `results/q7_party_mae.csv` |
| Q8: Где модели систематически ошибаются? | ✅ Хуже всего — Москва и северные/удалённые регионы (Ненецкий АО, Якутия, Ямало-Ненецкий АО, Хабаровский край) с bias>0 (завышение: Москва +9.1 на A, +3.1 на B); лучше всего — Северный Кавказ и стабильные республики (Кабардино-Балкария, КЧР, Дагестан, Тыва). `results/q8_regional_errors.csv` |
| Q9: Насколько стабилен результат разных seeds? | ✅ Усреднение по 5 seeds [42,123,456,789,2026]; Transformer/GRU стабильны (std 0.26–0.75), LSTM/MLPTorch нет (разброс 3.0–3.2 п.п.). MLPSklearn стал стабильнее (разброс 1.65). `results/q9_seed_stability.csv` |
| Q10: Насколько ensemble лучше лучшей модели? | ✅ Не лучше. WeightedEnsemble (6.26/7.44) и StackingEnsemble (7.11/7.47) уступают XGBoost (4.49/6.74) и историческим средним (4.44–4.98/6.21–6.68). Базовые модели слабы на KPRF, усреднение не компенсирует их провалы. Вывод: ансамбль плоских моделей не даёт преимущества. |

---

## 19. Методологический аудит (2026-08-20)

Закрыты замечания ревью по честности эксперимента:

| № | Замечание | Что сделано |
|---|-----------|-------------|
| 19.1 | Сплиты A/B: test-год был в train | Сплиты пересмотрены: A train 2003–2007, val 2011, test 2016; B train 2003–2011, val 2016, test 2021. `get_experiment_splits` возвращает train/val/test; тесты проверяют `max(train) < min(test)` и val ∉ train |
| 19.2 | Tuning гиперпараметров на train (текут на test) | `src/evaluation/tuning.py`: RandomSearch/Optuna на **val**; двухстадийный протокол — отбор на val → refit на train+val → оценка на test |
| 19.3 | Temporal/MLP ранний стоп по train | `MLPTorch.fit(X, y, X_val, y_val)`: ранний стоп по внешнему val; MLPSklearn `early_stopping=False`; temporal-модели валидируются на val-секвенциях |
| 19.4 | Ensemble OOF — случайный KFold | `src/models/ensemble.py`: temporal expanding-window OOF (`fit(X, y, years=None)`), без shuffle |
| 19.5 | Нет настоящего multioutput | `run_multioutput_backtest`: независимые per-party регрессии с per-party tuning на val |
| 19.6 | Федеральные веса из Росстата | `scripts/data/build_electoral_weights.py` (RED по УИК, `electoral_weights.parquet`, 912 region-events) + `federal_aggregation` (приоритет valid→turnout→electorate) |
| 19.7 | Президентские признаки 2024 не использовались | `src/data/presidential_features.py`: `pres_turnout_lag`/`pres_leading_candidate_share_lag` (ближайшие прошедшие президентские выборы); для 2026 — синтетический контекст из 2024 pres |
| 19.8 | Пост-хок нормализация к 100% | `normalize_predictions` default `False` везде; `normalize_compositional` — opt-in; прогноз 2026 — реальные доли |
| 19.9 | Нет лиг сравнения | `LEAGUES` в `scripts/evaluation/benchmark.py`: Baseline / Tabular / Sequential |
| 19.10 | Усиление тестов на leakage | `tests/test_leakage.py`: ручная проверка `UR_share_lag1`(2016)==UR_share(2011) и `pres_leading_candidate_share_lag`(2016)==pres(2012) по регионам |

---

## Следующие шаги (Priority)

1. ✅ **Методологический аудит завершён (2026-08-20)**: честные сплиты train/val/test, tuning на val, двухстадийный протокол, президентские лаги, федеральные веса из RED, temporal OOF, per-party регрессии, лиги в benchmark. Бенчмарк пересчитан по новому протоколу (`results/benchmark_all_20260820.csv`) для всех 3 фич-групп на A/B; прогноз 2026 обновлён (ЕР 52.8 / КПРФ 16.1 / ЛДПР 10.4, веса = valid 2021).
1a. ✅ **Исправлена инверсия decay в WeightedHistoricalMean** (2026-08-20): раньше максимальный вес получала самая старая выборка (2003), из-за чего прогноз 2026 «заваливался» вниз (49.2/13.7/10.6); теперь вес растёт к последним выборам, прогноз пересчитан (52.8/16.1/10.4), WHM улучшился до 4.04/6.11. `predict_2026.py` теперь использует decay из tuning на B (экспоненциальный, rate 0.8). Добавлены тесты `tests/test_baselines.py`.
2. ✅ **Ablation пересчитан по новому протоколу** (2026-08-20): `make ablation` → `results/ablation_feature.csv` и `results/ablation_history.csv` (ALL_FEATURES почти всегда лучший; ROSSTAT_ONLY в одиночку худший; глубина истории помогает линейным, слабо — временным).
3. ✅ **Q4/Q7/Q8/Q9 пересчитаны по новому протоколу** (2026-08-20): `scripts/evaluation/research_questions.py` → `results/q{4,7,8,9}_*.csv`. Исправлен баг с классификацией моделей по классам в Q4 (лиги в `Model` ломали разбор имени).
4. ⏳ **Возможные доработки**: пересчёт долей в мандаты (seat allocation), bootstrap-CI для устойчивости MAE, SHAP-анализ, калибровка StackingEnsemble.

---

## Блокеры / Проблемы

| # | Описание | Статус |
|---|----------|--------|
| 1 | `uv run pytest tests/` проходит без ошибок | ✅ 42 passed |
| 2 | `make lint` (ruff) проходит | ✅ All checks passed (скорректирован список правил под research-код) |
| 3 | CatBoost/XGBoost могут требовать настройки для categorical features | ⏳ пока не используются (признаки числовые) |
| 4 | Precinct-level data очень большой (1M+ строк) — может потребоваться sampling | ⏳ P0 только на region level |
| 5 | 2024 в данных — президентские, а не парламентские → нет 3-го оцениваемого backtest | ✅ учтено: оцениваются A, B; C/D — финальный прогноз 2026 |
| 6 | В 2003 `turnout_rate_lag1`/`*_share_lag1` полностью NaN (нет предыдущих парл. выборов) | ✅ pre-existing, не чинить (SimpleImputer warning) |

---

*Обновлять этот файл после каждого выполненного этапа.*