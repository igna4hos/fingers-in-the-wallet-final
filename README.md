# Прогноз дохода клиентов банка — финальное решение

Дано множество анонимизированных признаков клиента банка. Нужно предсказать его официальный доход.

Группы признаков: обороты по счетам, кредитная история БКИ, данные ПФР (зарплата, взносы, пенсионные
баллы, стаж), транзакции по MCC-категориям, гео, телеком-профиль, цифровой след.

**Результат на публичном лидерборде: WMAE = 74025.86.**

Финальная модель — бленд двух независимых предикторов:

```
submission = 0.80 · seed-bag(градиентный бустинг, N=3)  +  0.20 · NN-корректор
```

## 1. Как запустить

### Вариант A — Google Colab

1. Открыть `solution.ipynb` в Colab.
2. **Runtime → Change runtime type → GPU** (нейросеть ускорится; без GPU тоже запустится, но медленнее).
3. Загрузить `train.csv` и `test.csv` в рабочую папку (панель «Files» слева или `from google.colab import files; files.upload()`).
4. **Runtime → Run all.** По окончании появится `submission.csv`.

Устройство (`cuda` / `mps` / `cpu`) выбирается автоматически, версии библиотек в Colab совместимы.

### Вариант B — локально

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install numpy==2.4.6 pandas==3.0.3 lightgbm==4.6.0 xgboost==3.3.0 \
            torch==2.13.0 scikit-learn==1.9.0 scipy==1.18.0 jupyter nbconvert ipykernel
# train.csv и test.csv лежат рядом с solution.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=4200 solution.ipynb
```

### Библиотеки (точные версии, на которых отлажено)

| Пакет | Версия | | Пакет | Версия |
|---|---|---|---|---|
| python | 3.14.5 | | torch | 2.13.0 |
| numpy | 2.4.6 | | scikit-learn | 1.9.0 |
| pandas | 3.0.3 | | scipy | 1.18.0 |
| lightgbm | 4.6.0 | | xgboost | 3.3.0 |

### Путь от данных до сабмита

```
train.csv / test.csv
  → приведение типов (sep=';', decimal=',', ручной to_numeric для колонок БКИ)
  → feature engineering (ранги внутри месяца, отношения к медианам, target-encoding, H2)
  → CDF-13 стек (13 классификаторов «доход>порог» → 6 признаков формы хвоста)
  → tree-blend (2×LightGBM(MAE) + XGBoost) × seed-bag(3 сида)      ┐
  → NN-корректор (квантильная TabMLP с эмбеддингами сущностей)     ┴→ бленд 0.8/0.2 → submission.csv
```