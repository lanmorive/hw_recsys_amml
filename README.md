# AMML 2026 — ранжирование русскоязычных текстов

## Что делает решение

`solution.ipynb` обучает два listwise-reranker на `train.csv`, получает оценки для `test.csv` и сохраняет `submission.csv`. Используется Jina Reranker v3.5 с QLoRA. Финальный прогноз — среднее рангов двух моделей внутри каждого `query_id`.

## Запуск

Нужны Linux, CUDA и GPU с 24+ GiB свободной памяти (запуски проводились на RTX A6000 48 GB). При первом запуске требуется интернет для загрузки базовой модели.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook solution.ipynb
```

В репозитории только обязательные файлы, поэтому `train.csv` и `test.csv` не включены. Ноутбук ищет их в `AMML_DATA_DIR`, в текущей папке, в `amml-ranking-competition-2026/` или среди Kaggle inputs. На Kaggle результат сохраняется в `/kaggle/working/submission.csv`.

## Проверенные варианты

Оценка проводилась на holdout по запросам: `GroupShuffleSplit(test_size=0.2, random_state=2026, groups=query_id)`. Метрика — NDCG@10.

| Модель | NDCG@10 |
| --- | ---: |
| TF-IDF + CatBoost | 0.769777 |
| BGE reranker | 0.935278 |
| BGE + CatBoost | 0.936472 |
| Jina v3.5, mixed BCE + RankNet | 0.955786 |
| Jina v3.5, LambdaRank@10 | 0.954897 |
| Финальный rank blend | 0.955988 |

У обеих финальных моделей: NF4 4-bit QLoRA, rank 16, длина фрагмента 384 токена, learning rate `5e-5`, gradient accumulation 4. Mixed-loss модель обучается 875 шагов, LambdaRank — 750 шагов. Идентификаторы не используются как признаки.

Полный прогон на A6000 занимает несколько часов. Из-за 4-bit CUDA-вычислений возможны небольшие различия между запусками.
