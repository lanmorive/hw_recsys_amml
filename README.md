# AMML 2026 — Russian passage ranking

`solution.ipynb` is the final reproducible solution. It trains two Jina Reranker v3.5 QLoRA models on the complete competition train set and writes `submission.csv` in the repository root. The final prediction is the mean of the two models' within-query ranks.

## Run

The final run needs Linux, CUDA, an NVIDIA GPU with at least 24 GiB free VRAM (the experiments used RTX A6000 48 GB), and first-run Internet access for `jinaai/jina-reranker-v3.5`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook solution.ipynb
```

The requested format permits only these three files, so the competition CSVs are not committed. Before running, place the original `train.csv` and `test.csv` in the repository root, in `amml-ranking-competition-2026/`, set `AMML_DATA_DIR` to their directory, or attach them as a Kaggle input. The notebook validates the original schema before training; no external labels or texts are used.

## Experiments

Model selection used one fixed query-level holdout: `GroupShuffleSplit(test_size=0.2, random_state=2026, groups=query_id)`. NDCG@10 sorts ties by `id`.

| Experiment | Holdout NDCG@10 |
| --- | ---: |
| TF-IDF + CatBoost YetiRank | 0.769777 |
| BGE reranker | 0.935278 |
| BGE + CatBoost rank blend | 0.936472 |
| Jina v3.5 zero-shot listwise | 0.939814 |
| Jina v3.5 QLoRA: mixed BCE + RankNet | 0.955786 |
| Jina v3.5 QLoRA: BCE only | 0.953746 |
| Jina v3.5 QLoRA: LambdaRank@10 | 0.954897 |
| **rank mean: mixed + LambdaRank@10** | **0.955988** |

The last row is the selected solution. Each branch uses the same model, 4-bit NF4 QLoRA (rank 16), listwise prompts containing all candidates for a query, 384 tokens per passage, learning rate `5e-5`, and gradient accumulation 4. Full-data fitting uses 875 mixed-loss steps and 750 LambdaRank steps—equivalents of the holdout-selected 700 and 600 steps. IDs are never input features.

The notebook resets Python, NumPy, and PyTorch random seeds to 2026 before each model. 4-bit CUDA arithmetic can still cause minor floating-point variation, but should preserve the expected ranking quality. A full A6000 run takes several hours. `submission.csv` is generated output and is intentionally not tracked.
