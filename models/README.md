# Models Directory

Model weights are excluded from version control due to file size.
All other files (plots, metrics, .pkl) are committed directly.

## Regenerate DistilBERT weights
```bash
python src/train.py        # ~27 min on GPU/MPS
python src/evaluate.py     # generates plots + metrics.json
python src/model_comparison.py  # generates .pkl + comparison chart
```

## Production model performance

| Metric | Score |
|---|---|
| Accuracy | 98.30% |
| Precision | 98.14% |
| Recall | 98.47% |
| F1 Score | 98.30% |
| ROC-AUC | 0.9980 |