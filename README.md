# MammaPrint

Michal Kišš, Matěj Pekár

## Getting started

### Instalation

Install [UV](https://docs.astral.sh/uv/) package manager and install all dependencies using the following command:
```bash
uv sync
```

### Preprocessing

Use scripts/preprocessing scripts.

## Local tile-prediction heatmaps

Run a trained embedding MIL checkpoint over a labeled split and create the same
kind of local tile-prediction heatmap used by the prostate project:

```bash
uv run -m ml.predict_heatmaps \
  --checkpoint-uri \
  'mlflow-artifacts:/3/<run-id>/artifacts/checkpoints/best' \
  --split test
```

This starts a new MLflow prediction run and logs:

- `predictions/predictions.csv`, with `record_num`, slide id, ground truth and
  model predictions;
- `predictions/predictions.json`, the same table in MLflow's table viewer;
- `heatmaps/local_tile_prediction/luminal_a_probability/*.tiff` for
  classification models;
- `heatmaps/local_tile_prediction/mammaprint_index/*.tiff` for regression models.

For joint models both heatmap types and both sets of prediction columns are
written. The heatmap value is the model's output when that tile is used as a
one-instance bag. It is a local prediction diagnostic, not a causal attribution
of the full slide prediction.

Submit the same run on an H100 worker with:

```bash
uv run python scripts/ml/submit_predict_heatmaps.py \
  --checkpoint-uri \
  'mlflow-artifacts:/3/<run-id>/artifacts/checkpoints/best' \
  --split test
```
