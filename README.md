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

### Explain selected trained models

Explain one explicitly selected checkpoint per invocation on `feat/explainability`.
The command reuses its logged training configuration and stored embeddings, and
writes a separate MLflow explanation run. It does not train the model again.

Check that the selected checkpoint and configuration load before submission:

```bash
uv run -m ml.explain \
  --checkpoint-uri mlflow-artifacts:/3/d8d6791360b44c5e8c0ca34db1e04817/artifacts/checkpoints/best \
  --tracking-uri https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
  --dry-run
```

The cluster helper clones the remote repository, so publish the updated
`feat/explainability` branch before submitting. To explain that checkpoint's full
test split on one A40:

```bash
uv run python scripts/ml/submit_explain.py \
  --checkpoint-uri mlflow-artifacts:/3/d8d6791360b44c5e8c0ca34db1e04817/artifacts/checkpoints/best \
  --split test --gpu A40
```

The worker reads the training-mean baseline in batches of 1,024 tiles; change this
with `--override explain.ig.baseline_batch_size=512` if needed. It retains float64
accumulation and skips an entire corrupt baseline slide, including any batches
read before the error. Attention tile-removal calculations also limit float64
feature copies to the counterfactual batch size. Explanations retain every tile.

The selected split and baseline split are still downloaded in full. Their disk
usage is separate from RAM usage. Inference still holds one complete slide bag;
these changes reduce temporary allocations, not the model's input size. The
submission command accepts a single checkpoint and never enumerates sweep runs.


The review uses RationAI Reporting's unmodified `report_template.html`, with
per-slide predictions and signed masks opened in xOpat v2. Open the URL printed
as `PATHOLOGIST_REVIEW_URL` (also saved as a run tag). The S3-backed private
artifact mount defaults to `secure_mlflow`; use
`--override explain.output.artifact_mount=...` for another deployment.

For an already completed explanation run, publish this report using only saved
metadata and mask paths, without another GPU job:

```bash
uv run -m ml.explain_report \
  --run-id 9e348bc9ad384171bdabeaba12a05bd0
```

This adds `report/report.html` to that run and updates its review-link tag. It
preserves existing artifacts. Add `--no-upload` to render locally first. Future
explanation jobs produce the same standard report in `explanations/report.html`.

The comparison defaults to tile removal, single-tile output, and attention;
Integrated Gradients is disabled for new explanation runs and omitted when
republishing older runs. Historical scores and masks remain in MLflow.
Tile removal is visible initially in xOpat. The report-only command accepts
`--methods leave_one_out single attention` to specify the displayed methods.
