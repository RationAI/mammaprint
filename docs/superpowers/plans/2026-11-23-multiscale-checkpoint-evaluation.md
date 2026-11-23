# Multiscale checkpoint evaluation implementation plan

## Goal

Evaluate the two saved L4/L3/L2 slide-attention regression checkpoints on their original test split and add the results to the original MLflow runs. Each evaluation should produce the native `test/*` metrics, one continuous prediction per evaluated slide, and an audit of which slides the pyramid could evaluate.

| Mask | Original MLflow run ID | Checkpoint |
| --- | --- | --- |
| CM5 | `a24468e200d24c569e9e68c06107b70c` | `checkpoints/best/checkpoint.ckpt` |
| EAN5 | `6405fef8a83e4e7e985d3dd34738ee00` | `checkpoints/best/checkpoint.ckpt` |

Both runs have saved checkpoints and `configs/config-resolved.yaml`. They were trained with `test_after_fit=false`, seed 0, and a complete non-overlapping pyramid of levels `[4, 3, 2]`. Their logged Hydra overrides refer to screen presets that are absent from this checkout. The existing `ml.resume_test` path replays those presets and uses training autologging, which rewrites artifacts under `configs/`. Use the saved resolved config and an explicit MLflow write after successful inference. [CM5 run](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/3/runs/a24468e200d24c569e9e68c06107b70c/artifacts), [EAN5 run](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/3/runs/6405fef8a83e4e7e985d3dd34738ee00/artifacts)

## 1. Add a test-only checkpoint evaluator

- [ ] Add `scripts/ml/evaluate_multiscale_checkpoint.py` on `feat/modular-multiscale-mil`. Accept one `--run-id`, `--tracking-uri`, `--output-dir`, `--dry-run`, and explicit `--log-to-run`. Default to local outputs. Make dry-run read the run and config, verify the checkpoint entry, and report the test inputs without loading the embeddings or writing to MLflow.
- [ ] Check that the run is an active, finished MammaPrint run in experiment `3` with the expected multiscale regression configuration. Download the resolved config and saved checkpoint to an existing scratch directory. Reuse the checkpoint artifact validation in `ml/resume_test.py`. Record the original source commit and hashes of the resolved config and checkpoint in the result.
- [ ] Build `config.ml` and the lazy `config.datamodule` from the saved config. Preserve the test embedding URIs, label mapping, pyramid layout and selection, precision, and zero thresholds for predictions and targets. Load the checkpoint strictly. If the saved config or weights are incompatible, report the mismatch before writing results and recover compatible source rather than substituting a current preset or dropping weights.
- [ ] Call Lightning `Trainer.test` once with training callbacks and MLflow autologging disabled. Attach a test-only collector that pairs each model output with its batch label and `metadata["slide_id"]` from the same forward pass. Follow the collector pattern in `9d14879:ml/test_evaluation.py` and keep the training path independent.

## 2. Validate and publish the outputs

- [ ] Prepare `predictions.csv` with `slide_id`, continuous `score`, and continuous `target`. Export `pyramid_selection_audit.csv` from the test dataset's existing `audit_frame()`. Require unique slide IDs, finite scores and targets, and exact agreement between prediction IDs and the audit's accepted slides. Report candidate, evaluated, and excluded slide counts. The earlier mean-model audits provide a coverage cross-check of 305/308 CM5 and 303/308 EAN5 when the dataset and selection settings match.
- [ ] Prepare `metrics.json` with the `test/*` metrics, checkpoint provenance and step, thresholds, input versions, and coverage counts. Recompute MAE, MSE, and zero-cutoff F1 from the exported rows and check agreement with the native metrics within floating-point tolerance.
- [ ] With `--log-to-run`, use `MlflowClient` to upload the predictions and audit under `test_evaluation/best_checkpoint/` and append native `test/*` scalars to the same run ID at the checkpoint step. Limit writes to those artifact and metric paths. Publish `metrics.json` last as the completion record. On retry, skip identical artifacts and metric points, resume an incomplete matching write, and stop on conflicting results. Allow one writer per run.

## 3. Verify locally, then evaluate the two runs

- [ ] Add `tests/test_evaluate_multiscale_checkpoint.py`. Cover dry-run with a client that rejects writes, a tiny CPU test dataset and saved checkpoint, strict failure on incompatible weights, prediction and audit mismatch, and an interrupted publication followed by a retry. Confirm successful publication targets only the original run ID and the evaluation artifact prefix. Run `uv run --frozen pytest tests/test_evaluate_multiscale_checkpoint.py tests/test_resume_test.py -q` and obtain independent review before live evaluation.
- [ ] Add `.agents/skills/multiscale-checkpoint-evaluation/SKILL.md` with the evaluator command, required data mapping and MLflow access, verified target context and namespace, scratch requirements, expected outputs, and retry procedure. Use the existing job tooling for execution. Show the exact proposed submission, target context, `rationai-jobs-ns` namespace, and affected run ID before obtaining explicit cluster submission approval.
- [ ] Evaluate CM5 first. Read its MLflow run back and confirm that `test/mae`, `test/mse`, `test/thresholded_f1`, the prediction and audit files, and checkpoint provenance are present. Confirm the original configs and checkpoint are intact. Once that path is verified, use the same evaluator for EAN5 with approval covering its submission.

## Follow-up comparison

After both are evaluated, export predictions from the matching L2-only and multiscale mean checkpoints using the same slide-ID schema and an evaluation-only artifact path. Within each mask, join on shared slide IDs, verify targets, and use the existing patient mapping to bootstrap paired differences in MAE and fixed-cutoff F1. Keep the full eligible counts in the comparison so the common-cohort result can be interpreted alongside coverage.
