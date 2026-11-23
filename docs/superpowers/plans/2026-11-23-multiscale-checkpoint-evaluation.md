# Multiscale checkpoint evaluation implementation plan

## Goal

Recover or recreate the best L4/L3/L2 slide-attention regression checkpoints for CM5 and EAN5, then evaluate them on the original test split. Each evaluation should produce the native `test/*` metrics, one continuous prediction per evaluated slide, and a pyramid coverage audit. Publish metrics to the original run only when evaluating its recovered original checkpoint. Record a replay's metrics on its linked recovery run.

| Mask | Original MLflow run ID | Original GPU | Saved epoch and validation loss | Best logged epoch and validation loss |
| --- | --- | --- | --- | --- |
| CM5 | `a24468e200d24c569e9e68c06107b70c` | 1 × NVIDIA A40 | 0, `0.13345987` | 37, `0.05476137` |
| EAN5 | `6405fef8a83e4e7e985d3dd34738ee00` | 1 × NVIDIA A40 | 0, `0.07916635` | 40, `0.05952283` |

Epochs are zero-based. The console logs identify the GPU and a single visible CUDA device for each run. The downloaded `checkpoints/best/checkpoint.ckpt` files contain epoch 0 and match the first validation loss in each run. The training histories reached their minima at MLflow steps `54149` and `58096`, corresponding to checkpoint `global_step` values `54150` and `58097`. Each MLflow run has only the stale `best` checkpoint. Both named training Jobs were absent when checked in `kubas-cluster` / `rationai-jobs-ns`. [CM5 run](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/3/runs/a24468e200d24c569e9e68c06107b70c/artifacts), [EAN5 run](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/3/runs/6405fef8a83e4e7e985d3dd34738ee00/artifacts)

The upload fix is `f60f4b7` on `feat/concept-explainability`. It is absent from the ancestry of `feat/modular-multiscale-mil`, and both original runs used the upstream `rationai.mlkit` logger. The original source SHA `a8d498a59efa8bdcfd38310195cd1d4985c7bbdc` is absent from the local object store. The saved Hydra overrides refer to screen presets absent from this checkout. Use `configs/config-resolved.yaml` as the run's config source and verify the implementation and dependency pins separately.

## 1. Recover or recreate the best checkpoints

- [ ] Check the original training storage or backups for a later `best.ckpt`. Verify its provenance, `valid/loss` callback score, epoch, global step, and SHA-256 against the source run's history before selecting it for evaluation. The epoch-0 artifacts fail this check.
- [ ] If the original best file cannot be recovered, recover the original training source and dependency pins, then port `f60f4b7` and its logger tests onto that source. Confirm the resolved training config instantiates `ml.loggers.MLFlowLogger`. Test that repeated saves at `best.ckpt` replace the remote bytes after an improvement and that an upload failure can be retried.
- [ ] Reuse the training replay and verification pattern in `feat/concept-explainability` (`ml/recover_checkpoint.py` and `scripts/ml/run_checkpoint_recovery.py`). Verify compatibility with `PyramidScaleDataset` and use the original seed, preprocessing artifacts, split, pyramid selection, precision, and early stopping. Submit each necessary fit with `--gpu A40` after approval. Train in a linked recovery run, verify the uploaded checkpoint against its own validation history, and report any difference from the original history. A seeded replay produces a new checkpoint with its own provenance.

## 2. Add a test-only checkpoint evaluator

- [ ] Add `scripts/ml/evaluate_multiscale_checkpoint.py` on `feat/modular-multiscale-mil`. Accept one `--run-id`, `--tracking-uri`, `--output-dir`, `--dry-run`, and explicit `--log-to-run`. Default to local outputs. Make dry-run verify checkpoint metadata against the validation history and report the test inputs without loading the embeddings or writing to MLflow.
- [ ] Check that the run is an active, finished MammaPrint run in experiment `3` with the expected multiscale regression configuration. Download the resolved config and checkpoint to an existing scratch directory. Reuse the artifact validation in `ml/resume_test.py` and enforce the epoch, step, and score check against that run's validation history before launching inference. For a recovered original file, accept a local checkpoint path and apply the same check. Record the source commit and hashes of the resolved config and checkpoint in the result.
- [ ] Build `config.ml` and the lazy `config.datamodule` from the saved config. Preserve the test embedding URIs, label mapping, pyramid layout and selection, precision, and zero thresholds for predictions and targets. Load the checkpoint strictly. If the saved config or weights are incompatible, report the mismatch before writing results and recover compatible source rather than substituting a current preset or dropping weights.
- [ ] Call Lightning `Trainer.test` once with training callbacks and MLflow autologging disabled. Attach a test-only collector that pairs each model output with its batch label and `metadata["slide_id"]` from the same forward pass. Follow the collector pattern in `9d14879:ml/test_evaluation.py` and keep the training path independent.

## 3. Validate and publish the outputs

- [ ] Prepare `predictions.csv` with `slide_id`, continuous `score`, and continuous `target`. Export `pyramid_selection_audit.csv` from the test dataset's existing `audit_frame()`. Require unique slide IDs, finite scores and targets, and exact agreement between prediction IDs and the audit's accepted slides. Report candidate, evaluated, and excluded slide counts. The earlier mean-model audits provide a coverage cross-check of 305/308 CM5 and 303/308 EAN5 when the dataset and selection settings match.
- [ ] Prepare `metrics.json` with the `test/*` metrics, checkpoint provenance and step, thresholds, input versions, and coverage counts. Recompute MAE, MSE, and zero-cutoff F1 from the exported rows and check agreement with the native metrics within floating-point tolerance.
- [ ] With `--log-to-run`, use `MlflowClient` to upload the predictions and audit under `test_evaluation/best_checkpoint/` and append native `test/*` scalars to the same run ID at the checkpoint step. Limit writes to those artifact and metric paths. Publish `metrics.json` last as the completion record. On retry, skip identical artifacts and metric points, resume an incomplete matching write, and stop on conflicting results. Allow one writer per run.

## 4. Verify locally, then evaluate the two runs

- [ ] Add `tests/test_evaluate_multiscale_checkpoint.py`. Cover dry-run with a client that rejects writes, a tiny CPU test dataset and saved checkpoint, an epoch-0 file mislabeled as `best`, incompatible weights, prediction and audit mismatch, and publication retry after interruption. Confirm successful publication targets only the evaluated run ID and evaluation artifact prefix. Run `uv run --frozen pytest tests/test_evaluate_multiscale_checkpoint.py tests/test_resume_test.py -q` and obtain independent review before live evaluation.
- [ ] Add `.agents/skills/multiscale-checkpoint-evaluation/SKILL.md` with the recovery and evaluator commands, required data mapping and MLflow access, verified target context and namespace, scratch requirements, expected outputs, and retry procedure. Render each needed Job for one A40. Show the exact submission, `kubas-cluster` context, `rationai-jobs-ns` namespace, and affected run ID before obtaining explicit submission approval.
- [ ] Evaluate CM5 first. Read the evaluated run back and confirm that `test/mae`, `test/mse`, `test/thresholded_f1`, the prediction and audit files, and checkpoint provenance are present. Once that path is verified, apply it to EAN5 with approval covering its submission.

## Follow-up comparison

After both are evaluated, export predictions from the matching L2-only and multiscale mean checkpoints using the same slide-ID schema and an evaluation-only artifact path. Within each mask, join on shared slide IDs, verify targets, and use the existing patient mapping to bootstrap paired differences in MAE and fixed-cutoff F1. Keep the full eligible counts in the comparison so the common-cohort result can be interpreted alongside coverage.
