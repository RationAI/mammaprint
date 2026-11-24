---
name: shared-gpu-training
description: Use when packing independent MammaPrint fits onto one NVIDIA GPU or MIG instance, comparing seed concurrency, or recovering a partially failed training batch.
---

# Shared-GPU training

Run one process per training command with `scripts/ml/run_training_batch.py`. Each process inherits the allocated device. Use `scripts/ml/render_training_batch_job.py` to convert a MammaPrint experiment and explicit seeds into one Kubernetes Job. Each fit gets its own output, MLflow run, and seed. Run with concurrency one for a new workload and GPU profile, then measure two before raising the cap.

## Prerequisites and target checks

- Use the repository's Python 3.12 environment. The offline renderer requires the existing `kube_jobs` package in the submission environment. The worker uses the pinned Python dependencies, NVIDIA PyTorch, and the NVIDIA driver in an approved cluster image.
- Confirm the intended Kubernetes context and namespace using your configured credentials. For the private cluster they are `kubas-cluster` and `rationai-jobs-ns`. Verify the data and project PVCs are bound and pick a writable output subdirectory of `/mnt/projects`. The rendered Job uses the cluster-managed GitLab deploy Secret and internal MLflow tracking. Supply no credentials in overrides or manifests.
- Use a published Git commit for `--source-commit`. An unpublished launcher can be rendered by pairing its committed SHA with `--overlay-base PUBLISHED_SHA`. The renderer embeds only the committed runner and resource probe from the local SHA. The training recipe and dependencies come from the published base. Verify the base has the desired recipe and data card. MLflow tags identify both commits.
- Render and inspect the exact Job name, target, resource request, image, helper and recipe revisions, seed list, and output path. Obtain explicit approval after showing that scope before every cluster submission, including retry or diagnostic Jobs.

## Render a seed batch locally

```bash
uv run --no-sync python -m scripts.ml.render_training_batch_job \
  --context "$TRAIN_CONTEXT" --namespace "$TRAIN_NAMESPACE" \
  --job-name "$TRAIN_JOB" --batch-id "$TRAIN_BATCH" \
  --source-commit "$TRAIN_SOURCE_COMMIT" --image cerit.io/rationai/base:2.0.6 \
  --gpu mig-1g.10gb --cpu 16 --memory 48Gi --shm 8Gi --scratch-size 128Gi \
  --threads-per-fit 2 --active-deadline-seconds 21600 \
  --output-root /mnt/projects/mammaprint/training-batches \
  --experiment ml/train_mil_embeddings_abmil_mean_regression \
  --seeds 101 102 103 104 --max-parallel "$TRAIN_CONCURRENCY" \
  --override trainer.max_epochs=2 --override test_after_fit=false \
  > "$TRAIN_MANIFEST"
```

`--override` accepts single Hydra configuration overrides. Use absolute paths for dataset or checkpoint inputs. The renderer sets the seed, CUDA accelerator and device count, per-run paths, MLflow identity, and tracking provenance. The example runs validation only. The experiment controls its usual evaluation behavior when `test_after_fit` is omitted.

The renderer performs only local actions. Check `spec.template.spec.containers[0].resources` for exactly one NVIDIA resource. The Job metadata pins the target context and source commit. The bootstrap arguments include the literal seed commands. The Kubernetes escaping `$$` is decoded when the pod starts.

After explicit approval for that exact manifest, create it using `kubectl --context "$TRAIN_CONTEXT" --namespace "$TRAIN_NAMESPACE" create -f "$TRAIN_MANIFEST"`. Read status with `kubectl --context "$TRAIN_CONTEXT" --namespace "$TRAIN_NAMESPACE" get job "$TRAIN_JOB"`. Inspect pod events if unscheduled and get logs with `kubectl --context "$TRAIN_CONTEXT" --namespace "$TRAIN_NAMESPACE" logs job/"$TRAIN_JOB" --timestamps` after the container starts.

## Observe and compare

The output is `<output-root>/<batch-id>/<job-name>/<pod-uid>/`. `spec.json` is the input, `status.json` holds per-fit timestamps and states, `gpu.json` identifies the device, and `resources.jsonl` and `resources-summary.json` contain sampled measurements. Each `seed-*/console.log` includes training and data-download output. The runner also prints `training_batch_result=` and `training_batch_resources=` JSON lines in the Job logs. Scratch for a fit is removed after it exits. Checkpoints and logs under the output root survive pod deletion.

For a pilot, run the same seeds, recipe, epoch cap and host resources with concurrency `1` and `2` under distinct Job and batch IDs. Run them sequentially so they do not have to compete for the same host resources. Compare succeeded fits per container-active hour using the pod container's `startedAt` and `finishedAt`, and confirm child time intervals overlap in the concurrent arm. A useful starting threshold is 20% higher throughput with no OOM or artifact errors and around 20% sampled device memory headroom. Sampled peaks can miss short spikes. Use the MIG UUID attached to the samples, and treat an NVML error or missing MIG samples as unknown memory headroom.

Check MLflow for `training_batch_id`, `training_run_id`, `training_attempt_id`, `training_execution_id`, `training_source_commit`, and `training_recipe_commit`. Success includes distinct completed MLflow runs with expected validation metrics and checkpoints. Inspect per-fit logs for source-download times. Remote embedding loading downloads a copy per fit, so disk, host memory, or I/O can dominate. Filesystem free space is a scratch-pressure signal from the underlying filesystem, and `resources.jsonl` identifies samples where host counters were unavailable.

## Recover or clean up

- `failed` means the process failed or could not start. Inspect the error field and console log. If all children fail at startup, inspect the pod logs for clone, environment, GPU probe, and mount errors. If the Job is pending, check events for insufficient `nvidia.com/mig-1g.10gb`, CPU, or RAM.
- `interrupted`, `not_started`, and stale `running` entries can follow cancellation or pod loss. Reconcile questionable entries against the pod result, MLflow tags, and checkpoints. Render a new Job name with the original batch ID and only the selected seeds to retry, then show that exact scope for approval. Keep the prior execution directory for diagnosis.
- Use a Job name outside the old sweep retry prefixes such as `mp-abmil-mig10-a1`. To stop a live Job, identify the exact context, namespace, and Job and obtain approval before deletion. Leave durable outputs available for reconciliation.

## Run a generic local batch

Create a trusted JSON file with `version: 1`, a `batch_id`, a positive `max_parallel`, and ordered `runs` entries with unique `id` and `argv` arrays. Point the runner at a writable output and separate scratch root. The command starts from the calling checkout. Trainers can read `TRAIN_RUN_DIR` and `TRAIN_RUN_ID` for isolated output and identity.

```bash
python -m scripts.ml.run_training_batch --spec "$TRAIN_SPEC" \
  --output-root "$TRAIN_OUTPUT" --scratch-root "$TRAIN_SCRATCH" \
  --attempt-id local-a1 --execution-id local-run1 --dry-run
```

Remove `--dry-run` to run a local batch. The runner accepts commands as arguments and inherits the process environment and device mask. `scripts/ml/training_batch_resources.py --probe` is used by the pod startup to check that exactly one CUDA device is visible. The runner consumes the probe JSON via `--gpu-info` when collecting telemetry.
