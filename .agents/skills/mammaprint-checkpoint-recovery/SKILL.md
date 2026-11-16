---
name: mammaprint-checkpoint-recovery
description: Use when replaying selected MammaPrint MLflow runs to recover stale best checkpoints, submitting bounded recovery jobs, or diagnosing guarded checkpoint replacement.
---

# Selected-model checkpoint recovery

Use `scripts/ml/run_checkpoint_recovery.py` from the repository root. One invocation targets exactly one supplied MLflow link or run ID. It never reads the training Sheet or selects an entire sweep.

## Prerequisites and identities

- Preparing code or printing a plan does not authorize a cluster submission or artifact replacement. Obtain explicit authorization for the concrete selected runs before executing `--submit`. A submitted default recovery job **will replace the selected original best checkpoint** after its checks pass. Use `--no-replace-original` to retain a new linked recovery run only.
- Use the approved VPN/kubeconfig and existing `kube_jobs`/Kubernetes environment. The private helper target is context `kubas-cluster`, namespace `rationai-jobs-ns`; flags are identity expectations, not arbitrary routing. The reused submitter prints the configured endpoint, rejects existing jobs and confirms the created UID.
- Authentication uses the existing MLflow environment and helper ConfigMap/Secret references (`mlflow`, `gitlab-deploy-token`), secure DATA and PROJECTS mounts. Never put credentials in links, command arguments, source config overrides or committed files.
- The exact implementation SHA must be pushed to `https://github.com/RationAI/mammaprint.git`. The job checks it out detached. The original training SHA is recovered from the source run tag and checked separately; a missing object may be fetched from the verified origin without checking out historical code. `--training-sha` is only for an absent source tag and cannot override a conflict.
- Original train/validation data paths, label mapping and dependencies must remain available. Replay verifies native source and original dependency pins, but cannot prove historical data bytes or original runtime fingerprints that were never recorded. Same seed does not guarantee identical lost weights.
- Defaults request one H100, 16 CPUs, 64Gi RAM and 16Gi shared memory. These are configurable starting resources, not a guarantee of GPU capacity or sufficient memory. The established image is `cerit.io/rationai/base:2.0.6`; CUDA availability is checked in the pod and has not been remotely verified by local tests.

## Plan, then submit only selected runs

From a clean checkout containing the pushed implementation:

```bash
python scripts/ml/run_checkpoint_recovery.py \
  --source 'https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/3/runs/RUN_ID/model-metrics' \
  --tracking-uri https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
  --git-sha FULL_PUSHED_IMPLEMENTATION_SHA \
  --git-ref refs/heads/feat/concept-explainability \
  --output-root /mnt/projects/mammaprint/recovery/FRESH_SELECTED_RUN \
  --context kubas-cluster --namespace rationai-jobs-ns
```

The source URL endpoint must equal `--tracking-uri`. To use the internal pod endpoint `http://mlflow-s3.rationai-mlflow`, pass `--source RUN_ID` instead of the public URL.

Default output is a fully offline JSON job plan: no MLflow lookup, kubeconfig, process execution or cluster call. Inspect source, target, pins, output path, resources and `replace_original: true`. Append `--submit` to the same command only when the concrete job and replacement are authorized. For another selected model, invoke it again with its explicit ID and fresh output root. There is no `--all` option.

The pod runs `python -m ml.recover_checkpoint ... --execute --replace-original` after frozen dependency setup and CUDA preflight. It unsets inherited `MLFLOW_RUN_ID`. Existing general sweep scripts pointing at `feat/tiling-values` do not automatically use this fix; the recovery job pins the fixed implementation explicitly.

## Stopping rule and preserved training settings

Default selection uses the source ModelCheckpoint monitor and mode: audited runs use minimum `valid/loss`. Complete validation and epoch histories identify the earliest eligible best event. Replay starts from the original seed, preserves the model/data/optimizer/scheduler and original trainer horizon, and stops through a callback after that historical epoch. It does not spend the original early-stopping patience tail and does not run test evaluation.

If the intended goal is a different validation criterion, supply both `--monitor` and `--mode`, for example `--monitor valid/f1 --mode max` for BCE, or `--monitor valid/thresholded_f1 --mode max` for the regression configurations. This changes checkpoint selection and is recorded as such. Test metrics are forbidden for selection. A new trajectory may miss the recorded target; in that case the output is retained and **the original checkpoint is not replaced**.

For a bounded read-only metadata plan without a cluster job, use:

```bash
uv run --frozen -m ml.recover_checkpoint RUN_ID \
  --tracking-uri https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
  --implementation-sha FULL_PUSHED_IMPLEMENTATION_SHA
```

This core plan contacts MLflow for bounded config/history reads and uses temporary files; unlike launcher planning, it is not an offline operation. `--help` remains offline. Ambiguous, nonfinite, truncated or unsupported histories fail rather than guessing an epoch.

## Replacement and evidence

The job captures the original checkpoint hash before training, trains in a **new linked run in the same experiment**, verifies the new checkpoint and its validation target, then:

1. Verifies original/recovery identities, source config/Git binding and finished states.
2. Backs up the original checkpoint and MLmodel under a unique `checkpoint_repairs/OPERATION/backup` namespace outside `checkpoints/`; reads back their hashes.
3. Stages and verifies replacement bytes, rechecks the original hash, replaces only `checkpoints/best/checkpoint.ckpt` and `MLmodel`, then reads both back.
4. Records old/new hashes, recovery run, namespace and publication status. Original config, console and numerical histories remain untouched; historical test scores are explicitly marked **not revalidated** for retrained weights.

Only one repair writer may operate on a given source run at a time. MLflow multi-object writes are not atomic; a hash recheck is not a transaction or atomic compare-and-swap. Backups and a phase journal remain available after partial failure. A CLI failure prints the in-memory partial receipt even when journal storage fails.

Inspect the source run publication tags, linked recovery run, `recovery/recovery-manifest.json`, local `recovery-manifest.json`/`publication-receipt.json`, selected validation score/epoch, exact checkpoint hash and all provenance fields. Successful job creation alone is not successful training or replacement. No new test F1 is claimed unless a separately authorized test evaluation is performed.

For an already completed recovery, manual publication remains available without retraining:

```bash
uv run --frozen -m ml.recover_checkpoint RUN_ID \
  --tracking-uri https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
  --implementation-sha FULL_PUSHED_IMPLEMENTATION_SHA \
  --publish --replace-original --manifest /private/recovery-manifest.json \
  --expected-old-sha256 EXPECTED_ORIGINAL_CHECKPOINT_SHA256
```

## Monitoring, failure and cleanup

Use the exact job name from the plan:

```bash
kubectl --context kubas-cluster -n rationai-jobs-ns get job JOB_NAME -o wide
kubectl --context kubas-cluster -n rationai-jobs-ns logs job/JOB_NAME
kubectl --context kubas-cluster -n rationai-jobs-ns describe job JOB_NAME
```

- Pending GPU/OOM: inspect events and the failing stage; amend resources only for an approved rerun. Do not silently truncate bags or alter architecture.
- Provenance, dependencies, history or missing data: preserve the error and plan. Do not substitute a new training SHA, current config or shorter scheduler horizon.
- Target missed: retain the linked recovery result for review; no automatic extra epochs, changed tolerance or original overwrite.
- Original hash changed: another repair/update occurred. Stop and inspect; do not bypass the guard or run competing writers.
- Partial publication: retain the receipt namespace, phase and old/new hashes. Read canonical checkpoint/MLmodel and compare with those hashes before any retry. Do not assume either object is unchanged.
- Rollback needs separate explicit authorization. Download only that operation's verified `backup/checkpoint.ckpt` and `backup/MLmodel`; verify their recorded hashes, verify the current canonical state matches the failed-operation receipt, then use the standard MLflow artifact client/CLI to restore those two files under `checkpoints/best` and read-verify them again. Never delete or overwrite root configs, console, metric histories or unrelated backups. No automatic rollback is attempted.
- Deleting a job or output needs separate authorization naming the exact object/path. Job deletion does not undo MLflow changes; preserve recovery evidence before cleanup or TTL removal.

Local fake-client and tiny CPU tests establish script behavior, scheduler/cap semantics and guarded artifact handling. They do not establish remote data availability, image/GPU capacity, exact reproducibility, actual cluster execution or real replacement. No remote training or publication is implied by preparing this workflow.
