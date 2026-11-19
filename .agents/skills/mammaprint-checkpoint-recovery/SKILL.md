---
name: mammaprint-checkpoint-recovery
description: Use when retraining selected MammaPrint MLflow runs under their original protocol, submitting recovery jobs, or diagnosing guarded checkpoint replacement.
---

# Selected-model checkpoint recovery

Use `scripts/ml/run_checkpoint_recovery.py` from the repository root. One invocation targets exactly one supplied MLflow link or run ID. It never reads the training Sheet or selects an entire sweep.

## Prerequisites and identities

- Preparing code or printing a plan does not authorize a cluster submission or artifact replacement. Obtain explicit authorization for the concrete selected runs before executing `--submit`. A submitted default recovery job **will replace the selected original best checkpoint** after its checks pass. Use `--no-replace-original` to retain a new linked recovery run only.
- Use the approved VPN/kubeconfig and existing `kube_jobs`/Kubernetes environment. The private helper target is context `kubas-cluster`, namespace `rationai-jobs-ns`; flags are identity expectations, not arbitrary routing. The reused submitter prints the configured endpoint, rejects existing jobs and confirms the created UID.
- Authentication uses the existing MLflow environment and helper ConfigMap/Secret references (`mlflow`, `gitlab-deploy-token`), secure DATA and PROJECTS mounts. Never put credentials in links, command arguments, source config overrides or committed files.
- The exact implementation SHA must be pushed to `https://github.com/RationAI/mammaprint.git`. The job checks it out detached. The original training SHA is recovered from the source run tag and checked separately; a missing object may be fetched from the verified origin without checking out historical code. `--training-sha` is only for an absent source tag and cannot override a conflict.
- Original train/validation/test data paths, label mapping and dependencies must remain available. Replay verifies native source and original dependency pins, but cannot prove historical data bytes or original runtime fingerprints that were never recorded. Same seed does not guarantee identical lost weights.
- Defaults request one H100, 16 CPUs, 64Gi RAM and 16Gi shared memory. These are configurable starting resources, not a guarantee of GPU capacity or sufficient memory. The established image is `cerit.io/rationai/base:2.0.6`; CUDA availability is checked in the pod and has not been remotely verified by local tests.
- The existing helper's `mig-1g.10gb` profile is also accepted. For the selected L2 linear + gated-attention regression family, begin with `--gpu mig-1g.10gb --cpu 8 --memory 32Gi`; seed 2 historically completed on that profile, but recovery peak usage is not yet measured. Keep the default 16Gi shared-memory capacity; actual shared-memory usage counts toward host RAM. Keep the original batch size and worker settings; do not truncate bags to fit.

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

MIG planning can be checked locally without Kubernetes: run the example with the resource flags above and omit `--submit`. Expect `resources.gpu: mig-1g.10gb`, `gpu_limit: 1`, CPU 8, RAM 32Gi and default shared memory 16Gi; the installed helper resolves the GPU limit to `nvidia.com/mig-1g.10gb: 1`. Launcher tests cover resource forwarding using a mocked submitter; this does not establish live MIG availability or successful training.

The pod runs `python -m ml.recover_checkpoint ... --execute --replace-original` after frozen dependency setup and CUDA preflight. It unsets inherited `MLFLOW_RUN_ID`. Existing general sweep scripts pointing at `feat/tiling-values` do not automatically use this fix; the recovery job pins the fixed implementation explicitly.

## Stopping rule and preserved training settings

Default selection uses the source ModelCheckpoint monitor and mode: audited runs use minimum `valid/loss`. Training starts from the original seed and preserves the model, data, optimizer, scheduler, maximum epochs/steps and native early-stopping settings. There is no historical-score stop or historical-epoch cap. Complete validation and epoch histories provide a comparison only; a score worse than the historical best does not block replacement. For model 1 seed 0 (`b302bc81001f41f780086b6364e32b50`), the source specifies max_epochs 100 and EarlyStopping patience 10 on minimum valid/loss.

After fit finishes normally, the job explicitly loads the saved best validation checkpoint and calls native test evaluation once on the original held-out test split. Test scores never select the checkpoint or gate publication by performance. The linked run records the new test metrics and `recovery/test-evaluation.json`, bound to the selected checkpoint hash/epoch/step. Interrupted training, failed testing, empty or nonfinite test metrics, or artifact integrity failures prevent replacement. The source run's historical test scores are not reattributed to the replacement weights.

For single-output index regression / A-positive logit classification, `ml/test_evaluation.py` adds comparison metrics **after fit and checkpoint selection only**. It observes the existing native test forward calls; no extra inference, training configuration change, validation metric, callback during fit, RNG draw or new dependency pin is introduced. Original test metrics remain present. The linked run logs `test/comparison_slide/...` and, when a stable mapping key exists, `test/comparison_patient/...`:

- AUROC, balanced accuracy, support-weighted F1/precision/recall and separate A/B F1/precision/recall. A/low risk is label 1 (index >= 0), B/high risk label 0; the existing zero decision threshold is preserved.
- A-positive and B-positive trapezoidal AUPRC (`auprc_a_trapezoid`, `auprc_b_trapezoid`). B ranking negates continuous scores and reverses labels. Average precision is logged separately because it is a different PR summary; CPMP's precise integration convention is not confirmed.
- Spearman and Pearson continuous-index correlations for regression, with neither presented as an exact match to Agendia's unspecified `r=0.954`.

`recovery/comparison-test-metrics.json` records definitions, class counts, undefined statistics as JSON null plus reasons, source/recovery/checkpoint identities and the evaluation unit. Only finite statistics are logged as MLflow scalars and included in the existing checkpoint-bound test evaluation. Patient/case scores use equal-weight means across slides grouped by the existing `patient_id`, `case_id` or `record_num` mapping column, with identical targets required within a group. Missing identifiers are never inferred from slide names. Grouping does not establish historical split independence; original patient-level train/validation/test separation still needs independent evidence. This is one existing held-out evaluation (`n=1`, SD unavailable), with no bootstrap, new splits or extra training repetitions. Aggregate genuine repeated patient-level evaluations later, explicitly identifying their protocol; do not label seed variation on one split as repeated cross-validation.

References: [CPMP paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822445/) (five repetitions of patient-level fivefold cross-validation; weighted classification metrics), [authors' evaluation code](https://github.com/lyotvincent/CPMP/blob/main/utils/metrics_utils.py) (sklearn weighted classification report), [Agendia announcement](https://agendia.com/agendia-unveils-clinical-validation-of-ai-driven-digital-pathology-approach/) (correlation definition unconfirmed). Do not compare our historical A-positive binary F1 directly with CPMP's weighted F1.

If the intended goal is a different validation criterion, supply both `--monitor` and `--mode`, for example `--monitor valid/f1 --mode max` for BCE, or `--monitor valid/thresholded_f1 --mode max` for regression. These overrides change checkpoint selection only; original early-stopping settings remain untouched and the change is recorded. For the original protocol, omit both flags. Test metrics are forbidden for selection. No historical-score match is required.

For a bounded read-only metadata plan without a cluster job, use:

```bash
uv run --frozen -m ml.recover_checkpoint RUN_ID \
  --tracking-uri https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
  --implementation-sha FULL_PUSHED_IMPLEMENTATION_SHA
```

This core plan contacts MLflow for bounded config/history reads and uses temporary files; unlike launcher planning, it is not an offline operation. `--help` remains offline. Ambiguous, nonfinite, truncated or unsupported histories fail rather than guessing an epoch.

## Replacement and evidence

The job captures the original checkpoint hash before training, trains in a **new linked run in the same experiment**, verifies the new checkpoint and its bound test evaluation, then:

1. Verifies original/recovery identities, source config/Git binding and finished states.
2. Backs up the original checkpoint and MLmodel under a unique `checkpoint_repairs/OPERATION/backup` namespace outside `checkpoints/`; reads back their hashes.
3. Stages and verifies replacement bytes, rechecks the original hash, replaces only `checkpoints/best/checkpoint.ckpt` and `MLmodel`, then reads both back.
4. Records old/new hashes, recovery run, namespace and publication status. Original config, console and numerical histories remain untouched; source tags point to the linked recovery run and test-evaluation artifact for the replacement's measured performance. Historical test scores remain historical.

Only one repair writer may operate on a given source run at a time. MLflow multi-object writes are not atomic; a hash recheck is not a transaction or atomic compare-and-swap. Backups and a phase journal remain available after partial failure. A CLI failure prints the in-memory partial receipt even when journal storage fails.

Inspect the source run publication tags, linked recovery run, `recovery/recovery-manifest.json`, local `recovery-manifest.json`/`publication-receipt.json`, selected validation score/epoch, exact checkpoint hash and all provenance fields. Successful job creation alone is not successful training or replacement. Read the replacement's test metrics from the linked recovery run, not the source run's untouched numerical histories. Version-2 manifests use the original training protocol; legacy version-1 manual publication retains its former score/budget requirements and is not generated by new jobs.

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
- Historical score missed: informational only. A normally completed, verified fit/test is eligible for publication regardless of this comparison. Do not add epochs, change patience, or select using test performance.
- Fit/test failed: preserve the linked failed run and logs. No original replacement is attempted. A completed checkpoint alone is insufficient without the bound successful test evaluation.
- Original hash changed: another repair/update occurred. Stop and inspect; do not bypass the guard or run competing writers.
- Partial publication: retain the receipt namespace, phase and old/new hashes. Read canonical checkpoint/MLmodel and compare with those hashes before any retry. Do not assume either object is unchanged.
- Rollback needs separate explicit authorization. Download only that operation's verified `backup/checkpoint.ckpt` and `backup/MLmodel`; verify their recorded hashes, verify the current canonical state matches the failed-operation receipt, then use the standard MLflow artifact client/CLI to restore those two files under `checkpoints/best` and read-verify them again. Never delete or overwrite root configs, console, metric histories or unrelated backups. No automatic rollback is attempted.
- Deleting a job or output needs separate authorization naming the exact object/path. Job deletion does not undo MLflow changes; preserve recovery evidence before cleanup or TTL removal.

Local fake-client and tiny CPU tests establish original early-stopping behavior, later best-checkpoint selection, one test evaluation of those exact weights, publication despite a historical-score miss, and guarded artifact handling. They do not establish remote data availability, image/GPU capacity, exact reproducibility, actual cluster execution or real replacement. No remote training or publication is implied by preparing this workflow.
