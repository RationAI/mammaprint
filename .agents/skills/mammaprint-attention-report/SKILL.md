---
name: mammaprint-attention-report
description: Use when reproducing CPU tile-removal and single-tile HTML reports for recovered MammaPrint linear mean regression/BCE models or attention regressors.
---

# Frozen CPU tile HTML report

Use `scripts/ml/run_attention_report.py`. It reuses the successful November 17
M1 report worker, durably preserved with per-file hashes under
`third_party/rationai-report-worker/`. It changes only the selected job's plan,
not inference, attribution or report code. It never trains a model.

Prerequisites: VPN; approved kubeconfig context `kubas-cluster`, namespace
`rationai-jobs-ns`, server
`https://rancher.cloud.trusted.e-infra.cz/k8s/clusters/c-m-lr2wzbk8`; existing
`kubernetes` and PyYAML environment. Authentication stays in the existing MLflow
environment and pod ConfigMap/Secret references (`mlflow`,
`gitlab-deploy-token`). Never print raw pod environment or credentials.

Select a finished recovery run in experiment 3, verify its recovery manifest
and checkpoint SHA, and use a fresh job name, run name and scratch root:

```bash
python scripts/ml/run_attention_report.py \
  --source RUN_ID --checkpoint-sha256 VERIFIED_SHA256 \
  --job-name mammaprint-attention-UNIQUE \
  --run-name attention-UNIQUE \
  --output-root /mnt/projects/mammaprint/attention-UNIQUE \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --plan-file /private/tmp/attention-plan.json
```

Default planning reads only run/config metadata, verifies the frozen source
archive, and prints a bounded summary. `--plan-file` optionally writes the full
reviewable job locally; it contains source code and secret references, no
credential values. No Kubernetes call occurs without `--submit`. Use lowercase
DNS-safe names; uppercase `UNIQUE` above is a placeholder.

The launcher supports IdentityEncoder + LinearHead + Identity output activation
with three already-frozen profiles: AttentionMIL/MSE/index, MeanPool/MSE/index,
and MeanPool/BCEWithLogitsLoss/type. It selects the original config1/config2/config3
plan respectively; all scientific worker files remain byte-identical. Mean models
use tile removal and single-tile outputs with attention disabled. All profiles
currently use the same CPU8/RAM16Gi/shm1Gi request for the L2 reports. It reads all test slides (default 308), retains
every tile, uses the original saved config and checks exact checkpoint/config
hashes. Test scoring is explicitly part of this workflow, as in the previous
reports. It produces raw learned attention, exact tile-removal effects and
single-tile outputs. Integrated gradients remain disabled; patch-flipping
settings, display scales, xOpat links and the standard RationAI HTML assembler
remain the same as the earlier worker. Attention is not a signed class effect.

After explicit authorization for this exact source, append `--submit` to the
reviewed command. The launcher verifies server identity, rejects an existing
job and reads back its UID. It requests **CPU only, 8 CPUs, 16Gi RAM, 1Gi shared
memory and 20Gi ephemeral storage**, exactly as the previous attention report.
Secure DATA/PROJECTS mounts are reused. Embeddings/output use a fresh directory
on PROJECTS, with a 160Gi free-space guard; this is a budget, not measured usage.
The old attention worker measured 2.60Gi peak process RSS; this is not a bound
for another cohort. Do not silently alter data, trim bags or upgrade packages.

The worker uses its own frozen dependency lock inside the pod; repository
training dependencies remain unchanged. The source archive contains code only,
not model weights, WSI files, clinical records or credentials. Code is unpacked
only after checksum verification and with Python's data extraction filter.

Success requires pod exit 0, a new MLflow report run with
`prepared_submission.status=complete`, 308 successful/rendered slides and zero
failed slides, checkpoint/config hash identity, and reproduced saved test
metrics within the existing 1e-5 tolerance. The HTML is `report/report.html`;
the validation receipt is `report/submission-validation.json`. Use its
`pathologist_review_url` tag for the browser-facing xOpat link. Scientific
outputs remain under `explanations/`.

Inspect the exact job with Kubernetes and the linked MLflow run. Pending means
placement is unconfirmed; container startup is not completed inference. OOM or
native-library failures require stage evidence and an amended approved resource
plan. Identity/data/metric mismatches must not be bypassed. An existing MLflow
run name fails closed, so inspect before retrying. Preserve failed scratch and
logs. The worker removes only its own temporary embedding/cache directory after
successful validation, retaining the result/receipt and published artifacts.
Deleting jobs, failed outputs, or rolling back artifacts needs separate explicit
authorization; this workflow never changes the source run or checkpoint.

Local tests verify frozen file identity, plan-only source binding, resources,
shell syntax, safe extraction, and rejection contracts. Local checks do not
establish runtime data availability, memory sufficiency or a finished report.

## Defer both reports until one recovery is published

Use `scripts/ml/run_reports_after_recovery.py` when the explicitly selected
checkpoint recovery is still running. This is a **local waiting process**, not a
cluster controller. It allocates no cluster CPU/GPU while waiting. The local
computer, process, VPN, kubeconfig and clean pinned checkout must remain
available. It stops without submitting on a failed/killed recovery, a conflicting
checkpoint/test binding, timeout, or a changed local checkout. Network outages
are retried within the bounded timeout. It does not change or restart training.

```bash
python scripts/ml/run_reports_after_recovery.py \
  --source-run ORIGINAL_RUN_ID --recovery-run LINKED_RECOVERY_RUN_ID \
  --git-sha FULL_PUSHED_IMPLEMENTATION_SHA \
  --job-prefix mammaprint-selected-bce \
  --output-root /mnt/projects/mammaprint/reports/selected-bce-UNIQUE \
  --state-dir /private/tmp/selected-bce-UNIQUE-state \
  --context kubas-cluster --namespace rationai-jobs-ns
```

Default is an offline plan with no writes or remote calls. Only add `--submit`
when both the CPU HTML and A40 K16 concept jobs for this exact recovery are
explicitly authorized. The source IDs and fresh paths are placeholders; populate
all values. Polls once per minute, timeout 48 hours by default (configurable
`--timeout-hours`, 1–168). The selected recovery must be finished in experiment 3,
linked to the exact original, and published with matching checkpoint SHA and
checkpoint-bound test metrics. The recorded recovery implementation is passed as
the concept **training** SHA; the pushed implementation pin remains separate.

Once ready, invokes the two existing launchers: CPU8/RAM16Gi/shm1Gi test HTML,
then A40/CPU8/RAM64Gi/shm16Gi train-fit/validation-explain deployed K16 concepts.
There is no new explainability algorithm, test concept fit, K sweep, bootstrap,
or source-run mutation. Mean BCE has no attention map or continuous-index
additive section; mean regression receives the existing additive concept report.

For unattended waiting, run the explicitly authorized command under the local
terminal/session supervisor and retain logs. On macOS `caffeinate -i` may keep
that waiting process alive through idle sleep. Record its PID and exact command.
The state directory contains `pid`, `status.json`, verified recovery evidence,
submission commands and separate submission logs. `waiting_for_published_recovery`
means **no report job has been submitted**. `submitted_both` proves that both
launchers confirmed creation, not that reports finished. Inspect exact job names
and UIDs in logs, Kubernetes and eventual new MLflow report runs.

If it fails after one submission, preserve state and inspect that named job;
never blindly restart both. Existing job-name checks reject duplicates. There is
no automatic retry of submissions or deletion/rollback. To cancel only the local
waiter when explicitly requested, verify the PID command and source IDs before
terminating that process; this leaves submitted jobs and training untouched.
The state file is evidence, not a cluster scheduler, and cannot survive a reboot
without a separately configured process supervisor. Local tests cover readiness,
identity/hash binding, training-vs-implementation pins and offline planning;
actual deferred creation is unverified until the recovery completes.
