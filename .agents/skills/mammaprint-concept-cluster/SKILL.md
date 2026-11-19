---
name: mammaprint-concept-cluster
description: Use when planning, submitting, monitoring, or diagnosing pinned MammaPrint concept jobs, or repairing their MLflow HTML report presentation in place.
---

# MammaPrint concept cluster workflow

Use `scripts/ml/run_concept_explainability.py` from the repository root to plan one job per explicit source-run/training-commit/K combination. Ordinary Python can plan offline: no dependency setup, filesystem writes, Git commands, MLflow lookup, or cluster access occurs without `--submit`.

## Prerequisites and identities

- Obtain explicit authorization for the particular cluster submission before running `--submit`. Preparing or pushing code, inspecting this skill, and printing a plan do not authorize a cluster job. Test scoring requires an explicit `--include-test` choice.
- Submission needs VPN/private-cluster connectivity, a local kubeconfig containing `kubas-cluster`, and the established `kube_jobs` and `kubernetes` environment. These submission-only packages are not application dependencies in `pyproject.toml`. Do not install an arbitrary replacement helper.
- The installed private helper must resolve to context `kubas-cluster`, namespace `rationai-jobs-ns`. The script checks its effective `CLUSTER` and `JOB_NAMESPACE` constants before loading kubeconfig, explicitly loads the named context, and prints its configured server endpoint before creation. Compare this endpoint with the approved private target. `--context` and `--namespace` are expectations, not routing options; another target requires a separately reviewed workflow.
- The helper supplies ConfigMap `mlflow` / `tracking-uri`, Secret `gitlab-deploy-token` / `username` and `password`, and the secure DATA and PROJECTS storage objects. Preserve these references; never copy credentials into commands, scripts, logs, or plans. The explicit `--tracking-uri` overrides the pod endpoint and must be a credentials-free HTTP(S) URI without query or fragment. Authentication stays in the approved environment.
- The implementation SHA must be a full commit already reachable from `https://github.com/RationAI/mammaprint.git`. Supply `--git-ref` when a particular reachable branch/ref is needed. Clone is full history; the job fetches missing exact objects, checks out the implementation SHA detached, and verifies HEAD.
- Recover **each historical training SHA independently** from the source MLflow run's `mlflow.source.git.commit` tag using an authorized read-only MLflow view/client. If the tag is absent, recover the exact full training SHA from that run's original training records and retain the evidence. Supply it as `--source RUN_ID:TRAINING_SHA`; never substitute the implementation SHA. The core rejects conflicting source provenance and checks historical code compatibility. Fetching a training object does not execute or check out its historical code.
- Select a fresh absolute output directory beneath `/mnt/projects`, outside the checkout. The core refuses nonempty result destinations. Do not delete existing outputs to force a retry.
- Default image is the existing `cerit.io/rationai/base:2.0.6`; native CUDA, PyTorch, OpenSlide, and libvips availability in that image is **unverified**. Choose an approved compatible `cerit.io` image with `--image` if necessary. Jobs perform import/CUDA checks; do not add apt installs, pip upgrades, or model-stack changes as a workaround.

## Prepare and inspect an offline plan

Use shell variables containing actual full SHAs and a fresh run directory. The example values below name roles; populate them from reviewed implementation/source evidence before submission.

```bash
CONCEPT_IMPLEMENTATION_SHA='<full pushed implementation SHA>'
SOURCE_RUN_ID='<source MLflow run ID>'
SOURCE_TRAINING_SHA='<full historical training SHA>'
CONCEPT_OUTPUT_ROOT='/mnt/projects/mammaprint/concepts/<fresh-run-directory>'
python3 scripts/ml/run_concept_explainability.py \
  --source "${SOURCE_RUN_ID}:${SOURCE_TRAINING_SHA}" \
  --git-sha "$CONCEPT_IMPLEMENTATION_SHA" \
  --git-ref refs/heads/feat/concept-explainability \
  --tracking-uri http://mlflow-s3.rationai-mlflow \
  --output-root "$CONCEPT_OUTPUT_ROOT" \
  --experiment-name 'MammaPrint concepts' \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --gpu A40 --cpu 8 --memory 32Gi --shm 16Gi --ray-cpus 8
```

Inspect all emitted JSON jobs and scripts: source/training pin differs in purpose from implementation pin; output destinations are unique; target, image, resources and tracking endpoint match the intended run; `concepts_deployed`, `action=fit_explain`, `device=cuda`, K16 and `include_test=false` appear. Add repeated `--source` pairs for explicit additional sources. Add `--k 8 16 32` (or repeat `--k`) for explicit component counts. This creates distinct jobs, not a queue or resume scheduler. Defaults request one A40, 8 CPUs, 32Gi host RAM, 16Gi shared-memory capacity and 8 Ray CPUs. H100, A100 and mig-1g.10gb remain explicit alternatives. Ray CPUs must not exceed allocated CPUs; shared memory must not exceed RAM.

The CPU/RAM defaults are a starting request for a K16 run, not measured full-cohort minima. The unchanged fit settings include a 250,000-row shuffle buffer (~2.38 GiB raw FP32 embeddings), 25,600-row batches (~0.244 GiB each), Parquet decoding, Ray workers/object store and temporary arrays. The K16 dictionary itself is only ~0.156 MiB. Keep 16Gi shared-memory capacity within the 32Gi host budget. GPU memory is driven by a single inference bag, not dictionary fitting; estimated 1–3 GiB VRAM for supported L3 mean-linear inference is unmeasured. The core supports CPU inference, but this launcher still explicitly requires CUDA. A40 placement is a user choice; it is not a mathematical or VRAM requirement of the fitting method. Do not change numerical fit parameters just to reduce resources.

For a newly recovered model, prefer its finished linked recovery run as the source so the report and checkpoint refer to the new run. If `mlflow.source.git.commit` is absent, verify its `recovery.implementation_sha` against `recovery/recovery-manifest.json` and pass that actual training implementation as the explicit training SHA; `recovery.source_training_sha` describes the older experiment being reproduced. The explainability implementation pin can be newer. Check graph compatibility and checkpoint hash locally before scheduling a full-cohort job; no embedding or slide download is needed for that preflight.

The default fits the dictionary on train and explains validation in one process, sharing the dataset cache. Split auditing lists train/val/test artifact membership and applies the original mapping; it does not download test bags. Listed zero-row bags are conservatively included in leakage checks because a listing cannot verify row count. Test embedding download, scoring and report generation occur only with `--include-test`.

The deployed metric supports IdentityEncoder + MeanPool + single LinearHead with either BCE/type (`native_mean_linear_binary_classifier_v1`) or MSE/index (`native_mean_linear_index_regression_v1`), always with Identity output activation. The saved source objective selects the adapter; do not override labels or loss. Unsupported graphs (including attention, MLP and joint) fail before full-split embedding downloads; there is no automatic plain-dictionary fallback. Mean pooling has no attention score, so the upstream attention term is explicitly inapplicable. Output-metric estimation, semi-NMF fitting and NNLS encoding run on CPU. The GPU serves native PyTorch explanation inference; do not describe dictionary fitting as GPU accelerated.

For the first regression pilot, source run `656ef0c37f264d6eab1d15ed20867335` is configuration 2, seed 1. Its verified recovery implementation/training pin is `47938eb79651d91173c5ee0dc516fc440771ec0e`; verify that binding against its recovery manifest at submission time. Use the same launcher with `--source RUN_ID:TRAINING_SHA --k 16`, a newly pushed regression-capable implementation SHA, and a fresh output root. Fit a new model-bound dictionary on train; classifier metric dictionaries cannot be reused as regressor dictionaries. The intervention remains 0.25 SD and outputs stay in raw index units, without sigmoid, class-specific sign flips or clipping.

Regression validation artifacts additionally include `slide_concept_contributions.parquet` and `slide_index_decomposition.parquet`. Check `prediction = baseline + sum(contributions) + residual`, where baseline is the output at the tile-weighted training mean embedding and raw NNLS amounts determine the signed contributions. The report must show the residual after reveal and use the blinded expected-index-direction question. Inspect MAE/RMSE, zero-boundary agreement, residual P95 and recorded numerical roundoff; do not infer a clinical acceptance threshold. Attention/MLP do not receive this additive report. The first native index integration test uses bounded synthetic bags; no real regression cluster result is implied by local tests.

The 32Gi example/default is not a guaranteed minimum: the configuration-3 K16 full-cohort attempt was OOM-killed at 32Gi. Use an explicitly approved 64Gi request for a comparable new full-cohort run, retaining 16Gi shared memory unless separately approved. That retry is not evidence that every cohort/K will fit in 64Gi. Resource changes do not authorize submission; keep numerical fit settings unchanged.

## Submit only the authorized plan

After explicit approval for the concrete source/K jobs and target, rerun the same command in the approved helper environment with **only** `--submit` added. The script prints the complete plan first, loads the exact context, prints target identity, checks that every planned job name is absent, submits using the existing helper, and reads each named job back to confirm a new UID. A confirmation proves resource creation, not successful execution. A failed confirmation may follow successful creation: inspect before retrying. Multi-job submission can partially succeed; already-created jobs are not automatically rolled back.

The pod uses `set -euo pipefail`, `umask 077`, full clone, both exact Git object pins, frozen explainability dependencies, CUDA/native-library preflight, and `unset MLFLOW_RUN_ID`. It invokes one combined core action, creating a new destination MLflow run while preserving the source run, checkpoint, resolved config and training history.

## Observe and verify results

Use the actual `job_name` from the plan, retaining the same context and namespace in every command. Read-only remote inspection also needs authorized connectivity.

```bash
CONCEPT_JOB_NAME='<exact job_name from plan>'
kubectl --context kubas-cluster -n rationai-jobs-ns get job "$CONCEPT_JOB_NAME" -o wide
kubectl --context kubas-cluster -n rationai-jobs-ns get pods -l "app=$CONCEPT_JOB_NAME" -o wide
kubectl --context kubas-cluster -n rationai-jobs-ns logs "job/$CONCEPT_JOB_NAME"
kubectl --context kubas-cluster -n rationai-jobs-ns describe job "$CONCEPT_JOB_NAME"
kubectl --context kubas-cluster -n rationai-jobs-ns get events --sort-by=.lastTimestamp
```

For scheduling/container failures, describe the exact pod selected by the job's `app` label. Preserve logs, job UID/status, resolved pins and new destination MLflow run ID. Confirm success from job completion **and** artifacts under `OUTPUT_ROOT/SOURCE_RUN_ID/k-K/{dictionary,val}` (plus `test` only if opted in). Inspect manifests and destination MLflow tags: metric enabled/profile, transform/checksum and source binding, source training SHA, implementation revision, K, train fit membership, validation/test split identity, requested attention weight and its inapplicability reason, and CPU/GPU stage metadata where recorded. Check that no test report exists for the default plan and no source artifact was mutated. Metadata-only audit evidence must not be described as test-bag verification.

| Failure | Next evidence and bounded response |
| --- | --- |
| Pending GPU | Pod events and resource/node-selector requests; A40 availability is not guaranteed. Change GPU/resources only with an amended approved plan. |
| OOM / shared memory | Pod termination reason and stage logs; increase approved RAM/shm/CPU settings or investigate peak CPU arrays. A larger GPU does not solve CPU factorization memory. |
| CUDA / OpenSlide / libvips preflight | Container import error and CUDA details; use an approved compatible image. Do not upgrade dependencies inside the failed job. |
| Git pin unavailable | Verify implementation was pushed, optional ref is reachable, and historical object is available from origin. Do not replace either exact pin with a mutable training branch. |
| Unsupported graph | Read compatibility rejection; select a supported source or separately scope an adapter change. Do not disable the deployed metric to obtain an apparent success. |
| Mapping/provenance/leakage | Preserve error and source mapping/config evidence; recover the correct training SHA/mapping and resolve identity conflicts before a fresh run. Never alter source metadata to bypass a check. |
| Unconfirmed create / existing job | Inspect the exact name and UID before retries. A helper can catch a creation error; the launcher therefore requires read-back confirmation. |

## Cleanup and limits

### Repair existing report signs without rerunning inference

Use `scripts/ml/update_concept_report_signs.py` for presentation-only updates of a finished validation report. It uses the existing authenticated MLflow client environment (VPN/private HTTPS connectivity; credentials remain in the environment). No GPU or Kubernetes job is needed. The user must explicitly authorize the particular runs before `--apply`; read-only preparation is safe without write authorization.

```bash
python scripts/ml/update_concept_report_signs.py \
  --tracking-uri https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
  --experiment-id 84 --run-id '<exact explanation run ID>' \
  --work-dir '/private/tmp/concept-signs-<run>-prepare'
```

The default downloads only `val/report_manifest.json`, `val/concept_effects.parquet`, and `val/naming_kK.html` to a fresh local directory. Inspect `updated/naming_kK.html` and `receipt.json`. Rerun with a **different fresh work directory** and `--apply` for the authorized replacement. It checks experiment/status, manifest/report identities, concept counts and exact original tables before making changes. Unfamiliar templates or mismatched data fail closed.

Both displayed intervention columns use after minus before. Classification always uses P(A), undoing the legacy B-cohort sign flip; removal undoes the legacy before-minus-after convention. Regression stays in raw index units. Raw Parquet evaluation fields retain Adam's upstream convention; consumers must not interpret their `subtract` column using the new HTML display convention. Future reports use the same shared presentation function, and their report manifest records `effect_display_version`.

Before replacement the script uploads and downloads a byte-verified backup at `report_updates/after-minus-before-v1/val/original/naming_kK.html`. It detects intervening changes to the original HTML, effects or report manifest, uploads/read-verifies a prepared receipt, replaces only the HTML under `val`, verifies its bytes by a fresh download, and uploads/read-verifies the final receipt beside the backup. Existing report manifests, scientific tables, model artifacts, run status, metrics, viewer links, embedded images and form behavior are retained. The HTML marker and receipt describe the presentation revision; the original manifest continues to describe the original computation. Applying again validates the upgraded report and its backup/receipt without writes; if the last attempt stopped after HTML replacement, it completes the prepared receipt after verifying the artifacts.

Success evidence is `status=verified`, a matching remote HTML SHA-256, and a verified backup and receipt. If upload/read-back fails, inspect the exact artifact and backup before retrying; do not rerun training. To roll back an authorized repair, download the backup, verify its hash against `original_sha256` in the receipt, and use `MlflowClient.log_artifact(run_id, local_backup_path, 'val')`; download the restored HTML and compare its hash again. A different existing backup is never overwritten. MLflow has no compare-and-swap artifact write, so the last pre-write recheck reduces but cannot eliminate concurrent-edit races; avoid concurrent report edits.

Deletion needs separate explicit authorization naming the new job or result directory. If authorized, delete only the exact new job with `kubectl --context kubas-cluster -n rationai-jobs-ns delete job "$CONCEPT_JOB_NAME"`; preserve logs/evidence first. Results and new destination MLflow runs require their own explicit cleanup decision. Deleting a job does not roll back PROJECTS files or MLflow publications. Never delete or modify the source run/checkpoint. The helper currently retains completed jobs for seven days; export evidence before TTL cleanup.

Offline unit tests, syntax/lint checks, and plan inspection establish launcher behavior only. They do not establish remote Git reachability, image/native/CUDA availability, actual source compatibility, resource sufficiency, cluster execution, or scientific metric validity. Report those limits whenever no authorized remote run was performed.
