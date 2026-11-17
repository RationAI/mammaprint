---
name: mammaprint-concept-cluster
description: Use when planning, submitting, monitoring, or diagnosing pinned MammaPrint deployed-metric concept jobs on the private GPU cluster with scripts/ml/run_concept_explainability.py.
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

The deployed metric initially supports IdentityEncoder + MeanPool + single LinearHead + BCE/type. Unsupported model graphs fail before full-split embedding downloads; there is no automatic plain-dictionary fallback. Mean pooling has no attention score, so the upstream attention term is explicitly inapplicable. Output-metric estimation, semi-NMF fitting and NNLS encoding run on CPU. The GPU serves native PyTorch explanation inference; do not describe dictionary fitting as GPU accelerated.

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

Deletion needs separate explicit authorization naming the new job or result directory. If authorized, delete only the exact new job with `kubectl --context kubas-cluster -n rationai-jobs-ns delete job "$CONCEPT_JOB_NAME"`; preserve logs/evidence first. Results and new destination MLflow runs require their own explicit cleanup decision. Deleting a job does not roll back PROJECTS files or MLflow publications. Never delete or modify the source run/checkpoint. The helper currently retains completed jobs for seven days; export evidence before TTL cleanup.

Offline unit tests, syntax/lint checks, and plan inspection establish launcher behavior only. They do not establish remote Git reachability, image/native/CUDA availability, actual source compatibility, resource sufficiency, cluster execution, or scientific metric validity. Report those limits whenever no authorized remote run was performed.
