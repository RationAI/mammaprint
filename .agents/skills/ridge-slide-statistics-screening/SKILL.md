---
name: ridge-slide-statistics-screening
description: Use when preparing or running MammaPrint CPU ridge screens on existing Virchow2 slide embeddings, reproducing their cross-validation, or evaluating the locked ridge model.
---

# Ridge slide-statistics screening

Use the repo's pinned Python 3.12 environment. The two entry points are `scripts/ml/run_ridge_screening.py` for the CPU workflow and `scripts/ml/render_ridge_screening_job.py` for rendering a cluster allocation. The latter prints a Job manifest. Follow `AGENTS.md` and obtain explicit approval before each Kubernetes submission, including the pilot, a resumption, or a held-out test job.

## Freeze the real inputs

Copy `configs/sweeps/ridge_screening/screening.yaml` into a durable run directory. Pin write-once MLflow artifacts. The remote inventory records artifact paths and byte counts, so it cannot detect an in-place replacement with the same size. Pin the canonical `data_mapping`, the embedding `cards_dir` and any `sources` overrides, and the historical L2 cm5 reference. MLflow artifact URIs require the correct `MLFLOW_TRACKING_URI` and applicable managed credentials. A local fixture or pre-staged directory can be used as a source override. Add L1 when its registered card is available and preserve its stride, sampling, and mask provenance.

Set absolute `cache_dir`, `sanity_dir`, `screen_dir`, `lock_dir`, and `historical_reference` paths for the worker. Use separate directories for a pilot and production. Put `test_registry` outside the versioned run directory and use the same physical registry across cache versions and invocations for the same evaluation. Set `blas_threads` to fit the allocated CPU count. In the rendered Job, `/mnt/projects` is a read-only input mount and `/ridge-output` is the writable mount of the pre-created output PVC subdirectory. A versioned run could use `/ridge-output/v1/cache` and `/ridge-output/v1/screen` and keep the shared registry at `/ridge-output/test-registry`.

The `historical_reference` JSON pins `source_uri`, `train_uri`, `val_uri`, actual eligible `train_slides` and `val_slides`, and `transform` (`raw` or `ln`). In `runs`, record each `run_id`, `seed`, `code_revision`, `config`, `checkpoint`, `checkpoint_selection`, `f1`, and `mae`. Supply the verified scalar linear head and matching archived validation predictions in `parity` as `weights`, `intercept`, and `predictions` keyed by slide ID. Use `resolution` only to document an investigated performance gap after prediction parity passes. Freeze the reference before scoring ridge.

## Verify locally and render the job

Use synthetic embedding Parquets and a synthetic mapping for local checks. The cache dry-run resolves configured inventory and the screen dry-run shows planned contrasts. Keep the fixture output separate from real artifacts.

```bash
uv run --frozen --with pytest==9.2.0 python -m pytest --import-mode=importlib \
  -o consider_namespace_packages=true -q tests/preprocessing/test_slide_statistics.py \
  tests/test_ridge_screening.py tests/test_render_ridge_screening_job.py
uv run --frozen python -m scripts.ml.run_ridge_screening cache \
  --config "$RIDGE_FIXTURE_CONFIG" --output "$RIDGE_FIXTURE_CACHE" --dry-run
```

For the private cluster, verify the intended context and namespace and confirm that the input and output PVCs belong to that target. The output PVC subdirectory must exist and be writable by the worker UID. Request CPU and memory only. Use a digest-pinned image with the code checked out at `--workdir`, or supply a full `--source-commit` accessible from the repository cloned by the pod. For a base image requiring private package access, pass the target's existing `--gitlab-secret` with `username` and `password` keys. An existing `--env-secret` can supply MLflow authentication and input environment variables. Keep secret values outside configs and job manifests.

```bash
export RIDGE_CONTEXT=kubas-cluster
export RIDGE_NAMESPACE=rationai-jobs-ns
export RIDGE_CONFIG=/mnt/projects/mammaprint/ridge-screening/v1/screening.yaml
kubectl config get-contexts "$RIDGE_CONTEXT"
uv run --frozen python -m scripts.ml.render_ridge_screening_job cache \
  --context "$RIDGE_CONTEXT" --namespace "$RIDGE_NAMESPACE" \
  --job-name mammaprint-ridge-cache-pilot --image "$RIDGE_IMAGE" \
  --source-commit "$RIDGE_COMMIT" --gitlab-secret gitlab-deploy-token \
  --config "$RIDGE_CONFIG" --output-subpath mammaprint/ridge-screening \
  --output /ridge-output/v1/cache-pilot --pilot-slides 10 \
  --cpu 4 --memory 16Gi --scratch-size 64Gi > ridge-cache-pilot.job.json
```

If the ridge worker commit is local, also pass `--overlay-base PUBLISHED_BASE_SHA`. The renderer packages the committed ridge worker and input cards over that published ancestor, with both revisions recorded in Job annotations. For the cache stage, `--repo-config` can replace the mounted `--config` argument. It uses the checked-in defaults, including the canonical mapping under `/mnt/projects`. Confirm the rendered pilot still has the intended read/write scope.

Review the rendered namespace, context annotation, stage, pinned source, mount scopes, resource requests, and output path. Show the manifest and affected scope for approval. After approval, use the same explicit context and namespace when creating and inspecting the job.

```bash
kubectl --context "$RIDGE_CONTEXT" --namespace "$RIDGE_NAMESPACE" \
  create -f ridge-cache-pilot.job.json
kubectl --context "$RIDGE_CONTEXT" --namespace "$RIDGE_NAMESPACE" \
  logs job/mammaprint-ridge-cache-pilot
```

For the full cache, drop `--pilot-slides`, use the configured production cache path, and choose a new Job name. Increase scratch and memory only if the pilot indicates it. Render and request approval for that submission separately. For a job that runs another stage, replace the renderer's `cache` argument with that stage and pass its own output. Add `--screen all` to render the full screening matrix. An approved CPU allocation with the same mounts can execute the CLI directly.

## Run and verify the stages

Use `RIDGE_RUN` for the durable production run directory inside the allocation. Adjust paths in the config to match that mount. Inspect each command's files before proceeding.

```bash
uv run --frozen python -m scripts.ml.run_ridge_screening cache \
  --config "$RIDGE_CONFIG" --output "$RIDGE_RUN/cache"
uv run --frozen python -m scripts.ml.run_ridge_screening sanity \
  --config "$RIDGE_CONFIG" --output "$RIDGE_RUN/sanity"
uv run --frozen python -m scripts.ml.run_ridge_screening screen \
  --screen all --config "$RIDGE_CONFIG" --output "$RIDGE_RUN/screen"
```

- Check cache `manifest.json`, `coverage.parquet`, and the `stats/dev` and `stats/test` partition inventory. Reconcile available level/mask combinations and missing slides. Read attempted and usable slide counts and known input bytes from `pilot_diagnostics` in the pilot manifest. Measure wall time, cache bytes, peak memory, and free local scratch during the pilot. A complete manifest in the production directory establishes the reusable cache version.
- Check whether `sanity.json` clears the historical comparison before running screens. Use `folds.csv`, `metrics.csv`, `summary.csv`, `ranking.csv`, `contrasts.csv`, and `report.md` to confirm the grouped folds and paired differences. Review `coverage.csv`, `exclusions.csv`, and `learning_curve.png` when available. Repeat screening into a second output directory and compare numeric tables and fold ledgers under the same runtime.
- Review the chosen feature set, locked model, and realized data boundary before opening test. Check `lock.json` before running `test` with the established shared registry. Preserve the lock, `test_metrics.json`, and `held_out_report.md`, which includes the development results.

```bash
uv run --frozen python -m scripts.ml.run_ridge_screening lock \
  --config "$RIDGE_CONFIG" --output "$RIDGE_RUN/lock"
uv run --frozen python -m scripts.ml.run_ridge_screening test \
  --config "$RIDGE_CONFIG" --output "$RIDGE_RUN/test"
```

## Diagnose and resume

- If a job is pending or fails on startup, verify target PVCs, output subdirectory permissions, image pull, and the pinned checkout. Treat pod-local scratch exhaustion as a sizing or staging issue. Compare input artifact bytes with free scratch and check active resource limits before retrying.
- On artifact transfer failure, verify the pinned URIs, MLflow identity, proxy routing and scratch space. After the prior writer has exited, rerun with the same cache inputs and durable output. Verified completion markers reuse finished sources. Changed source, config, or membership requires a new cache version or a supported extension. Keep failed-slide evidence.
- If sanity blocks, compare the pinned split membership, raw and LayerNormed features, checkpoint parity, and sign convention before proceeding. If a screen reports infeasible folds, inspect class and record counts and coverage. Preserve those reports with the run.
- Preserve the test registry and frozen lock after a failed or successful evaluation. Use the same lock and registry for a recoverable publication retry. Keep published artifacts and reports. Reclaim pilot and scratch data after collecting sizing and failure evidence and confirming it is expendable.
