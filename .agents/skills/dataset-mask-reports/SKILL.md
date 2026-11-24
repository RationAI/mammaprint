---
name: dataset-mask-reports
description: Use when creating or updating MammaPrint pathologist reports with optional QC, tissue, epithelial, or cancer mask overlays in xOpat, or when previewing, applying, or rolling back an existing report.
---

# Dataset masks in doctor reports

Use the repository Python environment and managed MLflow credentials. Run preparation where the original level-0 WSIs and the dataset's QC and epithelial mask directories are readable under `/mnt`. Use `--mask-wsi-root` and `--mask-source-root` to point to local mirrors of that mount. Tissue and cancer sources can be MLflow artifacts. The explanation manifest identifies the exact report split, level, source run, and checkpoint. The resolver follows the recorded dataset lineage and displays only verified gates. The layer manifest records unavailable masks and selection evidence.

## New reports

Use the standard `ml.explain_report` entrypoint and the pinned RationAI worker from the [targeted tile-removal workflow](../targeted-tile-removal/SKILL.md). The publisher renders the draft and automatically prepares dataset layers. The overlays start off in xOpat. An `--output-dir` preview includes `dataset-mask-layers.json` when the report has sufficient metadata.

```bash
python -m ml.explain_report --worker-root "$worker_root" \
  --tracking-uri "$tracking_uri" --run-id "$explanation_run_id" \
  --output-dir "$draft" --no-upload
```

If a source needs conversion, the normal preview stages a content-addressed OME-TIFF under the draft directory in `dataset-mask-assets/`. For fresh publication, pass the draft files `report.html`, `manifest.json`, `slides.parquet`, and `dataset-mask-layers.json` as `--base-report`, `--report-manifest`, `--report-slides`, and `--layer-manifest`. Keep the staged assets next to the layer manifest and use the same report artifact path, tracking URI, explanation run, experiment, and xOpat host. On `--apply`, the publisher checks the reviewed `--expected-output-sha256`, verifies the live slide table, and uploads and reads back the staged TIFFs before publishing the HTML. Use both `--mask-export-dir "$export_dir"` and `--mask-export-viewer-prefix "$viewer_prefix"` if preview assets need a separately managed xOpat mount for visual review. Stage those exact TIFFs through the authorized artifact workflow and verify their hashes before review.

## Update an existing report

Read the exact MLflow experiment, explanation run, and its `report.artifact` tag first. The usual target is `report/report.html` and the explanation metadata root is `explanations`. Set `--explanation-artifact-root` if that root differs. For each report follow this preview and apply sequence.

```bash
target=(--tracking-uri "$tracking_uri" --experiment-id "$experiment_id"
        --run-id "$explanation_run_id" --artifact-path "$report_artifact")

python -m scripts.ml.backfill_dataset_mask_reports "${target[@]}" \
  --write-layer-manifest "$layer_manifest" --preview "$preview_html"

python -m scripts.ml.backfill_dataset_mask_reports "${target[@]}" \
  --layer-manifest "$layer_manifest" --apply \
  --expected-original-sha256 "$original_sha" --expected-updated-sha256 "$updated_sha"
```

For an existing report whose source TIFFs need conversion, add `--mask-export-dir "$export_dir"` and `--mask-export-viewer-prefix "$viewer_prefix"` to the preview command. Make the resulting TIFFs available at that prefix and verify them before approval. Keep the preview and layer manifest in a private directory.

For a local report use `--report "$html" --report-manifest "$manifest" --report-slides "$slides"` for preview and apply. Add the source tracking URI during preparation if layers need to be resolved. For a test with synthetic data run `python -m unittest tests.test_dataset_mask_reports tests.test_dataset_mask_assets tests.test_dataset_mask_lineage`.

Check `gallery_slides`, `slides_with_masks`, `layers_by_type`, and `missing_masks` in the preview receipt and inspect the per-slide reasons in the layer manifest. Open both links for a representative slide in xOpat. Toggle each layer and check known tissue landmarks at two zooms. Verify the prediction and original shaders. For a missing layer, confirm the recorded source and artifact mount, TIFF readability and registration, and the saved tiling or filter provenance. Obtain approval for the specific published report path and both hashes before overwriting it. Apply using one writer per artifact. A verified receipt records the backup and read-back hash. Keep the receipt and TIFFs used by any published report. Obtain separate explicit approval for every cluster submission.

To restore, use the same target with `--rollback "$receipt_artifact"`, preview the restore, and repeat with `--apply` after approval. For a local restore only the HTML target and receipt are needed. Re-download the restored published HTML and compare its digest with the receipt. An unexpected current hash means a later edit needs reconciliation before restoring.

## Preview from protected cluster mounts

When the local host cannot read the original slides, render a candidate job with `scripts/ml/render_dataset_mask_preview_job.py`. Use the original published report SHA-256 downloaded directly from exact run and path. Repeat `--run` for the reviewed scope.

```bash
python -m scripts.ml.render_dataset_mask_preview_job \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --job-name "$job_name" --experiment-id "$experiment_id" \
  --run "$explanation_run_id:$original_sha" --output "$job_manifest"
```

The renderer packages the current source for the preview worker. The preview renderer pins the same internal S3-backed MLflow service used by the original report jobs, `http://mlflow-s3.rationai-mlflow`. The namespace's default MLflow ConfigMap points at a different backend and will return `RESOURCE_DOES_NOT_EXIST` for these reports. Inspect the target, run IDs and original hashes, mounts, CPU and memory requests, and manifest and payload digests. The renderer makes no cluster request. Once the user explicitly approves the exact submission, confirm the local context, check that the named job does not exist in the target namespace, and create the reviewed manifest once.

```bash
kubectl --context kubas-cluster --namespace rationai-jobs-ns \
  get job "$job_name" --ignore-not-found
kubectl --context kubas-cluster --namespace rationai-jobs-ns \
  create -f "$job_manifest"
kubectl --context kubas-cluster --namespace rationai-jobs-ns \
  logs -f "job/$job_name"
```

The worker runs `scripts/ml/prepare_dataset_mask_previews.py` with managed MLflow credentials and the original slide and mask mounts read-only. It checks the original HTML before and after preparation, publishes only new files under `report_backfills/dataset_masks/previews/$job_name`, and reads them back to verify their hashes. Each successful run stages `layers.json`, `preview.html`, `receipt.json`, and any converted TIFFs under `dataset-mask-assets/`. Keep the staged TIFFs for as long as a published overlay references them. Use the MLflow artifact listing and the receipt to reconcile every selected run before review. If the run failed, inspect the job status and its per-run diagnostics before asking for approval for another submission. The job neither applies nor rolls back a published report. Follow the normal apply flow with the staged layer manifest and approved original and updated hashes.

Verify the renderer and staging behavior locally with `python -m unittest tests.test_render_dataset_mask_preview_job tests.test_prepare_dataset_mask_previews`.

### Reuse a verified preview after a partial job

If a preview job finishes one run before a deadline or another run fails, use the completed run as a donor. The worker rechecks the donor receipt, the layer manifest, the original slide paths, and both saved dataset lineages. It reuses only masks with matching source locations and verified viewer assets. For the target report it records the target dataset and its own selection rules. Keep the donor preview assets available while any report references them.

Prepare a local preview if every eligible layer can be reused. In this example the worker reads MLflow artifacts and writes the preview, layers, and receipt under the target run subdirectory in a private local output directory.

```bash
python -m scripts.ml.prepare_dataset_mask_previews \
  --tracking-uri "$tracking_uri" --experiment-id "$experiment_id" \
  --batch "$batch" --run "$target_run:$original_sha" \
  --reuse-layer-run "$donor_run" --reuse-layer-batch "$donor_batch" \
  --output-dir "$local_preview_dir"
```

If unique masks still need the mounted WSI, pass the same two reuse options to the job renderer and include only the unfinished targets. The worker prepares the outstanding mask assets, stages the complete preview, and checks its hashes. Use a new job name and obtain approval for that exact submission. Confirm all expected types in the receipt before considering an in-place apply.
