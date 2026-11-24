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
