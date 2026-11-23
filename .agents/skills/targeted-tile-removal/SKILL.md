---
name: targeted-tile-removal
description: Use when running MammaPrint validation tile-removal masks, rendering blinded xOpat review, or adding targeted tile removal to a new or existing HTML explanation report.
---

# MammaPrint targeted tile removal

Use frozen single-scale scalar regression checkpoints with mean or gated-attention pooling. Start with EAN5-L2-Lin-Mean-Reg-s1 and CM5-L2-Lin-Attn-Reg-s2. For an existing report, use its exact explanation run and source checkpoint. A recovered run can have the same model name as an earlier training run with different weights.

## Prepare and pilot the validation runner

Use the repository Python environment and managed MLflow authentication. The execution environment needs the original data mapping, validation Parquet bags, and mounted WSIs with native OpenSlide for geometry. Check whether the mapping, validation artifact, tiling parameters, and source Git revision match the resolved checkpoint configuration. Use actual SHA-256 digests, including for the frozen mapping snapshot. Read target IDs from the explanation manifest or MLflow metadata. Keep artifact authentication in the environment. Keep the study manifest and outputs in a private working directory.

Create a JSON study manifest with `split: "val"` and a `models` array. Use these keys per model.

| Field | Value |
| --- | --- |
| `model_id` | Study alias |
| `source_run_id`, `source_revision` | Full MLflow source run ID and source Git SHA |
| `checkpoint`, `checkpoint_sha256` | Exact .ckpt file path or MLflow artifact URI and digest |
| `config`, `config_sha256` | Resolved training config for that run and digest |
| `label_mapping`, `label_mapping_sha256` | Mapping location from `datamodule.val.data_mapping` and digest |
| `validation_source` | Exact `datamodule.val.levels.<level>.uris.val` location |
| `patient_column` | Optional verified patient or case column |
| `validation_predictions` | Optional object with `path` and `sha256` for a validation-only CSV containing `slide_id,prediction` |

Use explicit `tracking_uri` for MLflow sources. Optional `options` include `wsi_root` for a local mirror of the mount, `tissue_mask_dir` and `tumor_mask_dir` for registered `<slide_id>.tiff` rasters, and `index_band: 0.05` for the index-band follow-up. Review the resolved defaults in the dry run and freeze them before the cohort is scored.

```bash
python -m scripts.ml.run_tile_removal --manifest "$study" --output "$pilot" \
  --pilot-slide-id "$validation_slide_id" --dry-run
python -m scripts.ml.run_tile_removal --manifest "$study" --output "$pilot" \
  --pilot-slide-id "$validation_slide_id"
python -m scripts.ml.run_tile_removal --manifest "$study" --output "$bundle"
```

Check intact-score parity, full roster reconciliation, verified mask counts, unavailable cases, and memory/runtime on each real checkpoint before the full cohort. Check that the selected-level tile positions match the original WSI at two zooms. An output directory is an immutable snapshot. The runner verifies all output artifact digests and checks local checkpoint, config, mapping, and bag hashes on retry. For refreshed inputs, use a fresh output directory and keep the original snapshot for comparison. Bag hashes and input provenance are in the bundle.

## Publish xOpat overlays and prepare blinded review

Plan the explicit overlay run and artifact path before publishing. Use a dedicated active MLflow run tagged `targeted_tile_removal.role=validation-overlays`. Set `overlay_target` with `tracking_uri`, `experiment_id`, `run_id`, `artifact_root`, and `xopat_url`. The artifact mount defaults to `secure_mlflow`. Use a private `review_key_file` with at least 32 cryptographically random bytes. Generate it with `umask 077; openssl rand -out "$review_key_file" 32` and keep it outside shared artifacts. Keep the same key for retries.

```bash
python -m scripts.ml.run_tile_removal --manifest "$study" --output "$linked_bundle" \
  --publish-overlays --dry-run
python -m scripts.ml.run_tile_removal --manifest "$study" --output "$linked_bundle" \
  --publish-overlays
```

After approval, the apply command uploads TIFFs to the displayed run and content-addressed directory, then downloads and checks their hashes. Reuse the study and key for a partial upload retry. Existing overlay files are read back before links are emitted. Keep overlays used by any published report. The verification record is `manifest.json`, with `slides.parquet`, `masks.parquet`, `summary.md`, and each model's `report.html`. Main reports link the exact and compact overlays, a representative matched control per family, and a distinct successful alternate when available. Quantitative control rows remain in `masks.parquet`.

On a live pilot, check a control, an overlap, and a clipped footprint in xOpat. If a viewer link fails, verify the mounted WSI path, the artifact mount and run ID, the uploaded TIFF, and its coordinate alignment. Share only `blinded_review.html` during blinded review. Keep `unblinding_key.csv` private until review closes. Use the review IDs to join returned observations to masks and cases.

## New or existing tile-explanation reports

For a new report, compose the validation section with a standard RationAI draft. Extract the pinned worker into a fresh private directory and resolve its locked environment with managed access to its dependencies.

```bash
git show 2c7a34b549da9a25cdc94178b82d6cfe5de9907b:third_party/rationai-report-worker/worker.tar.gz | \
  tar -xzf - -C "$worker_root"
(cd "$worker_root" && uv sync --frozen --no-dev)
python -m ml.explain_report --worker-root "$worker_root" --tracking-uri "$tracking_uri" \
  --run-id "$explanation_run_id" --tile-removal-bundle "$linked_bundle" \
  --output-dir "$draft" --no-upload
```

Omit `--tile-removal-bundle` to generate the normal report alone. After approving the preview, fresh publication uses `--base-report "$draft/report.html" --report-manifest "$draft/manifest.json" --run-id "$explanation_run_id" --tracking-uri "$tracking_uri" --experiment-id "$experiment_id" --apply --expected-output-sha256 "$draft_sha"`. Use the backfill command for an existing artifact. An already composed draft retains its validation section on upload.

To compose a preexisting standard draft locally, use the same generator in draft mode.

```bash
python -m ml.explain_report --base-report "$draft/report.html" \
  --report-manifest "$draft/manifest.json" --report-slides "$draft/slides.parquet" \
  --tile-removal-bundle "$linked_bundle" --output "$new_report"
```

For an existing local report, preview first at a new path. Add `--report-slides` with the original slide table for validation reports that use the RationAI template.

```bash
python -m scripts.ml.backfill_tile_removal_reports --report "$report" \
  --report-manifest "$report_manifest" --bundle "$linked_bundle" --preview "$preview"
```

For an existing published report, use the exact active explanation run and experiment. The historical artifact is `report/report.html` and its provenance manifest is `explanations/manifest.json`. This is the explanation run, distinct from the checkpoint's training run.

```bash
python -m scripts.ml.backfill_tile_removal_reports --tracking-uri "$tracking_uri" \
  --experiment-id "$experiment_id" --run-id "$explanation_run_id" \
  --artifact-path report/report.html --bundle "$linked_bundle" --preview "$preview"
```

After blinded review closes, pass narrative findings in an approved text file with `--review-findings "$findings"` on new composition or backfill. Confirm split, checkpoint hash, slide counts, existing report controls, and xOpat links in the preview. For a test report, the added panel is labeled **Targeted tile removal on validation** and has its own slide list. Before overwriting a published artifact, obtain approval for the exact report and hashes shown in the preview. Use one writer per artifact. Apply with `--apply --expected-original-sha256 "$original_sha" --expected-updated-sha256 "$updated_sha"` and the same target and bundle. Success returns a verified receipt and backup path. For rollback, reuse the exact target arguments, replace `--bundle` with `--rollback "$receipt_artifact"`, preview, and repeat with `--apply` after approval. Re-download the final HTML and verify the receipt hash. Keep the backup and receipt.

Local smoke checks cover the frozen forward path, the synthetic validation cohort, renderer, and fake MLflow transactions. Verify real checkpoint inference on the original bags and xOpat links before publication. Obtain explicit approval for each cluster submission with its context, namespace, and workload.
