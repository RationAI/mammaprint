---
name: l2-embedding-augmentation
description: Use when preparing, profiling, running, or validating MammaPrint L2 Virchow2 H&E stain jitter, rotation probes, or clean-plus-augmented training.
---

# L2 embedding augmentation

Run on the pinned project checkout with the native OpenSlide library, `uv sync --frozen`, the authorized MLflow endpoint, and the secure DATA and PROJECTS mounts. Use the cluster-managed Hugging Face and GitLab credentials. The examples use one full A40 on `kubas-cluster` in `rationai-jobs-ns`. Confirm the actual target and present each exact rendered manifest for explicit approval before submission.

## Freeze inputs on CPU

Set an approved shared work root. Keep the frozen input roots separate from the A40 job's writable output subtree. Run the preparer as a CPU command in an existing authorized allocation.

```bash
export MPA_ROOT=/mnt/projects/mammaprint/l2-augmentation-v1
export MPA_MAPPING=/mnt/projects/mammaprint/data_mapping.csv
export MPA_TILING=mlflow-artifacts:/3/698ec2dcda3a4b7aa57b250dbbbfaec0/artifacts/mou_2_224_cancer_mask
export MPA_TRAIN=mlflow-artifacts:/3/b88229631c214c8e8d1161ce5f2ef23b/artifacts/l2_cancer_mask_5_embed_train
export MPA_VAL=mlflow-artifacts:/3/d47d498ad0524ad2b14a140779a66e4f/artifacts/l2_cancer_mask_5_embed_val
export MPA_TEST=mlflow-artifacts:/3/b1e75bf1510a4dc581d1c60ae4a061c5/artifacts/l2_cancer_mask_5_embed_test
mkdir -p "$MPA_ROOT/outputs"
uv run --frozen python -m preprocessing.l2_augmentation prepare \
  --raw-tiling "$MPA_TILING" --clean-uri "$MPA_TRAIN" --data-mapping "$MPA_MAPPING" \
  --split train --output "$MPA_ROOT/selection-train"
```

For the 90° validation probe, use `--split val --clean-uri "$MPA_VAL" --output "$MPA_ROOT/selection-probe" --slides` followed by the frozen validation stems. Keep the roster on secure storage. Choose cases spanning label range, tile count and source where known. Check selection counts, exclusions, split isolation and read geometry before embedding. Use an immutable WSI snapshot including companion files such as MRXS payloads. The worker verifies the main file's recorded size and modification time. The preparer reads clean Parquets one at a time to recover exact row order.

## Render an A40 job

Use the published Git SHA, a pinned 40-character Virchow2 HF revision and the approved image digest via `--image`. A tag such as the renderer default may move. Pin the digest before reusing a production view or checkpoint comparison. For archive reuse, first verify the original revision and matched fresh-clean probe parity. An unresolved historical revision calls for fresh, same-encoder control views.

```bash
uv run --no-sync python -m scripts.preprocessing.run_l2_augmentation_a40 \
  --job-name mp-l2-stain-pilot-1024-8 --context kubas-cluster --namespace rationai-jobs-ns \
  --source-commit "$MPA_CODE_SHA" --encoder-revision "$MPA_ENCODER_SHA" --image "$MPA_IMAGE" \
  --manifest "$MPA_ROOT/selection-train/manifest.json" \
  --output-root "$MPA_ROOT/outputs" --output "$MPA_ROOT/outputs/stain" \
  --view stain --seed 42 --batch-size 1024 --workers 8 --prefetch 2 \
  --cpu 24 --memory 64Gi --shm 12Gi --scratch-size 32Gi --ephemeral-storage 40Gi \
  --pilot stain-b1024-w8 --pilot-slides "${MPA_PILOT_SLIDES[@]}" \
  > mp-l2-stain-pilot-1024-8.json
```

Use a Bash array of fixed train slide stems for `MPA_PILOT_SLIDES`. Inspect the pinned code, WSI and mapping inputs, writable subtree, one-device resource limit and telemetry path in the JSON. After explicit approval of that exact manifest, create it with `kubectl --context kubas-cluster -n rationai-jobs-ns create -f mp-l2-stain-pilot-1024-8.json`.

Compare `512/8`, `1024/8`, `1024/12`, and `1536/12` batch/worker combinations at prefetch 2 on identical tiles. Use distinct Job and pilot names. Read `attempts/*/*.json` for sustained tiles/sec excluding model startup, loader and writer waits and peak reserved VRAM. Check `outputs/.telemetry/<job>/<pod UID>/` for sampled host memory, shared memory, GPU utilization and device memory. Compare steady-state measurements and repeat finalists in alternate order. Choose the fastest stable profile with roughly 20% RAM and VRAM headroom. Lower host RAM if a smaller allocation sustains that rate. Investigate reader wait, shared-storage throughput or writer wait when GPU utilization is low. Size durable output at 10,240 feature bytes per tile before compression plus metadata, and estimate GPU hours from observed tiles/sec.

For a full view, remove the two pilot arguments, set the winning profile, and use a new Job name. Use separate output roots for `clean`, `stain`, `rotation`, and `stain-rotation`. For the validation probe, embed the entire frozen subset once as `clean` and once as `rotation`. The latter always turns 90°. Production output is under `<output>/production`. To use direct MLflow storage, pass a dedicated run ID with `--mlflow-run-id` on a frozen production workload and benchmark that backend separately.

## Check the probe and publish the training card

Score the frozen validation subset on the secure work root. Choose thresholds for per-tile relative L2 parity, prediction drift and practical rotation effect before inspecting any scores. Start with archived seed 1. Its saved checkpoint matched the logged best step. The audit found stale saved steps for seeds 0 and 2, so those two need verified replacements for prediction or rank metrics. Treat those two logged MAEs as historical context until their best checkpoints are verified or clean controls are refit.

```bash
uv run --frozen python -m scripts.preprocessing.score_l2_rotation_probe \
  --manifest "$MPA_ROOT/selection-probe/manifest.json" --data-mapping "$MPA_MAPPING" \
  --clean-output "$MPA_ROOT/outputs/probe-clean/production" \
  --rotation-output "$MPA_ROOT/outputs/probe-rotation/production" \
  --run-id ef2eff36d88b4325b14f0acf67a681d0 \
  --parity-max-p99-relative-l2 "$MPA_VECTOR_TOL" \
  --parity-max-abs-drift "$MPA_PARITY_TOL" --rotation-min-mean-abs-shift "$MPA_ROTATION_MIN" \
  --output "$MPA_ROOT/rotation-probe.json"
```

Gate full rotation on measured sensitivity. Use separate stain and rotation arms first. Test combination with `stain-rotation` only after deciding to measure its additional effect. For an archive-parity-passed stain view, generate a card after the completed production inventory verifies.

```bash
uv run --frozen python -m preprocessing.l2_augmentation card \
  --manifest "$MPA_ROOT/selection-train/manifest.json" \
  --run-config "$MPA_ROOT/outputs/stain/production/run.json" \
  --augmented-uri "$MPA_ROOT/outputs/stain/production/embeddings" \
  --clean-val "$MPA_VAL" --clean-test "$MPA_TEST" \
  --output configs/data/embedded/cancer_mask_5/l2_stain.yaml
```

The card command verifies inventory, coordinate order, vector type and completion checksums. A direct-MLflow card uses the published embeddings URI and corroborates the local run JSON against the published metadata, then checks each file sequentially. For a fresh-control comparison, freeze the full validation selection with `--full-clean-val` and embed clean train and clean validation. Point the card at those outputs with `--clean-train`, `--clean-val`, and `--fresh-val-manifest` and omit the test URI until a recipe is selected. Use the same fresh card with `~dataset.levels.2.uris.train_augmented` for the clean-only baseline. To repeat the rotation probe with a fresh checkpoint, freeze a new probe against the complete fresh validation embeddings and score the same pinned runtime's probe views with the matching fresh checkpoint. Generate the same-encoder clean test view separately after recipe selection and explicit approval.

Fit the historical linear configuration for seeds 0, 1, and 2, using `+experiment=ml/train_mil_embeddings_l2_augmentation data/embedded=cancer_mask_5/l2_stain seed=<seed>`. Use a separate `metadata.run_name` per augmentation and fit seed. The preset enables split checks, validation-only evaluation and a dedicated view seed matched to the fit seed. Training selects one full clean or augmented bag per fetch. Keep zero data-loader workers for comparison and exact view-sampler resume. Evaluate clean full bags at the checkpoint selected by minimum validation loss. Plan fit scratch separately. An archived clean train and val already consume 132.25 GiB plus the augmented view if downloaded. Shared absolute augmentation paths load in place.

## Diagnose and resume

If a pod is pending, inspect whether the scheduler can meet the full A40, CPU, RAM, or ephemeral disk request. Investigate CUDA OOM from reported reserved bytes and host or IPC OOM from durable `resources.tsv`. Confirm the previous pod stopped and retain the same frozen manifest, view, seed, encoder, runtime and output on a resumed run. The writer verifies completion receipts before skipping slides. A run with a different transform or source needs a new output root. A direct MLflow run takes one writer and its original shared output root. Preserve partial outputs and attempt reports for diagnosis. Obtain approval for the revised job before submitting it.

See the [implementation plan](../../../docs/superpowers/plans/2026-11-25-l2-embedding-augmentation.md) for experiment acceptance.
