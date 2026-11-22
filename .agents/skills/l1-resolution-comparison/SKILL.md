---
name: l1-resolution-comparison
description: Run the MammaPrint Virchow2 L1 A40 embedding pilot, resumable shards, MLflow registration, and validation-only L1/L2 MIL regression training.
---

# L1 resolution comparison

Run from the MammaPrint repo on the pinned Linux runtime with `uv sync --frozen`, CUDA for embedding/training, the native OpenSlide library, secure DATA/PROJECTS mounts, an authenticated Hugging Face model cache or injected `HF_TOKEN`, and the approved MLflow endpoint. Use cluster-managed credentials. Check the context and namespace of the allocation against the approved target before running workload commands. The following commands run inside an existing allocation and do not submit Kubernetes jobs. Obtain explicit approval before creating any pilot, shard, publishing, training, or test job on a cluster.

## Submit one A40 embedding job with MLflow storage

Use this route when the user requests L1 embeddings directly and skips the benchmark. The job selects up to 3,000 non-overlapping tiles per slide with seed 42, then embeds all selected slides using FP32 Virchow2. It reads the secure DATA and PROJECTS mounts read-only and writes temporary files to a pod-local `emptyDir`. The selection, per-slide Parquets, upload receipts, and failure inventory remain in one MLflow run. L2 preparation and train/validation/test publication are subsequent stages.

Render the exact job before asking for submission approval. The renderer packages only the committed embedding code over the published Git base, references the cluster-managed Hugging Face Secret, and takes the cluster context and namespace explicitly.

```bash
uv run --frozen python -m scripts.preprocessing.run_l1_dataset_a40 \
  --source-commit "$(git rev-parse HEAD)" --context kubas-cluster \
  --namespace rationai-jobs-ns --job-name mammaprint-l1-embed-a40
```

After explicit approval, add `--submit` to create the job. Inspect its pod logs for the MLflow run ID. Expect `selection/manifest.json`, `selection/frozen.tar.gz`, `embeddings/`, `complete/`, and `metadata/l1_inventory.json`. Inspect the attempt report and missing-slide inventory if the run fails. To resume, confirm the prior pod stopped, retain the source commit, pass the recorded `--run-id` with a distinct job name, and obtain approval for the new submission. The worker restores the exact frozen selection and uses completed upload receipts to skip saved slides. Duplicate concurrent writers to the same run are unsupported.

## Prepare immutable input manifests

Run on a CPU allocation with enough local scratch for the projected tile metadata and shared space for manifests and selected L2 embeddings. Use the existing artifacts listed in the corresponding raw and embedded Hydra cards. Match the encoder SHA to the historical L2 artifact and record any provenance gap before interpreting the result. Use a pinned epithelium mask snapshot if available.

```bash
export RUN=/mnt/projects/mammaprint/l1-resolution-v1
export MAPPING=/mnt/projects/mammaprint/data_mapping.csv
export EPITHELIUM_MASKS=/mnt/projects/mammaprint/epithelium_onnx_masks
export L1_TILING='mlflow-artifacts:/3/7651a68400f6484db36d3626b1d64e83/artifacts/mou_1_224_cancer_mask'
export L2_TILING='mlflow-artifacts:/3/698ec2dcda3a4b7aa57b250dbbbfaec0/artifacts/mou_2_224_cancer_mask'
export L2_EMBEDDINGS='mlflow-artifacts:/3/ac2fdbe10e6047a99186e16ac149825c/artifacts/embeddings'
export SHARDS=1
mkdir -p "$RUN"

uv run --frozen python -m preprocessing.embedding_selection --level 1 \
  --tiling "$L1_TILING" --data-mapping "$MAPPING" --output "$RUN/l1" \
  --shards "$SHARDS" --mask-dir "epithelium_overlap=$EPITHELIUM_MASKS"
uv run --frozen python -m preprocessing.embedding_selection --level 2 \
  --tiling "$L2_TILING" --l2-embeddings "$L2_EMBEDDINGS" \
  --data-mapping "$MAPPING" --output "$RUN/l2"
```

Check per-slide geometry, `score_missing`, `mapping_without_source` and L2 coordinate mismatches in both manifests. Other unavailable score columns can be restored by passing another `--mask-dir SCORE=PATH_OR_URI` in a fresh selection root. The existing cancer and QC gates constrain every downstream mask. Set `SHARDS` to the intended number of independent work units. For a different shard count or a pilot-derived slide overhead weight, regenerate the manifest into a new root with the same cap and seed and optional `--slide-overhead-tiles`. Keep all production workers on the same resulting manifest and production output root.

## Benchmark the A40

Start with one full A40, 24 CPU cores, 64 GiB RAM, and 8–12 GiB `/dev/shm`. The loader holds uint8 RGB until transfer, giving an in-flight image payload floor of 2.30 GiB for 1,024 × 8 × 2 and 5.17 GiB for 1,536 × 12 × 2. Check CPU, host memory, IPC headroom and the GPU's free VRAM inside the pod. Use dedicated local scratch if testing staging. Confirm the shared production volume supports cross-pod `flock` and atomic same-directory rename.

Set `ENCODER_SHA` to the pinned 40-character HF commit and `RUNTIME_ID` to an identifier combining the pinned image and git revisions. Choose the same representative train/val slide stems for all profiles, covering WSI formats, sparse and capped bags. Use enough slides to reach steady state. Check that the GPU matches the allocation.

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
df -h /dev/shm "$RUN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 2 -f "$RUN/a40-utilization.csv"
```

Run monitoring in another terminal. Omit `--execute` to render the bounded jobs and inspect them. Execute in the approved pilot allocation.

```bash
uv run --frozen python -m scripts.preprocessing.run_resolution_embeddings pilot \
  --manifest "$RUN/l1/manifest.json" --output "$RUN/embed" --series a40-fp32-01 \
  --encoder-revision "$ENCODER_SHA" --runtime-id "$RUNTIME_ID" \
  --slides "${PILOT_SLIDES[@]}" --execute
```

Compare committed tiles/sec, loader wait, GPU inference, host-to-device and device-to-host copies, writer wait, PyTorch reserved VRAM, and the utilization log. Read/decode and writer totals are worker time and may overlap with GPU time. Select the fastest stable batch/worker pair with VRAM and host-memory headroom. Rerun finalists with new `--series` values and varied order to assess warm-cache effects. If I/O dominates, benchmark `--batch-size N --workers N --stage-dir /local/scratch` against mounted WSI reads. Staging copies complete single-file WSIs in the pilot for measurement. Allow space for the files and include staging time when projecting benefit.

Run a BF16 pilot with the winning batch/worker pair at the same tiles, then compare completed profiles. Review precision disagreement and compatibility with the historical FP32 L2 before using BF16 for the primary comparison. Keep FP32 as the production default. On stable pilots, divide the frozen tile count by measured sustained tiles/sec to estimate GPU hours, then add startup and failure overhead. If using multiple A40 pods, benchmark aggregate throughput under contention before projecting wall time.

```bash
uv run --frozen python -m scripts.preprocessing.run_resolution_embeddings pilot \
  --manifest "$RUN/l1/manifest.json" --output "$RUN/embed" --series a40-bf16-01 \
  --encoder-revision "$ENCODER_SHA" --runtime-id "$RUNTIME_ID" --precision bf16 \
  --batch-size "$BATCH" --workers "$WORKERS" --slides "${PILOT_SLIDES[@]}" --execute
uv run --frozen python -m scripts.preprocessing.run_resolution_embeddings compare-precision \
  --reference "$RUN/embed/pilot/$FP32_PROFILE" --candidate "$RUN/embed/pilot/$BF16_PROFILE"
```

## Embed and register

In each approved A40 allocation, set `SHARD_ID` to a distinct zero-based manifest shard. Substitute the benchmark winner for `BATCH` and `WORKERS`. Add `--dry-run` to inspect counts and output paths. Rerun the same shard after a recoverable interruption. Completed files are verified and reused, and all attempts are logged separately.

```bash
uv run --frozen python -m preprocessing.resolution_embeddings \
  --manifest "$RUN/l1/manifest.json" --output "$RUN/embed" --shard "$SHARD_ID" \
  --encoder-revision "$ENCODER_SHA" --runtime-id "$RUNTIME_ID" \
  --batch-size "$BATCH" --workers "$WORKERS"
```

Reconcile when all shards have stopped. Omit `--publish` to build and inspect the local inventory and matched bags first. Use `--allow-incomplete` only after reviewing and recording unresolved L1/L2 exclusions. Resolve large scale or coverage mismatches before interpreting a resolution result.

```bash
uv run --frozen python -m preprocessing.resolution_data \
  --l1-manifest "$RUN/l1/manifest.json" --l2-manifest "$RUN/l2/manifest.json" \
  --l1-output "$RUN/embed" --data-mapping "$MAPPING" --output "$RUN/data" --publish
```

Expect `l1_inventory.json`, `l1_data_card.json`, a verified MLflow run with all four artifact URIs, and the checksummed `comparison` bundle. Reissue the same publish command after an artifact transfer failure to resume the same MLflow run. For generic Hydra L1 callers, populate `MAMMAPRINT_L1_{ALL,TRAIN,VAL,TEST}_URI` from the generated data card. Preserve the frozen files and do not edit comparison Parquets in place.

## Train and compare by hand

Use a GPU allocation for fitting. The matrix runs sequentially by default. To distribute work across approved allocations, pass `--level 1 --pooling attention --seed 0` with the corresponding values for each fit. On an I/O-limited node, stage the immutable comparison bundle on local scratch and use the staged path as `--bundle`. Keep the same durable `--output` for checkpoints. Omit `--execute` to list the fit matrix.

```bash
uv run --frozen python -m scripts.ml.run_resolution_training \
  --bundle "$RUN/data/comparison" --data-mapping "$MAPPING" --output "$RUN/fits" --execute
```

Compare the MLflow runs by `resolution_level`, `resolution_pooling`, and `resolution_seed`. Read `valid/mae`, `valid/spearman`, and `valid/sign_auroc` at each run's minimum `valid/loss` step, which selects the saved best checkpoint. Pair L1 and L2 by pooling and seed. Record the chosen checkpoint before evaluating the held-out test data once through the existing test entry point.

## Diagnose and recover

- CUDA OOM during the pilot calls for a smaller batch. A worker bus error calls for checking `/dev/shm` and reducing prefetch or workers.
- Recheck WSI mounts, permissions and native OpenSlide errors for failed slides. Resume with the same manifest, encoder, precision, and runtime. An interrupted writer's temporary Parquet can be inspected and removed once no shard owns the lock.
- Keep pilot/training reports for cost accounting. Remove approved expendable scratch or complete pilot runs only after collecting evidence. Use a new output root for changed provenance or cohort. Preserve any published run and decision for reproducibility.
