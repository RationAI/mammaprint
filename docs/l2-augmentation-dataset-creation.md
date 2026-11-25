# Create the L2 augmentation datasets

These commands freeze the existing L2 cancer-mask-0.5 training bags, benchmark the stain embedding pipeline on one full A40, write the full augmented training artifact, and verify a YAML data card. The transforms operate on crops loaded from the original WSI files. The output contains every eligible historical tile in its original row order.

The launchers mount both historical WSI locations read-only, `/mnt/data/MOU/breast/mammaprint` and `/mnt/projects/quality_control/resampled_mammaprint`. The latter contains the resampled TIFFs referenced by the raw L2 tiling metadata.

## Set the inputs

Publish the implementation commit to the Git remote before launching a Job because each pod checks it out by SHA. Use the same code, image and Virchow2 revision for every compared view. The encoder SHA below is the pinned revision used by the project's recent L1 workflow. Its agreement with the older L2 embeddings still needs the clean parity probe described in the [augmentation runbook](../.agents/skills/l2-embedding-augmentation/SKILL.md).

Run the local renderer commands in Bash from the repository checkout using the existing cluster-launch environment, which supplies `kube_jobs` and `kubernetes`. Preserve those launcher packages when syncing project dependencies with `uv sync --frozen --inexact`. The renderer commands use `--no-sync` and perform local manifest generation. Set an approved immutable image digest for the full experiment. The default base tag is usable for an initial pilot.

```bash
uv sync --frozen --inexact
export MPA_CODE_SHA=$(git rev-parse HEAD)
export MPA_IMAGE=cerit.io/rationai/base:2.0.6
export MPA_ENCODER_SHA=3158645804b69e3f3bc4439d4116edddf0840a72
export MPA_ROOT=/mnt/projects/mammaprint/l2-augmentation-v1
export MPA_TILING=mlflow-artifacts:/3/698ec2dcda3a4b7aa57b250dbbbfaec0/artifacts/mou_2_224_cancer_mask
export MPA_TRAIN=mlflow-artifacts:/3/b88229631c214c8e8d1161ce5f2ef23b/artifacts/l2_cancer_mask_5_embed_train
export MPA_VAL=mlflow-artifacts:/3/d47d498ad0524ad2b14a140779a66e4f/artifacts/l2_cancer_mask_5_embed_val
```

Create `$MPA_ROOT` and `$MPA_ROOT/outputs` once on the cluster's projects mount from an authorized environment that already has it mounted. The launchers mount pre-existing PVC subdirectories. Use a new versioned root if those locations contain an earlier selection. Keep WSI files and companion payloads on an immutable snapshot while embedding and resuming.

Alternatively pass `--create-output-root` to the CPU preparation renderer. A short init container creates only that subtree and its `outputs` directory before the scoped main container starts.

```bash
mkdir -p "$MPA_ROOT/outputs"
```

## Prepare the training selection on CPU

```bash
uv run --no-sync python -m scripts.preprocessing.run_l2_augmentation_cpu prepare \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --job-name mp-l2-augment-prepare --source-commit "$MPA_CODE_SHA" \
  --image "$MPA_IMAGE" --output-root "$MPA_ROOT" \
  --raw-tiling "$MPA_TILING" --clean-uri "$MPA_TRAIN" \
  --data-mapping /mnt/projects/mammaprint/data_mapping.csv \
  --split train --output "$MPA_ROOT/selection-train" \
  > mp-l2-augment-prepare.json
```

Inspect the manifest, then submit it after approval of that exact Job and target. This CPU Job requests 8 cores and 16 GiB RAM with 32 GiB scratch and a 40 GiB ephemeral-storage limit. It stages one historical embedding Parquet at a time. It reads the raw tiling metadata and original WSI file stats. It writes `manifest.json`, per-slide frozen coordinates and `summary.json`.

```bash
kubectl --context kubas-cluster -n rationai-jobs-ns create -f mp-l2-augment-prepare.json
kubectl --context kubas-cluster -n rationai-jobs-ns wait \
  --for=condition=Complete job/mp-l2-augment-prepare --timeout=60s
kubectl --context kubas-cluster -n rationai-jobs-ns logs job/mp-l2-augment-prepare > mp-l2-augment-prepare.log
```

Repeat the bounded wait if still running. If failed, inspect logs before retrying. From the completed log, save the final JSON summary locally and use its four tile-count-spanning slide IDs for the pilot.

```bash
python3 - <<'PY'
import json
from pathlib import Path
summaries = []
for line in Path('mp-l2-augment-prepare.log').read_text().splitlines():
    try:
        item = json.loads(line)
    except ValueError:
        continue
    if isinstance(item, dict) and 'pilot_slides' in item and 'feature_bytes_per_view' in item:
        summaries.append(item)
if len(summaries) != 1:
    raise SystemExit('Expected exactly one completed selection summary')
Path('mp-l2-summary.json').write_text(json.dumps(summaries[0], indent=2) + '\n')
print(json.dumps(summaries[0], indent=2))
PY
MPA_PILOT_SLIDES=()
while IFS= read -r slide; do MPA_PILOT_SLIDES+=("$slide"); done < <(
  python3 -c 'import json; print("\n".join(json.load(open("mp-l2-summary.json"))["pilot_slides"]))'
)
```

Reserve durable output space from `feature_bytes_per_view`, allowing for Parquet metadata, receipts and all views to be retained. A full stain view needs approximately the same raw feature payload as its clean counterpart. The 32 GiB pod scratch request covers dependencies and incremental metadata downloads because embeddings are written to the shared output mount.

## Measure the A40 profile

```bash
uv run --no-sync python -m scripts.preprocessing.run_l2_augmentation_a40 \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --job-name mp-l2-stain-pilot-b1024-w8 --source-commit "$MPA_CODE_SHA" \
  --image "$MPA_IMAGE" --encoder-revision "$MPA_ENCODER_SHA" \
  --manifest "$MPA_ROOT/selection-train/manifest.json" \
  --output-root "$MPA_ROOT/outputs" --output "$MPA_ROOT/outputs/stain" \
  --view stain --seed 42 --batch-size 1024 --workers 8 --prefetch 2 \
  --cpu 24 --memory 64Gi --shm 12Gi --scratch-size 32Gi --ephemeral-storage 40Gi \
  --pilot b1024-w8 --pilot-slides "${MPA_PILOT_SLIDES[@]}" \
  > mp-l2-stain-pilot-b1024-w8.json
```

After reviewing and approving the rendered Job, create it using the same `kubectl ... create -f` form. Run the four profiles sequentially, changing the Job and pilot names with the batch and worker counts `512/8`, `1024/8`, `1024/12`, and `1536/12`. Use the same tile roster in every profile. The attempt JSON files report committed tiles/sec, loader and writer waits and reserved GPU memory. Host RAM, shared memory, GPU utilization and device memory are sampled under `$MPA_ROOT/outputs/.telemetry/<job>/<pod UID>/`.

Choose the fastest stable profile with approximately 20% measured RAM and GPU memory headroom. Repeat the finalist with a smaller RAM allocation if measurements suggest the same throughput. The script validates queue memory and shared memory against the requested batch and worker counts. Actual utilization depends on WSI read and decode throughput and shared output bandwidth.

## Create the full stain view

Use the winning pilot settings below. The initial values show the starting profile. Creating another Job requires reviewing and approving its rendered scope.

```bash
MPA_BATCH=1024
MPA_WORKERS=8
MPA_CPU=24
MPA_RAM=64Gi
uv run --no-sync python -m scripts.preprocessing.run_l2_augmentation_a40 \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --job-name mp-l2-stain-full --source-commit "$MPA_CODE_SHA" \
  --image "$MPA_IMAGE" --encoder-revision "$MPA_ENCODER_SHA" \
  --manifest "$MPA_ROOT/selection-train/manifest.json" \
  --output-root "$MPA_ROOT/outputs" --output "$MPA_ROOT/outputs/stain" \
  --view stain --seed 42 --batch-size "$MPA_BATCH" --workers "$MPA_WORKERS" --prefetch 2 \
  --cpu "$MPA_CPU" --memory "$MPA_RAM" --shm 12Gi --scratch-size 32Gi --ephemeral-storage 40Gi \
  > mp-l2-stain-full.json
```

Production output is `$MPA_ROOT/outputs/stain/production/`, containing `run.json`, `embeddings/`, `complete/` and `attempts/`. Each complete slide is checkpointed with a checksum. Resume an interrupted run using a distinct Job name and the original selection, encoder, seed, view, runtime and output after confirming its earlier pod stopped.

For a full rotation-only training view, use `--view rotation`, `--output "$MPA_ROOT/outputs/rotation"` and a distinct Job name. For the optional combined view, use `--view stain-rotation` with its own output and Job name. The agreed first rotation stage is the smaller validation probe. Freeze it on CPU with `--split val --clean-uri "$MPA_VAL" --slides <frozen validation stems>`, then embed that selection as `clean` and `rotation` in distinct output directories. Validation rotations are fixed at 90 degrees. Details and probe scoring commands are in the [runbook](../.agents/skills/l2-embedding-augmentation/SKILL.md).

## Verify and write the training card on CPU

Use this archive-control route after clean parity confirms that the pinned encoder is compatible with the historical L2 embeddings. The card references the durable shared artifact directly.

```bash
uv run --no-sync python -m scripts.preprocessing.run_l2_augmentation_cpu card \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --job-name mp-l2-stain-card --source-commit "$MPA_CODE_SHA" \
  --image "$MPA_IMAGE" --output-root "$MPA_ROOT" \
  --manifest "$MPA_ROOT/selection-train/manifest.json" \
  --run-config "$MPA_ROOT/outputs/stain/production/run.json" \
  --augmented-uri "$MPA_ROOT/outputs/stain/production/embeddings" \
  --clean-val "$MPA_VAL" --output "$MPA_ROOT/cards/l2_stain.yaml" \
  > mp-l2-stain-card.json
```

Submit the verified CPU manifest after approval. Successful card creation verifies the full slide inventory, ordered coordinates, vector schema and per-slide checksums. Copy the completed YAML into `configs/data/embedded/cancer_mask_5/l2_stain.yaml` in the checkout used for training. It carries `train`, `val` and `train_augmented` URIs. The normal validation-only augmentation training preset can then select it.

If historical encoder parity fails, create fresh clean train and clean validation views with the same encoder. Use `prepare --split val --full-clean-val` for the full validation selection and `embed --view clean` for both controls. The card launcher accepts `--clean-train`, `--clean-val` and `--fresh-val-manifest` to verify those controls. Preserve these stages as dataset creation work before starting separate model fits.
