# MammaPrint

Michal Kišš, Matěj Pekár

## Getting started

### Instalation

Install [UV](https://docs.astral.sh/uv/) package manager and install all dependencies using the following command:
```bash
uv sync
```

### Preprocessing

Use scripts/preprocessing scripts.

#### Epithelium masks with the ONNX model

The model consumes raw RGB image data, not embeddings. It was trained on 512 x
512 tiles with stride 256 at 0.468 um/px. Its training slides reported that as
pyramid level 0; pyramid indices are slide-specific, and the physically matching
level in the MammaPrint slides is L1 (about 0.485 um/px).

First submit the repository's tiling pipeline. Its new default experiment asks
for 0.468 um/px, causing ratiopath to select the nearest available MammaPrint
level (L1), and uses the exact 512/256 training geometry:

```bash
uv run --extra submit python scripts/preprocessing/run_tiling.py \
  --username <cluster-username>
```

This uses the slides in `/mnt/projects/mammaprint/data_mapping.csv`, inherited
from `configs/data/default.yaml`, and logs a raw tiled dataset named
`mou_epithelium_512_tissue` to MLflow. It uses tissue and QC masks to avoid
running inference on irrelevant tiles; it does not create or consume
embeddings.

The successful tiled artifact is registered in
`configs/data/tiled/tissue_only/epithelium_512.yaml`, so submit inference with:

```bash
uv run --extra submit python scripts/preprocessing/run_epithelium_masks.py \
  --username <cluster-username>
```

Use `--tiled-dataset <uri-or-directory>` to override the registered artifact.

The inference job downloads the ONNX model and tiled metadata, reads the raw RGB
pixels from the original slides at the recorded level and coordinates, blends
overlapping predictions, and writes one pyramidal TIFF per slide to
`/mnt/projects/mammaprint/epithelium_onnx_masks`. Filenames match the convention
used by `preprocessing/tiling.py`, so the directory can be used as slide-level
epithelium masks later. The model URI embeds the old MLflow artifact server as
`mlflow-artifacts://mlflow.rationai-mlflow:5000/...`, while the default tracking
URI remains the new `http://mlflow-s3.rationai-mlflow`. This downloads the model
from the old server but downloads the tiled dataset from—and logs the masks
to—the new server.

You can still skip the pre-tiling step and process the configured slide mapping
directly with `--data-mapping /mnt/projects/mammaprint/data_mapping.csv`, or pass
selected files with `--input`.
Use `--kind tile` only for already-extracted raw RGB images. Direct slide masks
are pyramidal BigTIFF files; ordinary image-tile masks are same-size PNG files.
Masks always retain the blended epithelial probabilities and store them as
grayscale values from 0 to 255 (`pixel / 255` recovers the approximate 0-to-1
probability). Add `--source-mpp <value>` if slide MPP metadata is missing, or
`--dry-run` to print commands without submitting.

For example, to process selected raw image tiles:

```bash
uv run --extra submit python scripts/preprocessing/run_epithelium_masks.py \
  --username <cluster-username> \
  --input /mnt/projects/mammaprint/my_tiles \
  --output-dir /mnt/projects/mammaprint/my_tile_masks \
  --kind tile
```

The default model is:

```text
mlflow-artifacts:/10/39f821ed5b964c71a603cc6db196f9fd/artifacts/checkpoints/epoch=19-step=32020/model.onnx/model.onnx
```
