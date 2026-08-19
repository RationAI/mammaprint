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

`scripts/preprocessing/run_epithelium_masks.py` submits a Kubernetes job that
downloads the epithelial segmentation model from MLflow and runs it directly on
mounted whole slides or image tiles:

```bash
uv run --extra submit python scripts/preprocessing/run_epithelium_masks.py \
  --username <cluster-username> \
  --input /mnt/data/Projects/mammaprint/slides \
  --output-dir /mnt/projects/mammaprint/epithelium_masks \
  --kind slide
```

Use `--kind tile` for already-extracted images. Slide masks are pyramidal
BigTIFF files at 0.468 um/px; tile masks are same-size PNG files. The default is
a binary 0/255 mask at threshold 0.5. Add `--output-type probability` to retain
probabilities, `--source-mpp <value>` if slide MPP metadata is missing, or
`--dry-run` to print the job commands without submitting.

The default model is:

```text
mlflow-artifacts:/10/39f821ed5b964c71a603cc6db196f9fd/artifacts/checkpoints/epoch=19-step=32020/model.onnx/model.onnx
```
