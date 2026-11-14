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

### Filtering existing embeddings with stricter QC

`preprocessing.filter_embeddings` joins a scored tiled dataset (`slides.parquet`
and `tiles.parquet`) to per-slide embedding Parquets by slide filename and `(x, y)`.
It reuses the embedding vectors and writes new files; no encoder or GPU is needed.

Use `--strategies qc_only` to keep the source dataset's existing tissue/cancer
selection and apply only additional QC gates. For example, to inspect a local
90%-clean variant:

```bash
uv run -m preprocessing.filter_embeddings \
  --level 2 \
  --tiling-uri /path/to/corresponding/scored-tiling \
  --embeddings-uri /path/to/source-embeddings \
  --data-mapping /path/to/original/data_mapping.csv \
  --strategies qc_only \
  --blur-threshold 0.9 \
  --folding-threshold 0.9 \
  --residual-threshold 0.9 \
  --output-dir /path/to/new/qc90-results \
  --no-upload
```

- Each enabled gate keeps `MASK_overlap > threshold`; equality is rejected.
  QC scores are **clean area fractions**, so higher thresholds remove more tiles.
  Omitted QC flags impose no additional gate for that mask.
- The `epithelium_only`, `or`, and `and` strategies also accept QC gates. QC is
  ANDed with their existing selection. Defaults remain the original three
  strategies with no additional QC filtering.
- Only scores used by the chosen strategies are required. QC-only filtering
  accepts the original cancer-mask tiling artifacts with null epithelium scores.
  Missing, non-finite, or out-of-range active scores cause an error.
- Use the **corresponding tiled dataset**: every scored coordinate must have an
  existing embedding. QC-only also requires scores for every source embedding,
  so a mismatched or narrower tiling fails instead of silently dropping tiles.
  Tightening can reuse existing vectors; recovering tiles
  removed before embedding needs a broader source with those embeddings.
- The supplied mapping preserves its train/validation/test assignments. Empty
  slides remain as empty Parquets; inspect reported `empty_slides` and
  `slides_below_100_tiles` before training. No replacement tiles are sampled.
- Thresholds appear in dataset names, filter state, and MLflow manifests/params.
  `--upload-only` verifies the same source URIs, strategies, and thresholds before
  reusing a completed output directory. Old non-QC filter states remain supported.
- `--no-upload` requires a fresh `--output-dir` and creates no MLflow runs or
  uploads. Without it, the existing workflow uploads new filtered datasets to
  MLflow. Both local paths and MLflow artifact URIs can be supplied as sources.

The CPU launcher `scripts/preprocessing/run_filter_embeddings.py` forwards the
same QC flags and supports `--dry-run` to print the command without submitting a
job. Set `--git-ref` to a published branch or commit containing these changes.
When supplying `--embeddings-uri`, an optional `--embeddings-path` must refer to
the same source; otherwise the launcher downloads the explicit URI.
