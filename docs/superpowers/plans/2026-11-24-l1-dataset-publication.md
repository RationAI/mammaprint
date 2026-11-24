# L1 dataset publication plan

Publish Hydra dataset cards and MLflow split artifacts for L1 using the same layout as the existing levels. The first release uses the completed L1 cancer-gated embeddings. A second release expands the tile pool to cover the tissue-only and broader epithelial variants.

## 1. Publish the variants available from the completed run

Use MLflow run `6d2c5bc4b2de4f2b9ef5987a163f2518` as the immutable parent. It has 2,040 eligible slides and 5,929,095 embedded tiles, selected from the non-overlapping L1 grid with a 3,000-tile cap and seed 42. Every embedded tile has cancer, epithelium, tissue and QC scores. All parent tiles pass the original cancer probability `> 0.5` gate.

The saved selection Parquets give these exact counts before publishing. The slide counts count nonempty bags.

| Hydra card | Mask | Train slides / tiles | Val slides / tiles | Test slides / tiles | Total tiles |
| --- | --- | ---: | ---: | ---: | ---: |
| `cancer_mask_5/l1` | Parent sample | 1,428 / 4,156,719 | 304 / 874,536 | 308 / 897,840 | 5,929,095 |
| `cancer_mask_9/l1` | Cancer `> 0.9` | 1,427 / 3,130,174 | 304 / 641,190 | 308 / 649,223 | 4,420,587 |
| `epithel_and_cancer_5/l1` | Epithelium `> 0.25` and cancer `> 0.5` | 1,428 / 3,487,278 | 304 / 733,697 | 308 / 738,386 | 4,959,361 |

1. Reuse the split-routing behavior in `preprocessing/split_dataset.py`, the mask logic in `preprocessing/filter_embeddings.py`, and the completion/resume patterns in `preprocessing/embedding_artifacts.py`. Add a publication entry point that reads the frozen inventory and applies masks to the scores already in each embedding Parquet. The current mask CLI expects a tissue-only embedding source plus a separate scored tiling, and the launcher selects levels 2–5. The L1 entry point should preserve the coordinates, embedding, sample rank and mask scores.
2. Verify the source inventory against the selection manifest and existing split map. Read each parent Parquet once, validate its receipt, and stream all three outputs to MLflow using pod-local scratch. Use the existing parent MLflow URI for `cancer_mask_5/l1` `uris.all`. Publish a flat `all` directory for each derived variant and `train`, `val` and `test` directories for all three variants. Persist job provenance, per-output completion records, and slide/tile counts to support an interrupted upload. Preserve the empty train bag for `cancer_mask_9` and report it, following the other filtered datasets. The training dataset loader excludes empty bags.
3. Before any cluster submission, test routing, strict thresholds, empty output, and restart on a local fixture. Render and review **`mammaprint-l1-publish-cancer-splits`** for **`kubas-cluster`**, namespace **`rationai-jobs-ns`**. Propose **one CPU job with 16 CPUs, 64 GiB RAM and 32 GiB pod-local scratch** and read-only input access. The parent is 57.85 GiB, and the projected MLflow upload is about 241 GiB with the existing flat `all` and split-directory layout. The job writes artifacts to MLflow. Obtain explicit approval for the rendered submission.
4. Verify the published file inventories and counts against the table, split disjointness, column schemas, and loader compatibility. Replace the environment placeholders in `configs/data/embedded/cancer_mask_5/l1.yaml` with verified URIs and add `configs/data/embedded/cancer_mask_9/l1.yaml` and `configs/data/embedded/epithel_and_cancer_5/l1.yaml`. Give each card concrete `all`, `train`, `val`, and `test` URIs, the nominal non-overlapping stride of 224, and provenance for the cap and source gate. Commit the cards after the URIs resolve and Hydra smoke tests pass. Include a storage budget and exact dry-run, resume and verification commands in the repository skill when implementing the publisher.

## 2. Add the broader tissue variants

The completed run can only supply tiles that pass cancer `> 0.5`. Within that parent, epithelium-only selects the same rows as epithelium AND cancer, and epithelium OR cancer selects the whole parent. The broader variants need the tissue tile pool, including tiles below the cancer threshold. The [existing L1 tissue tiling](https://mlflow-s3.rationai.cloud.trusted.e-infra.cz/#/experiments/3/runs/43602e9f922d46f18dc06910b66a3be1) is available. The older tissue-embedding run has no published Parquets.

| Order | Proposed job | Purpose and approval evidence |
| --- | --- | --- |
| 1 | `mammaprint-l1-prepare-tissue-selection` (CPU) | Start from the existing tissue tiling, keep the same non-overlapping grid, and apply the proposed 3,000-tile cap and seed 42 to the tissue/QC pool. Recover mask scores where required. Join by slide and tile coordinates to the existing embeddings and verify geometry and encoder compatibility. Save the frozen selection, reusable and missing tile counts, predicted GPU hours and upload footprint to MLflow. Show the manifest and resources for separate approval. |
| 2 | `mammaprint-l1-embed-tissue` (A40, if needed) | Embed the missing tissue coordinates using the same pinned Virchow2 revision and FP32 settings. Review the measured tile count, projected cost, and exact A40 submission first. Reuse compatible vectors from the finished L1 run. |
| 3 | `mammaprint-l1-publish-tissue-masks` (CPU) | Publish the broader tissue parent and filter it to epithelium `> 0.25` and epithelium `> 0.25` OR cancer `> 0.5` using the existing filter logic. Verify split integrity and card compatibility, then add `tissue_only/l1`, `epithel/l1` and `epithel_or_cancer_5/l1` with real MLflow URIs. Show the exact publishing job for approval after the tissue inputs are ready. |

The two releases produce six L1 cards matching the other levels’ layout. Each card identifies its sampling pool, score thresholds, realized coverage and mask provenance. Keep the existing cancer-based artifacts versioned as their own datasets. A broader capped sample and the existing cancer-gated capped sample have different coverage, which matters when comparing model performance. Processing every original overlapping L1 tile would require a separately scoped run.
