# L1 dataset publication plan

Publish Hydra dataset cards and MLflow split artifacts for the completed L1 cancer-gated embeddings, using the same layout as the existing levels. Publication uses the existing tile selection and runs entirely on CPU.

## Publish the variants available from the completed run

Use MLflow run `6d2c5bc4b2de4f2b9ef5987a163f2518` as the immutable parent. The completion inventory verifies all 5,929,095 selected tiles across 2,040 eligible slides, with zero missing or invalid outputs. The run used the non-overlapping L1 grid with a 3,000-tile cap and seed 42. Every embedded tile has cancer, epithelium, tissue and QC scores. All parent tiles pass the original cancer probability `> 0.5` gate.

The saved selection Parquets give these exact counts before publishing. The slide counts count nonempty bags.

| Hydra card | Mask | Train slides / tiles | Val slides / tiles | Test slides / tiles | Total tiles |
| --- | --- | ---: | ---: | ---: | ---: |
| `cancer_mask_5/l1` | Parent sample | 1,428 / 4,156,719 | 304 / 874,536 | 308 / 897,840 | 5,929,095 |
| `cancer_mask_9/l1` | Cancer `> 0.9` | 1,427 / 3,130,174 | 304 / 641,190 | 308 / 649,223 | 4,420,587 |
| `epithel_and_cancer_5/l1` | Epithelium `> 0.25` and cancer `> 0.5` | 1,428 / 3,487,278 | 304 / 733,697 | 308 / 738,386 | 4,959,361 |

1. Extend the existing `preprocessing/filter_embeddings.py` script and its `scripts/preprocessing/run_filter_embeddings.py` launcher. Add an opt-in mode for reading saved scores from each embedding Parquet and a cancer-only threshold rule alongside the existing epithelial AND cancer rule. Use the existing split-routing logic in `preprocessing/split_dataset.py` and completion/resume patterns in `preprocessing/embedding_artifacts.py`. Preserve the coordinates, embedding, sample rank and mask scores.
2. Verify the source inventory against the selection manifest and existing split map. Read each parent Parquet once, validate its receipt, and stream all three outputs to MLflow using pod-local scratch. Use the existing parent MLflow URI for `cancer_mask_5/l1` `uris.all`. Publish a flat `all` directory for each derived variant and `train`, `val` and `test` directories for all three variants. Persist job provenance, per-output completion records, and slide/tile counts to support an interrupted upload. Preserve the empty train bag for `cancer_mask_9` and report it, following the other filtered datasets. The training dataset loader excludes empty bags.
3. Before any cluster submission, test routing, strict thresholds, empty output, and restart on a local fixture. Render and review **`mammaprint-l1-publish-cancer-splits`** for **`kubas-cluster`**, namespace **`rationai-jobs-ns`**. Propose **one CPU job with 16 CPUs, 64 GiB RAM and 32 GiB pod-local scratch** and read-only input access. The parent is 57.85 GiB, and the projected MLflow upload is about 241 GiB with the existing flat `all` and split-directory layout. The job writes artifacts to MLflow. Obtain explicit approval for the rendered submission.
4. Verify the published file inventories and counts against the table, split disjointness, column schemas, and loader compatibility. Replace the environment placeholders in `configs/data/embedded/cancer_mask_5/l1.yaml` with verified URIs and add `configs/data/embedded/cancer_mask_9/l1.yaml` and `configs/data/embedded/epithel_and_cancer_5/l1.yaml`. Give each card concrete `all`, `train`, `val`, and `test` URIs, the nominal non-overlapping stride of 224, and provenance for the cap and source gate. Commit the cards after the URIs resolve and Hydra smoke tests pass. Include a storage budget and exact dry-run, resume and verification commands in the repository skill when implementing the publisher.

## What these cards cover

The three cards represent the completed, cancer-gated L1 sample. In this pool, epithelium-only selects the same rows as epithelium AND cancer, and epithelium OR cancer selects the whole parent. The broader `tissue_only`, `epithel`, and `epithel_or_cancer_5` definitions used by other levels start from the tissue pool, which includes tiles below the cancer threshold. Any expansion to that pool is a separate scope decision.
