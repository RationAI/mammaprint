# GPU memory reference for shared MammaPrint training

Use this reference before setting `--max-parallel` on [the Job renderer](render_training_batch_job.py). One Kubernetes Job requests one GPU or MIG instance. [The batch runner](run_training_batch.py) runs each seed in a separate process and starts another when a slot opens. The [shared-GPU training skill](../../.agents/skills/shared-gpu-training/SKILL.md) covers rendering, approval, monitoring, and recovery.

## Lookup table

Each cell gives **typical / planning GiB per fit** for the `cancer_mask_5` embedding card. Typical uses the mean bag size for each enabled stage and reports the largest stage estimate. Planning uses the largest recorded bag in each enabled stage and rounds the largest stage estimate up to the next 0.25 GiB. Both are model-based estimates. L1 substitutes the selection cap of 3,000 tiles for both bag sizes.

| Level | Mean/max + linear | Mean/max + MLP | Gated attention + linear | Gated attention + MLP | ABMIL mean | ABMIL attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 1.06 / 1.25 | 1.07 / 1.25 | 1.27 / 1.50 | 1.28 / 1.50 | 1.08 / 1.25 | 1.11 / 1.25 |
| L2 | 1.15 / 2.00 | 1.17 / 2.00 | 1.63 / 4.00 | 1.64 / 4.00 | 1.11 / 1.50 | 1.13 / 1.50 |
| L3 | 1.04 / 1.25 | 1.05 / 1.25 | 1.20 / 2.00 | 1.21 / 2.00 | 1.08 / 1.25 | 1.10 / 1.25 |
| L4 | 1.01 / 1.25 | 1.02 / 1.25 | 1.09 / 1.25 | 1.10 / 1.50 | 1.05 / 1.25 | 1.06 / 1.25 |
| L5 | 1.00 / 1.25 | 1.01 / 1.25 | 1.06 / 1.25 | 1.07 / 1.25 | 1.02 / 1.25 | 1.02 / 1.25 |

The table includes reference recipes and source cards from other MammaPrint revisions. Check that the selected recipe and active split URIs are present at the pinned training revision.

The first four columns use one uncapped bag per step in train, validation, and test. The ABMIL presets use four train bags with a 1,024-tile cap, one uncapped validation bag, and validation-only fits. All profiles use frozen Virchow2 embeddings of width 2,560 and FP32. The plain MLP head has width 256. Plain gated attention has width 512 and LayerNorm. The ABMIL projector has width 256, and the projected gated attention has width 128. Use the L1 row with the published 3,000-tile selection recipe and verify that the resolved L1 source uses that cap.

## Size a Job from the reference

Use the **planning** column to pack runs. The total of the per-fit budgets should stay within 80% of the visible device's GiB. For example, an A100 MIG 1g.10gb allocation in the pilot exposed 9.50 GiB, giving a 7.60 GiB planning budget. Two uncapped L2 gated-attention fits would require 8.00 GiB on this model, so the lookup screens them out. Two L2 ABMIL mean fits were observed to fit. A VRAM ceiling gives the maximum to consider. Also check the runnable seed count, CPU, host RAM, and scratch before choosing concurrency.

For `cancer_mask_9`, `epithel`, `epithel_and_cancer_5`, `epithel_or_cancer_5`, or `tissue_only` on L2–L5, use the same checked-in inventory through the estimator. Pass actual Hydra batch sizes and the training cap when they differ from the defaults. Set `--no-include-test` for validation-only experiments, or `--include-test` when running the test stage. Use `--train-tile-cap 0` for uncapped training. Evaluation still uses full bags.

```bash
python -m scripts.ml.estimate_training_batch_memory \
  --level 2 --mask tissue_only --architecture attention-mlp \
  --train-batch-size 1 --eval-batch-size 1 --no-include-test --device-gib 9.50
```

For the plain profiles, the first part of the estimator name is the Hydra `ml/aggregator` and the second is `ml/head`. For example, `attention-mlp` corresponds to `--override ml/aggregator=attention --override ml/head=mlp` on the rendered frozen-embedding experiment. `abmil-mean` and `abmil-attention` refer to the named ABMIL regression presets.

The JSON reports per-stage estimates, the per-fit planning budget, and `vram_fit_count_ceiling` when a device size is provided. Use the existing pilot evidence for a matching recipe and allocation. Start with one fit on an unprofiled recipe and device. Measure two together if the estimated budget accommodates them and record the real peak and throughput. Raw-image encoders, multiscale bags, and trainable tile backbones require their own sizing evidence.

Scratch scales with simultaneous MLflow downloads even when VRAM is low. Sum all enabled splits, including test when enabled, per active fit. For example, the pinned L2 `cancer_mask_5` train and validation files occupy 132.25 GiB per fit. The two-fit pilot used a 320 GiB scratch volume and a 336 GiB ephemeral-storage reservation to accommodate two copies, startup files, and logs.

## Basis and calibration

- [The bag-size inventory](training_batch_bag_sizes.csv) records the mean, nearest-rank p95, and maximum of nonempty `ok` bags separately for train, validation, and test. It also records the mean after capping each bag at 1,024. It contains aggregate counts for 24 L2–L5 level/mask combinations and zero slide identifiers or labels. The source is `/mnt/projects/mammaprint/ridge-screening/ridge-screening/cache/coverage.parquet`, SHA-256 `ef98adeb80ad64207af40d5cb4ab91ed7ad636c829cfb824fa055dc1295b924a`. The manifest's 24 source URIs matched the `uris.all` values in the corresponding embedded cards. The L1 resolution-comparison workflow supplies the 3,000-tile L1 selection cap. The audited cache had no L1 bag counts.
- [The estimator](estimate_training_batch_memory.py) adds a 1.0 GiB allowance per trainer for CUDA and workspace overhead, 20 bytes per parameter for weights, gradients, Adam states and optimizer scratch, and four bytes for each modeled per-tile float. The per-tile count includes the raw input and conservative allowances for simultaneously live activations and work buffers in each stage. The batch is ragged, so it estimates the total across bags without padding. A worst case uses the maximum per-slide count times the batch size. At a custom training cap other than 1,024, `min(mean, cap)` is an upper estimate of the capped mean. The largest observed bag applies to the inventory used here, and a different card revision can change it.
- **Measured anchor.** Recipe `65774dc` ran L2 `cancer_mask_5` ABMIL mean with batch 4, train cap 1,024, validation batch 1, and two epochs. On the same A100 MIG instance, `mp-abmil-mig10-serial-20261124-a2` peaked at **1.24 GiB** over 3,494 successful NVML samples and completed four fits. `mp-abmil-mig10-paired-20261124-a2` peaked at **2.45 GiB combined** over 1,468 successful samples and completed seeds 101 and 102 together. Seeds 103 and 104 failed while loading artifacts. This anchor supports two concurrent fits for that recipe. The other cells are analytical estimates. The durable run directories under `/mnt/projects/mammaprint/training-batches` hold each Job's `resources-summary.json` and `status.json`.

When a matching run yields a new reliable peak, record the recipe, device capacity, batch sizes, training cap, concurrency, peak, and outcome alongside the anchor. Update the inventory and the lookup table together when the data cards change.
