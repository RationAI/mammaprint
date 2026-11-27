# Paired multiscale comparison

The branch is based on local `feat/tiling-values` at `705e670`, which includes the shared-GPU runner. This comparison fills the missing seeds for the CM5 and EAN5 L4/L3/L2 regression screen.

## Exact fit matrix

| Experiment under `ml/multiscale/screen/` | Seeds to run | Adapter | Within level | Fusion | Slide pooling | Linear head input |
| --- | --- | --- | --- | --- | --- | --- |
| `cm5_mean_regression` | 1, 2 | identity | mean | concat | mean | 7680 |
| `ean5_mean_regression` | 1, 2 | identity | mean | concat | mean | 7680 |
| `cm5_projected_mean_regression` | 0, 1, 2 | shared linear, 256 | mean | concat | mean | 768 |
| `ean5_projected_mean_regression` | 0, 1, 2 | shared linear, 256 | mean | concat | mean | 768 |
| `cm5_slide_attention_regression` | 1, 2 | shared linear, 256 | mean | concat | gated attention, width 256 | 768 |
| `ean5_slide_attention_regression` | 1, 2 | shared linear, 256 | mean | concat | gated attention, width 256 | 768 |

All fits use frozen Virchow2 embeddings of width 2560, identity encoder and output activation, MSE, Adam with learning rate and weight decay both `0.001`, ReduceLROnPlateau on `valid/loss` with factor `0.5` and patience `3`, and early stopping on that loss with patience `10`. Maximum training length is 100 epochs. Batch size is one slide in every split, with zero loader workers and FP32. The shared projection uses per-level LayerNorm, GELU and dropout `0.1`. The three recipes have 7,681, 671,745 and 1,067,266 trainable parameters respectively.

Each pyramid has one L4 anchor, four L3 tokens and sixteen L2 tokens. Tile extent is 224, anchor level is 4, level ratios are `[1, 2, 4]`, completeness is required, overlap policy is `non_overlapping`, and selection seed is 0. CM5 requires cancer probability greater than 0.5. EAN5 requires epithelial overlap greater than 0.25 and cancer probability greater than 0.5. The restored per-level cards pin the original split artifacts. Label mapping is `/mnt/projects/mammaprint/data_mapping.csv`.

Each fit selects its minimum-validation-loss checkpoint and tests once with `test_after_fit=true`. Prediction and target cutoffs are zero. `comparison-plan.json` contains the common resolved model, trainer and split inputs plus the explicit seed list for each Job. Each Job manifest contains the per-fit seed, output-directory and MLflow overrides in its worker arguments. Each seed has its own directory and MLflow run, with recipe-qualified names and a shared `comparison_id` tag.

Reuse the valid seed-0 mean runs `7813dba42d87405a885b5807f1ee23e9` and `f09a4ad6c7fe45a79992e8a4b8ed4299`, and attention retries `c84e92d5aeb64b4eb7444938d7f162cf` and `3eefc7f92dbf401ea95fed92a7f22f6c`. Compare with the existing L2 attention + linear MSE baseline at seeds 0, 1 and 2 per mask after its checkpoint audit and prediction export. The new matrix has fourteen fits. The matched L2 tissue-selection control from the earlier recommendation is a subsequent experiment.

## Resources

The estimator's three `multiscale-*` profiles use the aggregate audits in `scripts/ml/multiscale_batch_bag_sizes.csv`. Every row records the source run, audit path and SHA-256. Maximum retained train bags are 360 regions for CM5 and 319 for EAN5, each with 21 tokens. All three recipes estimate 1.25 GiB planning VRAM per fit, or 2.5 GiB for two active fits. These are analytical estimates. The batch runner records sampled memory and throughput for validation.

| Family, three Jobs each | GPU per Job | Active fits | CPU | RAM | Scratch | Ephemeral storage |
| --- | --- | --- | --- | --- | --- | --- |
| CM5 | 1 A40 | 2 | 8 cores | 32 GiB | 470 GiB | 478 GiB |
| EAN5 | 1 A40 | 2 | 8 cores | 32 GiB | 401 GiB | 409 GiB |

Host requests use the successful 4-core, 16-GiB allocation per active fit. Disk covers separate train/validation/test downloads per active process, 10% working space, and a shared 12-GiB environment allowance. Another 8 GiB covers files outside scratch. Each Job uses 1 GiB shared memory and four compute threads per fit. The projected-mean batch queues seed 2 as a slot opens. Six independent Jobs can allocate six A40s, 48 cores and 192 GiB RAM in total. Create individual rendered files to limit the number of active Jobs.

## Render and run

Use Python 3.12 with project dependencies and the compatible `rationai-kube-jobs` helper, version 0.4.0 at commit `7dc4aed3b1f8a76f8aacd8468c670fc582a75c02`. The global version 0.2.0 has an incompatible storage API. Install the helper into the launcher environment if needed.

```bash
uv pip install --python .venv/bin/python \
  'git+ssh://git@gitlab.ics.muni.cz/rationai/infrastructure/kube-jobs@7dc4aed3b1f8a76f8aacd8468c670fc582a75c02'
```

Commit and publish the rebased worker source before submission. Each pod checks out the full `--source-commit`. Render from its clean matching checkout.

```bash
comparison_source=$(git rev-parse HEAD)
.venv/bin/python -m scripts.ml.run_multiscale_comparison \
  --context kubas-cluster --namespace rationai-jobs-ns \
  --batch-id ms-compare-v1 --source-commit "$comparison_source" \
  --output-dir /tmp/mammaprint-ms-compare-v1 --max-parallel 2
```

This creates six Job JSON files and `comparison-plan.json`. Read the latter for exact resolved configurations. For an initial serial measurement, use `--max-parallel 1` with a distinct batch ID and output directory. Compare resource summaries with the two-fit version before increasing concurrency further.

After explicit approval of the six rendered Jobs, context, and namespace, repeat with `--submit`. On partial submission, list the existing Jobs and create only the remaining reviewed JSON files. An existing name causes `kubectl create` to stop. A repeat requires a distinct batch ID and submission approval.

Verify Kubernetes resources and inspect the batch-level `status.json` and `resources-summary.json` under `<output-root>/<job-name>/<job-name>/<pod-uid>/`. Each fit has a `seed-*/console.log` and its own MLflow tags. Verify the checkpoint epoch and `best_model_score` against the minimum loss in its run's history.
