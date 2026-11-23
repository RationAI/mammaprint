---
name: multiscale-checkpoint-evaluation
description: Use when replaying the MammaPrint CM5 or EAN5 seed-0 slide-attention fits on A40 and testing their newly saved best checkpoints.
---

# Multiscale checkpoint evaluation

Use this workflow for the two historical L4/L3/L2 regression runs with epoch-0 checkpoints in MLflow. The worker starts a fresh fit, selects by minimum `valid/loss`, and tests once. Its MammaPrint MLflow run carries the original ID in `recovery_source_run_id`.

The original training revision is `a8d498a59efa8bdcfd38310195cd1d4985c7bbdc`. The published revision `aab79d6167031e92f69f45173b4394260c1aaa4f` adds the corrected checkpoint logger to that source. Confirm the full SHA is reachable from the private repository before rendering. The historical F1 at minimum validation loss was 77.25% for CM5 and 76.97% for EAN5. Compare the replay with those values after it finishes.

## Render and verify

The renderer requires Python 3 and only writes the JSON Job to stdout. Confirm access to the intended Kubernetes context and namespace. The approved namespace uses the `projects-rw` claim for the read-only label mapping mount and the `gitlab-deploy-token` Secret for locked GitLab dependencies. GitHub uses the base image's existing access mechanism. MLflow is reached through the namespace's internal service. Keep credential values out of commands and logs.

```bash
replay_context=kubas-cluster
replay_namespace=rationai-jobs-ns
replay_source=aab79d6167031e92f69f45173b4394260c1aaa4f
replay_manifests=$(mktemp -d)
for dataset in cm5 ean5; do
  python3 -m scripts.ml.render_multiscale_retry_job \
    --dataset "$dataset" --source-commit "$replay_source" \
    --context "$replay_context" --namespace "$replay_namespace" \
    --username kissmi > "$replay_manifests/$dataset.json"
done
kubectl --context "$replay_context" -n "$replay_namespace" create \
  --dry-run=server -f "$replay_manifests/cm5.json" -f "$replay_manifests/ean5.json"
```

Review the two names, source annotations, exact Hydra commands, and resources. Each job defaults to one A40, four CPU cores, and 16 GiB RAM. The original loader uses no subprocess workers. CM5 downloads about 208 GiB of embeddings across train, validation, and test, and EAN5 about 176 GiB. Keep the per-job scratch and ephemeral-storage requests at 230 and 200 GiB unless new measurements support a change.

## Submit and verify

Show the rendered jobs, context, namespace, and scope and obtain explicit submission approval. Check the names for existing Jobs. For the approved files, use `kubectl create` with the same context and namespace. Read back their specs and pod status. Check that the source and recovery tags match and that the A40 selector and one-GPU limit survived admission.

Use `kubectl --context "$replay_context" -n "$replay_namespace" logs -f "job/$job_name"` to follow a running pod, or read its status and events if it is pending. A disk-backed `emptyDir` contains the Git checkout, dependency cache, and embedding downloads. `Insufficient ephemeral-storage` is a node placement constraint. Compare the event with the measured embedding size before changing scratch. On failure, check pod termination and the linked MLflow run before retrying. Any replacement submission requires approval. Job deletion discards pod-local files, so retain the MLflow run ID, best checkpoint, and audit evidence before cleanup.

Completion evidence is a finished linked MLflow run with `test/mae`, `test/mse`, `test/thresholded_f1`, and a checkpoint whose embedded epoch and score match the replay's best `valid/loss` entry. If the checkpoint still contains the first epoch, check that the run used `ml.loggers.MLFlowLogger` and inspect the upload log.
