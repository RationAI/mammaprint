import itertools
import os
import time

from kube_jobs import storage, submit_job


# Each objective has a self-consistent experiment (head out_dim, label_mode,
# output activation, loss and metrics are set together). The generic mean-pool
# experiments are used as the base because their metadata remains valid when the
# aggregator is overridden; the transformer-specific experiments interpolate
# transformer-only fields and cannot be composed with mean/max/attention.
EXPERIMENTS = {
    "bce": "ml/train_mil_embeddings",
    "regression": "ml/train_mil_embeddings_regression",
    "joint": "ml/train_mil_embeddings_joint",
}
# The cancer_mask_5/l4 split is currently being materialised. Do not launch the
# sweep until its train/validation/test placeholders have been replaced.
DATASETS = {
    "cm5-l4": "cancer_mask_5/l4",
    "cm5-l5": "cancer_mask_5/l5",
    "cm9-l4": "cancer_mask_9/l4",
    "cm9-l5": "cancer_mask_9/l5",
}
AGGREGATORS = ("mean", "max", "attention", "transformer")
HEADS = ("mlp", "linear")
GPUS = ["A40"]
CODE_VERSION = 2
SUBMISSION_INTERVAL_SECONDS = 45 * 60
USERNAME = "kissmi"
POD_MLFLOW_TRACKING_URI = os.getenv(
    "POD_MLFLOW_TRACKING_URI", "http://mlflow-s3.rationai-mlflow"
)
MLFLOW_LOOKUP_URI = os.getenv("MLFLOW_LOOKUP_URI")
MLFLOW_EXPERIMENT = "MammaPrint"
JOB_NAME_PREFIX = "mammaprint-train-cmv2-"


def _job_name(
    dataset_key: str,
    head: str,
    aggregator: str,
    objective: str,
) -> str:
    return f"{JOB_NAME_PREFIX}{dataset_key}-{head}-{aggregator}-{objective}"


def _sweep_jobs() -> list[tuple[int, str, str, str, str, str, str]]:
    combinations = itertools.product(
        DATASETS.items(),
        EXPERIMENTS.items(),
        AGGREGATORS,
        HEADS,
    )
    return [
        (
            job_index,
            dataset_key,
            dataset_config,
            objective,
            experiment,
            aggregator,
            head,
        )
        for job_index, (
            (dataset_key, dataset_config),
            (objective, experiment),
            aggregator,
            head,
        ) in enumerate(combinations)
    ]


def _existing_mlflow_run_names(expected_names: set[str]) -> set[str]:
    """Fetch sweep runs that already exist, including failed/running runs."""
    if MLFLOW_LOOKUP_URI is None:
        print(
            "Skipping MLflow lookup: set MLFLOW_LOOKUP_URI to a tracking endpoint "
            "reachable from this machine to enable it."
        )
        return set()

    from mlflow import MlflowClient

    mlflow_client = MlflowClient(tracking_uri=MLFLOW_LOOKUP_URI)
    experiment = mlflow_client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        print(f"MLflow experiment {MLFLOW_EXPERIMENT!r} does not exist yet.")
        return set()

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name LIKE '{JOB_NAME_PREFIX}%'",
        max_results=50_000,
    )
    matching_runs = [run for run in runs if run.info.run_name in expected_names]
    existing = {run.info.run_name for run in matching_runs}
    print(f"Fetched {len(existing)} existing sweep run(s) from MLflow.")
    return existing


def _existing_kubernetes_job_names(expected_names: set[str]) -> set[str]:
    """Fetch jobs that were submitted but may not have reached MLflow yet."""
    from kube_jobs.constants import CLUSTER, JOB_NAMESPACE
    from kubernetes import client
    from kubernetes.config import load_kube_config

    load_kube_config(context=CLUSTER)
    batch = client.BatchV1Api()
    jobs = batch.list_namespaced_job(
        namespace=JOB_NAMESPACE,
        label_selector=f"created_by={USERNAME}",
    ).items
    matching_jobs = [
        job
        for job in jobs
        if job.metadata is not None and job.metadata.name in expected_names
    ]
    existing = {job.metadata.name for job in matching_jobs}
    print(f"Fetched {len(existing)} existing sweep job(s) from Kubernetes.")
    return existing


def main() -> None:
    jobs = _sweep_jobs()
    expected_names = {
        _job_name(dataset_key, head, aggregator, objective)
        for _, dataset_key, _, objective, _, aggregator, head in jobs
    }
    kubernetes_names = _existing_kubernetes_job_names(expected_names)
    mlflow_names = _existing_mlflow_run_names(expected_names)
    existing_names = mlflow_names | kubernetes_names
    pending_jobs = [
        job
        for job in jobs
        if _job_name(job[1], job[6], job[5], job[3]) not in existing_names
    ]

    print(
        f"Sweep contains {len(jobs)} jobs: {len(existing_names)} already submitted, "
        f"{len(pending_jobs)} pending."
    )

    for pending_index, (
        job_index,
        dataset_key,
        dataset_config,
        objective,
        experiment,
        aggregator,
        head,
    ) in enumerate(pending_jobs):
        if pending_index > 0:
            print("Waiting 45 minutes before submitting the next job.")
            time.sleep(SUBMISSION_INTERVAL_SECONDS)

        name = _job_name(dataset_key, head, aggregator, objective)
        gpu = GPUS[job_index % len(GPUS)]
        print(f"Submitting {name} to {gpu} ({pending_index + 1}/{len(pending_jobs)}).")
        submit_job(
            job_name=name,
            username=USERNAME,
            image="cerit.io/rationai/base:2.0.6",
            cpu=16,
            memory="48Gi",
            gpu=gpu,
            public=False,
            script=[
                "git clone https://github.com/rationAI/mammaprint workdir",
                "cd workdir",
                "git checkout feat/tiling-values",
                f"export MLFLOW_TRACKING_URI={POD_MLFLOW_TRACKING_URI}",
                # "export HF_TOKEN=",
                "uv sync --frozen",
                f"""
                uv run -m ml.train +experiment={experiment} \
                data/embedded={dataset_config} \
                ml/aggregator={aggregator} \
                ml/head={head} \
                metadata.run_name={name} \
                +logger.tags.code_version={CODE_VERSION} \
                +logger.tags.data_variant={dataset_key} \
                """,
            ],
            storage=[storage.secure.DATA, storage.secure.PROJECTS],
        )


if __name__ == "__main__":
    main()
