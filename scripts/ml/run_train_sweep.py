import itertools
import os
import time

from kube_jobs import storage, submit_job


# Each objective points to the matching training experiment. Edit these values,
# together with the dataset/model lists below, to define the Cartesian sweep.
EXPERIMENTS = {
    "bce": "ml/train_mil_embeddings",
    "regression": "ml/train_mil_embeddings_regression",
    "joint": "ml/train_mil_embeddings_joint",
}
DATASETS = {
    "tissue": "tissue_only",
    "cm5": "cancer_mask_5",
    "cm9": "cancer_mask_9",
    "e025": "epithel",
    "eor5": "epithel_or_cancer_5",
    "ean5": "epithel_and_cancer_5",
}
LEVELS = (
    # 2,
          3, 4, 5)
AGGREGATORS = ("mean", "max", "attention", "transformer")
HEADS = ("mlp", "linear")
SEEDS = tuple(range(20))
GPUS = [
    "mig-2g.20gb",
    # "H100",
    # "A40",
]
GIT_REF = "feat/tiling-values"
SUBMISSION_INTERVAL_SECONDS = 10
USERNAME = "kissmi"
IMAGE = "cerit.io/rationai/base:2.0.6"
POD_MLFLOW_TRACKING_URI = os.getenv(
    "POD_MLFLOW_TRACKING_URI", "http://mlflow-s3.rationai-mlflow"
)
MLFLOW_LOOKUP_URI = os.getenv(
    "MLFLOW_LOOKUP_URI",
    "https://mlflow-s3.rationai.cloud.trusted.e-infra.cz",
)
MLFLOW_EXPERIMENT = "MammaPrint"
JOB_NAME_PREFIX = "mammaprint-sweep-"


def _job_name(
    dataset_key: str,
    level: int,
    head: str,
    aggregator: str,
    objective: str,
    seed: int,
) -> str:
    return f"{JOB_NAME_PREFIX}{dataset_key}-l{level}-{head}-{aggregator}-{objective}-s{seed}"


def _sweep_jobs() -> list[tuple[int, str, str, int, str, str, str, str, int]]:
    combinations = itertools.product(
        SEEDS,
        DATASETS.items(),
        LEVELS,
        EXPERIMENTS.items(),
        AGGREGATORS,
        HEADS,
    )
    return [
        (
            job_index,
            dataset_key,
            dataset_config,
            level,
            objective,
            experiment,
            aggregator,
            head,
            seed,
        )
        for job_index, (
            seed,
            (dataset_key, dataset_config),
            level,
            (objective, experiment),
            aggregator,
            head,
        ) in enumerate(combinations)
    ]


def _gpu_for_job(job_index: int) -> str:
    configurations_per_seed = (
        len(DATASETS)
        * len(LEVELS)
        * len(EXPERIMENTS)
        * len(AGGREGATORS)
        * len(HEADS)
    )
    seed_index, configuration_index = divmod(job_index, configurations_per_seed)
    return GPUS[(configuration_index + seed_index) % len(GPUS)]


def _existing_mlflow_run_names(expected_names: set[str]) -> set[str]:
    """Fetch sweep runs that already exist, including failed/running runs."""
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
        _job_name(dataset_key, level, head, aggregator, objective, seed)
        for _, dataset_key, _, level, objective, _, aggregator, head, seed in jobs
    }
    kubernetes_names = _existing_kubernetes_job_names(expected_names)
    mlflow_names = _existing_mlflow_run_names(expected_names)
    existing_names = mlflow_names | kubernetes_names
    pending_jobs = [
        job
        for job in jobs
        if _job_name(job[1], job[3], job[7], job[6], job[4], job[8])
        not in existing_names
    ]

    print(
        f"Sweep contains {len(jobs)} jobs: {len(existing_names)} already submitted, "
        f"{len(pending_jobs)} pending."
    )

    for pending_index, (
        job_index,
        dataset_key,
        dataset_config,
        level,
        objective,
        experiment,
        aggregator,
        head,
        seed,
    ) in enumerate(pending_jobs):
        if pending_index > 0:
            print(
                f"Waiting {SUBMISSION_INTERVAL_SECONDS / 60} minutes before submitting the next job."
            )
            time.sleep(SUBMISSION_INTERVAL_SECONDS)

        name = _job_name(dataset_key, level, head, aggregator, objective, seed)
        gpu = _gpu_for_job(job_index)
        print(f"Submitting {name} to {gpu} ({pending_index + 1}/{len(pending_jobs)}).")
        submit_job(
            job_name=name,
            username=USERNAME,
            image=IMAGE,
            cpu=10,
            memory="36Gi",
            gpu=gpu,
            public=False,
            script=[
                "git clone https://github.com/rationAI/mammaprint workdir",
                "cd workdir",
                f"git checkout {GIT_REF}",
                f"export MLFLOW_TRACKING_URI={POD_MLFLOW_TRACKING_URI}",
                # "export HF_TOKEN=",
                "uv sync --frozen",
                f"""
                uv run -m ml.train +experiment={experiment} \
                data/embedded={dataset_config}/l{level} \
                ml/aggregator={aggregator} \
                ml/head={head} \
                seed={seed} \
                metadata.run_name={name} \
                +logger.tags.data_variant={dataset_config} \
                +logger.tags.level=level_{level} \
                +logger.tags.sweep_seed=seed_{seed} \
                """,
            ],
            storage=[storage.secure.DATA, storage.secure.PROJECTS],
        )


if __name__ == "__main__":
    main()
