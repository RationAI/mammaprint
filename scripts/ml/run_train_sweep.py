"""Refill GPU waiting queues every five minutes until the sweep is submitted."""

import argparse
import itertools
import os
import time
from collections import deque
from collections.abc import Mapping
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Any

from kube_jobs import storage, submit_job


if TYPE_CHECKING or __package__:
    from scripts.ml.list_pending_gpu_jobs import pending_gpu_counts
else:
    from list_pending_gpu_jobs import pending_gpu_counts


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
LEVELS = (2, 3, 4, 5)
AGGREGATORS = ("mean", "max", "attention", "transformer")
HEADS = ("mlp", "linear")
SEEDS = tuple(range(20))
GPUS = (
    "A40",
    "H100",
    "mig-2g.20gb",
    "mig-1g.10gb",
)
POLL_INTERVAL_SECONDS = 5 * 60
WAITING_JOB_TARGETS = {gpu: 1 if gpu == "H100" else 2 for gpu in GPUS}
BASIC_RESOURCES = {"cpu": 8, "memory": "16Gi"}

# GPU rules limit which entries from GPUS a matching job may use. Jobs without
# a matching rule may use the complete pool. All assignments share one set of
# projected waiting counts, including assignments restricted to one GPU type.
GPU_RULES: list[dict[str, Any]] = [
    {
        "match": {
            "level": (2, 3),
            "aggregator": ("transformer", "spatial_transformer"),
        },
        "only": ("H100",),
    },
    {
        "match": {
            "level": 4,
            "aggregator": ("transformer", "spatial_transformer"),
        },
        "exclude_gpus": ("mig-1g.10gb",),
    },
]

# Override host resources independently of GPU placement. Rules are applied in
# order, so later matches override only the fields they specify.
RESOURCE_RULES: list[dict[str, Any]] = [
    {
        "match": {
            "level": (2, 3),
            "aggregator": ("mean", "max"),
        },
        "resources": {"memory": "20Gi"},
    },
    {
        "match": {
            "level": (2, 3),
            "aggregator": "attention",
        },
        "resources": {"memory": "40Gi"},
    },
    {
        "match": {
            "level": (2, 3),
            "aggregator": ("transformer", "spatial_transformer"),
        },
        "resources": {"memory": "56Gi"},
    },
]
GIT_REF = "feat/tiling-values"
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

SweepJob = tuple[int, str, str, int, str, str, str, str, int]
ResolvedJob = tuple[SweepJob, dict[str, Any]]


def _job_name(
    dataset_key: str,
    level: int,
    head: str,
    aggregator: str,
    objective: str,
    seed: int,
) -> str:
    return f"{JOB_NAME_PREFIX}{dataset_key}-l{level}-{head}-{aggregator}-{objective}-s{seed}"


def _sweep_jobs() -> list[SweepJob]:
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


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        return actual in expected
    return actual == expected


def _job_matches(job: dict[str, object], conditions: dict[str, object]) -> bool:
    return all(_matches(job.get(key), value) for key, value in conditions.items())


def _eligible_gpus(job: dict[str, object]) -> tuple[str, ...]:
    eligible = list(GPUS)
    for rule in GPU_RULES:
        if not _job_matches(job, rule.get("match", {})):
            continue
        if "only" in rule:
            allowed = set(rule["only"])
            eligible = [gpu for gpu in eligible if gpu in allowed]
        excluded = set(rule.get("exclude_gpus", ()))
        eligible = [gpu for gpu in eligible if gpu not in excluded]

    if not eligible:
        raise ValueError(f"No configured GPU can run job {job!r}.")
    return tuple(eligible)


def _assign_gpu(
    job: dict[str, object],
    projected_waiting_counts: dict[str, int],
    waiting_targets: Mapping[str, int] | None = None,
) -> str | None:
    """Choose an eligible GPU with room in its queue; use GPUS order for ties."""
    eligible = _eligible_gpus(job)
    if waiting_targets is not None:
        eligible = tuple(
            gpu
            for gpu in eligible
            if projected_waiting_counts[gpu] < waiting_targets[gpu]
        )
    if not eligible:
        return None
    gpu = min(eligible, key=lambda candidate: projected_waiting_counts[candidate])
    projected_waiting_counts[gpu] += 1
    return gpu


def _resources_for_job(
    projected_waiting_counts: dict[str, int],
    dataset_key: str,
    dataset_config: str,
    level: int,
    objective: str,
    experiment: str,
    aggregator: str,
    head: str,
    seed: int,
    *,
    waiting_targets: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    job = {
        "dataset_key": dataset_key,
        "dataset_config": dataset_config,
        "level": level,
        "objective": objective,
        "experiment": experiment,
        "aggregator": aggregator,
        "head": head,
        "seed": seed,
    }
    resources: dict[str, Any] = dict(BASIC_RESOURCES)

    for rule in RESOURCE_RULES:
        matches = _job_matches(job, rule.get("match", {}))
        excluded = "exclude" in rule and _job_matches(job, rule["exclude"])
        if matches and not excluded:
            resources.update(rule["resources"])

    gpu = _assign_gpu(job, projected_waiting_counts, waiting_targets)
    if gpu is None:
        return None
    resources["gpu"] = gpu
    return resources


def _balanced_job_order(
    resolved_jobs: list[ResolvedJob], waiting_counts: Mapping[str, int]
) -> list[ResolvedJob]:
    """Interleave GPU queues by their projected waiting-job counts."""
    queues: dict[str, deque[ResolvedJob]] = {}
    first_job_index: dict[str, int] = {}
    for list_index, resolved_job in enumerate(resolved_jobs):
        gpu = resolved_job[1]["gpu"]
        if not isinstance(gpu, str):
            raise TypeError(f"Resolved GPU must be a string, got {gpu!r}.")
        queues.setdefault(gpu, deque()).append(resolved_job)
        first_job_index.setdefault(gpu, list_index)

    gpu_heap = [
        (waiting_counts.get(gpu, 0), first_job_index[gpu], gpu) for gpu in queues
    ]
    heapify(gpu_heap)

    ordered_jobs = []
    while gpu_heap:
        projected_count, tie_breaker, gpu = heappop(gpu_heap)
        ordered_jobs.append(queues[gpu].popleft())
        if queues[gpu]:
            heappush(gpu_heap, (projected_count + 1, tie_breaker, gpu))
    return ordered_jobs


def _jobs_to_submit(
    pending_jobs: list[SweepJob],
    waiting_counts: Mapping[str, int],
    max_jobs: int | None,
) -> tuple[list[ResolvedJob], int]:
    """Fill only vacant waiting slots, assigning jobs afresh on every poll."""
    projected_counts = {gpu: waiting_counts.get(gpu, 0) for gpu in GPUS}
    resolved_jobs: list[ResolvedJob] = []
    deferred_count = 0
    for job in pending_jobs:
        try:
            resources = _resources_for_job(
                projected_counts,
                *job[1:],
                waiting_targets=WAITING_JOB_TARGETS,
            )
        except ValueError as error:
            if not str(error).startswith("No configured GPU can run job "):
                raise
            deferred_count += 1
            continue
        if resources is not None:
            resolved_jobs.append((job, resources))
            if max_jobs is not None and len(resolved_jobs) >= max_jobs:
                break
    return _balanced_job_order(resolved_jobs, waiting_counts), deferred_count


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


def _submit_sweep_job(job: SweepJob, resources: dict[str, Any]) -> None:
    (
        _,
        dataset_key,
        dataset_config,
        level,
        objective,
        experiment,
        aggregator,
        head,
        seed,
    ) = job
    name = _job_name(dataset_key, level, head, aggregator, objective, seed)
    print(f"Submitting {name} with {resources}.")
    submit_job(
        job_name=name,
        username=USERNAME,
        image=IMAGE,
        cpu=resources["cpu"],
        memory=resources["memory"],
        gpu=resources["gpu"],
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
    # kube_jobs catches creation errors internally, so a normal return alone
    # does not mean the job was created.
    if name not in _existing_kubernetes_job_names({name}):
        raise RuntimeError(f"Kubernetes did not confirm submission of {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Submit at most this many jobs in total, then exit.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help="Seconds between queue refills (default: 300).",
    )
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be a positive integer")
    if args.poll_interval_seconds <= 0:
        parser.error("--poll-interval-seconds must be a positive integer")

    jobs = _sweep_jobs()
    expected_names = {
        _job_name(dataset_key, level, head, aggregator, objective, seed)
        for _, dataset_key, _, level, objective, _, aggregator, head, seed in jobs
    }
    submitted_names: set[str] = set()
    print(f"GPU waiting targets: {WAITING_JOB_TARGETS}")
    try:
        while True:
            try:
                kubernetes_names = _existing_kubernetes_job_names(expected_names)
                mlflow_names = _existing_mlflow_run_names(expected_names)
                existing_names = mlflow_names | kubernetes_names | submitted_names
                pending_jobs = [
                    job
                    for job in jobs
                    if _job_name(job[1], job[3], job[7], job[6], job[4], job[8])
                    not in existing_names
                ]
                print(
                    f"Sweep contains {len(jobs)} jobs: "
                    f"{len(existing_names)} already submitted, "
                    f"{len(pending_jobs)} left to submit."
                )
                if not pending_jobs:
                    print("All sweep jobs have been submitted.")
                    return

                waiting_counts = pending_gpu_counts(USERNAME)
                print(
                    "GPU waiting counts: "
                    f"{ {gpu: waiting_counts.get(gpu, 0) for gpu in GPUS} }"
                )
                remaining_limit = (
                    None
                    if args.max_jobs is None
                    else args.max_jobs - len(submitted_names)
                )
                jobs_to_submit, deferred_count = _jobs_to_submit(
                    pending_jobs, waiting_counts, remaining_limit
                )
                print(f"Submitting {len(jobs_to_submit)} job(s) to refill the queues.")
                if deferred_count:
                    print(
                        f"Deferred {deferred_count} job(s) with no eligible "
                        "configured GPU."
                    )
                    if deferred_count == len(pending_jobs):
                        print("No remaining jobs can use the configured GPU pool.")
                        return

                for job, resources in jobs_to_submit:
                    _submit_sweep_job(job, resources)
                    submitted_names.add(
                        _job_name(job[1], job[3], job[7], job[6], job[4], job[8])
                    )
                if args.max_jobs is not None and len(submitted_names) >= args.max_jobs:
                    print(f"Reached the submission limit ({args.max_jobs}).")
                    return
            except Exception as error:
                # Refresh cluster state before retrying an uncertain submission.
                print(f"Queue refill failed: {error}. Will retry after the interval.")

            print(f"Sleeping for {args.poll_interval_seconds} seconds.", flush=True)
            time.sleep(args.poll_interval_seconds)
    except KeyboardInterrupt:
        print("\nSweep queue refiller stopped.")


if __name__ == "__main__":
    main()
