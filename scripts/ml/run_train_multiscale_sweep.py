"""Submit the 24-run multiscale screen, refilling GPU queues every five minutes.

Edit the settings below to change the batch or resource limits. Preview with
``uv run python scripts/ml/run_train_multiscale_sweep.py --dry-run``.
Training code/configuration must be committed and pushed before submission;
pods check out the launcher's commit. Dry runs never contact cluster services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


# Experiment settings: 2 datasets x 4 recipes x 3 paired seeds = 24 jobs.
DATASETS = ("cm5", "ean5")
RECIPES = (
    "mean_regression",
    "projected_mean_regression",
    "slide_attention_regression",
    "mean_bce",
)
SEEDS = (0, 1, 2)
DATA_MAPPING: str | None = None  # None keeps /mnt/projects/mammaprint/data_mapping.csv.
GPUS = ("A40", "H100")
WAITING_JOB_TARGETS = {"A40": 2, "H100": 1}
CPU = 8
MEMORY = "64Gi"
POLL_INTERVAL_SECONDS = 5 * 60
GIT_REF = "feat/modular-multiscale-mil"
REPOSITORY_URL = "https://github.com/RationAI/mammaprint.git"
USERNAME = "kissmi"
IMAGE = "cerit.io/rationai/base:2.0.6"
POD_MLFLOW_TRACKING_URI = os.getenv(
    "POD_MLFLOW_TRACKING_URI", "http://mlflow-s3.rationai-mlflow"
)
MLFLOW_LOOKUP_URI = os.getenv(
    "MLFLOW_LOOKUP_URI", "https://mlflow-s3.rationai.cloud.trusted.e-infra.cz"
)
MLFLOW_EXPERIMENT = "MammaPrint"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SweepJob:
    dataset: str
    recipe: str
    seed: int

    @property
    def experiment(self) -> str:
        return f"ml/multiscale/screen/{self.dataset}_{self.recipe}"


def _source_revision() -> str:
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "ml",
            "configs",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=REPO_ROOT,
        text=True,
    )
    if dirty.strip():
        raise RuntimeError(
            "Commit and push training code/config changes before launching."
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _prefix(revision: str) -> str:
    mapping = DATA_MAPPING or "/mnt/projects/mammaprint/data_mapping.csv"
    mapping_id = hashlib.sha256(mapping.encode()).hexdigest()[:6]
    return f"mp-ms-{revision[:8]}-{mapping_id}-"


def _job_name(job: SweepJob, prefix: str) -> str:
    name = f"{prefix}{job.dataset}-{job.recipe.replace('_', '-')}-s{job.seed}"
    if len(name) > 60 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(f"Invalid job name or insufficient room for retries: {name}")
    return name


def _jobs() -> list[SweepJob]:
    if not GPUS or any(WAITING_JOB_TARGETS.get(gpu, 0) <= 0 for gpu in GPUS):
        raise ValueError("Each configured GPU needs a positive waiting-job target.")
    jobs = [
        SweepJob(dataset, recipe, seed)
        for seed in SEEDS
        for dataset in DATASETS
        for recipe in RECIPES
    ]
    if not jobs or len(jobs) != len(set(jobs)) or any(job.seed < 0 for job in jobs):
        raise ValueError(
            "The configured batch must contain unique jobs and non-negative seeds."
        )
    return jobs


def _validate_experiments(jobs: list[SweepJob]) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        for experiment in dict.fromkeys(job.experiment for job in jobs):
            config = compose(config_name="ml", overrides=[f"+experiment={experiment}"])
            OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


def _training_command(job: SweepJob, name: str) -> str:
    arguments = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "ml.train",
        f"+experiment={job.experiment}",
        f"seed={job.seed}",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "test_after_fit=false",
        f"metadata.run_name={name}",
        f"+logger.tags.multiscale_screen={job.recipe}",
        f"+logger.tags.data_variant={job.dataset}",
        f"+logger.tags.sweep_seed=seed_{job.seed}",
    ]
    if DATA_MAPPING is not None:
        arguments.append(f"dataset.data_mapping={json.dumps(DATA_MAPPING)}")
    return shlex.join(arguments)


def _pod_script(job: SweepJob, name: str, revision: str) -> list[str]:
    return [
        "set -eu",
        f"git clone --branch {shlex.quote(GIT_REF)} {shlex.quote(REPOSITORY_URL)} workdir",
        "cd workdir",
        f"git checkout --detach {shlex.quote(revision)}",
        f"export MLFLOW_TRACKING_URI={shlex.quote(POD_MLFLOW_TRACKING_URI)}",
        "uv sync --frozen",
        _training_command(job, name),
    ]


def _existing_mlflow_run_names(expected: set[str], prefix: str) -> set[str]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_LOOKUP_URI)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return set()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name LIKE '{prefix}%'",
        max_results=50_000,
    )
    # Existing failed runs also count; retrying them is an explicit separate action.
    return {run.info.run_name for run in runs if run.info.run_name in expected}


def _existing_kubernetes_job_names(expected: set[str]) -> set[str]:
    from kube_jobs.constants import CLUSTER, JOB_NAMESPACE
    from kubernetes import client
    from kubernetes.config import load_kube_config

    load_kube_config(context=CLUSTER)
    jobs = (
        client.BatchV1Api()
        .list_namespaced_job(
            namespace=JOB_NAMESPACE, label_selector=f"created_by={USERNAME}"
        )
        .items
    )
    names = {
        re.sub(r"-\d+$", "", job.metadata.name)
        for job in jobs
        if job.metadata is not None and job.metadata.name
    }
    return names & expected


def _require_submission_dependencies() -> None:
    try:
        import kube_jobs  # noqa: F401
        import kubernetes  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Submission requires rationai-kube-jobs and kubernetes; use the same "
            "Python environment as the other cluster run scripts."
        ) from error


def _waiting_counts() -> dict[str, int]:
    if TYPE_CHECKING or __package__:
        from scripts.ml.list_pending_gpu_jobs import pending_gpu_counts
    else:
        from list_pending_gpu_jobs import pending_gpu_counts
    return dict(pending_gpu_counts(USERNAME))


def _select_jobs(
    pending: list[SweepJob], waiting: dict[str, int], limit: int | None
) -> list[tuple[SweepJob, str]]:
    counts = {gpu: waiting.get(gpu, 0) for gpu in GPUS}
    selected = []
    for job in pending:
        eligible = [gpu for gpu in GPUS if counts[gpu] < WAITING_JOB_TARGETS[gpu]]
        if not eligible:
            break
        gpu = min(eligible, key=lambda candidate: counts[candidate])
        counts[gpu] += 1
        selected.append((job, gpu))
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _submit(job: SweepJob, gpu: str, prefix: str, revision: str) -> None:
    from kube_jobs import storage, submit_job

    name = _job_name(job, prefix)
    print(f"Submitting {name} to {gpu}.", flush=True)
    submit_job(
        job_name=name,
        username=USERNAME,
        image=IMAGE,
        cpu=CPU,
        memory=MEMORY,
        gpu=gpu,
        public=False,
        script=_pod_script(job, name, revision),
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )
    if name not in _existing_kubernetes_job_names({name}):
        raise RuntimeError(f"Kubernetes did not confirm submission of {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print all jobs and commands offline."
    )
    parser.add_argument(
        "--max-jobs", type=int, help="Submit at most this many new jobs, then exit."
    )
    parser.add_argument(
        "--poll-interval-seconds", type=int, default=POLL_INTERVAL_SECONDS
    )
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    if args.poll_interval_seconds <= 0:
        parser.error("--poll-interval-seconds must be positive")
    jobs = _jobs()
    _validate_experiments(jobs)
    revision = _source_revision()
    prefix = _prefix(revision)
    expected = {_job_name(job, prefix) for job in jobs}
    print(f"Multiscale screen: {len(jobs)} jobs; source commit {revision}")
    print(f"CPU: {CPU}; memory: {MEMORY}; waiting targets: {WAITING_JOB_TARGETS}")
    print(f"Data mapping: {DATA_MAPPING or 'experiment default'}")
    if args.dry_run:
        for job in jobs:
            name = _job_name(job, prefix)
            print(f"{name}: {_training_command(job, name)}")
        return

    _require_submission_dependencies()
    submitted: set[str] = set()
    try:
        while True:
            try:
                existing = (
                    _existing_kubernetes_job_names(expected)
                    | _existing_mlflow_run_names(expected, prefix)
                    | submitted
                )
                pending = [
                    job for job in jobs if _job_name(job, prefix) not in existing
                ]
                if not pending:
                    print("All screen jobs have been submitted.")
                    return
                remaining = (
                    None if args.max_jobs is None else args.max_jobs - len(submitted)
                )
                selected = _select_jobs(pending, _waiting_counts(), remaining)
                print(
                    f"{len(existing)} existing; {len(pending)} pending; submitting {len(selected)}."
                )
                for job, gpu in selected:
                    _submit(job, gpu, prefix, revision)
                    submitted.add(_job_name(job, prefix))
                if args.max_jobs is not None and len(submitted) >= args.max_jobs:
                    print(f"Reached submission limit ({args.max_jobs}).")
                    return
            except Exception as error:
                print(
                    f"Queue refill failed: {error}; refreshing after the interval.",
                    flush=True,
                )
            time.sleep(args.poll_interval_seconds)
    except KeyboardInterrupt:
        print("\nMultiscale queue refiller stopped.")


if __name__ == "__main__":
    main()
