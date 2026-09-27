"""Submit the configured D-optimal mask-comparison training sweep.

Edit the settings block below, then run this file without command-line flags.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from kube_jobs import storage, submit_job


EXPERIMENTS = {
    "bce": "ml/train_mil_embeddings",
    "regression": "ml/train_mil_embeddings_regression",
    "joint": "ml/train_mil_embeddings_joint",
}
MASK_CONFIGS = {
    "tissue_only": ("tissue_only", "tis"),
    "cancer_mask_5": ("cancer_mask_5", "c05"),
    "cancer_mask_9": ("cancer_mask_9", "c09"),
    "epithel_and_cancer_5": ("epithel_and_cancer_5", "eac05"),
    "epithel_or_cancer_5": ("epithel_or_cancer_5", "eoc05"),
}
DESIGN_DIR = Path("configs/sweeps/d_optimal_mask_comparison")
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

# Sweep settings. Edit these values before running this file.
DESIGN_SIZE = 48
DATA_MAPPING: str | None = None  # e.g. "/mnt/projects/mammaprint/my_mapping.csv"
REPEATS = 20
SEED_START = 0
GPUS = ("H100", "A40")
SUBMISSION_INTERVAL_MINUTES = 30.0
MAX_JOBS: int | None = None
DRY_RUN = False


@dataclass(frozen=True)
class SweepJob:
    design_row: int
    level: int
    head: str
    aggregator: str
    loss: str
    mask_type: str
    data_config: str
    job_token: str


@dataclass(frozen=True)
class WorkItem:
    job: SweepJob
    repeat: int
    seed: int


def _validate_settings() -> str:
    if DESIGN_SIZE not in (36, 48):
        raise ValueError(f"DESIGN_SIZE must be 36 or 48, got {DESIGN_SIZE}.")
    if DATA_MAPPING is None:
        raise ValueError(
            "Set DATA_MAPPING in the settings block to a path visible inside the pod."
        )
    if not GPUS:
        raise ValueError("GPUS must contain at least one GPU type.")
    unexpected_gpus = sorted(set(GPUS) - {"H100", "A40"})
    if unexpected_gpus:
        raise ValueError(f"Unsupported GPUs: {unexpected_gpus}")
    if SUBMISSION_INTERVAL_MINUTES < 0:
        raise ValueError("SUBMISSION_INTERVAL_MINUTES cannot be negative.")
    if MAX_JOBS is not None and MAX_JOBS <= 0:
        raise ValueError("MAX_JOBS must be positive or None.")
    return DATA_MAPPING


def _load_jobs(design_size: int) -> list[SweepJob]:
    path = DESIGN_DIR / f"d_optimal_{design_size}_combinations.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing design table {path}; run `uv run python main.py`."
        )

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != design_size:
        raise RuntimeError(f"Expected {design_size} rows in {path}, found {len(rows)}.")

    jobs = []
    for design_row, row in enumerate(rows, start=1):
        loss = row["loss"]
        if loss not in EXPERIMENTS:
            raise ValueError(f"Unknown loss {loss!r} in {path}.")
        mask_type = row["mask_type"]
        try:
            data_config, job_token = MASK_CONFIGS[mask_type]
        except KeyError as error:
            raise ValueError(f"Unknown mask type {mask_type!r} in {path}.") from error
        jobs.append(
            SweepJob(
                design_row=design_row,
                level=int(row["level"]),
                head=row["head"],
                aggregator=row["aggregator"],
                loss=loss,
                mask_type=mask_type,
                data_config=data_config,
                job_token=job_token,
            )
        )
    return jobs


def _mapping_id(data_mapping: str) -> str:
    return hashlib.sha256(data_mapping.encode()).hexdigest()[:6]


def _work_items(jobs: list[SweepJob], repeats: int, seed_start: int) -> list[WorkItem]:
    if repeats <= 0:
        raise ValueError(f"REPEATS must be positive, got {repeats}.")
    return [
        WorkItem(job=job, repeat=repeat, seed=seed_start + repeat)
        for repeat in range(repeats)
        for job in jobs
    ]


def _job_name(item: WorkItem, design_size: int, mapping_id: str) -> str:
    job = item.job
    name = (
        f"mp-doe{design_size}-{job.job_token}-l{job.level}-"
        f"{job.head}-{job.aggregator}-{job.loss}-s{item.seed}-m{mapping_id}"
    )
    # Leave room for the retry helper's '-2', '-3', ... suffixes.
    if len(name) > 60:
        raise ValueError(f"Kubernetes job name leaves no retry suffix room: {name}")
    return name


def _gpu_for(item: WorkItem, gpus: tuple[str, ...]) -> str:
    # Rotate each configuration across devices between repetitions instead of
    # permanently coupling a model configuration to one GPU type.
    gpu_index = (item.job.design_row - 1 + item.repeat) % len(gpus)
    return gpus[gpu_index]


def _validate_dataset_cards(jobs: list[SweepJob]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checked = set()
    for job in jobs:
        data_override = f"{job.data_config}/l{job.level}"
        if data_override in checked:
            continue
        checked.add(data_override)
        card = repo_root / "configs" / "data" / "embedded" / f"{data_override}.yaml"
        if not card.is_file():
            raise FileNotFoundError(f"Missing dataset card: {card}")
        text = card.read_text()
        missing = [
            split
            for split in ("all", "train", "val", "test")
            if f"  {split}:" not in text
        ]
        if missing or "<run_id>" in text:
            problem = ", ".join(missing) if missing else "placeholder URI"
            raise RuntimeError(f"Dataset card {card} is incomplete: {problem}.")


def _existing_mlflow_run_names(expected_names: set[str], prefix: str) -> set[str]:
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
    return {run.info.run_name for run in runs if run.info.run_name in expected_names}


def _existing_kubernetes_job_names(expected_names: set[str]) -> set[str]:
    from kube_jobs.constants import CLUSTER, JOB_NAMESPACE
    from kubernetes import client
    from kubernetes.config import load_kube_config

    load_kube_config(context=CLUSTER)
    batch = client.BatchV1Api()
    jobs = batch.list_namespaced_job(
        namespace=JOB_NAMESPACE,
        label_selector=f"created_by={USERNAME}",
    ).items
    existing = set()
    for job in jobs:
        if job.metadata is None:
            continue
        name = job.metadata.name
        match = re.match(r"^(.*)-(\d+)$", name)
        base_name = match.group(1) if match else name
        if base_name in expected_names:
            existing.add(base_name)
    return existing


def _submit(
    item: WorkItem,
    design_size: int,
    gpu: str,
    mapping_id: str,
    data_mapping: str,
) -> None:
    job = item.job
    name = _job_name(item, design_size, mapping_id)
    experiment = EXPERIMENTS[job.loss]
    data_override = f"{job.data_config}/l{job.level}"
    data_mapping_override = (
        f" {shlex.quote(f'dataset.data_mapping={json.dumps(data_mapping)}')}"
    )
    submit_job(
        job_name=name,
        username=USERNAME,
        image=IMAGE,
        cpu=16,
        memory="48Gi",
        gpu=gpu,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {GIT_REF}",
            f"export MLFLOW_TRACKING_URI={POD_MLFLOW_TRACKING_URI}",
            "uv sync --frozen",
            f"""
            uv run -m ml.train +experiment={experiment} \
            data/embedded={data_override} \
            ml/aggregator={job.aggregator} \
            ml/head={job.head} \
            seed={item.seed}{data_mapping_override} \
            test_after_fit=false \
            metadata.run_name={name} \
            +logger.tags.data_variant={job.mask_type} \
            +logger.tags.d_optimal_design=d_optimal_{design_size} \
            +logger.tags.d_optimal_design_row=row_{job.design_row} \
            +logger.tags.repeat=repeat_{item.repeat + 1} \
            +logger.tags.data_mapping_id=map_{mapping_id}
            """,
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


def main() -> None:
    data_mapping = _validate_settings()
    jobs = _load_jobs(DESIGN_SIZE)
    _validate_dataset_cards(jobs)
    items = _work_items(jobs, REPEATS, SEED_START)
    mapping_id = _mapping_id(data_mapping)
    expected_names = {_job_name(item, DESIGN_SIZE, mapping_id) for item in items}
    prefix = f"mp-doe{DESIGN_SIZE}-"

    if DRY_RUN:
        print(
            f"D-optimal design: {len(jobs)} configurations x {REPEATS} seeds "
            f"= {len(items)} jobs"
        )
        print(f"GPUs: {', '.join(GPUS)}")
        print(f"Interval: {SUBMISSION_INTERVAL_MINUTES:g} minutes")
        print(f"Data mapping: {data_mapping}")
        print(f"Data mapping ID: {mapping_id}")
        for item in items:
            gpu = _gpu_for(item, GPUS)
            print(f"{_job_name(item, DESIGN_SIZE, mapping_id)} -> {gpu}")
        return

    kubernetes_names = _existing_kubernetes_job_names(expected_names)
    mlflow_names = _existing_mlflow_run_names(expected_names, prefix)
    existing_names = kubernetes_names | mlflow_names
    pending = [
        item
        for item in items
        if _job_name(item, DESIGN_SIZE, mapping_id) not in existing_names
    ]
    if MAX_JOBS is not None:
        pending = pending[:MAX_JOBS]

    print(
        f"Design has {len(items)} jobs; {len(existing_names)} already exist and "
        f"{len(pending)} will be submitted now."
    )
    interval_seconds = SUBMISSION_INTERVAL_MINUTES * 60
    for pending_index, item in enumerate(pending):
        if pending_index > 0:
            print(
                f"Waiting {SUBMISSION_INTERVAL_MINUTES:g} minutes before the next job."
            )
            time.sleep(interval_seconds)
        gpu = _gpu_for(item, GPUS)
        name = _job_name(item, DESIGN_SIZE, mapping_id)
        print(f"Submitting {name} to {gpu} ({pending_index + 1}/{len(pending)}).")
        _submit(item, DESIGN_SIZE, gpu, mapping_id, data_mapping)


if __name__ == "__main__":
    main()
