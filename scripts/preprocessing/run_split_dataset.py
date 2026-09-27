"""Submit a Kubernetes job that materialises train/val/test MLflow artifacts.

Examples::

    uv run python scripts/preprocessing/run_split_dataset.py --kind tiled \
        --card configs/data/tiled/tissue_only/l5.yaml --name l5_tissue

    uv run python scripts/preprocessing/run_split_dataset.py --kind embedded \
        --uri mlflow-artifacts:/3/<run>/artifacts/embeddings \
        --name l5_tissue_embed

The submitted pod runs ``preprocessing.split_dataset``. The projects volume is
mounted so the worker can read ``/mnt/projects/mammaprint/data_mapping.csv``.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

from kube_jobs import storage, submit_job


USERNAME = "kissmi"
GIT_BRANCH = "feat/tiling-values"
IMAGE = "cerit.io/rationai/base:2.0.6"
CPU = 8
MEMORY = "64Gi"
SHM = "16Gi"
TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DATA_MAPPING = "/mnt/projects/mammaprint/data_mapping.csv"


def _slug(value: str) -> str:
    """Convert a dataset identifier to a Kubernetes-compatible name fragment."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _default_job_name(
    kind: str, card: str | None, uri: str | None, name: str | None
) -> str:
    if name:
        identifier = name
    elif card:
        path = Path(card)
        identifier = f"{path.parent.name}-{path.stem}"
    else:
        identifier = Path((uri or "dataset").rstrip("/")).name

    job_name = f"mammaprint-split-{kind}-{_slug(identifier)}"
    return job_name[:63].rstrip("-")


def _worker_command(args: argparse.Namespace) -> str:
    command = [
        "uv",
        "run",
        "-m",
        "preprocessing.split_dataset",
        "--kind",
        args.kind,
        "--data-mapping",
        args.data_mapping,
    ]
    if args.card:
        command.extend(["--card", args.card])
    else:
        command.extend(["--uri", args.uri])
    if args.name:
        command.extend(["--name", args.name])
    return shlex.join(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a Kubernetes job that splits one MLflow dataset artifact."
    )
    parser.add_argument("--kind", choices=["embedded", "tiled"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--card", help="Repository-relative level card containing an `all` URI."
    )
    source.add_argument("--uri", help="Whole MLflow artifact URI to split.")
    parser.add_argument("--name", help="Output base name (default: inferred by the worker).")
    parser.add_argument("--data-mapping", default=DATA_MAPPING)
    parser.add_argument("--job-name", help="Override the generated Kubernetes job name.")
    args = parser.parse_args()

    job_name = args.job_name or _default_job_name(
        args.kind, args.card, args.uri, args.name
    )
    print(f"Submitting {job_name} with {CPU} CPUs and {MEMORY} RAM.")
    submit_job(
        job_name=job_name,
        username=USERNAME,
        image=IMAGE,
        cpu=CPU,
        memory=MEMORY,
        shm=SHM,
        gpu=None,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {shlex.quote(GIT_BRANCH)}",
            f"export MLFLOW_TRACKING_URI={shlex.quote(TRACKING_URI)}",
            "uv sync --frozen",
            _worker_command(args),
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )
    print(f"Submitted {job_name}.")


if __name__ == "__main__":
    main()
