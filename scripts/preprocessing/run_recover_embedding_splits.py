"""Submit a resumable job that completes split uploads from an MLflow run's all artifact."""

from __future__ import annotations

import argparse
import re
import shlex

from kube_jobs import storage, submit_job


USERNAME = "kissmi"
GIT_BRANCH = "feat/tiling-values"
TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DATA_MAPPING = "/mnt/projects/mammaprint/data_mapping.csv"


def _job_name(run_id: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", run_id.lower()).strip("-")
    return f"mammaprint-recover-embedding-splits-{token[:12]}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-artifact-path", default="embeddings")
    parser.add_argument("--data-mapping", default=DATA_MAPPING)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--mark-finished", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = [
        "uv",
        "run",
        "-m",
        "preprocessing.recover_embedding_splits",
        "--run-id",
        args.run_id,
        "--source-artifact-path",
        args.source_artifact_path,
        "--data-mapping",
        args.data_mapping,
        "--max-retries",
        str(args.max_retries),
        "--apply",
    ]
    if args.mark_finished:
        command.append("--mark-finished")
    worker_command = shlex.join(command)
    job_name = _job_name(args.run_id)

    if args.dry_run:
        print(f"Job: {job_name}")
        print(f"Command: {worker_command}")
        return

    submit_job(
        job_name=job_name,
        username=USERNAME,
        image="cerit.io/rationai/base:2.0.6",
        cpu=4,
        memory="16Gi",
        shm="4Gi",
        gpu=None,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {GIT_BRANCH}",
            f"export MLFLOW_TRACKING_URI={TRACKING_URI}",
            "uv sync --frozen",
            worker_command,
        ],
        storage=[storage.secure.PROJECTS],
    )
    print(f"Submitted {job_name}.")


if __name__ == "__main__":
    main()
