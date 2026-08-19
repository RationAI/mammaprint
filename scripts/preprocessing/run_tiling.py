"""Submit the MammaPrint tiling pipeline as a Kubernetes job."""

from __future__ import annotations

import argparse
import shlex

from humanfriendly import parse_size
from kube_jobs import storage, submit_job


DEFAULT_BRANCH = "codex/epithelium-onnx-job"
DEFAULT_REPOSITORY = "https://github.com/RationAI/MammaPrint.git"
DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DEFAULT_EXPERIMENT = "preprocessing/tiling/tissue_only/mou_epithelium_512"
CPU = 48
MEMORY = "64Gi"


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit raw-slide tiling and log the tiled dataset to MLflow."
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--job-name", default="mammaprint-tiling-epithelium-512")
    parser.add_argument("--cpu", type=int, default=CPU)
    parser.add_argument("--memory", default=MEMORY)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def job_commands(args: argparse.Namespace) -> list[str]:
    ray_env = [
        f"export RAY_NUM_CPUS={args.cpu}",
        f"export RAY_MEMORY_BYTES={parse_size(args.memory, binary=True)}",
    ]
    return [
        shlex.join(
            [
                "git",
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.repository,
                "workdir",
            ]
        ),
        "cd workdir",
        f"export MLFLOW_TRACKING_URI={shlex.quote(args.tracking_uri)}",
        *ray_env,
        "uv sync --frozen",
        shlex.join(
            [
                "uv",
                "run",
                "-m",
                "preprocessing.tiling",
                f"+experiment={args.experiment}",
            ]
        ),
    ]


def main() -> None:
    args = argument_parser().parse_args()
    commands = job_commands(args)
    if args.dry_run:
        print("\n".join(commands))
        return

    submit_job(
        job_name=args.job_name,
        username=args.username,
        image="cerit.io/rationai/base:2.0.6",
        cpu=args.cpu,
        memory=args.memory,
        gpu=None,
        public=False,
        script=commands,
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
