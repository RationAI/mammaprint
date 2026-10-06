import argparse
import time

from humanfriendly import parse_size
from kube_jobs import storage, submit_job


CPU = 48
MEMORY = "64Gi"
SHM = "32Gi"

USERNAME = "kissmi"
GIT_BRANCH = "feat/tiling-values"
DEFAULT_LEVELS = (2, 4, 5)
DEFAULT_INTERVAL_MINUTES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit full-cohort all-mask tiling scans at a fixed interval."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=range(6),
        default=DEFAULT_LEVELS,
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=DEFAULT_INTERVAL_MINUTES,
    )
    return parser.parse_args()


def submit_level(level: int) -> None:
    experiment = f"preprocessing/tiling/threshold_scan/mou_{level}_224"
    job_name = f"mammaprint-tiling-all-masks-l{level}-threshold-scan"
    ray_env = [
        f"export RAY_NUM_CPUS={CPU}",
        f"export RAY_MEMORY_BYTES={parse_size(MEMORY, binary=True)}",
    ]

    print(f"Submitting {job_name}.", flush=True)
    submit_job(
        job_name=job_name,
        username=USERNAME,
        image="cerit.io/rationai/base:2.0.6",
        cpu=CPU,
        memory=MEMORY,
        shm=SHM,
        gpu=None,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {GIT_BRANCH}",
            "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
            *ray_env,
            "uv sync --frozen",
            f"uv run -m preprocessing.tiling +experiment={experiment}",
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


def main() -> None:
    args = parse_args()
    interval_seconds = args.interval_minutes * 60

    for index, level in enumerate(args.levels):
        if index:
            print(
                f"Waiting {args.interval_minutes:g} minutes before the next submission.",
                flush=True,
            )
            time.sleep(interval_seconds)
        submit_level(level)


if __name__ == "__main__":
    main()
