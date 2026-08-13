"""Submit labeled tile-prediction heatmaps to one H100 worker."""

import argparse
import shlex

from kube_jobs import storage, submit_job


DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DEFAULT_BRANCH = "codex/tile-probability-heatmaps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--job-name", default="mammaprint-tile-heatmaps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_args = [
        "uv",
        "run",
        "-m",
        "ml.predict_heatmaps",
        "--checkpoint-uri",
        args.checkpoint_uri,
        "--split",
        args.split,
    ]
    for override in args.override:
        predict_args.extend(("--override", override))

    submit_job(
        job_name=args.job_name,
        username="kissmi",
        image="cerit.io/rationai/base:2.0.6",
        cpu=16,
        memory="48Gi",
        gpu="H100",
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {shlex.quote(args.branch)}",
            f"export MLFLOW_TRACKING_URI={shlex.quote(args.tracking_uri)}",
            "uv sync --frozen",
            shlex.join(predict_args),
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
