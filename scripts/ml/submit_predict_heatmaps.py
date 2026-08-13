"""Submit tile prediction, heatmaps, and the pathologist report."""

import argparse
import shlex

from kube_jobs import storage, submit_job


DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DEFAULT_MLFLOW_UI_URI = "https://mlflow.rationai.cloud.trusted.e-infra.cz/"
DEFAULT_BRANCH = "codex/tile-probability-heatmaps"
CHECKPOINT_URI = (
    "mlflow-artifacts:/3/a4f8b526efd14f53b0bd657d7507e006/artifacts/checkpoints/best"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-uri", default=CHECKPOINT_URI)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--mlflow-ui-uri", default=DEFAULT_MLFLOW_UI_URI)
    parser.add_argument("--report-user", default="kissmi")
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", default="A40")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--job-name", default="mammaprint-tile-heatmaps-1")
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
        "--tracking-uri",
        args.tracking_uri,
        "--mlflow-ui-uri",
        args.mlflow_ui_uri,
        "--report-user",
        args.report_user,
    ]
    if not args.report:
        predict_args.append("--no-report")
    for override in args.override:
        predict_args.extend(("--override", override))

    submit_job(
        job_name=args.job_name,
        username="kissmi",
        image="cerit.io/rationai/base:2.0.6",
        cpu=16,
        memory="48Gi",
        gpu=args.gpu,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {shlex.quote(args.branch)}",
            f"export MLFLOW_TRACKING_URI={shlex.quote(args.tracking_uri)}",
            f"export MLFLOW_UI_URI={shlex.quote(args.mlflow_ui_uri)}",
            "uv sync --frozen",
            shlex.join(predict_args),
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
