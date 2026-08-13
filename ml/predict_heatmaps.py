"""Run labeled prediction and prostate-style tile heatmaps from an MLflow model.

The command reconstructs the source training run's Hydra overrides, downloads its
checkpoint, and launches ``ml.train`` in predict mode with the prediction-table and
local-tile heatmap callbacks enabled. The prediction is logged as a new MLflow run.

Example:
    uv run -m ml.predict_heatmaps \
      --checkpoint-uri \
      mlflow-artifacts:/3/<run-id>/artifacts/checkpoints/best \
      --split test
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from omegaconf import OmegaConf

from ml.pathologist_report import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_MLFLOW_UI_URI,
    build_pathologist_report,
)


HYDRA_ARTIFACT = "configs/hydra.yaml"
CHECKPOINT_FILENAME = "checkpoint.ckpt"
DROP_OVERRIDE_KEYS = {
    "mode",
    "checkpoint",
    "predict_split",
    "test_after_fit",
    "logger.run_id",
}
MLFLOW_ARTIFACT_URI = re.compile(
    r"^mlflow-artifacts:/[^/]+/(?P<run_id>[^/]+)/artifacts(?:/|$)"
)


def source_run_id(checkpoint_uri: str) -> str:
    """Extract the source run id from a standard MLflow artifact URI."""
    match = MLFLOW_ARTIFACT_URI.match(checkpoint_uri)
    if match is None:
        raise SystemExit(
            "--checkpoint-uri must be an mlflow-artifacts URI containing the "
            "source experiment and run id."
        )
    return match.group("run_id")


def original_overrides(client: MlflowClient, run_id: str, dst: Path) -> list[str]:
    """Load replayable task overrides recorded by mlkit autologging."""
    path = client.download_artifacts(run_id, HYDRA_ARTIFACT, str(dst))
    hydra_config = OmegaConf.load(path)
    task = OmegaConf.select(hydra_config, "overrides.task")
    if task is None:
        task = OmegaConf.select(hydra_config, "hydra.overrides.task")
    overrides = list(task or [])
    if not overrides:
        raise SystemExit(
            f"Run {run_id} has no Hydra task overrides in {HYDRA_ARTIFACT}."
        )
    return [
        override
        for override in overrides
        if override.split("=", 1)[0].lstrip("+~") not in DROP_OVERRIDE_KEYS
    ]


def merge_overrides(original: list[str], caller: list[str]) -> list[str]:
    """Apply caller overrides without giving Hydra duplicate-key errors."""

    def key(override: str) -> str:
        return override.split("=", 1)[0].lstrip("+~")

    caller_keys = {key(override) for override in caller}
    return [
        override for override in original if key(override) not in caller_keys
    ] + caller


def download_checkpoint(
    checkpoint_uri: str, dst: Path, tracking_uri: str | None = None
) -> Path:
    """Download a checkpoint file or ``checkpoints/best`` directory."""
    downloaded = Path(
        download_artifacts(
            artifact_uri=checkpoint_uri,
            dst_path=str(dst),
            tracking_uri=tracking_uri,
        )
    )
    checkpoint = downloaded / CHECKPOINT_FILENAME if downloaded.is_dir() else downloaded
    if not checkpoint.is_file():
        raise SystemExit(
            f"No {CHECKPOINT_FILENAME!r} found at checkpoint URI {checkpoint_uri!r}."
        )
    return checkpoint


def build_command(
    overrides: list[str],
    checkpoint: Path,
    checkpoint_uri: str,
    run_id: str,
    source_run_name: str,
    split: str,
    invocation_id: str | None = None,
) -> list[str]:
    """Construct the exact Hydra predict invocation."""
    run_name = f"🎯 Tile predictions: {source_run_name}"
    callbacks = [
        "+callbacks@trainer.callbacks.predictions_csv=predictions_csv",
        "+callbacks@trainer.callbacks.tile_probability_heatmaps="
        "tile_probability_heatmaps",
    ]
    command = [
        "uv",
        "run",
        "-m",
        "ml.train",
        *overrides,
        *callbacks,
        "mode=predict",
        f"predict_split={split}",
        f"checkpoint={checkpoint}",
        "test_after_fit=false",
        f"metadata.run_name={json.dumps(run_name)}",
        f"+logger.tags.source_run_id={run_id}",
        f"+logger.tags.source_checkpoint_uri={json.dumps(checkpoint_uri)}",
    ]
    if invocation_id is not None:
        command.append(f"+logger.tags.prediction_invocation_id={invocation_id}")
    return command


def find_prediction_run(client: MlflowClient, invocation_id: str) -> str:
    """Resolve the one new run carrying a unique invocation tag."""
    experiment_ids = [
        experiment.experiment_id for experiment in client.search_experiments()
    ]
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=(f"tags.prediction_invocation_id = '{invocation_id}'"),
        max_results=2,
    )
    if len(runs) != 1:
        raise RuntimeError(
            f"Expected one prediction run for invocation {invocation_id}, found "
            f"{len(runs)}."
        )
    return runs[0].info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-uri",
        required=True,
        help="MLflow checkpoint.ckpt URI or checkpoints/best directory URI.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI; defaults to MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override applied after the source run; repeatable.",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build and attach the RationAI pathologist report (default: on).",
    )
    parser.add_argument("--report-user", default=os.getenv("USER", "unknown"))
    parser.add_argument("--report-title")
    parser.add_argument("--report-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--mlflow-ui-uri",
        default=os.getenv("MLFLOW_UI_URI", DEFAULT_MLFLOW_UI_URI),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = source_run_id(args.checkpoint_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    source_run = client.get_run(run_id)
    source_run_name = source_run.data.tags.get("mlflow.runName", run_id)
    invocation_id = uuid4().hex

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        overrides = merge_overrides(
            original_overrides(client, run_id, tmp / "config"), args.override
        )
        checkpoint = download_checkpoint(
            args.checkpoint_uri, tmp / "checkpoint", args.tracking_uri
        )
        command = build_command(
            overrides,
            checkpoint,
            args.checkpoint_uri,
            run_id,
            source_run_name,
            args.split,
            invocation_id,
        )
        print(f"Source run: {run_id}")
        print(f"Checkpoint: {checkpoint}")
        print("Predict command:")
        print("  " + " ".join(command))
        if args.dry_run:
            if args.report:
                print(
                    "[dry-run] A report would be attached after resolving the new "
                    "prediction run id."
                )
            return 0
        environment = os.environ.copy()
        if args.tracking_uri is not None:
            environment["MLFLOW_TRACKING_URI"] = args.tracking_uri
        prediction = subprocess.run(command, check=False, env=environment)
        if prediction.returncode != 0:
            return prediction.returncode

        prediction_run_id = find_prediction_run(client, invocation_id)
        print(f"PREDICTION_RUN_ID={prediction_run_id}")
        if not args.report:
            return 0

        tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri is None:
            tracking_uri = mlflow.get_tracking_uri()
        return build_pathologist_report(
            client=client,
            prediction_run_id=prediction_run_id,
            tracking_uri=tracking_uri,
            user=args.report_user,
            title=args.report_title,
            config_dir=args.report_config_dir,
            mlflow_ui_uri=args.mlflow_ui_uri,
        )


if __name__ == "__main__":
    raise SystemExit(main())
