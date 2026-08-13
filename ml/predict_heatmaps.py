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

from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from omegaconf import OmegaConf


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
) -> list[str]:
    """Construct the exact Hydra predict invocation."""
    run_name = f"🎯 Tile predictions: {source_run_name}"
    callbacks = [
        "+callbacks@trainer.callbacks.predictions_csv=predictions_csv",
        "+callbacks@trainer.callbacks.tile_probability_heatmaps="
        "tile_probability_heatmaps",
    ]
    return [
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = source_run_id(args.checkpoint_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    source_run = client.get_run(run_id)
    source_run_name = source_run.data.tags.get("mlflow.runName", run_id)

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
        )
        print(f"Source run: {run_id}")
        print(f"Checkpoint: {checkpoint}")
        print("Predict command:")
        print("  " + " ".join(command))
        if args.dry_run:
            return 0
        environment = os.environ.copy()
        if args.tracking_uri is not None:
            environment["MLFLOW_TRACKING_URI"] = args.tracking_uri
        return subprocess.run(command, check=False, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
