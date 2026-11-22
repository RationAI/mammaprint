"""Select capped L1 tiles and persist Virchow2 embeddings directly to MLflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from preprocessing.embedding_selection import prepare_selection


if TYPE_CHECKING:
    from mlflow import MlflowClient


def _selection(
    args: argparse.Namespace, client: MlflowClient, run_id: str, scratch: Path
) -> Path:
    selection = scratch / "selection"
    location = "selection/frozen.tar.gz"
    if any(
        item.path == location for item in client.list_artifacts(run_id, "selection")
    ):
        archive = Path(client.download_artifacts(run_id, location, str(scratch)))
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(scratch, filter="data")
    else:
        if client.list_artifacts(run_id, "embeddings"):
            raise ValueError(
                f"Existing embeddings have no frozen tile selection in {run_id}"
            )
        prepare_selection(
            level=1,
            tiling=args.tiling,
            data_mapping=args.data_mapping,
            output=selection,
            shards=1,
            cap=3000,
            seed=42,
            mask_sources={"epithelium_overlap": args.epithelium_masks},
        )
        archive = scratch / "frozen.tar.gz"
        with tarfile.open(archive, "w:gz") as destination:
            destination.add(selection, arcname="selection")
        client.log_artifact(run_id, str(archive), artifact_path="selection")
    manifest = selection / "manifest.json"
    client.log_artifact(run_id, str(manifest), artifact_path="selection")
    archive.unlink()
    return manifest


def _run(args: argparse.Namespace) -> int:
    from mlflow import MlflowClient

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise ValueError("Set MLFLOW_TRACKING_URI to the intended artifact destination")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("MammaPrint")
    if experiment is None:
        raise ValueError("MammaPrint MLflow experiment does not exist")
    if args.run_id:
        run = client.get_run(args.run_id)
        if (
            run.info.experiment_id != experiment.experiment_id
            or run.data.tags.get("workflow") != "l1-virchow2-embedding"
            or run.info.lifecycle_stage != "active"
        ):
            raise ValueError(
                f"MLflow run {args.run_id} is not an active L1 dataset run"
            )
    else:
        run = client.create_run(
            experiment.experiment_id,
            run_name="L1 cancer mask Virchow2 capped",
            tags={"workflow": "l1-virchow2-embedding", "runtime_id": args.runtime_id},
        )
    run_id = run.info.run_id
    print(f"MLflow run {run_id}: {run.info.artifact_uri}", flush=True)
    scratch = args.work_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        manifest = _selection(args, client, run_id, scratch)
        command = [
            sys.executable,
            "-m",
            "preprocessing.resolution_embeddings",
            "--manifest",
            str(manifest),
            "--output",
            str(scratch / "embed"),
            "--mlflow-run-id",
            run_id,
            "--shard",
            "0",
            "--encoder-revision",
            args.encoder_revision,
            "--runtime-id",
            args.runtime_id,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
        ]
        status = subprocess.run(command, check=False).returncode
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id, status="FINISHED" if status == 0 else "FAILED")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiling", required=True, help="Existing cancer-mask L1 artifact"
    )
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--epithelium-masks", required=True)
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Empty pod-local scratch"
    )
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-id", help="Resume a prior L1 embedding MLflow run")
    args = parser.parse_args()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error("--work-dir must be empty")
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
