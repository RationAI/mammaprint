"""Embed the missing-only selection from a finished L1 tissue preparation run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from preprocessing.resolution_embeddings import load_selection_manifest


if TYPE_CHECKING:
    from mlflow import MlflowClient


WORKFLOW = "l1-virchow2-tissue-missing-embedding"


def _prepare(
    client: MlflowClient,
    preparation_id: str,
    experiment_id: str,
    encoder_revision: str,
    scratch: Path,
) -> tuple[Path, Path, str, int]:
    source = client.get_run(preparation_id)
    if (
        source.info.status != "FINISHED"
        or source.info.experiment_id != experiment_id
        or source.info.lifecycle_stage != "active"
        or source.data.tags.get("workflow") != "l1-prepare-tissue-selection"
    ):
        raise ValueError(f"{preparation_id} is not a finished L1 tissue preparation")

    def download(artifact: str) -> Path:
        return Path(client.download_artifacts(preparation_id, artifact, str(scratch)))

    audit = json.loads(download("metadata/tissue_reuse_audit.json").read_text())
    archive_path = download("missing_selection/frozen.tar.gz")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(scratch, filter="data")
    manifest_path = scratch / "missing_selection" / "manifest.json"
    manifest = load_selection_manifest(manifest_path)
    reuse = manifest["reuse_source"]
    if (
        manifest["level"] != 1
        or manifest["selection_kind"] != "missing_tissue_embeddings"
        or manifest["shards"] != 1
        or manifest["digest"] != audit["missing_selection_digest"]
        or any(manifest["score_missing"].values())
        or any(audit["score_missing"].values())
        or reuse["run_id"] != source.data.tags.get("reuse_source_run_id")
        or reuse["encoder_revision"] != encoder_revision
        or reuse["precision"] != "fp32"
        or reuse["normalization"] != "imagenet-rgb-fp32-v1"
    ):
        raise ValueError(
            "Missing-only selection or saved encoder settings differ from the preparation audit"
        )

    slides = manifest["slides"]
    if any(
        slide["shard"] != 0
        or not 0 <= slide["tile_count"] <= slide["full_tile_count"] <= manifest["cap"]
        or slide["tile_count"] + slide["reused_tile_count"] != slide["full_tile_count"]
        for slide in slides
    ):
        raise ValueError("Invalid tile counts in the missing-only selection")
    pending = sum(slide["tile_count"] for slide in slides)
    if pending == 0 or pending != audit["total"]["missing_embeddings"]:
        raise ValueError(
            "Missing tile count is zero or differs from the preparation audit"
        )
    return manifest_path, archive_path, manifest["digest"], pending


def _run(args: argparse.Namespace) -> int:
    from mlflow import MlflowClient

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise ValueError("Set MLFLOW_TRACKING_URI to the intended artifact destination")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("MammaPrint")
    if experiment is None:
        raise ValueError("MammaPrint MLflow experiment does not exist")
    scratch = args.work_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    manifest, archive, digest, tiles = _prepare(
        client,
        args.preparation_run_id,
        experiment.experiment_id,
        args.encoder_revision,
        scratch,
    )
    tags = {
        "workflow": WORKFLOW,
        "preparation_run_id": args.preparation_run_id,
        "missing_selection_digest": digest,
    }
    if args.run_id:
        run = client.get_run(args.run_id)
        if (
            run.info.experiment_id != experiment.experiment_id
            or run.info.lifecycle_stage != "active"
            or any(run.data.tags.get(key) != value for key, value in tags.items())
        ):
            raise ValueError(
                f"MLflow run {args.run_id} is outside this tissue embedding scope"
            )
    else:
        run = client.create_run(
            experiment.experiment_id,
            tags=tags,
            run_name="Embed missing capped L1 tissue tiles",
        )
    run_id = run.info.run_id
    print(
        f"L1 tissue embedding MLflow run {run_id}: {run.info.artifact_uri}", flush=True
    )
    print(
        f"Embedding {tiles:,} tiles from the frozen missing-only selection", flush=True
    )
    try:
        # Keep the same durable selection and embedding artifact layout as the original L1 run.
        files = {item.path for item in client.list_artifacts(run_id, "selection")}
        for source in (manifest, archive):
            if f"selection/{source.name}" not in files:
                client.log_artifact(run_id, str(source), artifact_path="selection")
        archive.unlink()
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
            "--precision",
            "fp32",
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
    parser.add_argument("--preparation-run-id", required=True)
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Empty pod-local scratch"
    )
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-id", help="Resume a stopped tissue embedding run")
    args = parser.parse_args()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error("--work-dir must be empty")
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
