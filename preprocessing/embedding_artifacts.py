"""Store L1 embedding results and resume receipts in a single MLflow run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from mlflow import MlflowClient


class EmbeddingArtifacts:
    """One writer streams completed slides to MLflow and releases local scratch."""

    def __init__(self, client: MlflowClient, run_id: str, scratch: Path):
        self.client = client
        self.run_id = run_id
        self.scratch = scratch
        self.embeddings: dict[str, int | None] = {}
        self.receipts: dict[str, dict[str, Any]] = {}

    def prepare(self, config: dict[str, Any]) -> None:
        self.scratch.mkdir(parents=True, exist_ok=True)
        existing = {
            info.path for info in self.client.list_artifacts(self.run_id, "metadata")
        }
        path = self.scratch / "run.json"
        if "metadata/run.json" in existing:
            saved = Path(
                self.client.download_artifacts(
                    self.run_id, "metadata/run.json", str(self.scratch)
                )
            )
            if json.loads(saved.read_text()) != config:
                raise ValueError(
                    f"Embedding settings differ from MLflow run {self.run_id}"
                )
        else:
            if self.client.list_artifacts(self.run_id, "embeddings"):
                raise ValueError(
                    f"Embedding artifacts are missing their settings in {self.run_id}"
                )
            path.write_text(json.dumps(config, sort_keys=True))
            self.client.log_artifact(self.run_id, str(path), artifact_path="metadata")
        self.embeddings = {
            Path(info.path).stem: info.file_size
            for info in self.client.list_artifacts(self.run_id, "embeddings")
            if not info.is_dir and info.path.endswith(".parquet")
        }
        self.receipts = {}
        receipt_files = self.client.list_artifacts(self.run_id, "complete")
        if receipt_files:
            directory = Path(
                self.client.download_artifacts(
                    self.run_id, "complete", str(self.scratch)
                )
            )
            for path in directory.glob("*.json"):
                try:
                    receipt = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if isinstance(receipt, dict):
                    self.receipts[path.stem] = receipt

    def complete(self, slide: dict[str, Any]) -> bool:
        stem = slide["slide_id"]
        saved = self.receipts.get(stem)
        size = saved.get("size") if saved else None
        return bool(
            saved
            and saved.get("tiles") == slide["tile_count"]
            and saved.get("selection_digest") == slide["selection_digest"]
            and isinstance(size, int)
            and size > 0
            and size == self.embeddings.get(stem)
            and saved.get("sha256")
        )

    def upload(self, parquet: Path, receipt: Path) -> None:
        committed = json.loads(receipt.read_text())
        committed["size"] = parquet.stat().st_size
        self.client.log_artifact(self.run_id, str(parquet), artifact_path="embeddings")
        receipt.write_text(json.dumps(committed, sort_keys=True))
        self.client.log_artifact(self.run_id, str(receipt), artifact_path="complete")
        self.embeddings[parquet.stem] = committed["size"]
        self.receipts[parquet.stem] = committed
        parquet.unlink()
        receipt.unlink()

    def report(self, path: Path, slides: list[dict[str, Any]]) -> bool:
        expected = {slide["slide_id"] for slide in slides if slide["tile_count"]}
        saved = {
            Path(info.path).stem: info.file_size
            for info in self.client.list_artifacts(self.run_id, "embeddings")
            if not info.is_dir and info.path.endswith(".parquet")
        }
        self.embeddings = saved
        complete = {
            slide["slide_id"]: {
                "split": slide["split"],
                **self.receipts[slide["slide_id"]],
            }
            for slide in slides
            if self.complete(slide)
        }
        path.write_text(
            json.dumps(
                {
                    "complete": complete,
                    "missing_or_invalid": sorted(expected - complete.keys()),
                    "unexpected": sorted(saved.keys() - expected),
                    "zero_eligible_tiles": sorted(
                        slide["slide_id"] for slide in slides if not slide["tile_count"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        self.client.log_artifact(self.run_id, str(path), artifact_path="metadata")
        return (
            bool(expected) and set(complete) == expected and not saved.keys() - expected
        )
