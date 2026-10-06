"""Resume split artifact uploads from a complete per-slide embedding artifact."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from mlflow import MlflowClient
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException

from preprocessing.filter_embeddings import (
    DEFAULT_DATA_MAPPING,
    DEFAULT_TRACKING_URI,
    SPLITS,
    _load_split_map,
)


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size: int

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True)
class RecoveryPlan:
    expected: dict[str, dict[str, ArtifactFile]]
    existing: dict[str, dict[str, ArtifactFile]]
    missing: dict[str, dict[str, ArtifactFile]]

    @property
    def missing_files(self) -> int:
        return sum(len(files) for files in self.missing.values())

    @property
    def missing_bytes(self) -> int:
        return sum(
            file.size for files in self.missing.values() for file in files.values()
        )


def _artifact_files(items: list[FileInfo], artifact_path: str) -> dict[str, ArtifactFile]:
    directories = [item.path for item in items if item.is_dir]
    if directories:
        raise ValueError(f"Unexpected directories under {artifact_path}: {directories[:10]}")

    files: dict[str, ArtifactFile] = {}
    for item in items:
        name = Path(item.path).name
        if Path(name).suffix != ".parquet":
            raise ValueError(f"Unexpected non-Parquet artifact: {item.path}")
        if name in files:
            raise ValueError(f"Duplicate artifact name under {artifact_path}: {name}")
        files[name] = ArtifactFile(item.path, int(item.file_size or 0))
    return files


def _build_recovery_plan(
    source: dict[str, ArtifactFile],
    destinations: dict[str, dict[str, ArtifactFile]],
    split_map: dict[str, str],
) -> RecoveryPlan:
    if not source:
        raise ValueError("The source embedding artifact is empty")

    unmapped = sorted(Path(name).stem for name in source if Path(name).stem not in split_map)
    if unmapped:
        raise ValueError(f"Source slides have no split mapping: {unmapped[:10]}")

    expected: dict[str, dict[str, ArtifactFile]] = {split: {} for split in SPLITS}
    for name, file in source.items():
        expected[split_map[Path(name).stem]][name] = file

    for split in SPLITS:
        unexpected = sorted(set(destinations[split]) - set(expected[split]))
        if unexpected:
            raise ValueError(
                f"embeddings_{split} contains unexpected files: {unexpected[:10]}"
            )
        mismatched = sorted(
            name
            for name, file in destinations[split].items()
            if file.size != expected[split][name].size
        )
        if mismatched:
            raise ValueError(
                f"embeddings_{split} contains size-mismatched files: {mismatched[:10]}"
            )

    missing = {
        split: {
            name: file
            for name, file in expected[split].items()
            if name not in destinations[split]
        }
        for split in SPLITS
    }
    return RecoveryPlan(expected=expected, existing=destinations, missing=missing)


def _retry[T](
    operation: Callable[[], T], *, label: str, max_retries: int, logger: logging.Logger
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except MlflowException:
            if attempt == max_retries:
                raise
            delay = 2**attempt
            logger.warning(
                "%s failed; retrying in %d second(s) (%d/%d)",
                label,
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _load_plan(
    client: MlflowClient,
    *,
    run_id: str,
    source_artifact_path: str,
    split_map: dict[str, str],
) -> RecoveryPlan:
    source = _artifact_files(
        client.list_artifacts(run_id, source_artifact_path), source_artifact_path
    )
    destinations = {
        split: _artifact_files(
            client.list_artifacts(run_id, f"embeddings_{split}"),
            f"embeddings_{split}",
        )
        for split in SPLITS
    }
    return _build_recovery_plan(source, destinations, split_map)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _print_plan(plan: RecoveryPlan) -> None:
    for split in SPLITS:
        print(
            f"{split}: {len(plan.existing[split])}/{len(plan.expected[split])} present; "
            f"{len(plan.missing[split])} missing"
        )
    print(
        f"Total transfer: {plan.missing_files} files, "
        f"{_format_bytes(plan.missing_bytes)}"
    )


def _recover(
    client: MlflowClient,
    *,
    run_id: str,
    plan: RecoveryPlan,
    max_retries: int,
    logger: logging.Logger,
) -> None:
    total = plan.missing_files
    completed = 0
    for split in SPLITS:
        artifact_path = f"embeddings_{split}"
        for name, source in sorted(plan.missing[split].items()):
            with tempfile.TemporaryDirectory(prefix="recover-embedding-split-") as tmp:
                local_path = Path(
                    _retry(
                        partial(
                            client.download_artifacts,
                            run_id,
                            source.path,
                            dst_path=tmp,
                        ),
                        label=f"Download {source.path}",
                        max_retries=max_retries,
                        logger=logger,
                    )
                )
                if local_path.stat().st_size != source.size:
                    raise ValueError(
                        f"Downloaded size mismatch for {source.path}: "
                        f"{local_path.stat().st_size} != {source.size}"
                    )
                _retry(
                    partial(
                        client.log_artifact,
                        run_id,
                        str(local_path),
                        artifact_path=artifact_path,
                    ),
                    label=f"Upload {artifact_path}/{name}",
                    max_retries=max_retries,
                    logger=logger,
                )
            completed += 1
            if completed % 25 == 0 or completed == total:
                logger.info("Recovered %d/%d split files", completed, total)


def _log_recovery_manifest(
    client: MlflowClient,
    *,
    run_id: str,
    source_artifact_path: str,
    plan: RecoveryPlan,
) -> None:
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_artifact_path": source_artifact_path,
        "recovered_at": datetime.now(UTC).isoformat(),
        "file_counts": {split: len(plan.expected[split]) for split in SPLITS},
    }
    with tempfile.TemporaryDirectory(prefix="embedding-recovery-manifest-") as tmp:
        path = Path(tmp) / "split_recovery_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        client.log_artifact(run_id, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-artifact-path", default="embeddings")
    parser.add_argument("--data-mapping", default=DEFAULT_DATA_MAPPING)
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform uploads. Without this flag, only print the recovery plan.",
    )
    parser.add_argument(
        "--mark-finished",
        action="store_true",
        help="Mark the recovered MLflow run FINISHED after complete validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if args.mark_finished and not args.apply:
        raise ValueError("--mark-finished requires --apply")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    split_map = _load_split_map(Path(args.data_mapping))
    plan = _load_plan(
        client,
        run_id=args.run_id,
        source_artifact_path=args.source_artifact_path,
        split_map=split_map,
    )
    _print_plan(plan)
    if not args.apply:
        print("Dry run only; pass --apply to upload missing files.")
        return

    _recover(
        client,
        run_id=args.run_id,
        plan=plan,
        max_retries=args.max_retries,
        logger=logger,
    )
    final_plan = _load_plan(
        client,
        run_id=args.run_id,
        source_artifact_path=args.source_artifact_path,
        split_map=split_map,
    )
    if final_plan.missing_files:
        raise RuntimeError(
            f"Recovery finished with {final_plan.missing_files} files still missing"
        )

    _log_recovery_manifest(
        client,
        run_id=args.run_id,
        source_artifact_path=args.source_artifact_path,
        plan=final_plan,
    )
    client.set_tag(run_id=args.run_id, key="split_upload_recovered", value="true")
    if args.mark_finished:
        client.set_terminated(args.run_id, status="FINISHED")
    print("Recovery complete; all split artifact files are present and size-validated.")


if __name__ == "__main__":
    main()
