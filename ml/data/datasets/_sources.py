"""Shared setup helpers for the slide datasets.

:class:`~ml.data.datasets.pyramid.PyramidSlideDataset` needs a few setup steps:
load + label the slide table, download the per-level artifacts, and find which
slides are actually present across levels. Extracting them here keeps the dataset
free of scaffolding and leaves them independently unit-testable.

The train/val/test split is NOT applied here — splitting is materialised into
physically separate artifacts upstream by ``scripts/preprocessing/split_dataset.py``
(the ``split`` column in ``data_mapping.csv`` is the sole split authority). A card's
per-split URI already points at a pure artifact, so no split filter is needed.
"""

import logging
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
from mlflow.artifacts import (
    download_artifacts as _download_artifacts,
)
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.utils.uri import append_to_uri_path

from ml.data.datasets.labels import LabelMode, get_target_columns, process_slides


logger = logging.getLogger(__name__)

MLFLOW_DOWNLOAD_MAX_RETRIES = 3
MLFLOW_DOWNLOAD_INITIAL_BACKOFF_SECONDS = 1.0


def _artifact_files(
    repository: ArtifactRepository, artifact_path: str
) -> list[FileInfo]:
    """Return every file below an artifact directory using metadata only."""
    files: list[FileInfo] = []
    pending = [artifact_path]
    while pending:
        for item in repository.list_artifacts(pending.pop()):
            if item.is_dir:
                pending.append(item.path)
            else:
                files.append(item)
    return files


def _is_complete_download(file_info: FileInfo, local_path: Path) -> bool:
    if not local_path.is_file():
        return False
    return (
        file_info.file_size is None or local_path.stat().st_size == file_info.file_size
    )


def _download_incomplete_artifacts(
    root_uri: str,
    artifact_path: str,
    destination: Path,
) -> Path:
    repository = get_artifact_repository(root_uri)
    remote_files = _artifact_files(repository, artifact_path)
    if not remote_files:
        # The URI may point to a single file rather than a directory.
        uri = append_to_uri_path(root_uri, artifact_path)
        return Path(_download_artifacts(artifact_uri=uri, dst_path=str(destination)))

    incomplete = [
        file_info
        for file_info in remote_files
        if not _is_complete_download(file_info, destination / file_info.path)
    ]
    logger.info(
        "Retrying %d incomplete MLflow artifact(s); keeping %d already downloaded.",
        len(incomplete),
        len(remote_files) - len(incomplete),
    )
    for file_info in incomplete:
        # Keep the original root and path. Reparsing a file URI drops the
        # directory prefix used by the initial directory download.
        with tempfile.TemporaryDirectory(dir=destination) as staging:
            downloaded = Path(
                repository.download_artifacts(file_info.path, dst_path=staging)
            )
            if not _is_complete_download(file_info, downloaded):
                raise MlflowException(
                    f"Incomplete MLflow artifact after retry: {file_info.path}"
                )
            local_path = destination / file_info.path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(local_path)

    return destination / artifact_path


def download_artifacts_with_retries(
    artifact_uri: str,
    *,
    max_retries: int = MLFLOW_DOWNLOAD_MAX_RETRIES,
    initial_backoff_seconds: float = MLFLOW_DOWNLOAD_INITIAL_BACKOFF_SECONDS,
) -> Path:
    """Download an MLflow artifact, resuming transient download failures.

    MLflow raises one aggregate :class:`MlflowException` when any file in a
    directory fails. The initial parallel download uses a stable temporary
    destination; retries preserve complete files and fetch only missing or
    size-mismatched files.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if initial_backoff_seconds < 0:
        raise ValueError("initial_backoff_seconds must be non-negative.")

    root_uri, artifact_path = _get_root_uri_and_artifact_path(artifact_uri)
    destination = Path(tempfile.mkdtemp(prefix="mammaprint-mlflow-artifacts-"))

    for retry in range(max_retries + 1):
        try:
            if retry == 0:
                return Path(
                    _download_artifacts(
                        artifact_uri=artifact_uri,
                        dst_path=str(destination),
                    )
                )
            return _download_incomplete_artifacts(
                root_uri,
                artifact_path,
                destination,
            )
        except MlflowException:
            if retry == max_retries:
                raise

            delay = initial_backoff_seconds * 2**retry
            logger.warning(
                "MLflow artifact download failed for %s; retrying in %.1f seconds "
                "(%d/%d).",
                artifact_uri,
                delay,
                retry + 1,
                max_retries,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def load_labeled_slides(
    data_mapping: str | Path,
    label_mode: LabelMode,
) -> pd.DataFrame:
    """Load the slide table and drop rows without a label.

    Args:
        data_mapping: Path to ``data_mapping.csv`` (``record_num``/``type``/
            ``mammaprint_index``/``path``).
        label_mode: Classification (``type``) or regression (``index``) target.

    Returns:
        A dataframe with a ``name`` (slide stem) column and the label column, keeping
        only rows that carry a label.
    """
    slides = pd.read_csv(data_mapping)
    raw_target_columns = {
        LabelMode.TYPE: ["type"],
        LabelMode.INDEX: ["mammaprint_index"],
        LabelMode.BOTH: ["type", "mammaprint_index"],
    }[label_mode]
    slides = slides.dropna(subset=raw_target_columns)
    slides = process_slides(slides, mode=label_mode)

    target_columns = get_target_columns(label_mode)
    keep = slides[target_columns].notna().all(axis=1)
    return slides[keep].reset_index(drop=True)


def download_level_sources(sources: Mapping[int, str]) -> dict[int, Path]:
    """Use local artifact directories in place and download remote sources."""
    directories = {}
    for level, uri in sources.items():
        parsed = urlsplit(uri)
        if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
            path = Path(unquote(parsed.path))
        elif not parsed.scheme and Path(uri).is_absolute():
            path = Path(uri)
        else:
            directories[level] = download_artifacts_with_retries(uri)
            continue
        if not path.is_dir():
            raise FileNotFoundError(
                f"Level {level} artifact directory is missing: {path}"
            )
        directories[level] = path
    return directories


def available_slides(dirs: Mapping[int, Path]) -> set[str]:
    """Slide stems present as ``<stem>.parquet`` in *every* level's directory."""
    per_level = ({p.stem for p in d.glob("*.parquet")} for d in dirs.values())
    return set.intersection(*per_level) if dirs else set()


def split_uri(card: Mapping[str, object], key: str, split: str, level: int) -> str:
    """Read a level card's per-split artifact URI (``card[key][split]``).

    Raises a clear error if the card lacks the URI map or the requested split.
    """
    uris = card.get(key)
    if not isinstance(uris, Mapping):
        raise KeyError(f"Level {level} data card is missing a '{key}' split->URI map.")
    uri = uris.get(split)
    if not isinstance(uri, str):
        raise KeyError(
            f"Level {level} data card '{key}' has no URI for split '{split}'."
        )
    return uri


__all__ = [
    "available_slides",
    "download_artifacts_with_retries",
    "download_level_sources",
    "load_labeled_slides",
    "split_uri",
]
