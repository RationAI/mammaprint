"""Persist and upload multiscale pyramid-selection audit tables."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PyramidAuditDataset(Protocol):
    """Dataset boundary needed by the training-side audit integration."""

    def audit_frame(self) -> pd.DataFrame: ...

    def audit_summary(self) -> pd.DataFrame: ...


class ArtifactLogger(Protocol):
    """Subset of the existing MLflow logger used for artifact uploads."""

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str | None = None,
    ) -> None: ...


def _write_table(frame: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)


def log_pyramid_audit(
    dataset: object,
    logger: ArtifactLogger,
    output_dir: Path,
    *,
    artifact_path: str = "pyramid_selection",
) -> bool:
    """Write and upload one split's selection audit, if the dataset supports it.

    ``output_dir`` is owned by the caller so training can use a managed temporary
    directory. Single-scale datasets do not expose the audit protocol and return
    without creating files or touching the logger.
    """
    if not isinstance(dataset, PyramidAuditDataset):
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(
        dataset.audit_frame(),
        output_dir / "pyramid_selection_audit.csv",
        output_dir / "pyramid_selection_audit.json",
    )
    _write_table(
        dataset.audit_summary(),
        output_dir / "pyramid_selection_summary_by_split_label.csv",
        output_dir / "pyramid_selection_summary_by_split_label.json",
    )
    logger.log_artifacts(str(output_dir), artifact_path=artifact_path)
    return True


__all__ = ["log_pyramid_audit"]
