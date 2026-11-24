"""Preview, apply, or roll back targeted tile removal in an existing report."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from ml.explainability.tile_removal_report import (
    ReportModel,
    load_report_model,
    missing_overlay_count,
    sha256,
    update_report_html,
)


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


_RECEIPT_VERSION = 1
_TILE_REMOVAL = "targeted_tile_removal"


class ReportStore(Protocol):
    artifact: str
    target: str

    def read(self, artifact: str) -> bytes: ...

    def write(self, artifact: str, data: bytes) -> None: ...

    def list(self, folder: str) -> set[str]: ...


def _artifact(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not re.fullmatch(r"[a-zA-Z0-9._/-]+", value)
        or path.is_absolute()
        or not path.name
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"Expected a relative artifact path, got {value!r}.")
    return path


def _digest(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Expected a SHA-256 digest.")
    return value


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            if previous_mode is not None:
                os.fchmod(handle.fileno(), previous_mode)
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class LocalReport:
    def __init__(self, path: Path):
        if path.suffix.lower() != ".html" or path.is_symlink() or not path.is_file():
            raise ValueError("Expected an existing non-symlink HTML report.")
        path = path.resolve(strict=True)
        self.root = path.parent
        self.artifact = path.name
        self.target = str(path)

    def _path(self, artifact: str) -> Path:
        path = self.root / (
            self.artifact if artifact == self.artifact else _artifact(artifact)
        )
        if path.resolve() != path:
            raise ValueError("Refusing to follow a report or backup symlink.")
        return path

    def read(self, artifact: str) -> bytes:
        return self._path(artifact).read_bytes()

    def write(self, artifact: str, data: bytes) -> None:
        _write_atomic(self._path(artifact), data)

    def list(self, folder: str) -> set[str]:
        directory = self._path(folder)
        if not directory.is_dir():
            return set()
        return {file.relative_to(self.root).as_posix() for file in directory.iterdir()}

    @contextmanager
    def lock(self) -> Iterator[None]:
        marker = self.root / f".{self.artifact}.tile-removal.lock"
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
        with os.fdopen(os.open(marker, flags, 0o600), "rb+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield


def validate_tracking_uri(tracking_uri: str) -> str:
    uri = urlsplit(tracking_uri)
    if (
        uri.scheme not in {"http", "https"}
        or not uri.hostname
        or any((uri.username is not None, uri.query, uri.fragment))
        or any(char.isspace() or char == "\\" for char in tracking_uri)
    ):
        raise ValueError("Expected an explicit credentials-free HTTP(S) tracking URI.")
    return tracking_uri.rstrip("/")


class MlflowReport:
    def __init__(
        self,
        client: Any,
        tracking_uri: str,
        experiment_id: str,
        run_id: str,
        artifact: str,
    ):
        tracking_uri = validate_tracking_uri(tracking_uri)
        if not re.fullmatch(r"[0-9a-f]{32}", run_id) or not re.fullmatch(
            r"[0-9]+", experiment_id
        ):
            raise ValueError("Expected an exact MLflow experiment ID and run ID.")
        if _artifact(artifact).suffix != ".html":
            raise ValueError("Expected an HTML artifact path.")
        run = client.get_run(run_id)
        if (
            run.info.run_id != run_id
            or str(run.info.experiment_id) != experiment_id
            or run.info.status != "FINISHED"
            or run.info.lifecycle_stage != "active"
            or run.data.tags.get("report.artifact") != artifact
        ):
            raise ValueError(
                "The active finished MLflow run does not publish that report artifact."
            )
        self.client = client
        self.run_id = run_id
        self.artifact = artifact
        self.target = (
            f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}/{artifact}"
        )

    def read(self, artifact: str) -> bytes:
        _artifact(artifact)
        with tempfile.TemporaryDirectory() as directory:
            path = self.client.download_artifacts(self.run_id, artifact, directory)
            return Path(path).read_bytes()

    def write(self, artifact: str, data: bytes) -> None:
        path = _artifact(artifact)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / path.name
            file.write_bytes(data)
            parent = (
                path.parent.as_posix() if path.parent != PurePosixPath(".") else None
            )
            self.client.log_artifact(self.run_id, str(file), artifact_path=parent)

    def list(self, folder: str) -> set[str]:
        _artifact(folder)
        return {file.path for file in self.client.list_artifacts(self.run_id, folder)}


def report_slide_ids(parquet: bytes) -> frozenset[str]:
    import pyarrow.parquet as pq

    rows = pq.read_table(BytesIO(parquet)).to_pylist()
    ids = [
        row["slide_id"]
        for row in rows
        if row.get("status") == "ok" and row.get("overlay_status") == "ok"
    ]
    if (
        not ids
        or any(not isinstance(slide, str) or not slide.strip() for slide in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("The published report's validation slide roster is invalid.")
    return frozenset(ids)


def _receipt_bytes(fields: Mapping[str, Any]) -> bytes:
    return (json.dumps(fields, indent=2, sort_keys=True) + "\n").encode()


def _check_readback(store: ReportStore, artifact: str, data: bytes) -> None:
    if sha256(store.read(artifact)) != sha256(data):
        raise RuntimeError(f"Read-back verification failed for {artifact}.")


def _archive_root(store: ReportStore, updated_sha256: str, kind: str) -> str:
    target_id = sha256(store.target.encode())[:16]
    return f"report_backfills/{kind}/{target_id}/{updated_sha256}"


def _recover_verified_receipt(
    store: ReportStore,
    current: bytes,
    expected: dict[str, Any],
    *,
    apply: bool,
    kind: str,
) -> dict[str, Any]:
    archive = _archive_root(store, expected["updated_sha256"], kind)
    for folder in sorted(store.list(archive), reverse=True):
        journal = f"{folder}/receipt.json"
        if journal not in store.list(folder):
            continue
        receipt = json.loads(store.read(journal))
        backup = f"{folder}/original.html"
        if (
            receipt.get("receipt_artifact") != journal
            or receipt.get("backup_artifact") != backup
            or receipt.get("status") not in {"prepared", "verified"}
            or any(
                receipt.get(field) != expected[field]
                for field in (
                    "version",
                    "target",
                    "updated_sha256",
                    "bundle_sha256",
                    "source_run_id",
                    "checkpoint_sha256",
                )
            )
        ):
            raise ValueError("An existing backfill receipt does not match the report.")
        if sha256(store.read(backup)) != receipt["original_sha256"]:
            raise ValueError("The original backup does not match the report receipt.")
        if receipt["status"] == "prepared" and apply:
            if store.read(store.artifact) != current:
                raise ValueError("The report changed while recovering the receipt.")
            receipt["status"] = "verified"
            payload = _receipt_bytes(receipt)
            store.write(journal, payload)
            _check_readback(store, journal, payload)
        return {
            **receipt,
            "status": "already_updated",
            "receipt_status": receipt["status"],
        }
    return {**expected, "status": "already_updated"}


def _apply_update(
    store: ReportStore,
    original: bytes,
    update: bytes,
    receipt: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    if store.read(store.artifact) != original:
        raise ValueError("The report changed after preview. Run the preview again.")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = _archive_root(store, receipt["updated_sha256"], kind)
    archive += f"/{stamp}-{receipt['original_sha256'][:12]}"
    backup = f"{archive}/original.html"
    journal = f"{archive}/receipt.json"
    receipt.update(backup_artifact=backup, receipt_artifact=journal, status="prepared")
    store.write(backup, original)
    _check_readback(store, backup, original)
    payload = _receipt_bytes(receipt)
    store.write(journal, payload)
    _check_readback(store, journal, payload)
    if store.read(store.artifact) != original:
        raise ValueError("The report changed after preview. Run the preview again.")
    try:
        store.write(store.artifact, update)
        _check_readback(store, store.artifact, update)
        receipt["status"] = "verified"
        payload = _receipt_bytes(receipt)
        store.write(journal, payload)
        _check_readback(store, journal, payload)
    except Exception as error:
        raise RuntimeError(
            f"Unable to verify the updated report. "
            f"Use receipt {journal} and backup {backup} to diagnose or roll back."
        ) from error
    return receipt


def backfill_report(
    store: ReportStore,
    bundle: Path,
    provenance: dict[str, Any],
    *,
    slide_ids: frozenset[str] | None = None,
    explanation_run_id: str | None = None,
    preview: Path | None = None,
    apply: bool = False,
    expected_original: str | None = None,
    expected_updated: str | None = None,
    review_findings: str | None = None,
) -> dict[str, Any]:
    model = load_report_model(bundle, provenance, explanation_run_id=explanation_run_id)
    if review_findings is not None:
        model = ReportModel(
            {**model.model, "review_findings": review_findings},
            model.split,
            model.bundle_sha256,
        )
    original = store.read(store.artifact)
    update = update_report_html(
        original,
        model,
        slide_ids=slide_ids,
        explanation_run_id=provenance.get("explanation_run_id"),
    )
    receipt = {
        "version": _RECEIPT_VERSION,
        "target": store.target,
        "report_split": model.split,
        "source_run_id": model.model["source_run_id"],
        "checkpoint_sha256": model.model["checkpoint_sha256"],
        "original_sha256": sha256(original),
        "updated_sha256": sha256(update),
        "bundle_sha256": model.bundle_sha256,
        "bundle_slide_count": len(model.model["slides"]),
        "validation_report_slide_count": update.count(
            b"<details data-tile-removal-slide-id="
        ),
        "missing_overlay_references": missing_overlay_count(model.model),
    }
    if review_findings is not None:
        receipt["review_findings_sha256"] = sha256(review_findings.encode("utf-8"))
    return finish_report_update(
        store,
        original,
        update,
        receipt,
        kind=_TILE_REMOVAL,
        preview=preview,
        apply=apply,
        expected_original=expected_original,
        expected_updated=expected_updated,
    )


def finish_report_update(
    store: ReportStore,
    original: bytes,
    update: bytes,
    receipt: dict[str, Any],
    *,
    kind: str,
    preview: Path | None,
    apply: bool,
    expected_original: str | None,
    expected_updated: str | None,
) -> dict[str, Any]:
    if preview is not None:
        with os.fdopen(
            os.open(preview, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
        ) as file:
            file.write(update)
    if (
        apply
        and expected_updated is not None
        and _digest(expected_updated) != receipt["updated_sha256"]
    ):
        raise ValueError("The proposed report differs from the approved preview.")
    if update == original:
        with (
            store.lock() if apply and isinstance(store, LocalReport) else nullcontext()
        ):
            return _recover_verified_receipt(
                store, original, receipt, apply=apply, kind=kind
            )
    if not apply:
        return {**receipt, "status": "preview"}
    if (
        expected_original is not None
        and _digest(expected_original) != receipt["original_sha256"]
    ):
        raise ValueError("The original report differs from the approved preview.")

    with store.lock() if isinstance(store, LocalReport) else nullcontext():
        return _apply_update(store, original, update, receipt, kind)


def rollback_report(
    store: ReportStore,
    receipt_artifact: str,
    *,
    apply: bool = False,
    kind: str = _TILE_REMOVAL,
) -> dict[str, Any]:
    path = _artifact(receipt_artifact)
    receipt = json.loads(store.read(receipt_artifact))
    expected_backup = (path.parent / "original.html").as_posix()
    if (
        not path.is_relative_to(f"report_backfills/{kind}")
        or path.name != "receipt.json"
        or receipt.get("version") != _RECEIPT_VERSION
        or receipt.get("receipt_artifact") != receipt_artifact
        or receipt.get("target") != store.target
        or receipt.get("status") not in {"prepared", "verified"}
        or receipt.get("backup_artifact") != expected_backup
    ):
        raise ValueError("The receipt does not identify a backfill for this report.")
    original = store.read(expected_backup)
    if sha256(original) != receipt["original_sha256"]:
        raise ValueError("The backup does not match the recorded original hash.")
    current = store.read(store.artifact)
    if sha256(current) == receipt["original_sha256"]:
        return {**receipt, "status": "already_original"}
    if sha256(current) != receipt["updated_sha256"]:
        raise ValueError("The current report differs from the backfill receipt.")
    if not apply:
        return {**receipt, "status": "rollback_preview"}
    with store.lock() if isinstance(store, LocalReport) else nullcontext():
        if store.read(store.artifact) != current:
            raise ValueError(
                "The report changed during rollback. Check the preview again."
            )
        store.write(store.artifact, original)
        _check_readback(store, store.artifact, original)
        rollback = {**receipt, "status": "rolled_back"}
        audit = (path.parent / "rollback.json").as_posix()
        payload = _receipt_bytes(rollback)
        store.write(audit, payload)
        _check_readback(store, audit, payload)
    return rollback


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--report", type=Path)
    target.add_argument("--run-id")
    parser.add_argument("--report-manifest", type=Path)
    parser.add_argument("--report-slides", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-id")
    parser.add_argument("--artifact-path")
    parser.add_argument("--manifest-artifact", default="explanations/manifest.json")
    parser.add_argument("--slides-artifact", default="explanations/slides.parquet")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--bundle", type=Path)
    action.add_argument("--rollback", metavar="RECEIPT_ARTIFACT")
    parser.add_argument("--preview", type=Path)
    parser.add_argument(
        "--review-findings", type=Path, help="Post-review findings in UTF-8 text."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-original-sha256")
    parser.add_argument("--expected-updated-sha256")
    args = parser.parse_args(argv)
    if args.rollback and (args.preview or args.review_findings):
        parser.error("--preview and --review-findings are options for --bundle.")

    if args.report is not None:
        if args.bundle and args.report_manifest is None:
            parser.error("--report-manifest is required for a local backfill.")
        if args.tracking_uri or args.experiment_id or args.artifact_path:
            parser.error("MLflow options require --run-id.")
        store: ReportStore = LocalReport(args.report)
    else:
        if not (args.tracking_uri and args.experiment_id and args.artifact_path):
            parser.error(
                "Specify --tracking-uri, --experiment-id, and --artifact-path."
            )
        if args.report_manifest or args.report_slides:
            parser.error("MLflow report provenance must come from the selected run.")
        if (
            args.apply
            and args.bundle
            and not (args.expected_original_sha256 and args.expected_updated_sha256)
        ):
            parser.error("Remote --apply requires both hashes from the preview.")
        from mlflow import MlflowClient

        uri = validate_tracking_uri(args.tracking_uri)
        store = MlflowReport(
            MlflowClient(tracking_uri=uri),
            uri,
            args.experiment_id,
            args.run_id,
            args.artifact_path,
        )

    if args.rollback:
        result = rollback_report(store, args.rollback, apply=args.apply)
    else:
        manifest_bytes = (
            store.read(args.manifest_artifact)
            if args.run_id
            else args.report_manifest.read_bytes()
        )
        provenance = json.loads(manifest_bytes)
        ids = None
        if provenance.get("split") == "val":
            if args.run_id:
                ids = report_slide_ids(store.read(args.slides_artifact))
            elif args.report_slides:
                ids = report_slide_ids(args.report_slides.read_bytes())
        result = backfill_report(
            store,
            args.bundle,
            provenance,
            slide_ids=ids,
            explanation_run_id=args.run_id,
            preview=args.preview,
            apply=args.apply,
            expected_original=args.expected_original_sha256,
            expected_updated=args.expected_updated_sha256,
            review_findings=(
                args.review_findings.read_text(encoding="utf-8")
                if args.review_findings
                else None
            ),
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
