"""Build a new tile-explanation HTML report from a RationAI report draft."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urljoin

from ml.explainability.tile_removal_report import sha256
from scripts.ml.backfill_tile_removal_reports import (
    LocalReport,
    _artifact,
    backfill_report,
    report_slide_ids,
    validate_tracking_uri,
)


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Sequence
    from typing import Any


_FROZEN_ENTRYPOINT = "a3ecb997d02eeff422c438e3322d7f1dd70b125bc6d6109285728acfc680e554"
_DEFAULT_XOPAT = "https://xopat.rationai.cloud.trusted.e-infra.cz/"
_DEFAULT_METHODS = ("leave_one_out", "single", "attention")


def _render_draft(args: Namespace, directory: Path) -> None:
    root = args.worker_root.resolve(strict=True)
    if sha256((root / "ml/explain_report.py").read_bytes()) != _FROZEN_ENTRYPOINT:
        raise ValueError("The worker root does not match the pinned report publisher.")
    python = args.worker_python or root / ".venv/bin/python"
    command = [
        str(python.absolute()),
        "-m",
        "ml.explain_report",
        "--no-upload",
        "--run-id",
        args.run_id,
        "--tracking-uri",
        args.tracking_uri,
        "--output-dir",
        str(directory),
        "--artifact-path",
        args.artifact_path,
        "--xopat-ui-uri",
        args.xopat_ui_uri,
        "--artifact-mount",
        args.artifact_mount,
        "--methods",
        *args.methods,
    ]
    if args.mlflow_ui_uri:
        command.extend(("--mlflow-ui-uri", args.mlflow_ui_uri))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(command, cwd=root, env=environment, stdout=sys.stderr, check=True)


def _publish_new(
    client: Any,
    args: Namespace,
    report: bytes,
    provenance: dict[str, Any],
) -> str:
    digest = sha256(report)
    if args.expected_output_sha256 != digest:
        raise ValueError("The HTML differs from the approved new-report preview.")
    artifact_folder = _artifact(args.report_artifact_path)
    artifact = (artifact_folder / "report.html").as_posix()
    run = client.get_run(args.run_id)
    if (
        run.info.run_id != args.run_id
        or str(run.info.experiment_id) != args.experiment_id
        or run.info.status != "FINISHED"
        or run.info.lifecycle_stage != "active"
        or provenance.get("explanation_run_id") != args.run_id
        or provenance.get("source_run_id") == args.run_id
        or not re.fullmatch(r"[a-f0-9]{32}", provenance.get("source_run_id") or "")
        or not re.fullmatch(r"[a-f0-9]{64}", provenance.get("checkpoint_sha256") or "")
        or args.run_id.encode() not in report
        or run.data.tags.get("report.artifact") not in {None, artifact}
    ):
        raise ValueError(
            "The selected run is not the active, finished explanation run."
        )
    mount = _artifact(args.artifact_mount)
    if len(mount.parts) != 1:
        raise ValueError("Expected one xOpat artifact mount component.")
    xopat = validate_tracking_uri(args.xopat_ui_uri)
    folder = mount / args.experiment_id / args.run_id / "artifacts" / artifact_folder
    url = (
        urljoin(xopat + "/", "index.php")
        + "?"
        + urlencode({"p": folder.as_posix(), "view": "report.html"})
    )
    with tempfile.TemporaryDirectory() as work:
        directory = Path(work)
        manifest_path = _artifact(args.artifact_path) / "manifest.json"
        verified_manifest = client.download_artifacts(
            args.run_id, manifest_path.as_posix(), str(directory)
        )
        remote_provenance = json.loads(Path(verified_manifest).read_bytes())
        if any(
            remote_provenance.get(field) != provenance.get(field)
            for field in (
                "explanation_run_id",
                "source_run_id",
                "checkpoint_sha256",
                "split",
            )
        ):
            raise ValueError("The live explanation manifest differs from the preview.")
        exists = any(
            item.path == artifact
            for item in client.list_artifacts(args.run_id, artifact_folder.as_posix())
        )
        if exists:
            path = client.download_artifacts(args.run_id, artifact, str(directory))
            if sha256(Path(path).read_bytes()) != digest:
                raise ValueError("The report already exists. Use the backfill command.")
        elif run.data.tags.get("report.artifact"):
            raise ValueError("The run already declares a published report artifact.")
        else:
            html_path = directory / "report.html"
            html_path.write_bytes(report)
            client.log_artifact(args.run_id, str(html_path), artifact_folder.as_posix())
            downloaded = directory / "read-back"
            downloaded.mkdir()
            path = client.download_artifacts(args.run_id, artifact, str(downloaded))
            if sha256(Path(path).read_bytes()) != digest:
                raise RuntimeError(
                    "The new report upload failed read-back verification."
                )
    for key, value in {
        "report.artifact": artifact,
        "report.template": "RationAI report_template.html",
        "report.methods": json.dumps(
            [
                method
                for method in args.methods
                if method in provenance.get("methods", ())
            ]
        ),
        "report.status": "ready",
        "pathologist_review_url": url,
    }.items():
        if run.data.tags.get(key) != value:
            client.set_tag(args.run_id, key, value)
    return url


def _render(
    args: Namespace, base: Path, manifest: Path, slides: Path | None, staging: Path
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    provenance = json.loads(manifest.read_bytes())
    if not args.tile_removal_bundle:
        return provenance, base.read_bytes(), {}
    slide_ids = (
        report_slide_ids(slides.read_bytes())
        if slides and provenance.get("split") == "val"
        else None
    )
    preview = staging / "composed.html"
    receipt = backfill_report(
        LocalReport(base),
        args.tile_removal_bundle,
        provenance,
        slide_ids=slide_ids,
        explanation_run_id=args.run_id,
        preview=preview,
        review_findings=(
            args.review_findings.read_text(encoding="utf-8")
            if args.review_findings
            else None
        ),
    )
    return provenance, preview.read_bytes(), receipt


def _write_new(path: Path, data: bytes) -> None:
    with os.fdopen(
        os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
    ) as file:
        file.write(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-report", type=Path)
    source.add_argument("--worker-root", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--report-manifest", type=Path)
    parser.add_argument("--report-slides", type=Path)
    parser.add_argument("--tile-removal-bundle", type=Path)
    parser.add_argument("--review-findings", type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-id")
    parser.add_argument("--artifact-path", default="explanations")
    parser.add_argument("--report-artifact-path", default="report")
    parser.add_argument("--xopat-ui-uri", default=_DEFAULT_XOPAT)
    parser.add_argument("--mlflow-ui-uri")
    parser.add_argument("--artifact-mount", default="secure_mlflow")
    parser.add_argument("--methods", nargs="+", default=_DEFAULT_METHODS)
    upload = parser.add_mutually_exclusive_group()
    upload.add_argument(
        "--no-upload", action="store_true", help="Render locally (default)."
    )
    upload.add_argument(
        "--apply", action="store_true", help="Publish a fresh MLflow artifact."
    )
    parser.add_argument("--expected-output-sha256")
    args = parser.parse_args(argv)
    if args.review_findings and not args.tile_removal_bundle:
        parser.error("--review-findings requires --tile-removal-bundle.")
    if args.worker_root and (args.report_manifest or args.report_slides):
        parser.error("The frozen worker supplies the report manifest and slide roster.")
    if args.base_report and not args.report_manifest:
        parser.error("--base-report requires --report-manifest.")
    if args.run_id and not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
        parser.error("Expected an exact MLflow run ID.")
    if (args.worker_root or args.apply) and not (args.run_id and args.tracking_uri):
        parser.error("Specify --run-id and --tracking-uri.")
    if args.apply and not (args.experiment_id and args.expected_output_sha256):
        parser.error("--apply requires --experiment-id and --expected-output-sha256.")
    if args.worker_python and not args.worker_root:
        parser.error("--worker-python requires --worker-root.")
    if args.experiment_id and not re.fullmatch(r"[0-9]+", args.experiment_id):
        parser.error("Expected an exact MLflow experiment ID.")
    if args.tracking_uri:
        args.tracking_uri = validate_tracking_uri(args.tracking_uri)
    output = args.output or (
        args.output_dir / "report.html" if args.output_dir else None
    )
    if not (output or args.apply):
        parser.error("Specify --output or --output-dir for a local render.")
    with tempfile.TemporaryDirectory() as work:
        staging = Path(work)
        if args.worker_root:
            _render_draft(args, staging)
            base, manifest, slides = (
                staging / "report.html",
                staging / "manifest.json",
                staging / "slides.parquet",
            )
        else:
            base, manifest, slides = (
                args.base_report,
                args.report_manifest,
                args.report_slides,
            )
        provenance, report, details = _render(args, base, manifest, slides, staging)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_new(output, report)
            if args.worker_root and args.output_dir:
                _write_new(args.output_dir / manifest.name, manifest.read_bytes())
                _write_new(args.output_dir / slides.name, slides.read_bytes())
        print_result = {**details, "output_sha256": sha256(report)}
        if output is not None:
            print_result["report"] = str(output.resolve())
        if args.apply:
            from mlflow import MlflowClient

            print_result["pathologist_review_url"] = _publish_new(
                MlflowClient(tracking_uri=args.tracking_uri), args, report, provenance
            )
    print(json.dumps(print_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
