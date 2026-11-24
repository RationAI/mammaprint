"""Build a new tile-explanation HTML report from a RationAI report draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urljoin

from ml.explainability.dataset_mask_report import (
    DEFAULT_XOPAT,
    LAYER_ASSETS,
    LAYER_MANIFEST,
    layer_manifest_bytes,
    patch_report_layers,
    prepare_layer_manifest,
    report_xopat_uri,
)
from ml.explainability.tile_removal_report import sha256
from scripts.ml.backfill_dataset_mask_reports import (
    add_mask_options,
    mask_preparation_options,
    validate_mask_options,
)
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
        args.xopat_ui_uri or DEFAULT_XOPAT,
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
    *,
    layer_manifest: dict[str, Any] | None = None,
    staged_assets: Path | None = None,
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
        or str(provenance.get("explanation_experiment_id", args.experiment_id))
        != args.experiment_id
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
    xopat = validate_tracking_uri(
        report_xopat_uri(
            provenance, args.xopat_ui_uri or (layer_manifest or {}).get("xopat_ui_uri")
        )
    )
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
                "level",
                "config_hash",
            )
        ):
            raise ValueError("The live explanation manifest differs from the preview.")

        def existing_report() -> bool:
            declared = client.get_run(args.run_id).data.tags.get("report.artifact")
            if declared not in {None, artifact}:
                raise ValueError(
                    "The run declares a different published report artifact."
                )
            if any(
                item.path == artifact
                for item in client.list_artifacts(
                    args.run_id, artifact_folder.as_posix()
                )
            ):
                with tempfile.TemporaryDirectory(dir=directory) as readback:
                    path = client.download_artifacts(args.run_id, artifact, readback)
                    if _file_digest(Path(path)) != digest:
                        raise ValueError(
                            "The report already exists. Use the backfill command."
                        )
                return True
            if declared:
                raise ValueError(
                    "The run already declares a published report artifact."
                )
            return False

        exists = existing_report()
        if layer_manifest and layer_manifest["assets"]:
            remote_slides = client.download_artifacts(
                args.run_id, f"{args.artifact_path}/slides.parquet", str(directory)
            )
            if _file_digest(Path(remote_slides)) != layer_manifest["slides_sha256"]:
                raise ValueError(
                    "The live report slide table differs from the preview."
                )
            _publish_staged_masks(
                client,
                args.run_id,
                artifact_folder.as_posix(),
                folder.as_posix(),
                layer_manifest,
                staged_assets,
                directory,
            )
            if not exists:
                exists = existing_report()
        if not exists:
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


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _staged_mask_files(
    run_id: str, viewer_folder: str, manifest: dict[str, Any]
) -> dict[str, str]:
    prefix = f"{viewer_folder}/{LAYER_ASSETS}/"
    run_root, separator, _ = viewer_folder.partition(f"/{run_id}/artifacts/")
    if not separator:
        raise ValueError("Invalid xOpat artifact folder for the selected run.")
    run_root += separator
    staged_files: dict[str, str] = {}
    for layers in manifest["assets"].values():
        for layer in layers.values():
            path = layer["viewer_path"]
            if not path.startswith(prefix):
                if path.startswith(run_root) and f"/{LAYER_ASSETS}/" in path:
                    raise ValueError(
                        "The staged dataset masks belong to a different report artifact path."
                    )
                continue
            name, digest = path.removeprefix(prefix), layer["sha256"]
            if name != f"{digest}.ome.tif":
                raise ValueError(
                    "A staged dataset mask filename does not match its digest."
                )
            staged_files[name] = digest
    return staged_files


def _publish_staged_masks(
    client: Any,
    run_id: str,
    artifact_folder: str,
    viewer_folder: str,
    manifest: dict[str, Any],
    staged: Path | None,
    scratch: Path,
) -> None:
    staged_files = _staged_mask_files(run_id, viewer_folder, manifest)
    if not staged_files:
        return

    folder = f"{artifact_folder}/{LAYER_ASSETS}"
    existing = {item.path for item in client.list_artifacts(run_id, folder)}
    for name, digest in sorted(staged_files.items()):
        artifact = f"{folder}/{name}"
        if artifact not in existing:
            local = staged / name if staged else None
            if (
                local is None
                or local.is_symlink()
                or not local.is_file()
                or _file_digest(local) != digest
            ):
                raise ValueError(
                    "A staged dataset mask is missing or differs from the preview."
                )
            client.log_artifact(run_id, str(local), folder)
        with tempfile.TemporaryDirectory(dir=scratch) as downloaded:
            path = Path(client.download_artifacts(run_id, artifact, downloaded))
            if _file_digest(path) != digest:
                raise ValueError("The published dataset mask differs from the preview.")


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


def _draft_viewer_folder(args: Namespace, provenance: dict[str, Any]) -> str | None:
    experiment = str(provenance.get("explanation_experiment_id"))
    run = provenance.get("explanation_run_id")
    if not re.fullmatch(r"[0-9]+", experiment) or not (
        isinstance(run, str) and re.fullmatch(r"[0-9a-f]{32}", run)
    ):
        return None
    mount = _artifact(args.artifact_mount).as_posix()
    folder = _artifact(args.report_artifact_path).as_posix()
    return f"{mount}/{experiment}/{run}/artifacts/{folder}"


def _copy_draft_assets(
    source: Path,
    destination: Path,
    run: str,
    viewer_folder: str,
    manifest: dict[str, Any],
) -> None:
    if source.resolve() == destination.resolve() or not source.is_dir():
        return
    files = _staged_mask_files(run, viewer_folder, manifest)
    for name, digest in files.items():
        old, new = source / name, destination / name
        if new.is_file() and not new.is_symlink() and _file_digest(new) == digest:
            continue
        if old.is_symlink() or not old.is_file():
            raise ValueError("A staged dataset mask is missing from the source draft.")
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as work:
            copy = Path(work) / name
            shutil.copyfile(old, copy)
            if _file_digest(copy) != digest:
                raise ValueError("A staged dataset mask differs from the preview.")
            os.link(copy, new)


def _dataset_masks(
    args: Namespace,
    client: Any,
    provenance: dict[str, Any],
    report: bytes,
    slides: Path | None,
    staged_assets: Path,
    viewer_folder: str | None,
) -> tuple[bytes, dict[str, Any], dict[str, Any] | None]:
    if args.layer_manifest is None and provenance.get("level") is None:
        return report, {"unresolved": "The report has no dataset level metadata."}, None
    root = _artifact(args.artifact_path).as_posix()
    if slides is None:
        if args.layer_manifest:
            raise ValueError(
                "A prepared layer manifest requires the exact report slide table."
            )
        return report, {"unresolved": "The report slide table is unavailable."}, None

    slides_data = slides.read_bytes()
    if args.layer_manifest:
        manifest = json.loads(args.layer_manifest.read_bytes())
    else:
        prep = mask_preparation_options(args)
        if client is not None and not prep["export_dir"] and viewer_folder:
            prep.update(
                export_dir=staged_assets,
                export_viewer_prefix=f"{viewer_folder}/{LAYER_ASSETS}",
            )
        manifest = prepare_layer_manifest(
            client,
            provenance,
            slides_data,
            artifact_root=root,
            artifact_mount=args.artifact_mount,
            **prep,
        )
    if not isinstance(manifest, dict):
        raise ValueError("Invalid dataset mask layer manifest.")
    xopat = report_xopat_uri(
        provenance, args.xopat_ui_uri or manifest.get("xopat_ui_uri")
    )
    manifest["xopat_ui_uri"] = xopat
    updated, details = patch_report_layers(
        report,
        provenance,
        slides_data,
        manifest,
        xopat_ui_uri=xopat,
        artifact_root=root,
        artifact_mount=args.artifact_mount,
    )
    return updated, details, manifest


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
    add_mask_options(parser)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-id")
    parser.add_argument("--artifact-path", default="explanations")
    parser.add_argument("--report-artifact-path", default="report")
    parser.add_argument("--xopat-ui-uri")
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
    validate_mask_options(parser, args)
    if args.worker_root and not args.xopat_ui_uri:
        args.xopat_ui_uri = DEFAULT_XOPAT
    if args.base_report and not any(
        (
            args.layer_manifest,
            args.mask_wsi_root,
            args.mask_source_root,
            args.mask_export_dir,
            args.report_config,
        )
    ):
        layers = args.base_report.with_name(LAYER_MANIFEST)
        if layers.is_file():
            args.layer_manifest = layers
    if args.layer_manifest and args.base_report and not args.report_slides:
        sibling_slides = args.layer_manifest.with_name("slides.parquet")
        if sibling_slides.is_file():
            args.report_slides = sibling_slides
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
        client = None
        if args.tracking_uri and (
            args.apply
            or (provenance.get("level") is not None and not args.layer_manifest)
            or (args.layer_manifest and slides is None)
        ):
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=args.tracking_uri)
        if (
            slides is None
            and client is not None
            and args.run_id
            and (args.layer_manifest or provenance.get("level") is not None)
        ):
            root = _artifact(args.artifact_path).as_posix()
            slides = Path(
                client.download_artifacts(
                    args.run_id, f"{root}/slides.parquet", str(staging)
                )
            )
        asset_parent = (
            args.layer_manifest.parent
            if args.layer_manifest
            else output.parent
            if output is not None
            else staging
        )
        staged_assets = asset_parent / LAYER_ASSETS
        viewer_folder = _draft_viewer_folder(args, provenance)
        report, mask_details, layer_manifest = _dataset_masks(
            args, client, provenance, report, slides, staged_assets, viewer_folder
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            if layer_manifest is not None and args.output_dir and viewer_folder:
                _copy_draft_assets(
                    staged_assets,
                    args.output_dir / LAYER_ASSETS,
                    provenance["explanation_run_id"],
                    viewer_folder,
                    layer_manifest,
                )
            _write_new(output, report)
            if args.output_dir and (args.worker_root or layer_manifest is not None):
                _write_new(args.output_dir / "manifest.json", manifest.read_bytes())
                if slides is not None:
                    _write_new(args.output_dir / "slides.parquet", slides.read_bytes())
            if layer_manifest is not None and args.output_dir:
                _write_new(
                    args.output_dir / LAYER_MANIFEST,
                    layer_manifest_bytes(layer_manifest),
                )
        print_result = {
            **details,
            "dataset_masks": mask_details,
            "output_sha256": sha256(report),
        }
        if output is not None:
            print_result["report"] = str(output.resolve())
        if args.apply:
            print_result["pathologist_review_url"] = _publish_new(
                client,
                args,
                report,
                provenance,
                layer_manifest=layer_manifest,
                staged_assets=staged_assets,
            )
    print(json.dumps(print_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
