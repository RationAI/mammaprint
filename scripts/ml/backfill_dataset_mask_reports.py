"""Preview, apply, or roll back dataset overlays in existing doctor reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml.explainability.dataset_mask_report import (
    layer_manifest_bytes,
    patch_report_layers,
    prepare_layer_manifest,
    report_xopat_uri,
)
from ml.explainability.tile_removal_report import sha256
from scripts.ml.backfill_tile_removal_reports import (
    LocalReport,
    MlflowReport,
    _artifact,
    finish_report_update,
    rollback_report,
    validate_tracking_uri,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from scripts.ml.backfill_tile_removal_reports import ReportStore


_KIND = "dataset_masks"


def add_mask_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layer-manifest", type=Path, help="Reuse a prepared layer manifest."
    )
    parser.add_argument(
        "--mask-wsi-root", type=Path, help="Local mirror of the WSI mount /mnt."
    )
    parser.add_argument(
        "--mask-source-root", type=Path, help="Local mirror of /mnt for mask files."
    )
    parser.add_argument(
        "--mask-export-dir", type=Path, help="Local directory for converted TIFFs."
    )
    parser.add_argument(
        "--mask-export-viewer-prefix", help="xOpat mount path for converted TIFFs."
    )
    parser.add_argument(
        "--report-config",
        type=Path,
        help="Exact resolved config for historical recovery.",
    )


def validate_mask_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if bool(args.mask_export_dir) != bool(args.mask_export_viewer_prefix):
        parser.error(
            "--mask-export-dir and --mask-export-viewer-prefix must be set together."
        )
    if args.layer_manifest and any(
        (
            args.mask_wsi_root,
            args.mask_source_root,
            args.mask_export_dir,
            args.report_config,
        )
    ):
        parser.error(
            "--layer-manifest reuses prepared assets and cannot prepare them again."
        )


def mask_preparation_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "resolved_config": args.report_config.read_bytes()
        if args.report_config
        else None,
        "wsi_root": args.mask_wsi_root,
        "mask_root": args.mask_source_root,
        "export_dir": args.mask_export_dir,
        "export_viewer_prefix": args.mask_export_viewer_prefix,
    }


def _write_new(path: Path, data: bytes) -> None:
    with os.fdopen(
        os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
    ) as file:
        file.write(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--report", type=Path)
    target.add_argument("--run-id", help="Exact explanation run ID.")
    parser.add_argument("--report-manifest", type=Path)
    parser.add_argument("--report-slides", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--artifact-path", help="Existing path from the run's report.artifact tag."
    )
    parser.add_argument("--explanation-artifact-root", default="explanations")
    parser.add_argument("--artifact-mount", default="secure_mlflow")
    parser.add_argument("--xopat-ui-uri")
    add_mask_options(parser)
    parser.add_argument("--write-layer-manifest", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--rollback", metavar="RECEIPT_ARTIFACT")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-original-sha256")
    parser.add_argument("--expected-updated-sha256")
    args = parser.parse_args(argv)
    validate_mask_options(parser, args)
    if args.rollback and any(
        (
            args.preview,
            args.write_layer_manifest,
            args.layer_manifest,
            args.report_config,
            args.mask_wsi_root,
            args.mask_source_root,
            args.mask_export_dir,
            args.expected_original_sha256,
            args.expected_updated_sha256,
        )
    ):
        parser.error("--rollback takes the target, receipt and optionally --apply.")
    if (
        not args.rollback
        and args.apply
        and not (
            args.layer_manifest
            and args.expected_original_sha256
            and args.expected_updated_sha256
        )
    ):
        parser.error(
            "--apply requires the prepared --layer-manifest and both preview hashes."
        )
    if args.report:
        if args.experiment_id or args.artifact_path:
            parser.error("--experiment-id and --artifact-path require --run-id.")
        if not args.rollback and not (args.report_manifest and args.report_slides):
            parser.error(
                "A local backfill requires --report-manifest and --report-slides."
            )
    elif not (args.tracking_uri and args.experiment_id and args.artifact_path):
        parser.error(
            "A published backfill requires --tracking-uri, --experiment-id and --artifact-path."
        )
    elif args.report_manifest or args.report_slides:
        parser.error("Published report metadata comes from the selected run.")

    client = None
    if args.tracking_uri:
        from mlflow import MlflowClient

        args.tracking_uri = validate_tracking_uri(args.tracking_uri)
        client = MlflowClient(tracking_uri=args.tracking_uri)
    store: ReportStore
    if args.report:
        store = LocalReport(args.report)
    else:
        store = MlflowReport(
            client,
            args.tracking_uri,
            args.experiment_id,
            args.run_id,
            args.artifact_path,
        )
    if args.rollback:
        result = rollback_report(store, args.rollback, apply=args.apply, kind=_KIND)
        print(json.dumps(result, indent=2))
        return 0

    root = _artifact(args.explanation_artifact_root).as_posix()
    if args.report:
        provenance = json.loads(args.report_manifest.read_bytes())
        slides = args.report_slides.read_bytes()
    else:
        provenance = json.loads(store.read(f"{root}/manifest.json"))
        slides = store.read(f"{root}/slides.parquet")
        if (
            provenance.get("explanation_run_id") != args.run_id
            or str(provenance.get("explanation_experiment_id", args.experiment_id))
            != args.experiment_id
        ):
            raise ValueError(
                "The explanation manifest does not identify the selected run."
            )
    if args.layer_manifest:
        manifest = json.loads(args.layer_manifest.read_bytes())
    else:
        manifest = prepare_layer_manifest(
            client,
            provenance,
            slides,
            artifact_root=root,
            artifact_mount=args.artifact_mount,
            **mask_preparation_options(args),
        )
    original = store.read(store.artifact)
    updated, summary = patch_report_layers(
        original,
        provenance,
        slides,
        manifest,
        xopat_ui_uri=report_xopat_uri(provenance, args.xopat_ui_uri),
        artifact_root=root,
        artifact_mount=args.artifact_mount,
    )
    if args.write_layer_manifest:
        _write_new(args.write_layer_manifest, layer_manifest_bytes(manifest))
    receipt = {
        "version": 1,
        "target": store.target,
        "explanation_run_id": provenance["explanation_run_id"],
        "source_run_id": provenance.get("source_run_id"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "report_split": provenance.get("split"),
        "original_sha256": sha256(original),
        "updated_sha256": sha256(updated),
        "bundle_sha256": sha256(layer_manifest_bytes(manifest)),
        **summary,
    }
    result = finish_report_update(
        store,
        original,
        updated,
        receipt,
        kind=_KIND,
        preview=args.preview,
        apply=args.apply,
        expected_original=args.expected_original_sha256,
        expected_updated=args.expected_updated_sha256,
    )
    if not summary["slides_with_masks"]:
        result["status"] = "no_verified_layers"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
