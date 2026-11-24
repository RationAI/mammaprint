"""Generate dataset mask previews and optionally stage them beside existing reports."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from ml.explainability.dataset_mask_assets import (
    _PREFIXES,
    _directory,
    prepare_mask_assets,
)
from ml.explainability.dataset_mask_lineage import resolve_mask_sources
from ml.explainability.dataset_mask_report import (
    LAYER_ASSETS,
    build_layer_manifest,
    layer_manifest_bytes,
    patch_report_layers,
    report_xopat_uri,
    slide_paths,
)
from scripts.ml.backfill_dataset_mask_reports import main as preview_report
from scripts.ml.backfill_tile_removal_reports import MlflowReport, validate_tracking_uri


if TYPE_CHECKING:
    from collections.abc import Sequence


REPORT = "report/report.html"
_MOUNT = PurePosixPath("secure_mlflow")
_BATCH_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"([0-9a-f]{32}):([0-9a-f]{64})", value)
    if match is None:
        raise argparse.ArgumentTypeError("Expected explanation-run-id:original-sha256.")
    return match[1], match[2]


def _stage_folder(batch: str) -> str:
    if not _BATCH_LABEL.fullmatch(batch):
        raise ValueError("Invalid preview batch label.")
    return f"report_backfills/dataset_masks/previews/{batch}"


def _read_reusable_layers(
    donor: MlflowReport,
    donor_folder: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    provenance = json.loads(donor.read("explanations/manifest.json"))
    slides = donor.read("explanations/slides.parquet")
    receipt = json.loads(donor.read(f"{donor_folder}/receipt.json"))
    manifest = donor.read(f"{donor_folder}/layers.json")
    preview = donor.read(f"{donor_folder}/preview.html")
    current_report = donor.read(donor.artifact)
    if (
        receipt.get("stage") != donor_folder
        or receipt.get("status") != "preview"
        or receipt.get("run_id") != donor.run_id
        or receipt.get("explanation_run_id") != donor.run_id
        or provenance.get("explanation_run_id") != donor.run_id
        or receipt.get("bundle_sha256") != hashlib.sha256(manifest).hexdigest()
        or receipt.get("updated_sha256") != hashlib.sha256(preview).hexdigest()
        or hashlib.sha256(current_report).hexdigest()
        not in {receipt.get("original_sha256"), receipt.get("updated_sha256")}
    ):
        raise ValueError(
            "Donor preview receipt does not verify its report and artifacts."
        )
    layers = json.loads(manifest)
    rendered, _ = patch_report_layers(
        current_report,
        provenance,
        slides,
        layers,
        xopat_ui_uri=report_xopat_uri(provenance),
        artifact_root="explanations",
        artifact_mount=str(_MOUNT),
    )
    if rendered != preview:
        raise ValueError("Donor layers do not produce the verified preview.")
    return provenance, slides, layers


def _prepare_reused_layers(
    client: Any,
    tracking_uri: str,
    experiment: str,
    provenance: dict[str, Any],
    slides: bytes,
    reuse: tuple[str, str],
    export_dir: Path,
    export_prefix: str,
) -> dict[str, Any]:
    donor_run, donor_batch = reuse
    if donor_run == provenance["explanation_run_id"]:
        raise ValueError("The reuse donor must be a different explanation run.")
    donor_folder = _stage_folder(donor_batch)
    donor = MlflowReport(client, tracking_uri, experiment, donor_run, REPORT)
    donor_provenance, donor_slides, donor_layers = _read_reusable_layers(
        donor, donor_folder
    )
    targets = slide_paths(slides)
    if targets != slide_paths(donor_slides):
        raise ValueError("The donor and target slide rosters differ.")
    basenames = {
        slide: PurePosixPath(path).with_suffix(".tiff").name
        for slide, path in targets.items()
    }
    if len(set(basenames.values())) != len(basenames):
        raise ValueError("The shared roster has ambiguous mask filenames.")
    donor_sources = resolve_mask_sources(client, donor_provenance, None)
    target_sources = resolve_mask_sources(client, provenance, None)
    if (
        donor_sources.unresolved
        or target_sources.unresolved
        or donor_sources.dataset_uri != donor_layers["dataset_uri"]
    ):
        raise ValueError("Could not verify the saved source lineage for mask reuse.")
    for kind, selection in donor_layers["selections"].items():
        source = donor_sources.layers.get(kind)
        if source is None or (
            source.selection != selection["selection"]
            or source.evidence != selection["evidence"]
        ):
            raise ValueError("Donor mask selections do not match its saved lineage.")
    folders = {}
    for kind in donor_sources.layers.keys() & target_sources.layers.keys():
        source = donor_sources.layers[kind]
        if source.location != target_sources.layers[kind].location:
            raise ValueError(
                f"The donor and target mask source locations differ for {kind}."
            )
        folders[kind] = _directory(client, source.location, _MOUNT, None)

    exports = f"{donor_folder}/{LAYER_ASSETS}"
    donor_inventory = {
        item.path
        for item in client.list_artifacts(donor_run, exports)
        if not item.is_dir
    }
    prefix = f"{_MOUNT}/{experiment}/{donor_run}/artifacts"
    assets: dict[str, dict[str, dict[str, str]]] = {}
    for slide, items in donor_layers["assets"].items():
        for kind, asset in items.items():
            if kind not in folders:
                continue
            directory = folders[kind]
            name = _PREFIXES[kind] + basenames[slide]
            if asset["source"] != f"{directory.source}/{name}":
                raise ValueError("Donor mask does not match the slide's source TIFF.")
            if asset["viewer_path"] != str(directory.viewer / name):
                artifact = f"{exports}/{asset['sha256']}.ome.tif"
                if (
                    asset["viewer_path"] != f"{prefix}/{artifact}"
                    or artifact not in donor_inventory
                ):
                    raise ValueError(
                        "Donor converted mask is absent from the pinned export inventory."
                    )
            assets.setdefault(slide, {})[kind] = asset

    groups: dict[tuple[str, ...], dict[str, str]] = defaultdict(dict)
    for slide, path in targets.items():
        needed = tuple(
            sorted(target_sources.layers.keys() - assets.get(slide, {}).keys())
        )
        if needed:
            groups[needed][slide] = path
    missing: dict[str, dict[str, str]] = {}
    for kinds, paths in groups.items():
        prepared, unavailable = prepare_mask_assets(
            client,
            {kind: target_sources.layers[kind] for kind in kinds},
            paths,
            artifact_mount=str(_MOUNT),
            export_dir=export_dir,
            export_viewer_prefix=export_prefix,
        )
        for slide, items in prepared.items():
            assets.setdefault(slide, {}).update(items)
        missing.update(unavailable)
    return build_layer_manifest(
        provenance,
        slides,
        dataset_uri=target_sources.dataset_uri,
        assets=assets,
        missing=missing,
        selections={
            kind: {"selection": source.selection, "evidence": source.evidence}
            for kind, source in target_sources.layers.items()
        },
    )


def _stage(client: Any, run: str, folder: str, files: list[Path]) -> None:
    if not files:
        return
    existing = {item.path for item in client.list_artifacts(run, folder)}
    for path in files:
        artifact = f"{folder}/{path.name}"
        expected = _hash(path)
        if artifact not in existing:
            client.log_artifact(run, str(path), folder)
        with tempfile.TemporaryDirectory() as work:
            uploaded = Path(client.download_artifacts(run, artifact, work))
            if _hash(uploaded) != expected:
                raise RuntimeError(
                    f"Staged artifact failed checksum verification: {artifact}"
                )


def prepare(
    client: Any,
    tracking_uri: str,
    experiment: str,
    run: str,
    original_sha: str,
    batch: str,
    *,
    stage: bool,
    reuse: tuple[str, str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    folder = _stage_folder(batch)
    viewer_path = f"{_MOUNT}/{experiment}/{run}/artifacts/{folder}"
    store = MlflowReport(client, tracking_uri, experiment, run, REPORT)
    with tempfile.TemporaryDirectory(prefix="mammaprint-mask-preview-") as work:
        root = Path(work)
        original = root / "report.html"
        slide_table = root / "slides.parquet"
        manifest = root / "manifest.json"
        layers = root / "layers.json"
        preview = root / "preview.html"
        original.write_bytes(store.read(store.artifact))
        if _hash(original) != original_sha:
            raise ValueError("The published report changed after the local preflight.")
        manifest.write_bytes(store.read("explanations/manifest.json"))
        provenance = json.loads(manifest.read_bytes())
        if (
            provenance.get("explanation_run_id") != run
            or str(provenance.get("explanation_experiment_id", experiment))
            != experiment
        ):
            raise ValueError("Explanation manifest does not match the preview target.")
        slide_table.write_bytes(store.read("explanations/slides.parquet"))
        print(json.dumps({"phase": "preparing", "run_id": run}), flush=True)
        export_dir = root / LAYER_ASSETS
        export_prefix = f"{viewer_path}/{LAYER_ASSETS}"
        if reuse is None:
            layer_args = [
                "--tracking-uri",
                tracking_uri,
                "--mask-export-dir",
                str(export_dir),
                "--mask-export-viewer-prefix",
                export_prefix,
            ]
        else:
            prepared = _prepare_reused_layers(
                client,
                tracking_uri,
                experiment,
                provenance,
                slide_table.read_bytes(),
                reuse,
                export_dir,
                export_prefix,
            )
            previous = root / "reuse-layers.json"
            previous.write_bytes(layer_manifest_bytes(prepared))
            layer_args = ["--layer-manifest", str(previous)]
        detail = io.StringIO()
        with redirect_stdout(detail):
            preview_report(
                [
                    "--report",
                    str(original),
                    "--report-manifest",
                    str(manifest),
                    "--report-slides",
                    str(slide_table),
                    "--write-layer-manifest",
                    str(layers),
                    "--preview",
                    str(preview),
                    *layer_args,
                ]
            )
        result = json.loads(detail.getvalue())
        result["target"] = store.target
        result["stage"] = folder if stage else None
        result["run_id"] = run
        if reuse is not None:
            result["reuse_layer_run"], result["reuse_layer_batch"] = reuse
        result["missing_reasons"] = dict(
            Counter(
                reason
                for missing in json.loads(layers.read_bytes())["missing"].values()
                for reason in missing.values()
            )
        )
        if result["original_sha256"] != original_sha or (
            hashlib.sha256(store.read(store.artifact)).hexdigest() != original_sha
        ):
            raise ValueError("The published report changed during preview preparation.")
        if result["updated_sha256"] != _hash(preview):
            raise ValueError("Preview HTML differs from the backfill receipt.")
        exports = sorted(export_dir.glob("*.ome.tif"))
        for path in exports:
            if path.name != f"{_hash(path)}.ome.tif":
                raise ValueError("A converted mask differs from its content address.")
        if exports and output_dir is not None and not stage:
            raise ValueError(
                "Use --stage before saving a preview that links new converted assets."
            )
        result["converted_files"] = len(exports)
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if stage:
            print(
                json.dumps(
                    {"phase": "staging", "run_id": run, "converted_files": len(exports)}
                ),
                flush=True,
            )
            _stage(client, run, f"{folder}/{LAYER_ASSETS}", exports)
            _stage(client, run, folder, [preview, layers, receipt])
        if output_dir is not None:
            output = output_dir / run
            output.mkdir(mode=0o700, parents=True)
            for path in (preview, layers, receipt):
                shutil.copyfile(path, output / path.name)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument(
        "--stage", action="store_true", help="Write only new preview artifacts."
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--reuse-layer-run")
    parser.add_argument("--reuse-layer-batch")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.tracking_uri or not re.fullmatch(r"[0-9]+", args.experiment_id):
        parser.error("Use managed MLflow tracking and an exact experiment ID.")
    if not _BATCH_LABEL.fullmatch(args.batch):
        parser.error(
            "Use a batch label of up to 50 lowercase alphanumeric or dash characters."
        )
    if len({run for run, _ in args.run}) != len(args.run):
        parser.error("Each selected report must occur once.")
    reuse = None
    if args.reuse_layer_run or args.reuse_layer_batch:
        if not re.fullmatch(
            r"[0-9a-f]{32}", args.reuse_layer_run or ""
        ) or not _BATCH_LABEL.fullmatch(args.reuse_layer_batch or ""):
            parser.error(
                "Supply both a valid --reuse-layer-run and --reuse-layer-batch."
            )
        reuse = args.reuse_layer_run, args.reuse_layer_batch

    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    uri = validate_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=uri)
    failures = 0
    for run, original_sha in args.run:
        try:
            result = prepare(
                client,
                uri,
                args.experiment_id,
                run,
                original_sha,
                args.batch,
                stage=args.stage,
                reuse=reuse,
                output_dir=args.output_dir,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        except (ValueError, RuntimeError, OSError) as exc:
            failures += 1
            print(json.dumps({"run_id": run, "error": str(exc)}), flush=True)
        except MlflowException as exc:
            failures += 1
            print(
                json.dumps({"run_id": run, "mlflow_error_code": exc.error_code}),
                flush=True,
            )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
