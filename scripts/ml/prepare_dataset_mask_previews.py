"""Generate dataset mask previews and optionally stage them beside existing reports."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml.explainability.dataset_mask_report import LAYER_ASSETS
from scripts.ml.backfill_dataset_mask_reports import main as preview_report
from scripts.ml.backfill_tile_removal_reports import MlflowReport, validate_tracking_uri


if TYPE_CHECKING:
    from collections.abc import Sequence


REPORT = "report/report.html"


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"([0-9a-f]{32}):([0-9a-f]{64})", value)
    if match is None:
        raise argparse.ArgumentTypeError("Expected explanation-run-id:original-sha256.")
    return match[1], match[2]


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
) -> dict[str, Any]:
    folder = f"report_backfills/dataset_masks/previews/{batch}"
    viewer_path = f"secure_mlflow/{experiment}/{run}/artifacts/{folder}"
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
                    "--tracking-uri",
                    tracking_uri,
                    "--write-layer-manifest",
                    str(layers),
                    "--preview",
                    str(preview),
                    "--mask-export-dir",
                    str(root / LAYER_ASSETS),
                    "--mask-export-viewer-prefix",
                    f"{viewer_path}/{LAYER_ASSETS}",
                ]
            )
        result = json.loads(detail.getvalue())
        result["target"] = store.target
        result["stage"] = folder if stage else None
        result["run_id"] = run
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
        if stage:
            exports = sorted((root / LAYER_ASSETS).glob("*.ome.tif"))
            for path in exports:
                if path.name != f"{_hash(path)}.ome.tif":
                    raise ValueError(
                        "A converted mask differs from its content address."
                    )
            result["converted_files"] = len(exports)
            print(
                json.dumps(
                    {"phase": "staging", "run_id": run, "converted_files": len(exports)}
                ),
                flush=True,
            )
            _stage(client, run, f"{folder}/{LAYER_ASSETS}", exports)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            _stage(client, run, folder, [preview, layers, receipt])
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
    args = parser.parse_args(argv)
    if not args.tracking_uri or not re.fullmatch(r"[0-9]+", args.experiment_id):
        parser.error("Use managed MLflow tracking and an exact experiment ID.")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?", args.batch):
        parser.error(
            "Use a batch label of up to 50 lowercase alphanumeric or dash characters."
        )
    if len({run for run, _ in args.run}) != len(args.run):
        parser.error("Each selected report must occur once.")

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
