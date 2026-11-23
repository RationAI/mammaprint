"""Run targeted tile deletion on a pinned validation cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import torch

from ml.explainability.tile_removal_bundle import (
    choose_review,
    markdown_summary,
    paired_summary,
    summarize,
)
from ml.explainability.tile_removal_evaluate import evaluate_slide
from ml.explainability.tile_removal_inputs import (
    fetch,
    prepare_model,
    sha256_file,
    verify_model_spec,
)
from ml.explainability.tile_removal_report import (
    render_blinded_review,
    render_standalone_report,
)
from ml.explainability.tile_removal_viewer import ViewerTarget


DEFAULTS = {
    "seed": 20261123,
    "control_repeats": 20,
    "area_tolerance": 0.10,
    "coverage_tolerance": 0.10,
    "distance_bands": [0.05, 0.15],
    "review_slides": 24,
    "tissue_mask_threshold": 0.5,
    "tumor_mask_threshold": 0.5,
    "index_band": None,
    "index_time_limit": 10.0,
}
LOG = logging.getLogger(__name__)


def _json(data: object) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _validate(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("split") != "val" or not plan.get("models"):
        raise ValueError(
            "An explicit validation plan with at least one pinned model is required."
        )
    for spec in plan["models"]:
        verify_model_spec(spec)
    ids = {spec["model_id"] for spec in plan["models"]}
    identities = {
        (spec["source_run_id"], spec["checkpoint_sha256"]) for spec in plan["models"]
    }
    if len(ids) != len(plan["models"]) or len(identities) != len(ids):
        raise ValueError("Model aliases and checkpoint identities must be unique.")
    raw = plan.get("options", {})
    allowed = DEFAULTS.keys() | {"wsi_root", "tissue_mask_dir", "tumor_mask_dir"}
    if unknown := set(raw) - allowed:
        raise ValueError(f"Unknown experiment options: {sorted(unknown)}.")
    options = {**DEFAULTS, **raw}
    for key in ("seed", "control_repeats", "review_slides"):
        if type(options[key]) is not int or options[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in (
        "area_tolerance",
        "coverage_tolerance",
        "tissue_mask_threshold",
        "tumor_mask_threshold",
    ):
        if not 0 < options[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
    bands = options["distance_bands"]
    if (
        not bands
        or bands != sorted(set(bands))
        or not all(math.isfinite(x) and x > 0 for x in bands)
    ):
        raise ValueError(
            "Distance boundaries must be finite, positive and strictly increasing."
        )
    if options["index_band"] is not None and not (
        math.isfinite(options["index_band"]) and options["index_band"] > 0
    ):
        raise ValueError("The index band radius must be positive.")
    if not (
        math.isfinite(options["index_time_limit"]) and options["index_time_limit"] > 0
    ):
        raise ValueError("Index search time limit must be positive.")
    return options


def _publish_overlays(
    target: ViewerTarget, models: list[dict[str, Any]], directory: Path
) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=target.tracking_uri)
    run = client.get_run(target.run_id)
    if (
        run.info.run_id != target.run_id
        or str(run.info.experiment_id) != target.experiment_id
        or run.info.lifecycle_stage != "active"
        or run.data.tags.get("targeted_tile_removal.role") != "validation-overlays"
        or run.info.status != "RUNNING"
    ):
        raise ValueError(
            "Expected the explicit active validation-overlay run in the stated experiment."
        )
    if any(model["source_run_id"] == target.run_id for model in models):
        raise ValueError("The overlay destination cannot be a source training run.")

    LOG.info(
        "Uploading overlays under run %s, path %s", target.run_id, target.artifact_root
    )
    remote_directory = f"{target.artifact_root}/overlays"
    artifacts = {
        path.path for path in client.list_artifacts(target.run_id, remote_directory)
    }
    with tempfile.TemporaryDirectory(prefix="verify-overlays-") as workdir:
        for model in models:
            for slide in model["slides"]:
                wsi = slide.pop("_wsi")
                for mask in slide["masks"]:
                    relative = mask["overlay_path"]
                    if not relative:
                        continue
                    try:
                        url = target.link(wsi, relative)
                    except ValueError:
                        mask["viewer_status"] = "wsi_not_mounted"
                        continue
                    local = directory / relative
                    artifact = f"{target.artifact_root}/{relative}"
                    if artifact not in artifacts:
                        client.log_artifact(target.run_id, str(local), remote_directory)
                    downloaded = Path(
                        client.download_artifacts(target.run_id, artifact, workdir)
                    )
                    if sha256_file(downloaded) != sha256_file(local):
                        raise RuntimeError(
                            f"Overlay read-back mismatch for {artifact}."
                        )
                    downloaded.unlink()
                    mask["viewer_url"] = url
                    mask["viewer_status"] = "published"


def _verify_local_inputs(previous: dict[str, Any], output: Path, root: Path) -> None:
    rows = json.loads((output / "report_data.json").read_bytes())["models"]
    by_id = {model["model_id"]: model["slides"] for model in rows}
    for spec in previous["models"]:
        for field in ("checkpoint", "config", "label_mapping"):
            location = spec[field]
            if urlsplit(location).scheme in {"", "file"}:
                path = fetch(location, root, output)
                if not path.is_file() or sha256_file(path) != spec[f"{field}_sha256"]:
                    raise ValueError(
                        f"Local {field} changed since the output snapshot."
                    )
        source = spec["validation_source"]
        if urlsplit(source).scheme not in {"", "file"}:
            continue
        directory = fetch(source, root, output)
        for row in by_id[spec["model_id"]]:
            bag = directory / f"{row['slide_id']}.parquet"
            recorded = row.get("bag_sha256")
            if recorded:
                changed = not bag.is_file() or sha256_file(bag) != recorded
            else:
                changed = row["status"] == "missing_embedding" and bag.exists()
            if changed:
                raise ValueError(
                    f"Validation bag changed since the output snapshot: {bag.name}."
                )


def _write_bundle(
    work: Path,
    manifest: dict[str, Any],
    models: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    options: dict[str, Any],
) -> None:
    all_rows = [row for model in models for row in model["slides"]]
    all_masks = [
        {"model_id": row["model_id"], "slide_id": row["slide_id"], **mask}
        for row in all_rows
        for mask in row["masks"]
    ]
    for row in all_rows:
        row.pop("_wsi", None)
    if all_masks:
        pd.DataFrame(all_masks).to_parquet(work / "masks.parquet", index=False)
    else:
        pd.DataFrame(columns=["model_id", "slide_id", "mask_id"]).to_parquet(
            work / "masks.parquet", index=False
        )
    slide_records = []
    for row in all_rows:
        record = {k: v for k, v in row.items() if k not in {"masks", "controls"}}
        for family in ("exact", "compact"):
            for field in ("matched", "requested", "seed"):
                record[f"{family}_control_{field}"] = (
                    row.get("controls", {}).get(family, {}).get(field)
                )
        slide_records.append(record)
    pd.DataFrame(slide_records).to_parquet(work / "slides.parquet", index=False)
    result = _json({"models": models, "paired": paired})
    (work / "report_data.json").write_bytes(result)
    (work / "summary.md").write_text(markdown_summary(models, paired), encoding="utf-8")
    review, unblinding = choose_review(models, options)
    for model in models:
        if not model["slides"]:
            continue
        destination = work / model["model_id"] / "report.html"
        destination.parent.mkdir()
        destination.write_text(render_standalone_report(model), encoding="utf-8")
    if review:
        (work / "blinded_review.html").write_text(
            render_blinded_review(review), encoding="utf-8"
        )
        with (work / "unblinding_key.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(unblinding[0]))
            writer.writeheader()
            writer.writerows(unblinding)
    manifest["review_regions"] = len(review)
    manifest["report_data_sha256"] = hashlib.sha256(result).hexdigest()
    manifest["artifacts_sha256"] = {
        file.relative_to(work).as_posix(): sha256_file(file)
        for file in sorted(work.rglob("*"))
        if file.is_file()
    }
    (work / "manifest.json").write_bytes(_json(manifest))


def run(
    plan_path: Path,
    output: Path,
    *,
    pilot: list[str] | None = None,
    device: str = "cpu",
    dry_run: bool = False,
    publish_overlays: bool = False,
) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_bytes())
    options = _validate(plan)
    root = plan_path.parent
    pilot = sorted(set(pilot or []))
    if publish_overlays and "overlay_target" not in plan:
        raise ValueError("Specify the explicit overlay publishing target in the plan.")
    target_details = plan.get("overlay_target") if publish_overlays else None
    if target_details and not re.fullmatch(r"[a-f0-9]{32}", target_details["run_id"]):
        raise ValueError("Use the full MLflow run ID for the overlay target.")
    key_path = plan.get("review_key_file")
    if target_details and not key_path:
        raise ValueError("Published blinded links require a private review_key_file.")
    review_key = (
        (root / key_path).read_bytes() if key_path else str(options["seed"]).encode()
    )
    if key_path and len(review_key) < 32:
        raise ValueError("Generate a private review key of at least 32 random bytes.")
    manifest = {
        "schema_version": 1,
        "split": "val",
        "models": plan["models"],
        "pilot_slide_ids": pilot,
        "options": dict(options),
        "review_key_sha256": hashlib.sha256(review_key).hexdigest()
        if key_path
        else None,
        "device": device,
        "tracking_uri": plan.get("tracking_uri"),
        "overlay_target": target_details,
    }
    digest = hashlib.sha256(_json({**manifest, "plan_dir": str(root)})).hexdigest()
    options["_review_key"] = review_key
    manifest["input_sha256"] = digest
    target = ViewerTarget(**target_details) if target_details else None
    if target is not None:
        target = ViewerTarget(
            target.tracking_uri,
            target.experiment_id,
            target.run_id,
            f"{target.artifact_root}/{digest}",
            target.xopat_url,
            target.artifact_mount,
        )
        manifest["overlay_artifact_root"] = target.artifact_root
    if dry_run:
        return {**manifest, "output": str(output), "dry_run": True}
    if output.exists():
        previous = json.loads((output / "manifest.json").read_bytes())
        if previous.get("input_sha256") != digest:
            raise ValueError(
                "Output directory was used with a different plan or settings."
            )
        for name, file_digest in previous["artifacts_sha256"].items():
            path = output / name
            if not path.is_file() or sha256_file(path) != file_digest:
                raise ValueError(f"Existing output is incomplete or changed: {name}.")
        _verify_local_inputs(previous, output, root)
        return previous

    from mlflow import set_tracking_uri

    if tracking_uri := plan.get("tracking_uri"):
        set_tracking_uri(tracking_uri)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tile-removal-", dir=output.parent
    ) as directory:
        base = Path(directory)
        work = base / "result"
        work.mkdir(mode=0o700)
        models = []
        seen: set[str] = set()
        manifest["models"] = []
        for spec in plan["models"]:
            LOG.info("Preparing %s", spec["model_id"])
            prepared = prepare_model(
                spec, root, base / "cache" / spec["model_id"], torch.device(device)
            )
            manifest["models"].append(prepared.provenance)
            rows = []
            for slide in prepared.roster:
                if pilot and slide.slide_id not in pilot:
                    continue
                seen.add(slide.slide_id)
                try:
                    row = evaluate_slide(
                        prepared,
                        slide,
                        options=options,
                        root=root,
                        output=work,
                        device=torch.device(device),
                    )
                except (RuntimeError, ValueError) as error:
                    LOG.warning(
                        "%s / %s evaluation failed: %s",
                        spec["model_id"],
                        slide.slide_id,
                        error,
                    )
                    row = {
                        "model_id": spec["model_id"],
                        "slide_id": slide.slide_id,
                        "status": "evaluation_error",
                        "intact_abs_index": None,
                        "target_class": None,
                        "masks": [],
                    }
                row["_wsi"] = slide.wsi
                rows.append(row)
            models.append(
                {
                    "model_id": spec["model_id"],
                    "source_run_id": spec["source_run_id"],
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "validation_source": spec["validation_source"],
                    "summary": summarize(rows, options["distance_bands"]),
                    "slides": rows,
                }
            )
        if missing := set(pilot) - seen:
            raise ValueError(
                f"Pilot contains slides absent from the validation roster: {sorted(missing)}."
            )
        if target is not None:
            _publish_overlays(target, models, work)
        _write_bundle(
            work,
            manifest,
            models,
            paired_summary(models, options["distance_bands"]),
            options,
        )
        os.rename(work, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-slide-id", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish-overlays", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run(
        args.manifest,
        args.output,
        pilot=args.pilot_slide_id,
        device=args.device,
        dry_run=args.dry_run,
        publish_overlays=args.publish_overlays,
    )
    print(_json(result).decode(), end="")


if __name__ == "__main__":
    main()
