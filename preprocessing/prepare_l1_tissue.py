"""Freeze scored L1 tissue tiles and the coordinates still needing embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.embedding_selection import (
    SCORE_COLUMNS,
    TILE_SCHEMA,
    _artifact_dir,
    _restore_scores,
    prepare_selection,
    selection_digest,
)


type Record = dict[str, Any]
MASK_SCORES = ("cancer_overlap", "epithelium_overlap")
GEOMETRY = (
    "path",
    "level",
    "extent_x",
    "extent_y",
    "tile_extent_x",
    "tile_extent_y",
    "stride_x",
    "stride_y",
    "grid_origin_x",
    "grid_origin_y",
    "mpp_x",
    "mpp_y",
)


def _digest(manifest: Record) -> str:
    fields = {key: value for key, value in manifest.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(fields, allow_nan=False, sort_keys=True).encode()
    ).hexdigest()


def _manifest(directory: Path) -> Record:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["level"] != 1 or manifest["digest"] != _digest(manifest):
        raise ValueError(f"{directory} is not a valid L1 selection")
    return manifest


def _write_json(path: Path, payload: Record) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )


def _read_tiles(directory: Path, slide: Record) -> pa.Table:
    table = pq.read_table(directory / "tiles" / f"{slide['slide_id']}.parquet")
    if (
        len(table) != slide["tile_count"]
        or selection_digest(table) != slide["selection_digest"]
    ):
        raise ValueError(f"Tile selection differs for {slide['slide_id']}")
    return table


def _prepare_slide(
    slide: Record,
    old_slide: Record | None,
    tissue: Path,
    cancer: Path,
    full: Path,
    missing: Path,
    mask_dirs: dict[str, Path],
) -> tuple[Record, Record, Counter[str]]:
    stem = slide["slide_id"]
    filename = f"{stem}.parquet"
    frame = _read_tiles(tissue, slide).to_pandas()
    reused = np.zeros(len(frame), dtype=bool)
    if old_slide and old_slide["split"] != slide["split"]:
        raise ValueError(f"Split assignment changed for {stem}")
    if old_slide and all(slide[key] == old_slide[key] for key in GEOMETRY):
        cached = _read_tiles(cancer, old_slide).to_pandas().set_index(["x", "y"])
        keys = pd.MultiIndex.from_frame(frame[["x", "y"]])
        reused = keys.isin(cached.index)
        aligned = cached.reindex(keys)
        for column in MASK_SCORES:
            # Raw tissue scores have priority. Cached scores fill the corresponding selected tiles.
            frame[column] = frame[column].fillna(
                pd.Series(aligned[column].to_numpy(), index=frame.index)
            )
    restored = Counter[str]()
    for column in MASK_SCORES:
        count = int(frame[column].isna().sum())
        if count:
            _restore_scores(frame, slide, column, mask_dirs[column])
        restored[column] = count - int(frame[column].isna().sum())
        valid = frame[column].dropna()
        if not valid.between(0, 1).all():
            raise ValueError(f"Invalid {column} on {stem}")
    table = pa.Table.from_pandas(frame, schema=TILE_SCHEMA, preserve_index=False)
    to_embed = table.filter(pa.array(~reused))
    pq.write_table(table, full / "tiles" / filename)
    pq.write_table(to_embed, missing / "tiles" / filename)

    def updated(rows: pa.Table, *, is_missing: bool) -> Record:
        status = "selected" if len(rows) else "zero_eligible_tiles"
        if is_missing and not len(rows) and len(table):
            status = "fully_reused"
        return {
            **slide,
            "tile_count": len(rows),
            "status": status,
            "selection_digest": selection_digest(rows),
            "score_missing": {name: rows[name].null_count for name in SCORE_COLUMNS},
            "full_tile_count": len(table),
            "reused_tile_count": int(reused.sum()),
        }

    known = frame[list(MASK_SCORES)].notna().all(axis=1)
    passes_or = (frame["cancer_overlap"] > 0.5) | (frame["epithelium_overlap"] > 0.25)
    summary = Counter(
        slides=int(len(table) > 0),
        selected_tiles=len(table),
        reusable_tiles=int(reused.sum()),
        missing_embeddings=len(to_embed),
        epithelium_or_cancer_tiles=int((known & passes_or).sum()),
        epithelium_or_cancer_slides=int((known & passes_or).any()),
        restored_cancer_scores=restored["cancer_overlap"],
        restored_epithelium_scores=restored["epithelium_overlap"],
    )
    return updated(table, is_missing=False), updated(to_embed, is_missing=True), summary


def build_tissue_selections(
    tissue: Path,
    cancer: Path,
    output: Path,
    mask_dirs: dict[str, Path],
    reuse_provenance: Record,
    workers: int,
) -> Record:
    """Preserve sampling order and write one full and one GPU-work-only selection."""
    tissue_manifest, cancer_manifest = _manifest(tissue), _manifest(cancer)
    if (
        "cancer_overlap" in tissue_manifest["source_gates"]
        or cancer_manifest["source_gates"].get("cancer_overlap") != 0.5
    ):
        raise ValueError(
            "Expected a tissue pool and the existing cancer-gated selection"
        )
    if (tissue_manifest["cap"], tissue_manifest["seed"]) != (
        cancer_manifest["cap"],
        cancer_manifest["seed"],
    ):
        raise ValueError("Tissue and cancer selections must use the same cap and seed")
    full, missing = output / "selection", output / "missing_selection"
    for dest in (full, missing):
        (dest / "tiles").mkdir(parents=True)
    by_slide = {row["slide_id"]: row for row in cancer_manifest["slides"]}
    full_slides, missing_slides = [], []
    by_split: dict[str, Counter[str]] = defaultdict(Counter)

    def process(slide: Record) -> tuple[Record, Record, Counter[str]]:
        return _prepare_slide(
            slide,
            by_slide.get(slide["slide_id"]),
            tissue,
            cancer,
            full,
            missing,
            mask_dirs,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (row, new, counts) in enumerate(
            pool.map(process, tissue_manifest["slides"]), start=1
        ):
            full_slides.append(row)
            missing_slides.append(new)
            by_split[row["split"]].update(counts)
            if i % 100 == 0:
                print(
                    f"Scored {i}/{len(tissue_manifest['slides'])} tissue slides",
                    flush=True,
                )

    def write_selection(directory: Path, rows: list[Record], kind: str) -> Record:
        result = {
            **tissue_manifest,
            "slides": rows,
            "selection_kind": kind,
            "reuse_source": {
                **reuse_provenance,
                "selection_digest": cancer_manifest["digest"],
            },
            "score_missing": {
                key: sum(row["score_missing"][key] for row in rows)
                for key in SCORE_COLUMNS
            },
        }
        result["digest"] = _digest(result)
        _write_json(directory / "manifest.json", result)
        return result

    full_manifest = write_selection(full, full_slides, "full_tissue")
    missing_manifest = write_selection(
        missing, missing_slides, "missing_tissue_embeddings"
    )
    total = Counter[str]()
    for counts in by_split.values():
        total.update(counts)
    missing_scores = {
        row["slide_id"]: {
            key: value for key, value in row["score_missing"].items() if value
        }
        for row in full_slides
        if any(row["score_missing"].values())
    }
    return {
        "full_selection_digest": full_manifest["digest"],
        "missing_selection_digest": missing_manifest["digest"],
        "score_missing": full_manifest["score_missing"],
        "slides_with_missing_mask_scores": len(missing_scores),
        "missing_scores_by_slide": missing_scores,
        "by_split": dict(by_split),
        "total": total,
    }


def _run(args: argparse.Namespace) -> None:
    from mlflow import MlflowClient

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise ValueError("Set MLFLOW_TRACKING_URI to the intended artifact destination")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("MammaPrint")
    source = client.get_run(args.cancer_run_id)
    if (
        experiment is None
        or source.info.experiment_id != experiment.experiment_id
        or source.info.status != "FINISHED"
        or source.data.tags.get("workflow") != "l1-virchow2-embedding"
    ):
        raise ValueError(
            "Expected the completed L1 cancer embedding source in MammaPrint"
        )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="l1-tissue-preparation-", dir=args.work_dir
    ) as directory:
        work = Path(directory)

        def download(artifact: str) -> Path:
            return Path(
                client.download_artifacts(args.cancer_run_id, artifact, str(work))
            )

        config = json.loads(download("metadata/run.json").read_text())
        if (
            config["precision"] != "fp32"
            or config["normalization"] != "imagenet-rgb-fp32-v1"
        ):
            raise ValueError(
                "Source embeddings use incompatible precision or normalization"
            )
        inventory = json.loads(download("metadata/l1_inventory.json").read_text())
        if inventory["missing_or_invalid"] or inventory["unexpected"]:
            raise ValueError("Source cancer embedding inventory is incomplete")
        archived = download("selection/frozen.tar.gz")
        with tarfile.open(archived, "r:gz") as archive:
            archive.extractall(work / "cancer", filter="data")
        archived.unlink()
        cancer = work / "cancer" / "selection"
        frozen = _manifest(cancer)
        if frozen["digest"] != config["manifest_digest"]:
            raise ValueError("Source run and frozen cancer selection differ")
        eligible = {
            row["slide_id"]: row for row in frozen["slides"] if row["tile_count"]
        }
        receipts = inventory["complete"]
        if eligible.keys() != receipts.keys() or any(
            (row["tile_count"], row["selection_digest"], row["split"])
            != (
                receipts[stem]["tiles"],
                receipts[stem]["selection_digest"],
                receipts[stem]["split"],
            )
            or not re.fullmatch(r"[0-9a-f]{64}", receipts[stem]["sha256"])
            for stem, row in eligible.items()
        ):
            raise ValueError(
                "Source embedding receipts differ from frozen cancer selection"
            )
        source_files = {
            Path(item.path).stem: item.file_size
            for item in client.list_artifacts(args.cancer_run_id, "embeddings")
            if not item.is_dir and item.path.endswith(".parquet")
        }
        if source_files != {
            stem: receipt["size"] for stem, receipt in receipts.items()
        }:
            raise ValueError(
                "Source embedding listing differs from completion receipts"
            )

        tags = {
            "workflow": "l1-prepare-tissue-selection",
            "reuse_source_run_id": args.cancer_run_id,
        }
        run_id = client.create_run(
            experiment.experiment_id,
            tags=tags,
            run_name="Prepare capped L1 tissue selection",
        ).info.run_id
        print(f"L1 tissue selection MLflow run {run_id}", flush=True)
        try:
            tissue = work / "tissue-input"
            prepare_selection(
                level=1,
                tiling=args.tiling_uri,
                data_mapping=args.data_mapping,
                output=tissue,
                shards=1,
                cap=frozen["cap"],
                seed=frozen["seed"],
                source_gate="tissue",
            )
            masks = {
                "cancer_overlap": _artifact_dir(args.cancer_masks_uri),
                "epithelium_overlap": _artifact_dir(args.epithelium_masks),
            }
            output = work / "scored"
            reuse = {
                "run_id": args.cancer_run_id,
                "encoder_revision": config["encoder_revision"],
                "precision": config["precision"],
                "normalization": config["normalization"],
                "score_sources": {
                    "cancer_overlap": args.cancer_masks_uri,
                    "epithelium_overlap": args.epithelium_masks,
                },
            }
            audit = build_tissue_selections(
                tissue, cancer, output, masks, reuse, args.workers
            )
            audit_path = work / "tissue_reuse_audit.json"
            _write_json(audit_path, audit)
            client.log_artifact(run_id, str(audit_path), artifact_path="metadata")
            if any(audit["score_missing"].values()):
                raise ValueError("The saved tissue audit lists missing mask scores")
            archive_path = work / "frozen.tar.gz"
            for name in ("selection", "missing_selection"):
                with tarfile.open(archive_path, "w:gz") as archive:
                    archive.add(output / name, arcname=name)
                client.log_artifact(
                    run_id, str(output / name / "manifest.json"), artifact_path=name
                )
                client.log_artifact(run_id, str(archive_path), artifact_path=name)
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        client.set_terminated(run_id)
        print(json.dumps(audit, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiling-uri", required=True)
    parser.add_argument("--cancer-run-id", required=True)
    parser.add_argument("--cancer-masks-uri", required=True)
    parser.add_argument("--epithelium-masks", required=True)
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    _run(args)


if __name__ == "__main__":
    main()
