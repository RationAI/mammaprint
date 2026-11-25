"""Freeze, embed, and register matched L2 cancer-mask-0.5 tile views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore[import-untyped]

from preprocessing.embedding_selection import (
    SCORE_COLUMNS,
    TILE_SCHEMA,
    _artifact_dir,
    _L2Source,
    _tile_partitions,
    selection_digest,
)
from preprocessing.resolution_embeddings import (
    FEATURES,
    _run,
    _safe_id,
    _selection,
    file_digest,
    load_selection_manifest,
    write_json,
)
from preprocessing.tile_augmentation import VIEWS, TileView


WORKFLOW = "l2-augmentation-v1"


def _roster(mapping: Path, clean: _L2Source, split: str) -> set[str]:
    metadata = pd.read_csv(
        mapping,
        usecols=["path", "split", "record_num", "mammaprint_index"],
        dtype={"record_num": str},
    ).dropna(subset=["path"])
    metadata["slide_id"] = metadata["path"].map(lambda path: Path(path).stem)
    if metadata["slide_id"].duplicated().any():
        raise ValueError("The data mapping contains duplicate slide IDs")
    if (
        not metadata["split"].isin(["train", "val", "test"]).all()
        or metadata["record_num"].isna().any()
        or metadata["record_num"].str.strip().eq("").any()
        or metadata.groupby(metadata["record_num"].str.strip())["split"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "The mapping contains invalid splits or records spanning splits"
        )
    entries = metadata.set_index("slide_id").reindex(clean.files)
    if entries["split"].ne(split).any():
        raise ValueError(f"Clean artifact includes unknown or non-{split} slides")
    targets = entries["mammaprint_index"].dropna().astype(float)
    if not np.isfinite(targets).all():
        raise ValueError("The clean cohort contains a nonfinite MammaPrint index")
    return set(targets.index)


def _raw_slides(source: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    table = pq.read_table(
        source / "slides.parquet",
        columns=["id", "path", "level", "tile_extent_x", "tile_extent_y"],
    )
    slides, buckets = {}, {}
    for index, row in enumerate(table.to_pylist()):
        raw_id, path = row.pop("id"), row["path"]
        if raw_id is None or not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Expected raw slide IDs and absolute WSI paths")
        source_id = str(raw_id)
        stem = _safe_id(Path(path).stem)
        if stem in slides or source_id in buckets:
            raise ValueError(f"Duplicate raw slide identity for {stem}")
        slides[stem] = {**row, "slide_id": stem, "source_slide_id": source_id}
        buckets[source_id] = index % 64
    return slides, buckets


def _coordinates(path: Path) -> pa.Table:
    data = pq.read_table(path, columns=["x", "y"])
    if any(not pa.types.is_integer(col.type) or col.null_count for col in data.columns):
        raise ValueError(f"Invalid integer coordinates in {path.name}")
    return data.cast(pa.schema([("x", pa.int64()), ("y", pa.int64())]))


def _freeze_coordinates(
    clean: _L2Source, slide: dict[str, Any], raw: pd.DataFrame, output: Path
) -> bool:
    stem = slide["slide_id"]
    with clean.opened(stem) as path:
        coords = _coordinates(path)
    if not coords.num_rows:
        return False
    if (
        not isinstance(slide["level"], int)
        or slide["level"] < 0
        or (slide["tile_extent_x"], slide["tile_extent_y"]) != (224, 224)
    ):
        raise ValueError(f"Expected historical 224 x 224 crop geometry for {stem}")
    clean_index = pd.MultiIndex.from_frame(coords.to_pandas())
    raw_index = pd.MultiIndex.from_frame(raw[["x", "y"]])
    if (
        clean_index.has_duplicates
        or raw_index.has_duplicates
        or not clean_index.isin(raw_index).all()
    ):
        raise ValueError(f"Duplicate or unmatched historical coordinates for {stem}")
    count = coords.num_rows
    # The existing embedding worker carries optional score columns alongside the bag.
    columns = [*coords.columns, pa.array(np.arange(count, dtype=np.int64))]
    columns.extend(pa.array(np.full(count, np.nan)) for _ in SCORE_COLUMNS)
    table = pa.Table.from_arrays(columns, schema=TILE_SCHEMA)
    pq.write_table(table, output / "tiles" / f"{stem}.parquet")
    source_stat = Path(slide["path"]).stat()
    slide.update(
        tile_count=count,
        selection_digest=selection_digest(table),
        source_stat=[source_stat.st_size, source_stat.st_mtime_ns],
        shard=0,
    )
    return True


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"A frozen selection already exists at {args.output}")
    if args.full_clean_val and (args.split != "val" or args.slides):
        raise ValueError(
            "--full-clean-val applies only to a full clean validation control"
        )
    if args.split == "val" and not (args.slides or args.full_clean_val):
        raise ValueError(
            "Freeze a validation probe with --slides or use --full-clean-val"
        )
    clean = _L2Source(args.clean_uri)
    mapping_digest = file_digest(args.data_mapping)
    eligible = _roster(args.data_mapping, clean, args.split)
    ids = set(args.slides) if args.slides else eligible
    if not ids or not ids <= eligible or (args.slides and len(ids) != len(args.slides)):
        raise ValueError(
            "Requested slides must be unique and eligible in the clean split"
        )
    raw = _artifact_dir(args.raw_tiling)
    slides, buckets = _raw_slides(raw)
    if ids - slides.keys():
        raise ValueError(
            f"Historical slides absent from raw tiling: {sorted(ids - slides.keys())[:5]}"
        )
    groups: dict[int, list[str]] = {}
    for stem in sorted(ids):
        groups.setdefault(buckets[slides[stem]["source_slide_id"]], []).append(stem)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent)
    )
    try:
        partition = work / "partition"
        partition.mkdir()
        (work / "tiles").mkdir()
        _tile_partitions(
            raw / "tiles.parquet",
            buckets,
            partition,
            selected_ids={slides[stem]["source_slide_id"] for stem in ids},
        )
        selected, empty = [], []
        for bucket, stems in sorted(groups.items()):
            path = partition / f"{bucket}.parquet"
            coords = (
                pq.read_table(path, columns=["slide_id", "x", "y"]).to_pandas()
                if path.exists()
                else pd.DataFrame(columns=["slide_id", "x", "y"])
            )
            by_slide = coords.groupby("slide_id", sort=False)
            for stem in stems:
                slide = {**slides[stem], "split": args.split}
                key = slide["source_slide_id"]
                space = (
                    by_slide.get_group(key)
                    if key in by_slide.groups
                    else coords.iloc[:0]
                )
                if _freeze_coordinates(clean, slide, space, work):
                    selected.append(slide)
                else:
                    empty.append(stem)
        if not selected or (args.slides and empty):
            raise ValueError(
                "An explicit probe requires nonempty bags for every requested slide"
            )
        if file_digest(args.data_mapping) != mapping_digest:
            raise ValueError("Data mapping changed while preparing the selection")
        shutil.rmtree(partition)
        manifest = {
            "workflow": WORKFLOW,
            "level": 2,
            "split": args.split,
            "shards": 1,
            "clean_uri": args.clean_uri,
            "raw_tiling": args.raw_tiling,
            "mapping_sha256": mapping_digest,
            "subset": bool(args.slides),
            "excluded_empty_slides": empty,
            "slides": sorted(selected, key=lambda slide: slide["slide_id"]),
        }
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        manifest["digest"] = hashlib.sha256(encoded).hexdigest()
        write_json(work / "manifest.json", manifest)
        ordered = sorted(
            selected, key=lambda slide: (slide["tile_count"], slide["slide_id"])
        )
        pilot_indices = sorted(
            set(np.linspace(0, len(ordered) - 1, min(4, len(ordered)), dtype=int))
        )
        summary = {
            "manifest": str(args.output / "manifest.json"),
            "slides": len(selected),
            "tiles": sum(slide["tile_count"] for slide in selected),
            "empty": len(empty),
            "feature_bytes_per_view": sum(slide["tile_count"] for slide in selected)
            * FEATURES
            * 4,
            "pilot_slides": [ordered[index]["slide_id"] for index in pilot_indices],
        }
        write_json(work / "summary.json", summary)
        os.rename(work, args.output)
        print(json.dumps(summary))
    finally:
        if work.exists():
            shutil.rmtree(work)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_selection_manifest(path)
    ids = [_safe_id(slide["slide_id"]) for slide in manifest["slides"]]
    if (
        manifest.get("workflow") != WORKFLOW
        or manifest.get("level") != 2
        or manifest.get("split") not in ("train", "val")
        or manifest.get("shards") != 1
        or not ids
        or len(ids) != len(set(ids))
        or any(
            slide["split"] != manifest["split"]
            or slide["shard"] != 0
            or slide["tile_count"] < 1
            for slide in manifest["slides"]
        )
    ):
        raise ValueError("Expected a complete, single-split L2 augmentation selection")
    return manifest


def embed(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    view = TileView(args.view, args.seed, manifest["split"])
    if view.split == "val" and view.kind == "rotation" and not manifest["subset"]:
        raise ValueError("The rotation probe needs an explicit validation subset")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", args.encoder_revision):
        raise ValueError("Pin Virchow2 with a 40-character HF commit SHA")
    if min(args.batch_size, args.prefetch, args.loader_timeout) < 1 or args.workers < 0:
        raise ValueError("Invalid batch, worker, prefetch or loader timeout settings")
    if args.output.resolve().is_relative_to(args.manifest.resolve().parent):
        raise ValueError("Embedding output must live outside the frozen selection")
    if bool(args.pilot) != bool(args.pilot_slides):
        raise ValueError("Provide --pilot and --pilot-slides together")
    if not args.dry_run and (
        torch.cuda.device_count() != 1 or "A40" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Expose exactly one full NVIDIA A40 to this worker")
    return _run(args, view=view)


def card(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"A data card already exists at {args.output}")
    manifest = _load_manifest(args.manifest)
    if manifest["split"] != "train" or manifest["subset"]:
        raise ValueError("A training card requires the full training selection")
    config = json.loads(args.run_config.read_text())
    view = config.get("tile_view", {})
    if (
        config.get("manifest_digest") != manifest["digest"]
        or view.get("kind") not in VIEWS[1:]
        or view.get("split") != "train"
        or config.get("precision") != "fp32"
        or "pilot_slides" in config
        or type(view.get("seed")) is not int
        or view != TileView(view["kind"], view["seed"], "train").provenance()
        or not re.fullmatch(r"[a-fA-F0-9]{40}", config.get("encoder_revision", ""))
    ):
        raise ValueError(
            "The embedding run does not match the selected augmentation cohort"
        )
    uris = {
        "train": manifest["clean_uri"],
        "val": args.clean_val,
        "train_augmented": args.augmented_uri,
    }
    if args.clean_test:
        uris["test"] = args.clean_test
    if args.clean_train:
        if args.clean_test or not args.fresh_val_manifest:
            raise ValueError(
                "A fresh control card requires a full validation manifest and omits test until recipe selection"
            )
        val = _load_manifest(args.fresh_val_manifest)
        if (
            val["split"] != "val"
            or val["subset"]
            or val["mapping_sha256"] != manifest["mapping_sha256"]
        ):
            raise ValueError(
                "The fresh clean validation selection differs from the training split map"
            )
        for uri, selection_path, selection in (
            (args.clean_train, args.manifest, manifest),
            (args.clean_val, args.fresh_val_manifest, val),
        ):
            # Keep the fresh control route on one verified shared filesystem.
            if ":" in uri:
                raise ValueError(
                    "Use a shared filesystem URI for the fresh clean control"
                )
            fresh_config = Path(uri).resolve().parent / "run.json"
            fresh = json.loads(fresh_config.read_text())
            if (
                fresh.get("manifest_digest") != selection["digest"]
                or fresh.get("tile_view", {}).get("kind") != "clean"
                or fresh.get("tile_view", {}).get("split") != selection["split"]
                or any(
                    fresh.get(key) != config.get(key)
                    for key in (
                        "encoder_revision",
                        "normalization",
                        "precision",
                        "runtime_id",
                    )
                )
            ):
                raise ValueError(
                    "A fresh clean view has different encoder or selection provenance"
                )
            _verify_dataset(uri, fresh_config, selection_path, selection)
        uris["train"] = args.clean_train
    elif args.fresh_val_manifest:
        raise ValueError("Provide --clean-train with --fresh-val-manifest")
    sources = uris.values()
    if len({source.rstrip("/") for source in sources}) != len(sources) or any(
        not source for source in sources
    ):
        raise ValueError("Provide distinct clean and augmented split sources")
    _verify_dataset(args.augmented_uri, args.run_config, args.manifest, manifest)
    body = {
        "defaults": ["/data/levels/l2@_here_", "_self_"],
        "name": f"l2_cancer_mask_5_{view['kind']}",
        "uris": uris,
        "augmentation": {
            **view,
            "encoder_revision": config["encoder_revision"],
            "manifest_digest": manifest["digest"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        staged = Path(temp) / "card.yaml"
        staged.write_text(
            "# @package dataset.levels.2\n" + yaml.safe_dump(body, sort_keys=False)
        )
        os.link(staged, args.output)


def _verify_dataset(
    uri: str, run_config: Path, selection: Path, manifest: dict[str, Any]
) -> None:
    augmented = _L2Source(uri)
    if set(augmented.files) != {slide["slide_id"] for slide in manifest["slides"]}:
        raise ValueError(
            "The augmented artifact slide inventory differs from the clean cohort"
        )
    receipts = _receipts(uri, run_config)
    for slide in manifest["slides"]:
        stem = slide["slide_id"]
        original = _selection(selection.parent, slide)
        if selection_digest(original) != slide["selection_digest"]:
            raise ValueError(f"Frozen coordinates changed for {stem}")
        receipt = json.loads((receipts / f"{stem}.json").read_text())
        with augmented.opened(stem) as path:
            if (
                receipt.get("tiles") != slide["tile_count"]
                or receipt.get("selection_digest") != slide["selection_digest"]
                or receipt.get("sha256") != file_digest(path)
                or not _coordinates(path).equals(original.select(["x", "y"]))
                or pq.read_schema(path).field("embedding").type
                != pa.list_(pa.float32(), FEATURES)
            ):
                raise ValueError(f"Augmented Parquet did not verify for {stem}")


def _receipts(uri: str, run_config: Path) -> Path:
    if ":" not in uri:
        path = Path(uri).resolve()
        if path != run_config.resolve().parent / "embeddings":
            raise ValueError(
                "The local embedding artifact and run config must share an output root"
            )
        return path.parent / "complete"
    from mlflow.artifacts import download_artifacts, load_text

    if not uri.endswith("/embeddings"):
        raise ValueError("Use the run's embeddings artifact URI")
    parent = uri.removesuffix("/embeddings")
    if json.loads(load_text(f"{parent}/metadata/run.json")) != json.loads(
        run_config.read_text()
    ):
        raise ValueError("Local and published augmentation run metadata differ")
    return Path(download_artifacts(artifact_uri=f"{parent}/complete"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    selection = modes.add_parser("prepare")
    selection.add_argument("--raw-tiling", required=True)
    selection.add_argument("--clean-uri", required=True)
    selection.add_argument("--data-mapping", required=True, type=Path)
    selection.add_argument("--split", required=True, choices=("train", "val"))
    selection.add_argument("--slides", nargs="+")
    selection.add_argument("--full-clean-val", action="store_true")
    selection.add_argument("--output", required=True, type=Path)
    worker = modes.add_parser("embed")
    worker.add_argument("--manifest", required=True, type=Path)
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--view", choices=VIEWS, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--encoder-revision", required=True)
    worker.add_argument("--runtime-id", required=True)
    worker.add_argument("--batch-size", type=int, default=1024)
    worker.add_argument("--workers", type=int, default=8)
    worker.add_argument("--prefetch", type=int, default=2)
    worker.add_argument("--loader-timeout", type=int, default=300)
    worker.add_argument("--pilot")
    worker.add_argument("--pilot-slides", nargs="+")
    worker.add_argument("--mlflow-run-id")
    worker.add_argument("--dry-run", action="store_true")
    worker.set_defaults(device="cuda:0", precision="fp32", shard=0, stage_dir=None)
    publish = modes.add_parser("card")
    publish.add_argument("--manifest", required=True, type=Path)
    publish.add_argument("--run-config", required=True, type=Path)
    publish.add_argument("--augmented-uri", required=True)
    publish.add_argument("--clean-val", required=True)
    publish.add_argument("--clean-test")
    publish.add_argument("--clean-train")
    publish.add_argument("--fresh-val-manifest", type=Path)
    publish.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "embed":
        return embed(args)
    if args.action == "prepare":
        prepare(args)
    else:
        card(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
