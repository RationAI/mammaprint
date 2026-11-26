"""Publish existing, scored L1 embeddings as split MLflow datasets."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

from preprocessing.filter_embeddings import (
    SPLITS,
    _load_split_map,
    _selection_masks,
    _threshold_mask,
)
from preprocessing.split_dataset import _assign_or_die


if TYPE_CHECKING:
    from mlflow import MlflowClient


BASE, STRICT, EPITHELIAL = "cancer_mask_5", "cancer_mask_9", "epithel_and_cancer_5"
VARIANTS = (BASE, STRICT, EPITHELIAL)
WORKFLOW = "l1-publish-cancer-splits"
NAMES = {
    BASE: "l1_cancer_mask_5_embed",
    STRICT: "l1_cancer_mask_9_embed",
    EPITHELIAL: "l1_epithel_and_cancer_025_05_embed",
}
SCORE_FIELDS = (
    "tissue_overlap",
    "blur_overlap",
    "folding_overlap",
    "residual_overlap",
    "cancer_overlap",
    "epithelium_overlap",
)
type Metadata = dict[str, Any]


def _download_json(
    client: MlflowClient, run_id: str, artifact: str, scratch: Path
) -> Metadata:
    return json.loads(
        Path(client.download_artifacts(run_id, artifact, str(scratch))).read_text()
    )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _list_files(
    client: MlflowClient, run_id: str, directory: str
) -> dict[str, int | None]:
    return {
        Path(item.path).name: item.file_size
        for item in client.list_artifacts(run_id, directory)
        if not item.is_dir and item.path.endswith(".parquet")
    }


def _part(variant: str, split: str) -> str:
    return (
        f"{variant}/embeddings" if split == "all" else f"{variant}/embeddings_{split}"
    )


def _routes(split: str) -> dict[str, tuple[str, ...]]:
    return {
        variant: (_part(variant, split),)
        if variant == BASE
        else (_part(variant, "all"), _part(variant, split))
        for variant in VARIANTS
    }


def _source_inventory(
    client: MlflowClient, source_id: str, data_mapping: Path, work: Path
) -> tuple[Metadata, dict[str, str], Metadata]:
    source = client.get_run(source_id)
    if (
        source.info.status != "FINISHED"
        or source.data.tags.get("workflow") != "l1-virchow2-embedding"
    ):
        raise ValueError(f"{source_id} is not a finished L1 scored-embedding run")
    manifest = _download_json(client, source_id, "selection/manifest.json", work)
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "digest"
    }
    manifest_digest = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if manifest.get("digest") != manifest_digest or manifest.get("level") != 1:
        raise ValueError("Invalid L1 selection manifest")
    if manifest["source_gates"].get("cancer_overlap") != 0.5 or any(
        manifest["score_missing"].get(score) != 0 for score in SCORE_FIELDS
    ):
        raise ValueError(
            "Expected complete saved scores and the original cancer > 0.5 gate"
        )
    inventory = _download_json(client, source_id, "metadata/l1_inventory.json", work)
    if (
        inventory["missing_or_invalid"]
        or inventory["unexpected"]
        or not inventory["complete"]
    ):
        raise ValueError("Source L1 publication inventory is incomplete")
    slides = inventory["complete"]
    selected = {row["slide_id"]: row for row in manifest["slides"] if row["tile_count"]}
    if slides.keys() != selected.keys():
        raise ValueError("Source inventory and frozen selected slides differ")
    split_map = _assign_or_die(sorted(slides), _load_split_map(data_mapping))
    if set(split_map.values()) != set(SPLITS):
        raise ValueError("Expected source embeddings in train, validation and test")
    expected_sizes = {}
    for stem, receipt in slides.items():
        if Path(stem).name != stem or stem in ("", ".", ".."):
            raise ValueError(f"Invalid source slide stem: {stem!r}")
        row = selected[stem]
        if row["split"] != split_map[stem]:
            raise ValueError(
                f"Current split differs from the frozen assignment for {stem}"
            )
        if (receipt["split"], receipt["tiles"], receipt["selection_digest"]) != (
            split_map[stem],
            row["tile_count"],
            row["selection_digest"],
        ):
            raise ValueError(f"Invalid assignment or receipt for {stem}")
        expected_sizes[f"{stem}.parquet"] = receipt["size"]
    if _list_files(client, source_id, "embeddings") != expected_sizes:
        raise ValueError("Source embedding listing differs from completion inventory")
    metadata = {
        "source_run_id": source_id,
        "source_artifact_uri": source.info.artifact_uri.rstrip("/") + "/embeddings",
        "selection_digest": manifest_digest,
        "data_mapping_sha256": manifest["data_mapping_sha256"],
        "tile_cap": manifest["cap"],
        "seed": manifest["seed"],
        "zero_eligible_tiles": inventory["zero_eligible_tiles"],
        "thresholds": {"base_cancer": 0.5, "strict_cancer": 0.9, "epithelium": 0.25},
    }
    return slides, split_map, metadata


def _write_json(path: Path, value: Metadata) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _read_receipts(
    client: MlflowClient, run_id: str, work: Path
) -> dict[str, Metadata]:
    receipts = {}
    if client.list_artifacts(run_id, "publication_receipts"):
        directory = Path(
            client.download_artifacts(run_id, "publication_receipts", str(work))
        )
        for file in directory.glob("*.json"):
            try:
                payload = json.loads(file.read_text())
                if isinstance(payload, dict):
                    receipts[file.stem] = payload
            except (OSError, ValueError):
                continue
    return receipts


def _receipt_ok(
    receipt: Metadata,
    parent: Metadata,
    routes: dict[str, tuple[str, ...]],
    listings: dict[str, dict[str, int | None]],
    filename: str,
) -> bool:
    if receipt.get("source_sha256") != parent["sha256"]:
        return False
    outputs = receipt.get("variants")
    if not isinstance(outputs, dict) or outputs.keys() != routes.keys():
        return False
    for variant, paths in routes.items():
        row = outputs[variant]
        if not isinstance(row, dict):
            return False
        tiles, size = row.get("tiles"), row.get("size")
        if not isinstance(tiles, int) or not 0 <= tiles <= parent["tiles"]:
            return False
        if not isinstance(size, int) or size <= 0:
            return False
        if variant == BASE and (tiles, size) != (parent["tiles"], parent["size"]):
            return False
        if any(listings[part].get(filename) != size for part in paths):
            return False
    return True


def _publish_slide(
    client: MlflowClient,
    source_id: str,
    run_id: str,
    stem: str,
    parent: Metadata,
    routes: dict[str, tuple[str, ...]],
    work: Path,
) -> Metadata:
    filename = f"{stem}.parquet"
    with tempfile.TemporaryDirectory(prefix="slide-", dir=work) as directory:
        stage = Path(directory)
        source = Path(
            client.download_artifacts(source_id, f"embeddings/{filename}", directory)
        )
        if _digest(source) != parent["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch for {stem}")
        table = pq.read_table(source)
        required = {"x", "y", "sample_rank", "embedding", *SCORE_FIELDS}
        if table.num_rows != parent["tiles"] or required - set(table.column_names):
            raise ValueError(f"Unexpected source schema or row count for {stem}")
        scores = table.select(["cancer_overlap", "epithelium_overlap"]).to_pandas()
        if (
            not np.isfinite(scores.to_numpy()).all()
            or not _threshold_mask(scores, "cancer_overlap", 0.5).all()
        ):
            raise ValueError(f"Unexpected mask values for {stem}")
        masks = {
            STRICT: _threshold_mask(scores, "cancer_overlap", 0.9),
            EPITHELIAL: _selection_masks(scores, 0.25, 0.5)["and"],
        }
        receipt: Metadata = {"source_sha256": parent["sha256"], "variants": {}}
        derived_dir = stage / "derived"
        derived_dir.mkdir()
        for variant, paths in routes.items():
            output = source
            count = table.num_rows
            if variant != BASE:
                output = derived_dir / filename
                selected = table.filter(masks[variant])
                count = selected.num_rows
                pq.write_table(selected, output)
            for artifact_path in paths:
                client.log_artifact(run_id, str(output), artifact_path=artifact_path)
            receipt["variants"][variant] = {
                "tiles": count,
                "size": output.stat().st_size,
            }
        receipt_path = stage / f"{stem}.json"
        _write_json(receipt_path, receipt)
        client.log_artifact(
            run_id, str(receipt_path), artifact_path="publication_receipts"
        )
        return receipt


def _finalize(
    client: MlflowClient,
    run_id: str,
    work: Path,
    slides: Metadata,
    splits: dict[str, str],
    provenance: Metadata,
    receipts: dict[str, Metadata],
) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tiles": 0, "nonempty_slides": 0, "empty_slides": 0}
    )
    files: dict[str, dict[str, int]] = {
        _part(variant, split): {}
        for variant in VARIANTS
        for split in (*SPLITS, "all")
        if variant != BASE or split != "all"
    }
    for stem in slides:
        for variant, paths in _routes(splits[stem]).items():
            info = receipts[stem]["variants"][variant]
            for part in paths:
                files[part][f"{stem}.parquet"] = info["size"]
                counts[part]["tiles"] += info["tiles"]
                counts[part]["nonempty_slides"] += int(info["tiles"] > 0)
                counts[part]["empty_slides"] += int(info["tiles"] == 0)
    for part, expected in files.items():
        if _list_files(client, run_id, part) != expected:
            raise ValueError(f"Published file inventory differs for {part}")
    artifact_root = client.get_run(run_id).info.artifact_uri.rstrip("/")
    cards = {}
    for variant in VARIANTS:
        uris = {split: f"{artifact_root}/{_part(variant, split)}" for split in SPLITS}
        uris["all"] = (
            provenance["source_artifact_uri"]
            if variant == BASE
            else f"{artifact_root}/{_part(variant, 'all')}"
        )
        cards[variant] = {
            "name": NAMES[variant],
            "level": 1,
            "stride": 224,
            "uris": uris,
        }
    path = work / "l1_dataset_cards.json"
    _write_json(
        path, {"cards": cards, "counts": dict(counts), "provenance": provenance}
    )
    client.log_artifact(run_id, str(path), artifact_path="metadata")


def publish_l1(
    client: MlflowClient,
    source_id: str,
    publication_id: str | None,
    data_mapping: Path,
    work_dir: Path,
    experiment_name: str,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment {experiment_name!r} does not exist")
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="l1-publication-", dir=work_dir
    ) as directory:
        work = Path(directory)
        slides, split_map, provenance = _source_inventory(
            client, source_id, data_mapping, work
        )
        tags = {"workflow": WORKFLOW, "source_run_id": source_id}
        if publication_id:
            run = client.get_run(publication_id)
            if (
                run.info.experiment_id != experiment.experiment_id
                or run.info.lifecycle_stage != "active"
                or any(run.data.tags.get(key) != value for key, value in tags.items())
            ):
                raise ValueError("Resume run is outside this L1 publication scope")
        else:
            run = client.create_run(
                experiment.experiment_id,
                tags=tags,
                run_name="Publish capped L1 mask datasets",
            )
        run_id = run.info.run_id
        print(
            f"L1 publication MLflow run {run_id}: {run.info.artifact_uri}", flush=True
        )
        try:
            config_name = "publication_config.json"
            if any(
                item.path == f"metadata/{config_name}"
                for item in client.list_artifacts(run_id, "metadata")
            ):
                if (
                    _download_json(client, run_id, f"metadata/{config_name}", work)
                    != provenance
                ):
                    raise ValueError("Publication settings differ from the resumed run")
            else:
                config = work / config_name
                _write_json(config, provenance)
                client.log_artifact(run_id, str(config), artifact_path="metadata")
            receipts = _read_receipts(client, run_id, work)
            parts = {
                part
                for split in SPLITS
                for paths in _routes(split).values()
                for part in paths
            }
            listings = {
                part: _list_files(client, run_id, part) for part in sorted(parts)
            }
            print(f"Publishing {len(slides)} L1 slides", flush=True)
            for index, (stem, parent) in enumerate(sorted(slides.items()), start=1):
                routes = _routes(split_map[stem])
                if not _receipt_ok(
                    receipts.get(stem, {}), parent, routes, listings, f"{stem}.parquet"
                ):
                    receipts[stem] = _publish_slide(
                        client, source_id, run_id, stem, parent, routes, work
                    )
                if index % 100 == 0:
                    print(
                        f"Verified or published {index}/{len(slides)} slides",
                        flush=True,
                    )
            _finalize(client, run_id, work, slides, split_map, provenance, receipts)
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        client.set_terminated(run_id)
        return run_id
