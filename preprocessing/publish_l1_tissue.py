"""Assemble frozen tissue bags from existing L1 vectors and publish tissue and OR splits."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.embedded_mask_publication import (
    SCORE_FIELDS,
    _digest,
    _download_json,
    _list_files,
    _part,
    _read_receipts,
    _write_json,
)
from preprocessing.embedding_selection import selection_digest
from preprocessing.filter_embeddings import SPLITS, _load_split_map, _selection_masks


if TYPE_CHECKING:
    from mlflow import MlflowClient


VARIANTS = ("tissue_only", "epithel_or_cancer_5")
WORKFLOW = "l1-publish-tissue-masks"
type Record = dict[str, Any]


def _manifest(path: Path) -> Record:
    import hashlib

    manifest = json.loads(path.read_text())
    fields = {k: v for k, v in manifest.items() if k != "digest"}
    if (
        manifest["digest"]
        != hashlib.sha256(
            json.dumps(fields, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    ):
        raise ValueError(f"Invalid selection digest for {path}")
    return manifest


def _inventory(client: MlflowClient, run_id: str, work: Path) -> Record:
    dest = work / run_id
    dest.mkdir(exist_ok=True)
    inventory = _download_json(client, run_id, "metadata/l1_inventory.json", dest)
    if inventory["missing_or_invalid"] or inventory["unexpected"]:
        raise ValueError(f"Incomplete source embedding inventory {run_id}")
    files = _list_files(client, run_id, "embeddings")
    if files != {
        f"{stem}.parquet": row["size"] for stem, row in inventory["complete"].items()
    }:
        raise ValueError(f"Source listing differs from receipts {run_id}")
    return inventory["complete"]


def _vectors(
    client: MlflowClient,
    run_id: str,
    stem: str,
    receipt: Record,
    split: str,
    work: Path,
) -> pa.Table:
    path = Path(
        client.download_artifacts(run_id, f"embeddings/{stem}.parquet", str(work))
    )
    if receipt["split"] != split or _digest(path) != receipt["sha256"]:
        raise ValueError(f"Invalid source receipt for {stem} from {run_id}")
    table = pq.read_table(path)
    if (
        len(table) != receipt["tiles"]
        or selection_digest(table) != receipt["selection_digest"]
        or table.schema.field("embedding").type != pa.list_(pa.float32(), 2560)
    ):
        raise ValueError(f"Invalid source embedding schema or selection for {stem}")
    path.unlink()
    return table


def assemble_slide(
    full: pa.Table,
    pending: pa.Table,
    cached: pa.Table | None,
    computed: pa.Table | None,
) -> pa.Table:
    """Attach exactly one vector to every frozen coordinate, preserving full row order."""
    if not set(SCORE_FIELDS) <= set(full.column_names):
        raise ValueError("Full selection lacks mask scores")
    scores = full.select(SCORE_FIELDS).to_pandas().to_numpy()
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Full selection contains invalid mask scores")
    frames = [
        table.select(["x", "y", "embedding"]).to_pandas()
        for table in (cached, computed)
        if table is not None
    ]
    keys = full.select(["x", "y"]).to_pandas()
    pending_keys = pending.select(["x", "y"]).to_pandas()
    if keys.duplicated().any() or pending_keys.duplicated().any():
        raise ValueError("Duplicated frozen coordinates")
    all_keys = set(map(tuple, keys.to_numpy()))
    pending_set = set(map(tuple, pending_keys.to_numpy()))
    cached_keys = (
        set()
        if cached is None
        else set(map(tuple, cached.select(["x", "y"]).to_pandas().to_numpy()))
    )
    if not pending_set <= all_keys or cached_keys & all_keys != all_keys - pending_set:
        raise ValueError(
            "Cached and missing vectors differ from the frozen coordinate partition"
        )
    if computed is None:
        if len(pending):
            raise ValueError("Missing computed vectors")
    elif (
        computed.select(["x", "y"]).to_pylist()
        != pending.select(["x", "y"]).to_pylist()
    ):
        raise ValueError("Computed vectors differ from missing-only coordinates")
    if not frames:
        if len(full):
            raise ValueError("No source vectors for nonempty full selection")
        values = np.empty((0, 2560), dtype=np.float32)
    else:
        vectors = pd.concat(frames, ignore_index=True)
        # Cached cancer bags include coordinates outside the new tissue sample.
        vectors = keys.merge(vectors, on=["x", "y"], how="left", validate="one_to_one")
        if len(vectors) != len(full) or vectors["embedding"].isna().any():
            raise ValueError("Missing or duplicate vectors in final tissue bag")
        values = (
            np.stack(vectors["embedding"].to_numpy()).astype(np.float32)
            if len(full)
            else np.empty((0, 2560), np.float32)
        )
    if values.shape != (len(full), 2560) or not np.isfinite(values).all():
        raise ValueError("Invalid final embedding vectors")
    return full.append_column(
        "embedding", pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 2560)
    )


def _routes(split: str) -> dict[str, tuple[str, str]]:
    return {
        variant: (_part(variant, "all"), _part(variant, split)) for variant in VARIANTS
    }


def _receipt_ok(
    receipt: Record,
    selection: str,
    routes: dict[str, tuple[str, str]],
    listings: Record,
    stem: str,
    tile_count: int,
) -> bool:
    if receipt.get("selection_digest") != selection:
        return False
    outputs = receipt.get("variants", {})
    if not isinstance(outputs, dict) or set(outputs) != set(VARIANTS):
        return False
    for variant, paths in routes.items():
        row = outputs[variant]
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("size"), int)
            or row["size"] <= 0
            or not isinstance(row.get("tiles"), int)
            or not 0 <= row["tiles"] <= tile_count
            or (variant == "tissue_only" and row["tiles"] != tile_count)
        ):
            return False
        if any(listings[part].get(f"{stem}.parquet") != row["size"] for part in paths):
            return False
    return True


def publish(client: MlflowClient, args: argparse.Namespace) -> str:
    experiment = client.get_experiment_by_name("MammaPrint")
    if experiment is None:
        raise ValueError("MammaPrint experiment is missing")
    expected = (
        (args.preparation_run_id, "l1-prepare-tissue-selection"),
        (args.embedding_run_id, "l1-virchow2-tissue-missing-embedding"),
        (args.cancer_run_id, "l1-virchow2-embedding"),
    )
    runs = {}
    for run_id, workflow in expected:
        run = client.get_run(run_id)
        if (
            run.info.status != "FINISHED"
            or run.info.lifecycle_stage != "active"
            or run.info.experiment_id != experiment.experiment_id
            or run.data.tags.get("workflow") != workflow
        ):
            raise ValueError(f"Invalid source MLflow run {run_id}")
        runs[run_id] = run
    if (
        runs[args.embedding_run_id].data.tags.get("preparation_run_id")
        != args.preparation_run_id
        or runs[args.preparation_run_id].data.tags.get("reuse_source_run_id")
        != args.cancer_run_id
    ):
        raise ValueError("Source runs do not form the expected preparation chain")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tissue-publication-", dir=args.work_dir
    ) as tmp:
        work = Path(tmp)
        for name in ("selection", "missing_selection"):
            archive_path = Path(
                client.download_artifacts(
                    args.preparation_run_id, f"{name}/frozen.tar.gz", str(work)
                )
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(work, filter="data")
            archive_path.unlink()
        full = _manifest(work / "selection" / "manifest.json")
        missing = _manifest(work / "missing_selection" / "manifest.json")
        audit = _download_json(
            client, args.preparation_run_id, "metadata/tissue_reuse_audit.json", work
        )
        if (
            full["digest"] != audit["full_selection_digest"]
            or missing["digest"] != audit["missing_selection_digest"]
            or any(full["score_missing"].values())
            or full["reuse_source"] != missing["reuse_source"]
            or full["reuse_source"]["run_id"] != args.cancer_run_id
        ):
            raise ValueError("Frozen preparation differs from its audit")
        settings = []
        for source_id in (args.cancer_run_id, args.embedding_run_id):
            dest = work / source_id
            dest.mkdir(exist_ok=True)
            settings.append(
                _download_json(client, source_id, "metadata/run.json", dest)
            )
        for key in ("encoder_revision", "normalization", "precision"):
            if (
                settings[0][key] != settings[1][key]
                or settings[0][key] != full["reuse_source"][key]
            ):
                raise ValueError(f"Incompatible embedding {key}")
        if (
            settings[1]["manifest_digest"] != missing["digest"]
            or settings[0]["manifest_digest"]
            != full["reuse_source"]["selection_digest"]
            or settings[0]["precision"] != "fp32"
        ):
            raise ValueError("Wrong GPU selection or precision")
        old_receipts = _inventory(client, args.cancer_run_id, work)
        new_receipts = _inventory(client, args.embedding_run_id, work)
        rows = {row["slide_id"]: row for row in full["slides"]}
        pending = {row["slide_id"]: row for row in missing["slides"]}
        if (
            len(rows) != len(full["slides"])
            or len(pending) != len(missing["slides"])
            or rows.keys() != pending.keys()
        ):
            raise ValueError("Frozen selections have different slide inventories")
        if set(new_receipts) != {
            stem for stem, row in pending.items() if row["tile_count"]
        }:
            raise ValueError(
                "Missing-only embedding inventory differs from preparation"
            )
        split_map = _load_split_map(args.data_mapping)
        if {row["split"] for row in rows.values()} != set(SPLITS):
            raise ValueError("Expected nonempty slide inventories in all three splits")
        for stem, row in rows.items():
            if (
                Path(stem).name != stem
                or stem in (".", "..", "")
                or split_map.get(stem) != row["split"]
                or pending[stem]["split"] != row["split"]
            ):
                raise ValueError(f"Invalid frozen split or stem {stem}")
        tags = {
            "workflow": WORKFLOW,
            "preparation_run_id": args.preparation_run_id,
            "embedding_run_id": args.embedding_run_id,
            "cancer_run_id": args.cancer_run_id,
        }
        if args.publication_run_id:
            run = client.get_run(args.publication_run_id)
            if (
                run.info.experiment_id != experiment.experiment_id
                or run.info.lifecycle_stage != "active"
                or any(run.data.tags.get(k) != v for k, v in tags.items())
            ):
                raise ValueError("Publication resume run differs")
        else:
            run = client.create_run(
                experiment.experiment_id,
                tags=tags,
                run_name="Publish L1 tissue and epithelium OR cancer",
            )
        run_id = run.info.run_id
        print(
            f"L1 tissue publication MLflow run {run_id}: {run.info.artifact_uri}",
            flush=True,
        )
        provenance = {
            **tags,
            "selection_digest": full["digest"],
            "missing_selection_digest": missing["digest"],
            "source_receipts": {"cancer": old_receipts, "new": new_receipts},
        }
        try:
            config = "metadata/publication_config.json"
            if any(
                item.path == config
                for item in client.list_artifacts(run_id, "metadata")
            ):
                if _download_json(client, run_id, config, work) != provenance:
                    raise ValueError("Resume selection changed")
            else:
                path = work / "publication_config.json"
                _write_json(path, provenance)
                client.log_artifact(run_id, str(path), artifact_path="metadata")
            receipts = _read_receipts(client, run_id, work)
            parts = {
                p
                for split in SPLITS
                for paths in _routes(split).values()
                for p in paths
            }
            listings = {p: _list_files(client, run_id, p) for p in parts}
            for i, (stem, row) in enumerate(sorted(rows.items()), 1):
                routes = _routes(row["split"])
                if _receipt_ok(
                    receipts.get(stem, {}),
                    row["selection_digest"],
                    routes,
                    listings,
                    stem,
                    row["tile_count"],
                ):
                    continue
                with tempfile.TemporaryDirectory(dir=work) as stage:
                    directory = Path(stage)
                    selected = pq.read_table(
                        work / "selection" / "tiles" / f"{stem}.parquet"
                    )
                    pending_table = pq.read_table(
                        work / "missing_selection" / "tiles" / f"{stem}.parquet"
                    )
                    for table, info in (
                        (selected, row),
                        (pending_table, pending[stem]),
                    ):
                        if (
                            len(table) != info["tile_count"]
                            or selection_digest(table) != info["selection_digest"]
                        ):
                            raise ValueError(f"Frozen tile file changed for {stem}")
                    cached = None
                    if row["reused_tile_count"]:
                        cached = _vectors(
                            client,
                            args.cancer_run_id,
                            stem,
                            old_receipts[stem],
                            row["split"],
                            directory,
                        )
                    computed = None
                    if pending[stem]["tile_count"]:
                        computed = _vectors(
                            client,
                            args.embedding_run_id,
                            stem,
                            new_receipts[stem],
                            row["split"],
                            directory,
                        )
                    table = assemble_slide(selected, pending_table, cached, computed)
                    if len(table) != row["tile_count"] or pending[stem][
                        "tile_count"
                    ] + row["reused_tile_count"] != len(table):
                        raise ValueError("Final bag has wrong tile count")
                    mask = _selection_masks(
                        table.select(
                            ["cancer_overlap", "epithelium_overlap"]
                        ).to_pandas(),
                        0.25,
                        0.5,
                    )["or"]
                    receipt: Record = {
                        "selection_digest": row["selection_digest"],
                        "variants": {},
                    }
                    output = directory / f"{stem}.parquet"
                    for variant, paths in routes.items():
                        result = (
                            table
                            if variant == "tissue_only"
                            else table.filter(pa.array(mask))
                        )
                        pq.write_table(result, output)
                        for part in paths:
                            client.log_artifact(run_id, str(output), artifact_path=part)
                        receipt["variants"][variant] = {
                            "size": output.stat().st_size,
                            "tiles": len(result),
                        }
                    path = directory / f"{stem}.json"
                    _write_json(path, receipt)
                    client.log_artifact(
                        run_id, str(path), artifact_path="publication_receipts"
                    )
                    receipts[stem] = receipt
                if i % 100 == 0:
                    print(f"Published {i}/{len(rows)} tissue slides", flush=True)
            counts: Record = defaultdict(
                lambda: {"tiles": 0, "nonempty_slides": 0, "empty_slides": 0}
            )
            for part in parts:
                variant = part.split("/")[0]
                split = next((s for s in SPLITS if part.endswith(f"_{s}")), "all")
                stems = [
                    s for s, r in rows.items() if split == "all" or r["split"] == split
                ]
                expected_files = {}
                for stem in stems:
                    info = receipts[stem]["variants"][variant]
                    expected_files[f"{stem}.parquet"] = info["size"]
                    counts[part]["tiles"] += info["tiles"]
                    counts[part]["nonempty_slides"] += int(info["tiles"] > 0)
                    counts[part]["empty_slides"] += int(info["tiles"] == 0)
                if _list_files(client, run_id, part) != expected_files:
                    raise ValueError(f"Published inventory differs at {part}")
            for variant, key in (
                ("tissue_only", "selected_tiles"),
                ("epithel_or_cancer_5", "epithelium_or_cancer_tiles"),
            ):
                if counts[_part(variant, "all")]["tiles"] != audit["total"][key]:
                    raise ValueError(
                        f"Final counts differ from preparation for {variant}"
                    )
                for split in SPLITS:
                    if (
                        counts[_part(variant, split)]["tiles"]
                        != audit["by_split"][split][key]
                    ):
                        raise ValueError(
                            f"Final {split} tile count differs for {variant}"
                        )
            root = run.info.artifact_uri.rstrip("/")
            cards = {
                variant: {
                    "name": f"l1_{'tissue' if variant == 'tissue_only' else 'epithel_or_cancer_025_05'}_embed",
                    "level": 1,
                    "stride": 224,
                    "uris": {
                        split: f"{root}/{_part(variant, split)}"
                        for split in (*SPLITS, "all")
                    },
                }
                for variant in VARIANTS
            }
            path = work / "l1_dataset_cards.json"
            _write_json(
                path, {"cards": cards, "counts": dict(counts), "provenance": provenance}
            )
            client.log_artifact(run_id, str(path), artifact_path="metadata")
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        client.set_terminated(run_id)
        return run_id


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-run-id", "embedding-run-id", "cancer-run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--publication-run-id")
    args = parser.parse_args()
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("Set MLFLOW_TRACKING_URI")
    publish(mlflow.MlflowClient(), args)


if __name__ == "__main__":
    main()
