"""Export verified development feature blocks without requiring a repository checkout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


if TYPE_CHECKING:
    from typing import Any, BinaryIO


SOURCE = re.compile(r"l[1-9][0-9]*/[a-z][a-z0-9_]*")
FEATURES = {"mean_raw", "std_raw", "mean_ln", "std_ln"}


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest(cache: Path, mapping: Path) -> dict[str, Any]:
    data = json.loads((cache / "manifest.json").read_text())
    expected = data.pop("digest", None)
    encoded = json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if (
        data.get("schema_version") != 1
        or hashlib.sha256(encoded.encode()).hexdigest() != expected
    ):
        raise ValueError("Invalid slide-statistics cache manifest")
    if data.get("pilot") or data.get("pilot_slides"):
        raise ValueError("A completed production cache is required")
    if _checksum(mapping) != data.get("mapping_sha256"):
        raise ValueError("Data mapping does not match the cache manifest")
    return data | {"digest": expected}


def _development(mapping: Path) -> dict[str, tuple[str, str, float]]:
    entries = {}
    seen = set()
    dev_records = set()
    test_records = set()
    with mapping.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        required = {"path", "split", "record_num", "mammaprint_index"}
        if required - set(rows.fieldnames or ()):
            raise ValueError("Invalid data mapping columns")
        for row in rows:
            path = row["path"]
            if not path or not path.strip():
                continue
            slide_id = Path(path).stem
            split = row["split"]
            record = (row["record_num"] or "").strip()
            if (
                not slide_id
                or slide_id in seen
                or not record
                or split not in ("train", "val", "test")
            ):
                raise ValueError(f"Invalid membership for slide {slide_id!r}")
            seen.add(slide_id)
            if split == "test":
                test_records.add(record)
                continue
            dev_records.add(record)
            try:
                target = float(row["mammaprint_index"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(target):
                entries[slide_id] = (split, record, target)
    if dev_records & test_records:
        raise ValueError("At least one record spans development and test")
    if not entries:
        raise ValueError("No finite labeled development slides")
    return dict(sorted(entries.items()))


def _requested(blocks: list[str]) -> dict[str, list[str]]:
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("Request one or more distinct feature blocks")
    requested: dict[str, list[str]] = {}
    for block in blocks:
        source, _, column = block.partition(":")
        if not SOURCE.fullmatch(source) or column not in FEATURES:
            raise ValueError(f"Invalid feature block {block!r}")
        requested.setdefault(source, []).append(column)
    return requested


def _project_blocks(
    path: Path,
    columns: list[str],
    roster: dict[str, int],
    width: int,
) -> dict[str, np.ndarray]:
    table = pq.read_table(
        path,
        columns=["slide_id", *columns],
        filters=[("slide_id", "in", list(roster))],
    )
    slide_ids = table["slide_id"].to_pylist()
    if len(set(slide_ids)) != len(slide_ids):
        raise ValueError(f"Duplicate eligible slide IDs in {path}")
    index = [roster[slide_id] for slide_id in slide_ids]
    blocks = {}
    expected_type = pa.list_(pa.float32(), width)
    for column in columns:
        values = table[column].combine_chunks()
        if values.type != expected_type or values.null_count:
            raise ValueError(f"Malformed cached feature {column!r} in {path}")
        flat = pc.list_flatten(values)
        if flat.null_count:
            raise ValueError(f"Null coordinates in feature {column!r} in {path}")
        matrix = flat.to_numpy().reshape(len(index), width)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite coordinates in feature {column!r} in {path}")
        block = np.full((len(roster), width), np.nan, dtype=np.float32)
        block[index] = matrix
        blocks[column] = block
    return blocks


def export_development(
    cache: Path, mapping: Path, blocks: list[str], output: BinaryIO
) -> None:
    requested = _requested(blocks)
    cache = cache.resolve()
    manifest = _manifest(cache, mapping)
    membership = _development(mapping)
    roster = {slide_id: index for index, slide_id in enumerate(membership)}
    splits, records, targets = zip(*membership.values(), strict=True)
    arrays = {
        "ids": np.array(list(roster)),
        "splits": np.array(splits),
        "records": np.array(records),
        "targets": np.array(targets, dtype=np.float64),
    }
    block_arrays: dict[str, str] = {}
    unavailable: set[str] = set()
    file_sha256 = {}
    source_uris = {}
    for source, columns in requested.items():
        entry = manifest["sources"].get(source, {})
        if entry.get("status") != "available":
            unavailable.update(f"{source}:{column}" for column in columns)
            continue
        relative = f"stats/dev/{source}.parquet"
        path = cache / relative
        expected = entry["files"].get(relative)
        if path.resolve() != path or not expected or _checksum(path) != expected:
            raise ValueError(
                f"Checksum or path mismatch for development partition {source}"
            )
        file_sha256[source] = expected
        source_uris[source] = entry["uri"]
        width = manifest["settings"]["embedding_dim"]
        for column, matrix in _project_blocks(path, columns, roster, width).items():
            key = f"block_{len(block_arrays)}"
            block_arrays[f"{source}:{column}"] = key
            arrays[key] = matrix
    metadata = {
        "cache_digest": manifest["digest"],
        "mapping_sha256": manifest["mapping_sha256"],
        "blocks": block_arrays,
        "unavailable": [block for block in blocks if block in unavailable],
        "file_sha256": file_sha256,
        "source_uris": source_uris,
    }
    np.savez_compressed(
        output,
        allow_pickle=False,
        **arrays,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--data-mapping", required=True, type=Path)
    parser.add_argument(
        "--block", action="append", required=True, metavar="SOURCE:COLUMN"
    )
    args = parser.parse_args()
    export_development(args.cache, args.data_mapping, args.block, sys.stdout.buffer)


if __name__ == "__main__":
    main()
