"""Freeze a capped non-overlapping sample from existing cancer-mask tiling."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import struct
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


SCORE_COLUMNS = (
    "tissue_overlap",
    "blur_overlap",
    "folding_overlap",
    "residual_overlap",
    "cancer_overlap",
    "epithelium_overlap",
)
MASK_GATES = {
    "tissue_overlap": 0.25,
    "blur_overlap": 0.5,
    "folding_overlap": 0.1,
    "residual_overlap": 0.5,
    "cancer_overlap": 0.5,
}
SPLITS = ("train", "val", "test")
TILE_SCHEMA = pa.schema(
    [(key, pa.int64()) for key in ("x", "y", "sample_rank")]
    + [(key, pa.float64()) for key in SCORE_COLUMNS]
)
EMBED_DIM = 2560
MASK_PREFIXES = {
    "blur_overlap": "Piqe_piqe_median_activity_mask_",
    "folding_overlap": "FoldingFunction_folding_test_",
    "residual_overlap": "ResidualArtifactsAndCoverage_coverage_mask_",
}
SOURCE_SLIDE_COLUMNS = (
    "id",
    "path",
    "level",
    "extent_x",
    "extent_y",
    "tile_extent_x",
    "tile_extent_y",
    "stride_x",
    "stride_y",
    "mpp_x",
    "mpp_y",
)


def selection_digest(table: pa.Table) -> str:
    """Hash ordered coordinates, ranks, and scores, independent of Parquet encoding."""
    digest = hashlib.sha256(b"mammaprint-selection-v1\0")
    digest.update(struct.pack("<Q", len(table)))
    for name in TILE_SCHEMA.names:
        values = table[name].to_numpy()
        if name in SCORE_COLUMNS:
            values = np.array(values, dtype="<f8", copy=True)
            values[np.isnan(values)] = np.nan
        else:
            values = np.asarray(values, dtype="<i8")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _require_columns(
    columns: list[str], required: tuple[str, ...], source: Path
) -> None:
    if missing := sorted(set(required) - set(columns)):
        raise ValueError(f"{source} lacks columns {missing}")


def _mapping(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    frame = pd.read_csv(path, usecols=["path", "split"], dtype=str).dropna()
    if not frame["split"].isin(SPLITS).all():
        raise ValueError(f"{path} has invalid split assignments")
    frame["stem"] = frame["path"].map(lambda value: Path(value).stem)
    if frame["stem"].duplicated().any():
        raise ValueError(f"{path} has duplicate slide stems")
    entries = frame.set_index("stem").to_dict(orient="index")
    return entries, hashlib.sha256(path.read_bytes()).hexdigest()


def _load_slides(
    tiling: Path, mapping: Mapping[str, dict[str, str]]
) -> list[dict[str, Any]]:
    path = tiling / "slides.parquet"
    schema = pq.read_schema(path)
    _require_columns(schema.names, SOURCE_SLIDE_COLUMNS, path)
    optional = [
        key
        for key in ("grid_origin_x", "grid_origin_y", "downsample")
        if key in schema.names
    ]
    slides = pq.read_table(path, columns=[*SOURCE_SLIDE_COLUMNS, *optional]).to_pylist()
    if not slides:
        raise ValueError(f"{path} contains no slides")
    stems = set()
    ids = set()
    for slide in slides:
        if not isinstance(slide["path"], str) or not slide["path"]:
            raise ValueError(f"{path} has a missing WSI path")
        stem = Path(slide["path"]).stem
        source_id = slide["id"]
        if source_id is None or source_id in ids or not stem or stem in stems:
            raise ValueError(
                f"{path} has a missing or duplicate slide ID or stem for {stem}"
            )
        if stem not in mapping:
            raise ValueError(f"No split mapping for source slide {stem}")
        stems.add(stem)
        ids.add(source_id)
        slide["slide_id"] = stem
        slide["split"] = mapping[stem]["split"]
        slide["source_slide_id"] = str(slide.pop("id"))
        if not isinstance(slide["level"], int) or slide["level"] < 0:
            raise ValueError(f"Invalid pyramid level for {stem}")
        for axis in ("x", "y"):
            origin = slide.setdefault(f"grid_origin_{axis}", 0)
            stride = slide[f"stride_{axis}"]
            extent = slide[f"extent_{axis}"]
            footprint = slide[f"tile_extent_{axis}"]
            mpp = slide[f"mpp_{axis}"]
            if not (
                all(
                    isinstance(value, int)
                    for value in (origin, stride, extent, footprint)
                )
                and 0 <= origin < extent
                and 0 < stride < footprint <= 2 * stride
                and isinstance(mpp, (int, float))
                and np.isfinite(mpp)
                and mpp > 0
            ):
                raise ValueError(
                    f"Invalid half-overlap grid geometry for {stem} on {axis}"
                )
    if len({slide["source_slide_id"] for slide in slides}) != len(slides):
        raise ValueError(f"{path} slide IDs collide after string conversion")
    return sorted(slides, key=lambda slide: slide["slide_id"])


def _tile_partitions(
    source: Path, bucket_by_id: Mapping[str, int], scratch: Path
) -> list[str]:
    """Bound input memory and open files by partitioning projected rows into buckets."""
    parquet = pq.ParquetFile(source)
    names = parquet.schema_arrow.names
    x, y = ("x", "y") if "x" in names else ("tile_x", "tile_y")
    _require_columns(names, ("slide_id", x, y), source)
    if not all(
        pa.types.is_integer(parquet.schema_arrow.field(key).type) for key in (x, y)
    ):
        raise ValueError(f"{source} has non-integer tile coordinates")
    scores = [key for key in SCORE_COLUMNS if key in names]
    with ExitStack() as stack:
        writers: dict[int, pq.ParquetWriter] = {}
        for batch in parquet.iter_batches(
            batch_size=262_144, columns=["slide_id", x, y, *scores]
        ):
            frame = batch.to_pandas().rename(columns={x: "x", y: "y"})
            if frame[["slide_id", "x", "y"]].isna().any().any():
                raise ValueError(f"{source} has missing tile coordinates or slide IDs")
            frame["slide_id"] = frame["slide_id"].astype(str)
            buckets = frame["slide_id"].map(bucket_by_id)
            if buckets.isna().any():
                unknown = sorted(frame.loc[buckets.isna(), "slide_id"].unique())
                raise ValueError(
                    f"{source} references unknown source slide IDs {unknown[:10]}"
                )
            for bucket, rows in frame.groupby(buckets):
                table = pa.Table.from_pandas(rows, preserve_index=False)
                if bucket not in writers:
                    writers[bucket] = stack.enter_context(
                        pq.ParquetWriter(scratch / f"{bucket}.parquet", table.schema)
                    )
                writers[bucket].write_table(table)
    return scores


def _grid_rows(tiles: pd.DataFrame, slide: dict[str, Any]) -> pd.DataFrame:
    stem = slide["slide_id"]
    if tiles.duplicated(["x", "y"]).any():
        raise ValueError(f"Duplicate tile coordinates on {stem}")
    keep = np.ones(len(tiles), dtype=bool)
    for axis in ("x", "y"):
        origin = slide[f"grid_origin_{axis}"]
        stride = slide[f"stride_{axis}"]
        footprint = slide[f"tile_extent_{axis}"]
        extent = slide[f"extent_{axis}"]
        # The source grid keeps its last partially padded tile at a regular stride.
        max_position = (extent - origin - footprint + stride - 1) // stride
        position, remainder = np.divmod(
            tiles[axis].to_numpy(dtype=np.int64) - origin, stride
        )
        if np.any((position < 0) | (position > max_position) | (remainder != 0)):
            raise ValueError(
                f"Coordinates do not match the stored source grid for {stem} on {axis}"
            )
        keep &= (position % 2) == 0
    return tiles.loc[keep].copy()


def _sample(grid: pd.DataFrame, slide_id: str, cap: int, seed: int) -> pd.DataFrame:
    # Hash before the spatial sort so read locality cannot alter the random subset.
    key = hashlib.sha256(f"{seed}\0{slide_id}".encode()).digest()
    hashed = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    struct.pack("<qq", x, y), key=key, digest_size=8
                ).digest(),
                "little",
            )
            for x, y in zip(grid["x"], grid["y"], strict=True)
        ),
        dtype=np.uint64,
        count=len(grid),
    )
    chosen = (
        grid.assign(hash_rank=hashed).sort_values(["hash_rank", "x", "y"]).head(cap)
    )
    chosen = chosen.assign(sample_rank=np.arange(len(chosen), dtype=np.int64))
    return (
        chosen.sort_values(["y", "x"]).drop(columns="hash_rank").reset_index(drop=True)
    )


def _restore_scores(
    chosen: pd.DataFrame,
    slide: dict[str, Any],
    score: str,
    mask_dir: Path,
) -> None:
    # Read only missing scores on selected coordinates using the tiling overlay routine.
    from ratiopath.tiling.overlays import _tile_overlay
    from shapely.geometry import box

    missing = chosen[score].isna()
    basename = Path(slide["path"]).with_suffix(".tiff").name
    mask = mask_dir / (MASK_PREFIXES.get(score, "") + basename)
    if not missing.any() or not mask.is_file():
        return
    width, height = slide["tile_extent_x"], slide["tile_extent_y"]
    roi = (
        box(width // 4, height // 4, width // 4 + width // 2, height // 4 + height // 2)
        if score == "tissue_overlap"
        else box(0, 0, width, height)
    )
    indices = chosen.index[missing]
    # Bound RAM because the scorer materializes all image regions for one batch.
    for start in range(0, len(indices), 128):
        rows = chosen.loc[indices[start : start + 128]]
        overlays = _tile_overlay(
            roi,
            pa.repeat(str(mask), len(rows)),
            pa.array(rows["x"], type=pa.int64()),
            pa.array(rows["y"], type=pa.int64()),
            pa.repeat(slide["mpp_x"], len(rows)),
            pa.repeat(slide["mpp_y"], len(rows)),
        )
        values = []
        for overlay in overlays:
            pixels = overlay.compressed()
            if not len(pixels):
                values.append(0.0)
            elif score == "cancer_overlap":
                values.append(float(np.mean(pixels) / 255.0))
            elif score in MASK_PREFIXES:
                values.append(float(np.mean(pixels == 0)))
            else:
                values.append(float(np.mean(pixels != 0)))
        chosen.loc[rows.index, score] = values


def _artifact_dir(source: str) -> Path:
    if ":" in source:
        from mlflow.artifacts import download_artifacts

        return Path(download_artifacts(artifact_uri=source))
    path = Path(source).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Artifact directory does not exist: {path}")
    return path


class _L2Source:
    def __init__(self, source: str) -> None:
        self.source = source.rstrip("/")
        self.remote = ":" in source
        if self.remote:
            from mlflow.artifacts import list_artifacts

            names = [
                Path(file.path).name
                for file in list_artifacts(artifact_uri=self.source)
                if not file.is_dir and file.path.endswith(".parquet")
            ]
        else:
            names = [file.name for file in _artifact_dir(source).glob("*.parquet")]
        self.files = {name.removesuffix(".parquet"): name for name in names}
        if not self.files:
            raise ValueError(f"No per-slide L2 embeddings under {source}")

    @contextmanager
    def opened(self, slide_id: str) -> Iterator[Path]:
        name = self.files[slide_id]
        if self.remote:
            from mlflow.artifacts import download_artifacts
            from mlflow.utils.uri import append_to_uri_path

            with tempfile.TemporaryDirectory(prefix="l2-slide-") as temp:
                yield Path(
                    download_artifacts(
                        artifact_uri=append_to_uri_path(self.source, name),
                        dst_path=temp,
                    )
                )
        else:
            yield Path(self.source).expanduser() / name


def _matched_l2(
    path: Path,
    source: pd.DataFrame,
    grid: pd.DataFrame,
    chosen: pd.DataFrame,
) -> tuple[pa.Array | None, dict[str, int], str]:
    parquet = pq.ParquetFile(path)
    _require_columns(parquet.schema_arrow.names, ("x", "y", "embedding"), path)
    coords = parquet.read(columns=["x", "y"]).to_pandas()
    source_index = pd.MultiIndex.from_frame(source[["x", "y"]])
    embedded_index = pd.MultiIndex.from_frame(coords)
    grid_index = pd.MultiIndex.from_frame(grid[["x", "y"]])
    mismatch = {
        "missing_grid": len(grid_index.difference(embedded_index)),
        "missing_source": len(source_index.difference(embedded_index)),
        "unexpected": len(embedded_index.difference(source_index)),
        "duplicate": int(embedded_index.duplicated().sum()),
        "null": int(coords.isna().any(axis=1).sum()),
    }
    if any(
        mismatch[key] for key in ("missing_grid", "unexpected", "duplicate", "null")
    ):
        return None, mismatch, "l2_coordinate_mismatch"
    if chosen.empty:
        return None, mismatch, "zero_eligible_tiles"

    locations = embedded_index.get_indexer(pd.MultiIndex.from_frame(chosen[["x", "y"]]))
    # Pull only selected rows from each row group. Legacy files may have one large group.
    embedding_parts = []
    output_positions = []
    offset = 0
    for index in range(parquet.num_row_groups):
        end = offset + parquet.metadata.row_group(index).num_rows
        picks = np.flatnonzero((locations >= offset) & (locations < end))
        if len(picks):
            vectors = parquet.read_row_group(index, columns=["embedding"])["embedding"]
            embedding_parts.extend(
                vectors.take(pa.array(locations[picks] - offset)).to_pylist()
            )
            output_positions.extend(picks.tolist())
        offset = end
    try:
        embeddings = np.asarray(embedding_parts, dtype=np.float32)[
            np.argsort(output_positions)
        ]
    except (TypeError, ValueError):
        return None, mismatch, "l2_invalid_embeddings"
    if (
        embeddings.shape != (len(chosen), EMBED_DIM)
        or not np.isfinite(embeddings).all()
    ):
        return None, mismatch, "l2_invalid_embeddings"
    array = pa.FixedSizeListArray.from_arrays(pa.array(embeddings.ravel()), EMBED_DIM)
    return array, mismatch, "selected"


def _verify_scores(tiles: pd.DataFrame, slide_id: str) -> None:
    for key in SCORE_COLUMNS:
        if key not in tiles:
            tiles[key] = np.nan
        score = tiles[key].astype(np.float64).dropna()
        if (score < 0).any() or (score > 1).any():
            raise ValueError(f"Out-of-range {key} value on {slide_id}")
        if key in MASK_GATES and (score <= MASK_GATES[key]).any():
            raise ValueError(
                f"{slide_id} contains tiles outside the expected {key} source gate"
            )


def _build_selection(
    slides: list[dict[str, Any]],
    source_file: Path,
    work: Path,
    *,
    l2: _L2Source | None,
    cap: int,
    seed: int,
    mask_dirs: Mapping[str, Path],
) -> list[str]:
    tiles_dir = work / "tiles"
    tiles_dir.mkdir()
    if l2:
        (work / "embeddings").mkdir()
    bucket_count = min(len(slides), 64)
    bucket_by_id = {
        slide["source_slide_id"]: index % bucket_count
        for index, slide in enumerate(slides)
    }
    with tempfile.TemporaryDirectory(prefix="partitions-", dir=work) as temporary:
        partition_dir = Path(temporary)
        scores = _tile_partitions(source_file, bucket_by_id, partition_dir)
        empty = pd.DataFrame(columns=["slide_id", "x", "y", *scores])
        for bucket in range(bucket_count):
            partition = partition_dir / f"{bucket}.parquet"
            frame = pd.read_parquet(partition) if partition.is_file() else empty
            groups = frame.groupby("slide_id", sort=False)
            for slide in slides[bucket::bucket_count]:
                source_id, stem = slide["source_slide_id"], slide["slide_id"]
                tiles = (
                    groups.get_group(source_id).copy()
                    if source_id in groups.indices
                    else empty.copy()
                )
                _verify_scores(tiles, stem)
                grid = _grid_rows(tiles, slide)
                chosen = _sample(grid, stem, cap, seed)
                features = None
                status = "selected" if len(chosen) else "zero_eligible_tiles"
                if l2 and stem in l2.files:
                    with l2.opened(stem) as embedding_path:
                        features, mismatch, status = _matched_l2(
                            embedding_path, tiles, grid, chosen
                        )
                    slide["l2_coordinate_counts"] = mismatch
                elif l2 and len(grid):
                    status = "l2_embedding_missing"

                if status != "selected":
                    chosen = chosen.iloc[:0].copy()
                elif mask_dirs:
                    for score, mask_dir in mask_dirs.items():
                        _restore_scores(chosen, slide, score, mask_dir)
                    _verify_scores(chosen, stem)

                table = pa.Table.from_pandas(
                    chosen[TILE_SCHEMA.names], schema=TILE_SCHEMA, preserve_index=False
                )
                pq.write_table(table, tiles_dir / f"{stem}.parquet")
                slide.update(
                    status=status,
                    source_tile_count=len(tiles),
                    grid_count=len(grid),
                    tile_count=len(table),
                    score_missing={key: table[key].null_count for key in SCORE_COLUMNS},
                    selection_digest=selection_digest(table),
                )
                if features is not None:
                    pq.write_table(
                        table.append_column("embedding", features),
                        work / "embeddings" / f"{stem}.parquet",
                    )
                    slide["l2_embedding_digest"] = hashlib.sha256(
                        np.asarray(features.values.to_numpy(), dtype="<f4").tobytes()
                    ).hexdigest()
    return scores


def prepare_selection(
    *,
    level: int,
    tiling: str,
    data_mapping: Path,
    output: Path,
    shards: int,
    cap: int = 3000,
    seed: int = 42,
    l2_embeddings: str | None = None,
    slide_overhead_tiles: int = 0,
    mask_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if level not in (1, 2) or (level == 2) != (l2_embeddings is not None):
        raise ValueError("An existing embedding artifact is required for L2 only")
    if not 1 <= cap <= 3000 or shards < 1 or slide_overhead_tiles < 0:
        raise ValueError(
            "Require a cap of 1-3000, positive shard count and nonnegative overhead"
        )
    if output.exists():
        raise FileExistsError(f"Frozen selection output already exists: {output}")
    tiling_dir = _artifact_dir(tiling)
    mapping, mapping_digest = _mapping(data_mapping)
    slides = _load_slides(tiling_dir, mapping)
    mask_sources = dict(mask_sources or {})
    if unknown := set(mask_sources) - set(SCORE_COLUMNS):
        raise ValueError(f"Unknown score columns: {sorted(unknown)}")
    mask_dirs = {score: _artifact_dir(source) for score, source in mask_sources.items()}
    l2 = _L2Source(l2_embeddings) if l2_embeddings else None
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        scores = _build_selection(
            slides,
            tiling_dir / "tiles.parquet",
            work,
            l2=l2,
            cap=cap,
            seed=seed,
            mask_dirs=mask_dirs,
        )
        shards_by_load = [(0, shard) for shard in range(shards)]
        for slide in sorted(
            slides, key=lambda item: (-item["tile_count"], item["slide_id"])
        ):
            load, shard = heapq.heappop(shards_by_load)
            slide["shard"] = shard
            heapq.heappush(
                shards_by_load,
                (load + slide["tile_count"] + slide_overhead_tiles, shard),
            )

        stems = {slide["slide_id"] for slide in slides}
        manifest = {
            "schema_version": 1,
            "level": level,
            "tiling_source": tiling,
            "l2_embeddings_source": l2_embeddings,
            "data_mapping_sha256": mapping_digest,
            "shards": shards,
            "seed": seed,
            "cap": cap,
            "slide_overhead_tiles": slide_overhead_tiles,
            "grid": {
                "coordinate_units": "pyramid-level pixels",
                "phase": [0, 0],
                "edge": "keep",
            },
            "source_gates": MASK_GATES,
            "source_score_columns": scores,
            "recovered_score_sources": mask_sources,
            "score_missing": {
                key: sum(slide["score_missing"][key] for slide in slides)
                for key in SCORE_COLUMNS
            },
            "mapping_without_source": [
                {"slide_id": stem, **mapping[stem]}
                for stem in sorted(set(mapping) - stems)
            ],
            "l2_unknown_embedding_slides": sorted(set(l2.files) - stems) if l2 else [],
            "slides": slides,
        }
        serialized = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        manifest["digest"] = hashlib.sha256(serialized).hexdigest()
        (work / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        os.rename(work, output)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True, choices=(1, 2))
    parser.add_argument(
        "--tiling", required=True, help="Raw tiling directory or artifact URI"
    )
    parser.add_argument("--data-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--cap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--l2-embeddings", help="Existing L2 embedding directory or artifact URI"
    )
    parser.add_argument("--slide-overhead-tiles", type=int, default=0)
    parser.add_argument(
        "--mask-dir",
        action="append",
        default=[],
        metavar="SCORE=PATH_OR_URI",
        help="Pinned mask source for recovering absent scores on selected tiles",
    )
    args = parser.parse_args()
    mask_sources = {}
    for item in args.mask_dir:
        score, separator, source = item.partition("=")
        if (
            not separator
            or score not in SCORE_COLUMNS
            or not source
            or score in mask_sources
        ):
            parser.error(f"Invalid or repeated --mask-dir {item!r}")
        mask_sources[score] = source
    manifest = prepare_selection(
        level=args.level,
        tiling=args.tiling,
        data_mapping=args.data_mapping,
        output=args.output,
        shards=args.shards,
        cap=args.cap,
        seed=args.seed,
        l2_embeddings=args.l2_embeddings,
        slide_overhead_tiles=args.slide_overhead_tiles,
        mask_sources=mask_sources,
    )
    count = sum(slide["tile_count"] for slide in manifest["slides"])
    print(
        f"Froze {count:,} level-{args.level} tiles in {args.output / 'manifest.json'}"
    )
    missing = {key: value for key, value in manifest["score_missing"].items() if value}
    if missing:
        print(f"Unavailable mask scores among selected tiles: {missing}")


if __name__ == "__main__":
    main()
