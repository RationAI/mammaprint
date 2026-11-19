"""Pathologist-facing artifacts for tile-level MIL explanations.

The embedding parquet coordinates are expressed at the selected WSI pyramid
level.  Every helper in this module therefore requires explicit slide geometry;
slide dimensions are never inferred from the tile extent.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import tifffile
from numpy.typing import ArrayLike, NDArray


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from ml.explainability.data import SlideGeometry


OUTPUT_SCHEMA_VERSION = "1.1"
TILE_TABLE_BASE_COLUMNS = (
    "slide_id",
    "tile_index",
    "x",
    "y",
    "target",
    "prediction",
    "label",
)
TILE_ATTRIBUTION_COLUMNS = (
    "exact_mean",
    "integrated_gradients",
    "leave_one_out",
    "single",
    "attention",
)
TILE_TABLE_COLUMNS = TILE_TABLE_BASE_COLUMNS + TILE_ATTRIBUTION_COLUMNS


@dataclass(frozen=True, slots=True)
class RasterizedMask:
    """A signed attribution raster and the number of tiles covering each pixel."""

    values: NDArray[np.float32]
    coverage: NDArray[np.uint32]
    downsample: int


@dataclass(frozen=True, slots=True)
class SignedMaskPaths:
    """Paths and integer downsample factors of a signed OME-TIFF pair."""

    positive: Path
    negative: Path
    pyramid_factors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _AssembledScalarMask:
    """Compact ratiopath mask plus its selected-level pixel size."""

    values: NDArray[np.float32]
    coverage: NDArray[np.uint32]
    cell_height: int
    cell_width: int


def make_tile_table(
    *,
    slide_id: str,
    x: ArrayLike,
    y: ArrayLike,
    attributions: Mapping[str, Mapping[str, ArrayLike]],
    predictions: Mapping[str, float],
    labels: Mapping[str, float | None] | None = None,
) -> pd.DataFrame:
    """Build the canonical lossless wide table for one slide.

    ``attributions`` is nested as ``target -> method -> one score per tile``.
    The output has one row per tile and target, with stable nullable columns for
    every supported method. Native attention is repeated across targets, but its
    column name and manifest semantics make clear that it is not target-specific.
    All floating-point values are stored as lossless float64 in Parquet.
    """
    x_array = _one_dimensional_array("x", x, np.int64)
    y_array = _one_dimensional_array("y", y, np.int64)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must contain the same number of tile coordinates.")
    if not slide_id:
        raise ValueError("slide_id must not be empty.")
    if not attributions:
        raise ValueError("attributions must contain at least one target and method.")

    tile_count = len(x_array)
    tile_index: NDArray[np.int64] = np.arange(tile_count, dtype=np.int64)
    frames: list[pd.DataFrame] = []
    for target, methods in attributions.items():
        if target not in predictions:
            raise KeyError(f"Missing prediction for target {target!r}.")
        if not methods:
            raise ValueError(f"Target {target!r} has no attribution methods.")
        unknown_methods = set(methods) - set(TILE_ATTRIBUTION_COLUMNS)
        if unknown_methods:
            raise ValueError(
                f"Target {target!r} contains unsupported methods: "
                f"{sorted(unknown_methods)}."
            )
        prediction = float(predictions[target])
        if not np.isfinite(prediction):
            raise ValueError(f"Prediction for target {target!r} is not finite.")
        label = None if labels is None else labels.get(target)
        if label is not None and not np.isfinite(float(label)):
            raise ValueError(f"Label for target {target!r} is not finite.")

        columns: dict[str, Any] = {
            "slide_id": slide_id,
            "tile_index": tile_index,
            "x": x_array,
            "y": y_array,
            "target": str(target),
            "prediction": prediction,
            "label": label,
        }
        for method in TILE_ATTRIBUTION_COLUMNS:
            values = methods.get(method)
            if values is None:
                columns[method] = pd.array([pd.NA] * tile_count, dtype="Float64")
                continue
            scores = _one_dimensional_array(
                f"attributions[{target!r}][{method!r}]", values, np.float64
            )
            if len(scores) != tile_count:
                raise ValueError(
                    f"Attribution {target!r}/{method!r} has {len(scores)} scores; "
                    f"expected {tile_count}."
                )
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"Attribution {target!r}/{method!r} contains NaN or infinity."
                )
            columns[method] = scores
        frames.append(pd.DataFrame(columns))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[:, TILE_TABLE_COLUMNS]
    frame = frame.astype(
        {
            "slide_id": "string",
            "tile_index": "int64",
            "x": "int64",
            "y": "int64",
            "target": "string",
            "prediction": "float64",
        }
    )
    frame["label"] = pd.array(frame["label"], dtype="Float64")
    for method in TILE_ATTRIBUTION_COLUMNS:
        frame[method] = pd.array(frame[method], dtype="Float64")
    validate_tile_table(frame)
    return frame


def validate_tile_table(frame: pd.DataFrame) -> None:
    """Validate the canonical wide tile-table schema and row uniqueness."""
    missing = set(TILE_TABLE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Tile table is missing columns: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("Tile table must not be empty.")

    required = [column for column in TILE_TABLE_BASE_COLUMNS if column != "label"]
    if frame[required].isna().any().any():
        raise ValueError("Tile table contains missing required values.")
    if frame.duplicated(["slide_id", "tile_index", "target"]).any():
        raise ValueError("Tile table contains duplicate tile/target rows.")

    for column in ("tile_index", "x", "y"):
        values = frame[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Tile table column {column!r} must be integral.")
    for column in ("prediction",):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Tile table column {column!r} must be finite.")
    labels = frame["label"].dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("Tile table labels must be finite or null.")
    for method in TILE_ATTRIBUTION_COLUMNS:
        values = frame[method].dropna().to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Tile table attribution column {method!r} must be finite or null."
            )
    if not any(frame[method].notna().any() for method in TILE_ATTRIBUTION_COLUMNS):
        raise ValueError("Tile table contains no attribution values.")


def write_tile_table(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write a tile table with lossless float64 values."""
    validate_tile_table(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, TILE_TABLE_COLUMNS].to_parquet(
        destination,
        index=False,
        compression="zstd",
    )
    return destination


def rasterize_tile_scores(
    x: ArrayLike,
    y: ArrayLike,
    scores: ArrayLike,
    *,
    geometry: SlideGeometry,
    tile_extent: int,
    stride: int,
    raster_downsample: int = 1,
    max_pixels: int = 64_000_000,
) -> RasterizedMask:
    """Rasterize signed tile scores with ratiopath overlap aggregation.

    Ratiopath expands scalar tile outputs on its compact GCD-aligned grid and uses
    ``MeanAggregator`` to average overlaps. This helper samples that grid at the
    requested selected-level downsample for previews or analysis. Non-finite tile
    scores are absent and do not contribute to coverage.
    """
    _validate_geometry(geometry)
    if tile_extent <= 0:
        raise ValueError("tile_extent must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if raster_downsample <= 0:
        raise ValueError("raster_downsample must be positive.")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive.")

    height = math.ceil(geometry.height / raster_downsample)
    width = math.ceil(geometry.width / raster_downsample)
    if height * width > max_pixels:
        raise MemoryError(
            f"Requested raster has {height * width:,} pixels; use a larger "
            "raster_downsample or write_signed_ome_tiffs for streaming output."
        )

    assembled = _assemble_scalar_mask(
        x,
        y,
        scores,
        geometry=geometry,
        tile_extent=tile_extent,
        stride=stride,
    )
    values, coverage = _sample_assembled_mask(
        assembled,
        height=height,
        width=width,
        factor=raster_downsample,
    )
    return RasterizedMask(
        values=values,
        coverage=coverage,
        downsample=raster_downsample,
    )


def cohort_percentile_scale(
    score_sets: Iterable[ArrayLike],
    percentile: float = 99.0,
) -> float:
    """Return one cohort-wide absolute percentile scale for a method/target.

    NaN and infinity are ignored. An empty or all-zero cohort returns ``1.0`` so
    the scaling remains defined and produces black masks.
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100].")
    finite_parts: list[NDArray[np.float64]] = []
    for values in score_sets:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = np.abs(array[np.isfinite(array)])
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return 1.0
    scale = float(np.percentile(np.concatenate(finite_parts), percentile))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def scale_signed_scores(
    values: ArrayLike,
    scale: float,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Map signed scores to separate positive/negative uint8 masks."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    array = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(array), array, 0.0)
    positive = np.rint(np.clip(finite / scale, 0.0, 1.0) * 255).astype(np.uint8)
    negative = np.rint(np.clip(-finite / scale, 0.0, 1.0) * 255).astype(np.uint8)
    return positive, negative


def write_signed_ome_tiffs(
    output_dir: str | Path,
    stem: str,
    x: ArrayLike,
    y: ArrayLike,
    scores: ArrayLike,
    *,
    geometry: SlideGeometry,
    tile_extent: int,
    stride: int,
    scale: float,
    pyramid_factors: Sequence[int] | None = None,
    tile_size: int = 512,
) -> SignedMaskPaths:
    """Write positive/negative pyramidal OME-BigTIFF masks without a dense WSI.

    Ratiopath first assembles scalar scores and averages overlaps on a compact
    GCD-aligned mask. This module then streams that mask into OME-TIFF storage
    tiles because ratiopath's generic BigTIFF writer does not emit OME metadata.
    Pyramid factors are integer downsample factors relative to the selected WSI
    level.
    """
    _validate_geometry(geometry)
    if tile_extent <= 0:
        raise ValueError("tile_extent must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    if tile_size <= 0 or tile_size % 16:
        raise ValueError("tile_size must be a positive multiple of 16.")
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename stem, not a path.")

    factors = _pyramid_factors(geometry, tile_size, pyramid_factors)
    assembled = _assemble_scalar_mask(
        x,
        y,
        scores,
        geometry=geometry,
        tile_extent=tile_extent,
        stride=stride,
    )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    positive = directory / f"{stem}_positive.ome.tif"
    negative = directory / f"{stem}_negative.ome.tif"
    _write_one_ome_tiff(
        positive,
        stem=f"{stem}_positive",
        assembled=assembled,
        factors=factors,
        geometry=geometry,
        scale=scale,
        tile_size=tile_size,
        polarity="positive",
    )
    _write_one_ome_tiff(
        negative,
        stem=f"{stem}_negative",
        assembled=assembled,
        factors=factors,
        geometry=geometry,
        scale=scale,
        tile_size=tile_size,
        polarity="negative",
    )
    return SignedMaskPaths(
        positive=positive,
        negative=negative,
        pyramid_factors=factors,
    )


def build_tile_geojson(
    *,
    slide_id: str,
    x: ArrayLike,
    y: ArrayLike,
    scores: ArrayLike,
    geometry: SlideGeometry,
    tile_extent: int,
    target: str,
    method: str,
    fraction: float = 0.05,
) -> dict[str, Any]:
    """Build QuPath-compatible top/bottom tile polygons in level-0 pixels."""
    _validate_geometry(geometry)
    if tile_extent <= 0:
        raise ValueError("tile_extent must be positive.")
    if not 0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5].")
    x_array, y_array, score_array = _tile_arrays(x, y, scores)
    finite_indices = np.flatnonzero(np.isfinite(score_array))
    if finite_indices.size == 0:
        return {"type": "FeatureCollection", "features": []}

    ordered = finite_indices[np.argsort(score_array[finite_indices], kind="stable")]
    count = max(1, math.ceil(len(ordered) * fraction))
    selections = (
        ("bottom", ordered[:count], [0, 130, 200]),
        ("top", ordered[-count:][::-1], [230, 25, 75]),
    )
    percentage = f"{100 * fraction:g}%"

    features: list[dict[str, Any]] = []
    for direction, indices, color in selections:
        for index in indices:
            x0 = max(0, int(x_array[index]))
            y0 = max(0, int(y_array[index]))
            x1 = min(geometry.width, int(x_array[index]) + tile_extent)
            y1 = min(geometry.height, int(y_array[index]) + tile_extent)
            if x0 >= x1 or y0 >= y1:
                continue
            level0_x0 = x0 * geometry.downsample
            level0_y0 = y0 * geometry.downsample
            level0_x1 = x1 * geometry.downsample
            level0_y1 = y1 * geometry.downsample
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [level0_x0, level0_y0],
                                [level0_x1, level0_y0],
                                [level0_x1, level0_y1],
                                [level0_x0, level0_y1],
                                [level0_x0, level0_y0],
                            ]
                        ],
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": {
                            "name": (
                                f"{direction.capitalize()} {percentage} attention "
                                "diagnostic"
                                if method == "attention"
                                else f"{direction.capitalize()} {percentage} {target}"
                            ),
                            "color": color,
                        },
                        "slide_id": slide_id,
                        "tile_index": int(index),
                        "target": target,
                        "method": method,
                        "direction": direction,
                        "effect": (
                            "diagnostic_weight"
                            if method == "attention"
                            else "raises"
                            if score_array[index] > 0
                            else "lowers"
                            if score_array[index] < 0
                            else "neutral"
                        ),
                        "score": float(score_array[index]),
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def write_tile_geojson(
    path: str | Path,
    *,
    slide_id: str,
    x: ArrayLike,
    y: ArrayLike,
    scores: ArrayLike,
    geometry: SlideGeometry,
    tile_extent: int,
    target: str,
    method: str,
    fraction: float = 0.05,
) -> Path:
    """Write top/bottom tile regions as QuPath-importable GeoJSON."""
    feature_collection = build_tile_geojson(
        slide_id=slide_id,
        x=x,
        y=y,
        scores=scores,
        geometry=geometry,
        tile_extent=tile_extent,
        target=target,
        method=method,
        fraction=fraction,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(feature_collection, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def write_summary_png(
    path: str | Path,
    *,
    slide_id: str,
    record_num: str | int | None = None,
    thumbnail: ArrayLike,
    attribution_maps: Mapping[str, ArrayLike],
    attribution_coverages: Mapping[str, ArrayLike] | None = None,
    attribution_limits: Mapping[str, float] | None = None,
    prediction_text: str,
    label_text: str | None = None,
    faithfulness: Mapping[str, float] | None = None,
    curves: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
) -> Path:
    """Write WSI-aligned attribution overlays and faithfulness curves."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    panels = 1 + len(attribution_maps) + int(bool(curves))
    columns = min(3, panels)
    rows = math.ceil(panels / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    flat_axes = np.atleast_1d(axes).reshape(-1)

    thumbnail_array = np.asarray(thumbnail)
    if thumbnail_array.ndim not in {2, 3}:
        raise ValueError("thumbnail must be a 2D grayscale or 3D color image.")
    flat_axes[0].imshow(thumbnail_array)
    flat_axes[0].set_title("WSI thumbnail")
    flat_axes[0].axis("off")
    panel_index = 1
    for name, values in attribution_maps.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Attribution map {name!r} must be two-dimensional.")
        configured_limit = (
            None if attribution_limits is None else attribution_limits.get(name)
        )
        finite = np.abs(array[np.isfinite(array)])
        limit = (
            float(configured_limit)
            if configured_limit is not None
            else float(np.percentile(finite, 99))
            if finite.size
            else 1.0
        )
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        axis = flat_axes[panel_index]
        axis.imshow(thumbnail_array)
        coverage_values = (
            None if attribution_coverages is None else attribution_coverages.get(name)
        )
        coverage = (
            np.isfinite(array)
            if coverage_values is None
            else np.asarray(coverage_values) > 0
        )
        if coverage.shape != array.shape:
            raise ValueError(
                f"Attribution coverage {name!r} has shape {coverage.shape}; "
                f"expected {array.shape}."
            )
        magnitude_alpha = np.clip(np.abs(array) / limit, 0.0, 1.0) * 0.78
        alpha = np.where(coverage & np.isfinite(array), magnitude_alpha, 0.0)
        height, width = thumbnail_array.shape[:2]
        image = axis.imshow(
            array,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            alpha=alpha,
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        )
        axis.set_title(name)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        panel_index += 1

    if curves:
        curve_axis = flat_axes[panel_index]
        for name, (fractions, predictions) in curves.items():
            curve_axis.plot(fractions, predictions, label=name)
        curve_axis.set_xlabel("Fraction of tiles removed")
        curve_axis.set_ylabel("Raw model output")
        curve_axis.set_title("Patch flipping")
        curve_axis.grid(alpha=0.25)
        curve_axis.legend(fontsize="small")
        panel_index += 1

    for axis in flat_axes[panel_index:]:
        axis.axis("off")
    details = [prediction_text]
    if label_text:
        details.append(label_text)
    if faithfulness:
        details.extend(f"{name}={value:.4g}" for name, value in faithfulness.items())
    identity = (
        f"Record {record_num} · slide {slide_id}"
        if record_num is not None
        else slide_id
    )
    figure.suptitle(f"{identity} | " + " | ".join(details))
    figure.tight_layout()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a deterministic JSON manifest with the output schema version."""
    document = dict(payload)
    document.setdefault("schema_version", OUTPUT_SCHEMA_VERSION)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return destination


def _one_dimensional_array(
    name: str,
    values: ArrayLike,
    dtype: np.dtype[Any] | type[np.generic],
) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    return array


def _tile_arrays(
    x: ArrayLike,
    y: ArrayLike,
    scores: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    x_array = _one_dimensional_array("x", x, np.int64)
    y_array = _one_dimensional_array("y", y, np.int64)
    score_array = _one_dimensional_array("scores", scores, np.float64)
    if not (len(x_array) == len(y_array) == len(score_array)):
        raise ValueError("x, y, and scores must have the same length.")
    return x_array, y_array, score_array


def _validate_geometry(geometry: SlideGeometry) -> None:
    if geometry.width <= 0 or geometry.height <= 0:
        raise ValueError("Slide width and height must be positive.")
    if not np.isfinite(geometry.downsample) or geometry.downsample <= 0:
        raise ValueError("Slide level downsample must be finite and positive.")
    for name, value in (("mpp_x", geometry.mpp_x), ("mpp_y", geometry.mpp_y)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Slide {name} must be finite and positive.")


def _assemble_scalar_mask(
    x: ArrayLike,
    y: ArrayLike,
    scores: ArrayLike,
    *,
    geometry: SlideGeometry,
    tile_extent: int,
    stride: int,
) -> _AssembledScalarMask:
    """Assemble scalar tile outputs through ratiopath's compact mean mask."""
    from ratiopath.masks.mask_builders import MaskBuilder, MeanAggregator

    x_array, y_array, score_array = _tile_arrays(x, y, scores)
    if (x_array < 0).any() or (y_array < 0).any():
        raise ValueError("Ratiopath mask coordinates must be non-negative.")
    if (x_array % stride).any() or (y_array % stride).any():
        raise ValueError(
            "Tile coordinates are not aligned to the configured stride; refusing "
            "to snap explanation masks to a different grid."
        )

    builder = MaskBuilder(
        source_extents=(geometry.height, geometry.width),
        source_tile_extent=(tile_extent, tile_extent),
        output_tile_extent=(1, 1),
        stride=(stride, stride),
        n_channels=1,
        storage="inmemory",
        aggregation=MeanAggregator,
        dtype=np.float32,
    )
    try:
        maximum_y = int(builder.span[0] - tile_extent)
        maximum_x = int(builder.span[1] - tile_extent)
        if (x_array > maximum_x).any() or (y_array > maximum_y).any():
            raise ValueError(
                "Tile coordinates fall outside the regular grid represented by "
                "the configured slide extent, tile extent, and stride."
            )

        finite = np.isfinite(score_array)
        if finite.any():
            coordinates = np.stack((y_array[finite], x_array[finite]), axis=1)
            values = score_array[finite].astype(np.float32, copy=False)[:, np.newaxis]
            builder.update_batch(values, coordinates)
        result = builder.finalize()
        mask = np.asarray(result["mask"][0], dtype=np.float32).copy()
        coverage = np.asarray(result["overlap_counter"][0], dtype=np.uint32).copy()
        span = np.asarray(builder.span, dtype=np.int64)
        mask_extents = np.asarray(builder.mask_extents, dtype=np.int64)
        if (span % mask_extents).any():
            raise RuntimeError(
                "Ratiopath produced a non-integral scalar-mask pixel extent."
            )
        cell_height, cell_width = (span // mask_extents).tolist()
    finally:
        builder.cleanup()

    return _AssembledScalarMask(
        values=mask,
        coverage=coverage,
        cell_height=int(cell_height),
        cell_width=int(cell_width),
    )


def _sample_assembled_mask(
    assembled: _AssembledScalarMask,
    *,
    height: int,
    width: int,
    factor: int,
    output_y: int = 0,
    output_x: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Sample a compact ratiopath mask in selected-level output coordinates."""
    source_y = (np.arange(height, dtype=np.int64) + output_y) * factor
    source_x = (np.arange(width, dtype=np.int64) + output_x) * factor
    mask_y = np.minimum(
        source_y // assembled.cell_height,
        assembled.values.shape[0] - 1,
    )
    mask_x = np.minimum(
        source_x // assembled.cell_width,
        assembled.values.shape[1] - 1,
    )
    return (
        assembled.values[np.ix_(mask_y, mask_x)],
        assembled.coverage[np.ix_(mask_y, mask_x)],
    )


def _pyramid_factors(
    geometry: SlideGeometry,
    tile_size: int,
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    if requested is None:
        factors = [1]
        while (
            max(
                math.ceil(geometry.width / factors[-1]),
                math.ceil(geometry.height / factors[-1]),
            )
            > tile_size
        ):
            factors.append(factors[-1] * 2)
        return tuple(factors)

    requested_factors = tuple(int(factor) for factor in requested)
    if not requested_factors or requested_factors[0] != 1:
        raise ValueError("pyramid_factors must start with 1.")
    if any(factor <= 0 for factor in requested_factors):
        raise ValueError("pyramid_factors must be positive integers.")
    if any(current <= previous for previous, current in pairwise(requested_factors)):
        raise ValueError("pyramid_factors must be strictly increasing.")
    if any(
        float(value) != factor
        for value, factor in zip(requested, requested_factors, strict=True)
    ):
        raise ValueError("pyramid_factors must contain integers.")
    return requested_factors


def _iter_scaled_mask_tiles(
    assembled: _AssembledScalarMask,
    *,
    height: int,
    width: int,
    storage_tile_size: int,
    factor: int,
    scale: float,
    polarity: str,
) -> Iterator[NDArray[np.uint8]]:
    tile_rows = math.ceil(height / storage_tile_size)
    tile_columns = math.ceil(width / storage_tile_size)
    for tile_row in range(tile_rows):
        global_y0 = tile_row * storage_tile_size
        global_y1 = min(height, global_y0 + storage_tile_size)
        for tile_column in range(tile_columns):
            global_x0 = tile_column * storage_tile_size
            global_x1 = min(width, global_x0 + storage_tile_size)
            values: NDArray[np.float32] = np.zeros(
                (storage_tile_size, storage_tile_size), dtype=np.float32
            )
            sampled, _ = _sample_assembled_mask(
                assembled,
                height=global_y1 - global_y0,
                width=global_x1 - global_x0,
                factor=factor,
                output_y=global_y0,
                output_x=global_x0,
            )
            values[: sampled.shape[0], : sampled.shape[1]] = sampled
            if polarity == "positive":
                scaled = np.clip(values / scale, 0.0, 1.0)
            else:
                scaled = np.clip(-values / scale, 0.0, 1.0)
            yield np.rint(scaled * 255).astype(np.uint8)


def _write_one_ome_tiff(
    path: Path,
    *,
    stem: str,
    assembled: _AssembledScalarMask,
    factors: Sequence[int],
    geometry: SlideGeometry,
    scale: float,
    tile_size: int,
    polarity: str,
) -> None:
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as writer:
        for level_index, factor in enumerate(factors):
            height = math.ceil(geometry.height / factor)
            width = math.ceil(geometry.width / factor)
            metadata = None
            subifds = None
            subfiletype = None
            if level_index == 0:
                metadata = {
                    "axes": "YX",
                    "Name": stem,
                    "PhysicalSizeX": geometry.mpp_x,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": geometry.mpp_y,
                    "PhysicalSizeYUnit": "µm",
                }
                subifds = len(factors) - 1
            else:
                subfiletype = 1
            writer.write(
                data=_iter_scaled_mask_tiles(
                    assembled,
                    height=height,
                    width=width,
                    storage_tile_size=tile_size,
                    factor=factor,
                    scale=scale,
                    polarity=polarity,
                ),
                shape=(height, width),
                dtype=np.uint8,
                photometric="minisblack",
                tile=(tile_size, tile_size),
                compression="deflate",
                compressionargs={"level": 6},
                predictor=True,
                resolution=(
                    10_000 / (geometry.mpp_x * factor),
                    10_000 / (geometry.mpp_y * factor),
                ),
                resolutionunit="CENTIMETER",
                subifds=subifds,
                subfiletype=subfiletype,
                metadata=metadata,
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "TILE_ATTRIBUTION_COLUMNS",
    "TILE_TABLE_BASE_COLUMNS",
    "TILE_TABLE_COLUMNS",
    "RasterizedMask",
    "SignedMaskPaths",
    "build_tile_geojson",
    "cohort_percentile_scale",
    "make_tile_table",
    "rasterize_tile_scores",
    "scale_signed_scores",
    "validate_tile_table",
    "write_manifest",
    "write_signed_ome_tiffs",
    "write_summary_png",
    "write_tile_geojson",
    "write_tile_table",
]
