"""Pathologist-facing artifacts for tile-level MIL explanations.

The embedding parquet coordinates are expressed at the selected WSI pyramid
level.  Every helper in this module therefore requires explicit slide geometry;
slide dimensions are never inferred from the tile extent.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tifffile
from numpy.typing import ArrayLike, NDArray


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from ml.explainability.data import SlideGeometry


OUTPUT_SCHEMA_VERSION = "1.0"
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


def write_pathologist_report(
    path: str | Path,
    *,
    slides: Iterable[Mapping[str, Any]],
    summary_paths: Mapping[str, str | Path] | None = None,
    viewer_urls: Mapping[str, str] | None = None,
    title: str = "MIL tile-explanation review",
    metadata: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> Path:
    """Write a self-contained cohort report for pathologist review.

    ``record_num`` is treated as the primary clinical key and ``slide_id`` as a
    secondary technical identifier. ``summary_paths`` and ``viewer_urls`` may be
    keyed by either value; record number is checked first. Summary PNGs are
    embedded as compact data-URL previews so the report has no asset dependency.
    An optional
    per-slide ``viewer_url`` field in ``slides`` is also accepted as a fallback.

    The review controls are intentionally browser-local. The download action
    creates a CSV on the reviewer's computer and does not write back to MLflow.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    slide_rows = [dict(row) for row in slides]
    summaries = {} if summary_paths is None else summary_paths
    viewers = {} if viewer_urls is None else viewer_urls

    status_counts: dict[str, int] = {}
    task_values: set[str] = set()
    cards: list[str] = []
    for row in slide_rows:
        status = _report_text(row.get("status"), fallback="unknown")
        task = _report_task(row)
        status_counts[status] = status_counts.get(status, 0) + 1
        task_values.add(task)
        identifiers = _report_identifiers(row)
        summary_path = _report_mapping_value(summaries, identifiers)
        viewer_url = _report_mapping_value(viewers, identifiers)
        if viewer_url is None and not _report_is_missing(row.get("viewer_url")):
            viewer_url = str(row["viewer_url"])
        cards.append(
            _pathologist_slide_card(
                row=row,
                task=task,
                status=status,
                summary_path=summary_path,
                viewer_url=viewer_url,
            )
        )

    status_options = "".join(
        f'<option value="{_report_escape(status)}">'
        f"{_report_escape(status.title())} ({count})</option>"
        for status, count in sorted(status_counts.items())
    )
    task_options = "".join(
        f'<option value="{_report_escape(task)}">'
        f"{_report_escape(_report_task_label(task))}</option>"
        for task in sorted(task_values)
    )
    metadata_html = _report_metadata(metadata)
    warning_items = [
        (
            "These maps explain the model output; they are not causal evidence or a "
            "diagnosis."
        ),
        (
            "Red regions raise the named raw output and blue regions lower it. "
            "Color strength represents attribution magnitude on the cohort-wide scale."
        ),
        (
            "Attention, when available, is a non-target-specific diagnostic and "
            "should not be interpreted as the primary explanation."
        ),
        *[str(value) for value in warnings],
    ]
    warnings_html = "".join(
        f"<li>{_report_escape(value)}</li>" for value in warning_items
    )
    cards_html = (
        "".join(cards)
        if cards
        else (
            '<div class="empty-state">No slides were available for this report.</div>'
        )
    )
    total = len(slide_rows)
    ok_count = status_counts.get("ok", 0)

    document = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
        "connect-src 'none'; base-uri 'none'; form-action 'none';\">"
        f"<title>{_report_escape(title)}</title>"
        """
<style>
:root { color-scheme: light; --ink:#17202a; --muted:#5c6875; --line:#dce2e8;
  --paper:#fff; --canvas:#f4f6f8; --brand:#155b75; --brand-soft:#e7f3f7;
  --ok:#177245; --warn:#9b5d00; --bad:#a12622; --raise:#d73027; --lower:#2474b5; }
* { box-sizing:border-box; }
body { margin:0; background:var(--canvas); color:var(--ink);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.45; }
.shell { width:min(1440px,calc(100% - 32px)); margin:0 auto; padding:32px 0 64px; }
.hero,.guide,.toolbar,.card,.empty-state { background:var(--paper); border:1px solid var(--line);
  border-radius:14px; box-shadow:0 1px 2px rgba(20,31,43,.04); }
.hero { padding:28px; border-top:5px solid var(--brand); }
h1,h2,h3,p { margin-top:0; } h1 { margin-bottom:8px; font-size:clamp(1.65rem,3vw,2.35rem); }
.lede { color:var(--muted); max-width:78ch; margin-bottom:20px; }
.counts { display:flex; flex-wrap:wrap; gap:10px; }
.count { padding:8px 12px; border-radius:999px; background:var(--brand-soft); font-weight:650; }
.meta { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px;
  margin:20px 0 0; }
.meta div,.facts div,.diagnostics div { min-width:0; }
dt { color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.04em;
  text-transform:uppercase; } dd { margin:2px 0 0; overflow-wrap:anywhere; }
.guide { margin-top:18px; padding:22px; display:grid; grid-template-columns:1fr 1fr; gap:22px; }
.method { border-left:4px solid var(--brand); padding-left:13px; }
.method h3 { margin-bottom:4px; font-size:1rem; }
.method p,.guide li { color:#3f4b56; }
.legend { display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 0; }
.swatch { width:13px; height:13px; display:inline-block; border-radius:3px; margin-right:6px; }
.raise { background:var(--raise); } .lower { background:var(--lower); }
.warnings { grid-column:1/-1; margin:0; padding:14px 18px 14px 34px;
  background:#fff8e8; border-radius:9px; }
.toolbar { position:sticky; top:8px; z-index:3; margin:18px 0; padding:14px;
  display:grid; grid-template-columns:minmax(220px,1fr) 180px 180px auto; gap:10px; }
input,select,textarea,button { font:inherit; }
.toolbar input,.toolbar select,.review input,.review select,.review textarea { width:100%; border:1px solid #b9c3cc;
  border-radius:8px; background:#fff; padding:9px 10px; color:var(--ink); }
button,.viewer-link { border:0; border-radius:8px; padding:10px 14px; font-weight:700;
  cursor:pointer; text-decoration:none; text-align:center; }
button { background:var(--brand); color:#fff; }
.viewer-link { display:inline-flex; align-items:center; justify-content:center; color:#fff;
  background:var(--brand); margin-top:10px; }
.result-count { color:var(--muted); margin:0 2px 12px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr)); gap:18px; }
.card { overflow:hidden; }
.card[hidden] { display:none; }
.card-head { padding:18px 20px 14px; display:flex; justify-content:space-between; gap:16px; }
.eyebrow { display:block; color:var(--muted); font-size:.74rem; font-weight:750;
  letter-spacing:.08em; text-transform:uppercase; }
.card h2 { margin:2px 0 1px; font-size:1.32rem; overflow-wrap:anywhere; }
.slide-id { color:var(--muted); font-size:.88rem; overflow-wrap:anywhere; }
.badges { display:flex; align-items:flex-start; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.badge { white-space:nowrap; padding:5px 9px; border-radius:999px; background:#edf1f4;
  font-size:.76rem; font-weight:750; }
.badge-ok { background:#e7f5ed; color:var(--ok); } .badge-failed { background:#fdebea; color:var(--bad); }
.badge-unavailable,.badge-unknown { background:#fff2d8; color:var(--warn); }
.preview { margin:0; border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  background:#eef1f3; min-height:240px; display:grid; place-items:center; }
.preview img { width:100%; max-height:640px; object-fit:contain; display:block; }
.preview-missing { color:var(--muted); padding:80px 20px; text-align:center; }
.body { padding:18px 20px 20px; }
.facts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px 20px; }
.fact-group { padding:12px; background:#f7f9fa; border-radius:9px; }
.fact-group h3 { margin-bottom:9px; font-size:.92rem; }
.fact-group dl { display:grid; grid-template-columns:1fr; gap:9px; margin:0; }
.error { margin:14px 0 0; padding:11px 13px; border-left:4px solid var(--bad);
  background:#fff0ef; overflow-wrap:anywhere; }
details { margin-top:14px; border-top:1px solid var(--line); padding-top:12px; }
summary { cursor:pointer; font-weight:700; }
.diagnostics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 18px;
  margin:12px 0 0; }
.review { margin-top:16px; padding:14px; border-radius:10px; background:var(--brand-soft); }
.review h3 { margin-bottom:3px; font-size:1rem; }.review-note { color:var(--muted); font-size:.82rem; }
.review-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:11px; }
.review label { display:block; font-size:.84rem; font-weight:700; }.review textarea { min-height:74px; resize:vertical; }
.comment { grid-column:1/-1; }.no-results { display:none; }
.empty-state,.no-results { padding:36px; text-align:center; color:var(--muted); }
@media (max-width:760px) { .shell { width:min(100% - 18px,1440px); padding-top:12px; }
  .guide,.toolbar { grid-template-columns:1fr; }.facts,.review-grid,.diagnostics { grid-template-columns:1fr; }
  .comment,.warnings { grid-column:auto; }.toolbar { position:static; } }
@media print { body { background:#fff; }.toolbar,.review,.viewer-link { display:none!important; }
  .shell { width:100%; padding:0; }.card { break-inside:avoid; box-shadow:none; } }
</style></head><body><main class="shell">
"""
        f'<section class="hero"><span class="eyebrow">Pathologist overview</span>'
        f"<h1>{_report_escape(title)}</h1>"
        '<p class="lede">Review predictions together with tile-level evidence from '
        "the complete MIL aggregator and prediction head. Use the controls on each "
        "slide to capture whether the prediction and highlighted regions are "
        "clinically coherent.</p>"
        f'<div class="counts"><span class="count">{total} slides</span>'
        f'<span class="count">{ok_count} completed</span></div>{metadata_html}</section>'
        """
<section class="guide" aria-labelledby="reading-guide"><div>
<h2 id="reading-guide">How to read the maps</h2>
<div class="method"><h3>Integrated gradients (primary)</h3>
<p>Attributes the final prediction through both the aggregator and head, relative
to the training-cohort mean embedding.</p></div>
<div class="method"><h3>Leave-one-tile-out (primary)</h3>
<p>Measures the exact change in the final prediction when one tile is removed from
the bag.</p></div>
<div class="legend" aria-label="Attribution color legend">
<span><i class="swatch raise"></i>Raises the named target</span>
<span><i class="swatch lower"></i>Lowers the named target</span></div></div>
<div><h2>Target semantics</h2><p><strong>Luminal A logit:</strong> positive evidence
raises the model's Luminal A evidence; negative evidence lowers it.</p>
<p><strong>MammaPrint index:</strong> positive evidence raises the predicted index;
negative evidence lowers it.</p>
<details><summary>Optional diagnostic methods</summary><p><strong>Single-tile
sufficiency</strong> asks what each tile predicts alone. <strong>Patch flipping</strong>
tests whether removing highly ranked tiles changes the output as expected.
<strong>Attention</strong> is shown only as a non-target-specific model diagnostic.</p>
</details></div>
"""
        f'<ul class="warnings">{warnings_html}</ul></section>'
        """
<section class="toolbar" aria-label="Report controls">
<input id="search" type="search" placeholder="Search record number or slide ID"
 aria-label="Search record number or slide ID">
<select id="status-filter" aria-label="Filter by status"><option value="">All statuses</option>
"""
        f"{status_options}</select>"
        '<select id="task-filter" aria-label="Filter by task"><option value="">All tasks</option>'
        f"{task_options}</select>"
        '<button id="download-review" type="button">Download review CSV</button></section>'
        '<p id="result-count" class="result-count" aria-live="polite"></p>'
        f'<div id="cards" class="cards">{cards_html}</div>'
        '<div id="no-results" class="no-results">No slides match these filters.</div>'
        """
</main><script>
(() => {
  "use strict";
  const cards = Array.from(document.querySelectorAll(".card"));
  const search = document.getElementById("search");
  const status = document.getElementById("status-filter");
  const task = document.getElementById("task-filter");
  const count = document.getElementById("result-count");
  const empty = document.getElementById("no-results");
  const applyFilters = () => {
    const needle = search.value.trim().toLocaleLowerCase();
    let visible = 0;
    cards.forEach((card) => {
      const matchesSearch = !needle || card.dataset.search.includes(needle);
      const matchesStatus = !status.value || card.dataset.status === status.value;
      const matchesTask = !task.value || card.dataset.task === task.value;
      const show = matchesSearch && matchesStatus && matchesTask;
      card.hidden = !show;
      if (show) visible += 1;
    });
    count.textContent = `${visible} of ${cards.length} slides shown`;
    empty.style.display = cards.length && !visible ? "block" : "none";
  };
  [search, status, task].forEach((control) => control.addEventListener("input", applyFilters));
  applyFilters();

  const csvCell = (value) => {
    let text = String(value ?? "");
    if (/^[=+@-]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  document.getElementById("download-review").addEventListener("click", () => {
    const header = ["record_num", "slide_id", "prediction_makes_sense",
      "highlighted_regions_make_sense", "missed_relevant_regions", "reviewer",
      "review_comment"];
    const rows = cards.map((card) => [card.dataset.recordNum, card.dataset.slideId,
      card.querySelector('[data-review="prediction"]').value,
      card.querySelector('[data-review="regions"]').value,
      card.querySelector('[data-review="missed-regions"]').value,
      card.querySelector('[data-review="reviewer"]').value,
      card.querySelector('[data-review="review-comment"]').value]);
    const csv = [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\r\n");
    const blob = new Blob(["\ufeff" + csv], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "pathologist_review.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });
})();
</script></body></html>
"""
    )
    destination.write_text(document, encoding="utf-8")
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


def _pathologist_slide_card(
    *,
    row: Mapping[str, Any],
    task: str,
    status: str,
    summary_path: Any | None,
    viewer_url: Any | None,
) -> str:
    record_num = _report_text(row.get("record_num"), fallback="Unavailable")
    slide_id = _report_text(row.get("slide_id"), fallback="Unavailable")
    search_text = f"{record_num} {slide_id}".lower()
    status_class = {
        "ok": "badge-ok",
        "failed": "badge-failed",
        "unavailable": "badge-unavailable",
    }.get(status.lower(), "badge-unknown")

    embedded_preview = _report_embedded_png(summary_path)
    if embedded_preview is None:
        preview_html = (
            '<div class="preview-missing" role="img" '
            f'aria-label="Preview unavailable for record {_report_escape(record_num)}">'
            "Preview unavailable</div>"
        )
    else:
        preview_html = (
            f'<img src="{embedded_preview}" loading="lazy" '
            f'alt="Explanation summary for record {_report_escape(record_num)}">'
        )

    safe_viewer_url = _report_safe_viewer_url(viewer_url)
    viewer_html = ""
    if safe_viewer_url is not None:
        viewer_html = (
            f'<a class="viewer-link" href="{_report_escape(safe_viewer_url)}" '
            'target="_blank" rel="noopener noreferrer">'
            "Open interactive overlay in xOpat ↗</a>"
        )

    facts = _report_prediction_facts(row, task)
    facts_html = "".join(
        '<section class="fact-group">'
        f"<h3>{_report_escape(group_name)}</h3><dl>"
        + "".join(
            f"<div><dt>{_report_escape(label)}</dt>"
            f"<dd>{_report_escape(value)}</dd></div>"
            for label, value in values
        )
        + "</dl></section>"
        for group_name, values in facts
        if values
    )
    if not facts_html:
        facts_html = (
            '<section class="fact-group"><h3>Prediction</h3>'
            "<p>Prediction values unavailable.</p></section>"
        )

    error_html = ""
    error = row.get("error")
    if not _report_is_missing(error):
        error_html = (
            '<p class="error"><strong>Slide processing warning:</strong> '
            f"{_report_escape(error)}</p>"
        )

    diagnostic_keys = [
        key
        for key in sorted(row)
        if key == "tile_count"
        or key == "level"
        or key == "runtime_seconds"
        or key == "overlay_status"
        or key == "summary_status"
        or key == "leave_one_out_status"
        or key.startswith(
            (
                "prediction_raw/",
                "ig_baseline_raw/",
                "ig_completeness_",
                "faithfulness_",
            )
        )
    ]
    diagnostics_html = "".join(
        f"<div><dt>{_report_escape(_report_diagnostic_label(key))}</dt>"
        f"<dd>{_report_escape(_report_text(row.get(key)))}</dd></div>"
        for key in diagnostic_keys
        if not _report_is_missing(row.get(key))
    )
    if not diagnostics_html:
        diagnostics_html = "<p>No optional diagnostics were recorded.</p>"

    return (
        f'<article class="card" data-search="{_report_escape(search_text)}" '
        f'data-status="{_report_escape(status)}" data-task="{_report_escape(task)}" '
        f'data-record-num="{_report_escape(record_num)}" '
        f'data-slide-id="{_report_escape(slide_id)}">'
        '<header class="card-head"><div><span class="eyebrow">Record number</span>'
        f"<h2>{_report_escape(record_num)}</h2>"
        f'<div class="slide-id">Slide {_report_escape(slide_id)}</div></div>'
        f'<div class="badges"><span class="badge {status_class}">'
        f'{_report_escape(status.title())}</span><span class="badge">'
        f"{_report_escape(_report_task_label(task))}</span></div></header>"
        f'<figure class="preview">{preview_html}</figure>'
        f'<div class="body"><div class="facts">{facts_html}</div>{error_html}'
        f"{viewer_html}"
        "<details><summary>Model diagnostics</summary>"
        f'<dl class="diagnostics">{diagnostics_html}</dl></details>'
        '<section class="review"><h3>Pathologist review</h3>'
        '<p class="review-note">Entries stay in this browser page until you download '
        "the CSV. They are not sent back to MLflow.</p>"
        '<div class="review-grid"><label>Prediction makes sense'
        f'<select data-review="prediction" aria-label="Prediction review for record '
        f'{_report_escape(record_num)}"><option value="">Not reviewed</option>'
        '<option value="yes">Yes</option><option value="no">No</option>'
        '<option value="uncertain">Uncertain</option></select></label>'
        "<label>Highlighted regions make sense"
        f'<select data-review="regions" aria-label="Region review for record '
        f'{_report_escape(record_num)}"><option value="">Not reviewed</option>'
        '<option value="yes">Yes</option><option value="no">No</option>'
        '<option value="uncertain">Uncertain</option></select></label>'
        "<label>Were relevant regions missed?"
        f'<select data-review="missed-regions" aria-label="Missed region review for record '
        f'{_report_escape(record_num)}"><option value="">Not reviewed</option>'
        '<option value="yes">Yes</option><option value="no">No</option>'
        '<option value="uncertain">Uncertain</option></select></label>'
        f'<label>Reviewer<input data-review="reviewer" maxlength="200" '
        f'aria-label="Reviewer for record {_report_escape(record_num)}" '
        f'placeholder="Name or initials"></label>'
        f'<label class="comment">Comment<textarea data-review="review-comment" '
        f'maxlength="4000" aria-label="Review comment for record {_report_escape(record_num)}" '
        f'placeholder="Optional note"></textarea></label></div></section></div></article>'
    )


def _report_prediction_facts(
    row: Mapping[str, Any],
    task: str,
) -> list[tuple[str, list[tuple[str, str]]]]:
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    if task in {"class", "both"}:
        observed_class = _report_first(
            row,
            "class",
            "class_label",
            "label/luminal_a_logit",
        )
        predicted_class = _report_first(
            row,
            "prediction_class",
            "prediction_class_label",
        )
        logit = _report_first(
            row, "prediction_luminal_a_logit", "prediction_raw/luminal_a_logit"
        )
        logit_number = _report_float(logit)
        if _report_is_missing(predicted_class) and logit_number is not None:
            predicted_class = "Luminal A" if logit_number >= 0 else "Luminal B"
        probability = _report_first(
            row,
            "prediction_luminal_a_probability",
            "prediction_probability/luminal_a_logit",
        )
        class_values: list[tuple[str, str]] = []
        if not _report_is_missing(observed_class):
            class_values.append(("Ground truth", _report_text(observed_class)))
        if not _report_is_missing(predicted_class):
            class_values.append(("Model prediction", _report_text(predicted_class)))
        if not _report_is_missing(probability):
            class_values.append(
                ("Luminal A probability", _report_number(probability, decimals=3))
            )
        groups.append(("Class", class_values))

    if task in {"index", "both"}:
        observed_index = _report_first(
            row,
            "mammaprint_index",
            "label/mammaprint_index",
        )
        predicted_index = _report_first(
            row,
            "prediction_mammaprint_index",
            "prediction_raw/mammaprint_index",
        )
        index_values: list[tuple[str, str]] = []
        if not _report_is_missing(observed_index):
            index_values.append(
                ("Ground truth", _report_number(observed_index, decimals=4))
            )
        if not _report_is_missing(predicted_index):
            index_values.append(
                ("Model prediction", _report_number(predicted_index, decimals=4))
            )
        groups.append(("MammaPrint index", index_values))
    return groups


def _report_task(row: Mapping[str, Any]) -> str:
    configured = _report_text(row.get("task"), fallback="").strip().lower()
    configured = {"type": "class", "classification": "class"}.get(
        configured, configured
    )
    if configured in {"class", "index", "both"}:
        return configured
    has_class = any(
        not _report_is_missing(row.get(key))
        for key in (
            "class",
            "class_label",
            "prediction_class",
            "prediction_raw/luminal_a_logit",
        )
    )
    has_index = any(
        not _report_is_missing(row.get(key))
        for key in (
            "mammaprint_index",
            "prediction_mammaprint_index",
            "prediction_raw/mammaprint_index",
        )
    )
    if has_class and has_index:
        return "both"
    return "class" if has_class else "index" if has_index else "unknown"


def _report_task_label(task: str) -> str:
    return {
        "class": "Class",
        "index": "Index",
        "both": "Class + index",
        "unknown": "Task unavailable",
    }.get(task, task)


def _report_identifiers(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value)
        for value in (row.get("record_num"), row.get("slide_id"))
        if not _report_is_missing(value)
    )


def _report_mapping_value(
    values: Mapping[str, Any],
    identifiers: Sequence[str],
) -> Any | None:
    for identifier in identifiers:
        if identifier in values:
            return values[identifier]
    return None


def _report_embedded_png(path: Any | None) -> str | None:
    if _report_is_missing(path):
        return None
    source = Path(str(path))
    if source.suffix.lower() != ".png":
        raise ValueError(f"Report summary must be a PNG file: {source}")
    try:
        from PIL import Image

        with Image.open(source) as image:
            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=82,
                optimize=True,
            )
        payload = buffer.getvalue()
        media_type = "image/jpeg"
    except (ImportError, OSError):
        try:
            payload = source.read_bytes()
        except OSError:
            return None
        media_type = "image/png"
    return f"data:{media_type};base64," + base64.b64encode(payload).decode("ascii")


def _report_safe_viewer_url(value: Any | None) -> str | None:
    if _report_is_missing(value):
        return None
    url = str(value).strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def _report_metadata(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return ""
    items = "".join(
        f"<div><dt>{_report_escape(key)}</dt>"
        f"<dd>{_report_escape(_report_text(value))}</dd></div>"
        for key, value in metadata.items()
        if not _report_is_missing(value)
    )
    return f'<dl class="meta">{items}</dl>' if items else ""


def _report_first(row: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = row.get(key)
        if not _report_is_missing(value):
            return value
    return None


def _report_number(value: Any, *, decimals: int) -> str:
    number = _report_float(value)
    if number is None:
        return _report_text(value)
    return f"{number:.{decimals}f}"


def _report_float(value: Any) -> float | None:
    if _report_is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _report_diagnostic_label(key: str) -> str:
    return key.replace("/", " · ").replace("_", " ").strip().title()


def _report_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _report_text(value: Any, *, fallback: str = "Unavailable") -> str:
    if _report_is_missing(value):
        return fallback
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return f"{number:.5g}" if np.isfinite(number) else fallback
    return str(value)


def _report_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


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
    "write_pathologist_report",
    "write_signed_ome_tiffs",
    "write_summary_png",
    "write_tile_geojson",
    "write_tile_table",
]
