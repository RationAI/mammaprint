"""Geometry and viewer overlays for validation tile masks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import quote, urljoin, urlsplit
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import tifffile


if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd


@dataclass(frozen=True)
class SlideGeometry:
    width: int
    height: int
    level0_width: int
    level0_height: int
    downsample: float
    mpp_x: float
    mpp_y: float
    level: int | None = None

    @classmethod
    def from_wsi(cls, wsi: Path, mpp: float) -> SlideGeometry:
        if not wsi.is_file():
            raise FileNotFoundError(wsi)
        if wsi.name.lower().endswith((".ome.tif", ".ome.tiff")):
            from ratiopath.tifffile import TiffFile

            with TiffFile(wsi) as slide:
                level = slide.closest_level(mpp)
                height, width, *_ = slide.get_main_page(level).shape
                level0_height, level0_width, *_ = slide.get_main_page(0).shape
                mpp_x, mpp_y = slide.slide_resolution(level)
            return cls(
                width,
                height,
                level0_width,
                level0_height,
                level0_width / width,
                mpp_x,
                mpp_y,
                level,
            )
        from openslide import OpenSlideError
        from ratiopath.openslide import OpenSlide

        try:
            with OpenSlide(str(wsi)) as slide:
                # Tiling uses read_slides(mpp=...), which selects a level per WSI.
                level = slide.closest_level(mpp)
                width, height = slide.level_dimensions[level]
                level0_width, level0_height = slide.dimensions
                mpp_x, mpp_y = slide.slide_resolution(level)
                return cls(
                    width,
                    height,
                    level0_width,
                    level0_height,
                    float(slide.level_downsamples[level]),
                    mpp_x,
                    mpp_y,
                    level,
                )
        except OpenSlideError as error:
            raise ValueError(
                f"OpenSlide could not read WSI geometry: {error}"
            ) from error

    def level0_rectangles(self, rectangles: np.ndarray) -> np.ndarray:
        # Match OpenSlide.read_region_relative's truncation of tile starts.
        starts = np.floor(rectangles[:, :2] * self.downsample)
        ends = starts + np.ceil(
            (rectangles[:, 2:] - rectangles[:, :2]) * self.downsample
        )
        ends = np.minimum(ends, [self.level0_width, self.level0_height])
        return np.hstack((starts, ends)).astype(np.int64)


def tile_rectangles(
    frame: pd.DataFrame, extent: int, geometry: SlideGeometry | None
) -> np.ndarray:
    coordinates = frame[["x", "y"]].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(coordinates).all()
        or np.any(coordinates < 0)
        or np.any(coordinates != np.floor(coordinates))
        or np.any(coordinates > 2**31 - 1)
    ):
        raise ValueError("Tile positions must be finite nonnegative integer pixels.")
    coordinates = coordinates.astype(np.int64)
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Duplicate tile positions make geometry ambiguous.")
    bounds = coordinates + extent
    if geometry is not None:
        bounds = np.minimum(bounds, [geometry.width, geometry.height])
        if np.any(bounds <= coordinates):
            raise ValueError("Tile positions fall outside the selected WSI level.")
    return np.hstack((coordinates, bounds))


class MaskCoverage:
    """Coverage of a footprint union on a co-registered mask's pixel grid."""

    def __init__(
        self,
        path: Path,
        rectangles: np.ndarray,
        geometry: SlideGeometry,
        threshold: float,
    ) -> None:
        with tifffile.TiffFile(path) as image:
            mask = image.series[0].levels[0].asarray().squeeze()
        if (
            mask.ndim != 2
            or not mask.size
            or not np.isfinite(mask).all()
            or np.min(mask) < 0
            or np.max(mask) > 255
        ):
            raise ValueError("Expected a finite 2-D binary or probability mask.")
        height, width = mask.shape
        # The mask must span the same whole-slide frame. Allow rounding by a pixel.
        if abs(width - height * geometry.level0_width / geometry.level0_height) > max(
            2, 2 * geometry.level0_width / geometry.level0_height
        ):
            raise ValueError("Mask dimensions do not match the WSI aspect ratio.")
        cutoff = threshold if np.max(mask) <= 1 else threshold * 255
        self.integral = np.pad(
            (mask >= cutoff).cumsum(axis=0, dtype=np.int64).cumsum(axis=1),
            ((1, 0), (1, 0)),
        )
        scale = np.array(
            [width / geometry.level0_width, height / geometry.level0_height]
        )
        self.rectangles = np.ceil(rectangles * np.tile(scale, 2) - 0.5).astype(np.int64)

    def totals(self, rows: tuple[int, ...] | np.ndarray) -> tuple[int, int]:
        if not len(rows):
            return 0, 0
        rectangles = self.rectangles[np.asarray(rows)]
        events: dict[int, list[tuple[int, int, int]]] = {}
        for x0, y0, x1, y1 in rectangles.tolist():
            if x0 < x1 and y0 < y1:
                events.setdefault(y0, []).append((x0, x1, 1))
                events.setdefault(y1, []).append((x0, x1, -1))
        active: dict[tuple[int, int], int] = {}
        area = positive = 0
        edges = sorted(events)
        for y0, y1 in pairwise(edges):
            for x0, x1, delta in events[y0]:
                key = (x0, x1)
                active[key] = active.get(key, 0) + delta
                if not active[key]:
                    del active[key]
            if not active:
                continue
            merged: list[tuple[int, int]] = []
            for x0, x1 in sorted(active):
                if merged and x0 <= merged[-1][1]:
                    merged[-1] = merged[-1][0], max(x1, merged[-1][1])
                else:
                    merged.append((x0, x1))
            for x0, x1 in merged:
                area += (x1 - x0) * (y1 - y0)
                table = self.integral
                positive += int(
                    table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
                )
        return positive, area

    def fraction(self, rows: tuple[int, ...]) -> float:
        positive, area = self.totals(rows)
        return positive / area if area else 0.0


def write_overlay(path: Path, rectangles: np.ndarray, geometry: SlideGeometry) -> None:
    """Stream a binary OME-TIFF pyramid on the selected level's WSI frame."""
    tile = 256
    factors = [1]
    while math.ceil(max(geometry.width, geometry.height) / factors[-1]) > tile:
        factors.append(factors[-1] * 2)

    def tiles(factor: int, width: int, height: int) -> Iterator[np.ndarray]:
        intersections: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for x0, y0, x1, y1 in rectangles.tolist():
            x0, y0 = x0 // factor, y0 // factor
            x1, y1 = math.ceil(x1 / factor), math.ceil(y1 / factor)
            for x in range(x0 // tile, (x1 - 1) // tile + 1):
                for y in range(y0 // tile, (y1 - 1) // tile + 1):
                    intersections.setdefault((x, y), []).append((x0, y0, x1, y1))
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                pixels = np.zeros((tile, tile), dtype=np.uint8)
                for x0, y0, x1, y1 in intersections.get((x // tile, y // tile), ()):
                    pixels[max(0, y0 - y) : y1 - y, max(0, x0 - x) : x1 - x] = 255
                yield pixels

    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as writer:
        for i, factor in enumerate(factors):
            width = math.ceil(geometry.width / factor)
            height = math.ceil(geometry.height / factor)
            metadata = {
                "axes": "YX",
                "UUID": str(uuid5(NAMESPACE_URL, "mammaprint:" + path.name)),
                "PhysicalSizeX": geometry.mpp_x,
                "PhysicalSizeY": geometry.mpp_y,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
            }
            writer.write(
                tiles(factor, width, height),
                shape=(height, width),
                dtype=np.uint8,
                photometric="minisblack",
                tile=(tile, tile),
                compression="deflate",
                resolution=(
                    10000 / (geometry.mpp_x * factor),
                    10000 / (geometry.mpp_y * factor),
                ),
                resolutionunit="CENTIMETER",
                metadata=None if i else metadata,
                subifds=None if i else len(factors) - 1,
                subfiletype=1 if i else 0,
            )


@dataclass(frozen=True)
class ViewerTarget:
    tracking_uri: str
    experiment_id: str
    run_id: str
    artifact_root: str
    xopat_url: str
    artifact_mount: str = "secure_mlflow"

    def __post_init__(self) -> None:
        for url in (self.tracking_uri, self.xopat_url):
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Expected a credentials-free HTTP(S) viewer or tracking URL."
                )
        for component in (self.experiment_id, self.run_id, self.artifact_mount):
            if not component or component in {".", ".."} or "/" in component:
                raise ValueError("Invalid xOpat artifact mount component.")
        _relative_path(self.artifact_root)

    def link(self, wsi: str, layer_path: str) -> str:
        # Standalone fragment format used by the historical explanation viewer.
        # The opaque overlay path and generic layer label are shared by both views.
        background = PurePosixPath(wsi)
        if background.is_absolute():
            if background.parts[1:2] != ("mnt",):
                raise ValueError("The xOpat WSI must be mounted below /mnt.")
            background = PurePosixPath(*background.parts[2:])
        artifact = (
            PurePosixPath(self.artifact_mount)
            / self.experiment_id
            / self.run_id
            / "artifacts"
            / self.artifact_root
            / _relative_path(layer_path)
        )
        settings = {
            "params": {},
            "data": [_relative_path(str(background)), artifact.as_posix()],
            "background": [{"dataReference": 0, "lossless": False}],
            "visualizations": [
                {
                    "name": "Region review",
                    "lossless": True,
                    "shaders": {
                        "region": {
                            "name": "Region",
                            "type": "heatmap",
                            "dataReferences": [1],
                            "fixed": False,
                            "visible": True,
                            "params": {
                                "use_channel0": "r",
                                "color": "#f2c14e",
                                "opacity": 0.5,
                            },
                        }
                    },
                }
            ],
        }
        payload = quote(json.dumps(settings, separators=(",", ":")), safe="")
        return (
            urljoin(self.xopat_url.rstrip("/") + "/", "xopat/redirect.php")
            + "#"
            + payload
        )


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("Expected a safe relative viewer path.")
    return path.as_posix()
