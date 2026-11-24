"""Verify slide registration and prepare viewer-compatible dataset mask TIFFs."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5
from xml.etree import ElementTree

import numpy as np
import tifffile


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ml.explainability.dataset_mask_lineage import MaskSource


_PREFIXES = {
    "blur": "Piqe_piqe_median_activity_mask_",
    "folding": "FoldingFunction_folding_test_",
    "residual": "ResidualArtifactsAndCoverage_coverage_mask_",
    "tissue": "",
    "epithelium": "",
    "cancer": "",
}
_TILE = 256
_DECODE_LIMIT = 64 * 1024 * 1024


class _UnavailableError(ValueError):
    """A safe per-slide preparation reason for preview and audit output."""


@dataclass(frozen=True)
class _Frame:
    width: int
    height: int
    mpp_x: float
    mpp_y: float


@dataclass(frozen=True)
class _AssetDirectory:
    viewer: PurePosixPath
    source: str
    local: Path | None = None
    run: str | None = None
    artifact: PurePosixPath | None = None

    def open(self, client: Any, name: str, scratch: Path) -> Path:
        if self.local is not None:
            return self.local / name
        assert self.run is not None and self.artifact is not None
        try:
            return Path(
                client.download_artifacts(
                    self.run, str(self.artifact / name), str(scratch)
                )
            )
        except Exception as error:
            raise _UnavailableError("Mask artifact download failed.") from error


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or re.search(r"[\\%?#:\x00-\x1f\x7f]", value)
    ):
        raise _UnavailableError("Expected a safe relative viewer path.")
    return path


def _mount_relative(value: str) -> PurePosixPath:
    if not value.startswith("/mnt/"):
        raise _UnavailableError("The source path is outside /mnt.")
    return _relative(value[len("/mnt/") :])


def _directory(
    client: Any, location: str, mount: PurePosixPath, root: Path | None
) -> _AssetDirectory:
    if location.startswith("/"):
        relative = _mount_relative(location)
        return _AssetDirectory(relative, location, (root or Path("/mnt")) / relative)

    artifact = re.fullmatch(
        r"mlflow-artifacts:/(\d+)/([0-9a-f]{32})/artifacts/(.+)", location
    )
    if artifact is None:
        raise _UnavailableError("Expected a pinned MLflow mask directory.")
    experiment, run, path = artifact.groups()
    directory = _relative(path)
    try:
        info = client.get_run(run).info
    except Exception as error:
        raise _UnavailableError("Mask source run is unavailable.") from error
    if str(info.experiment_id) != experiment or info.run_id != run:
        raise _UnavailableError("Mask source run or experiment differs from its URI.")
    return _AssetDirectory(
        mount / experiment / run / "artifacts" / directory,
        location,
        run=run,
        artifact=directory,
    )


def _rational(tag: Any) -> float:
    return float(tag.value[0]) / float(tag.value[1])


def _tiff_frame(image: tifffile.TiffFile, *, mask: bool) -> _Frame | None:
    page = image.pages[0]
    if not isinstance(page, tifffile.TiffPage):
        raise _UnavailableError("TIFF has no readable base page.")
    tags = page.tags
    if "Orientation" in tags and tags["Orientation"].value != 1:
        raise _UnavailableError("TIFF orientation differs from the WSI raster frame.")
    if mask and any(
        _rational(tags[name]) != 0
        for name in ("XPosition", "YPosition")
        if name in tags
    ):
        raise _UnavailableError("Mask source has a shifted raster origin.")

    unit = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 1
    units = {2: 25400, 3: 10000, 4: 1000, 5: 1.0}
    tag_mpp = None
    if unit in units and all(name in tags for name in ("XResolution", "YResolution")):
        tag_mpp = tuple(
            units[unit] / _rational(tags[name])
            for name in ("XResolution", "YResolution")
        )

    ome_mpp = None
    if image.ome_metadata:
        pixels = ElementTree.fromstring(image.ome_metadata).find(".//{*}Pixels")
        if pixels is not None:
            if mask and any(
                float(plane.get(f"Position{axis}", "0")) != 0
                for plane in pixels.findall("{*}Plane")
                for axis in ("X", "Y")
            ):
                raise _UnavailableError("Mask source has a shifted OME origin.")
            if all(f"PhysicalSize{axis}" in pixels.attrib for axis in ("X", "Y")):
                conversion = {"µm": 1.0, "um": 1.0, "nm": 0.001, "mm": 1000.0}
                ome_mpp = tuple(
                    float(pixels.attrib[f"PhysicalSize{axis}"])
                    * conversion[pixels.get(f"PhysicalSize{axis}Unit", "µm")]
                    for axis in ("X", "Y")
                )

    if (
        tag_mpp is not None
        and ome_mpp is not None
        and any(
            not math.isclose(a, b, rel_tol=0.001)
            for a, b in zip(tag_mpp, ome_mpp, strict=True)
        )
    ):
        raise _UnavailableError("TIFF and OME physical scales disagree.")
    mpp = ome_mpp or tag_mpp
    if mpp is None:
        return None
    if any(not math.isfinite(value) or value <= 0 for value in mpp):
        raise _UnavailableError("TIFF has an invalid calibrated physical scale.")
    return _Frame(page.imagewidth, page.imagelength, mpp[0], mpp[1])


def _wsi_frame(path: Path) -> _Frame:
    if not path.is_file():
        raise _UnavailableError("Original WSI is unavailable for registration.")
    if path.suffix.lower() in {".tif", ".tiff"}:
        with suppress(tifffile.TiffFileError), tifffile.TiffFile(path) as image:
            frame = _tiff_frame(image, mask=False)
            if frame is not None:
                return frame

    try:
        from ratiopath.openslide import OpenSlide

        with OpenSlide(str(path)) as slide:
            width, height = slide.dimensions
            mpp_x, mpp_y = slide.slide_resolution(0)
    except Exception as error:
        raise _UnavailableError(
            "Original WSI lacks readable level-0 geometry."
        ) from error
    if min(width, height) <= 0 or any(
        not math.isfinite(value) or value <= 0 for value in (mpp_x, mpp_y)
    ):
        raise _UnavailableError("Original WSI lacks calibrated level-0 geometry.")
    return _Frame(width, height, mpp_x, mpp_y)


def _require_registration(mask: _Frame, wsi: _Frame) -> None:
    for count, mpp, wsi_count, wsi_mpp in (
        (mask.width, mask.mpp_x, wsi.width, wsi.mpp_x),
        (mask.height, mask.mpp_y, wsi.height, wsi.mpp_y),
    ):
        if abs(count * mpp - wsi_count * wsi_mpp) > max(mpp, wsi_mpp):
            raise _UnavailableError("Mask physical extent differs from the WSI frame.")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pyramid_is_usable(image: tifffile.TiffFile) -> bool:
    levels = image.series[0].levels
    h, w = levels[0].shape
    for level in levels:
        page = level.pages[0]
        if (
            not isinstance(page, tifffile.TiffPage)
            or not page.is_tiled
            or page.compression not in {1, 5, 8, 32946}
            or page.photometric != tifffile.PHOTOMETRIC.MINISBLACK
            or level.dtype != np.uint8
            or len(level.shape) != 2
            or any(
                abs(actual - expected) > 1
                for actual, expected in zip(level.shape, (h, w), strict=True)
            )
        ):
            return False
        h, w = ((n + 1) // 2 for n in level.shape)
    page = levels[-1].pages[0]
    return isinstance(page, tifffile.TiffPage) and all(
        actual <= tile
        for actual, tile in zip(
            levels[-1].shape, (page.tilelength, page.tilewidth), strict=True
        )
    )


def _tiles(pixels: np.ndarray, *, binary: bool) -> Iterator[np.ndarray]:
    height, width = pixels.shape
    for y in range(0, height, _TILE):
        for x in range(0, width, _TILE):
            region = pixels[y : y + _TILE, x : x + _TILE]
            tile = np.zeros((_TILE, _TILE), dtype=np.uint8)
            h, w = region.shape
            tile[:h, :w] = (region != 0) * 255 if binary else region
            yield tile


def _reduce(pixels: np.ndarray, output: np.memmap, *, binary: bool) -> None:
    height, width = output.shape
    for y in range(0, height, _TILE):
        for x in range(0, width, _TILE):
            h, w = min(_TILE, height - y), min(_TILE, width - x)
            region = pixels[2 * y : 2 * (y + h), 2 * x : 2 * (x + w)]
            values = np.zeros((2 * h, 2 * w), dtype=np.uint16)
            values[: region.shape[0], : region.shape[1]] = (
                region != 0 if binary else region
            )
            cells = values.reshape(h, 2, w, 2)
            if binary:
                reduced = cells.max(axis=(1, 3)) * 255
            else:
                rows = np.minimum(2, region.shape[0] - 2 * np.arange(h))
                columns = np.minimum(2, region.shape[1] - 2 * np.arange(w))
                count = rows[:, None] * columns[None, :]
                reduced = (cells.sum(axis=(1, 3)) + count // 2) // count
            output[y : y + h, x : x + w] = reduced


def _export(
    page: tifffile.TiffPage,
    frame: _Frame,
    folder: Path,
    source_digest: str,
    *,
    binary: bool,
) -> tuple[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        temporary = Path(scratch)
        output = temporary / "mask.ome.tif"
        raw = temporary / "pixels"
        pixels = page.asarray(out=str(raw), maxworkers=1, buffersize=_DECODE_LIMIT)
        n_sublevels = (max(frame.width, frame.height) - 1).bit_length() - 8
        n_sublevels = max(0, n_sublevels)
        with tifffile.TiffWriter(output, bigtiff=True, ome=True) as writer:
            for level in range(n_sublevels + 1):
                factor = 2**level
                writer.write(
                    _tiles(pixels, binary=binary),
                    shape=pixels.shape,
                    dtype=np.uint8,
                    photometric="minisblack",
                    tile=(_TILE, _TILE),
                    compression="deflate",
                    maxworkers=1,
                    resolution=(
                        10000 / (frame.mpp_x * factor),
                        10000 / (frame.mpp_y * factor),
                    ),
                    resolutionunit="CENTIMETER",
                    metadata=None
                    if level
                    else {
                        "axes": "YX",
                        "UUID": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"mammaprint:dataset-mask:{source_digest}:{binary}",
                            )
                        ),
                        "PhysicalSizeX": frame.mpp_x,
                        "PhysicalSizeY": frame.mpp_y,
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeYUnit": "µm",
                    },
                    subifds=None if level else n_sublevels,
                    subfiletype=1 if level else 0,
                )
                if level < n_sublevels:
                    reduced_path = temporary / f"level-{level + 1}"
                    reduced = np.memmap(
                        reduced_path,
                        dtype=np.uint8,
                        mode="w+",
                        shape=tuple((dimension + 1) // 2 for dimension in pixels.shape),
                    )
                    _reduce(pixels, reduced, binary=binary)
                    del pixels
                    raw.unlink(missing_ok=True)
                    pixels, raw = reduced, reduced_path
        digest = _digest(output)
        name = f"{digest}.ome.tif"
        target = folder / name
        try:
            os.link(output, target)
        except FileExistsError:
            if target.is_symlink() or _digest(target) != digest:
                raise _UnavailableError(
                    "Existing content-addressed overlay differs from its hash."
                ) from None
    return name, digest


def _prepare_mask(
    path: Path,
    wsi: _Frame,
    viewer_path: PurePosixPath,
    source: str,
    *,
    binary: bool,
    export_dir: Path | None,
    export_prefix: PurePosixPath | None,
) -> dict[str, str]:
    with tifffile.TiffFile(path) as image:
        if len(image.series) != 1:
            raise _UnavailableError("Mask TIFF contains multiple image series.")
        page = image.pages[0]
        if not isinstance(page, tifffile.TiffPage):
            raise _UnavailableError("TIFF has no readable base page.")
        frame = _tiff_frame(image, mask=True)
        if frame is None:
            raise _UnavailableError("Mask TIFF has no calibrated physical scale.")
        _require_registration(frame, wsi)
        if (
            page.shape != (frame.height, frame.width)
            or page.photometric != tifffile.PHOTOMETRIC.MINISBLACK
        ):
            raise _UnavailableError(
                "Mask TIFF is not a single-channel foreground raster."
            )
        if page.dtype != np.uint8 and not (binary and page.dtype == np.bool_):
            raise _UnavailableError(
                "Mask TIFF encoding does not match the dataset layer."
            )
        segment_shape = (
            (page.tilelength, page.tilewidth)
            if page.is_tiled
            else (min(page.rowsperstrip, frame.height), frame.width)
        )
        if math.prod(segment_shape) > _DECODE_LIMIT:
            raise _UnavailableError(
                "Mask TIFF decode segment exceeds the memory bound."
            )

        direct = _pyramid_is_usable(image)
        if direct and binary:
            direct = all(
                segment is None or not np.any((segment != 0) & (segment != 255))
                for segment, _, _ in page.segments(
                    maxworkers=1, buffersize=_DECODE_LIMIT
                )
            )
        elif direct:
            # Confirm we can read the original encoding before linking a remote TIFF.
            next(page.segments(maxworkers=1))
        digest = _digest(path)
        if not direct:
            if export_dir is None or export_prefix is None:
                raise _UnavailableError(
                    "Mask needs a normalized TIFF export on a viewer mount."
                )
            name, digest = _export(page, frame, export_dir, digest, binary=binary)
            viewer_path = export_prefix / name
    return {"viewer_path": str(viewer_path), "source": source, "sha256": digest}


def prepare_mask_assets(
    client: Any,
    sources: Mapping[str, MaskSource],
    slide_paths: Mapping[str, str],
    *,
    artifact_mount: str,
    wsi_root: Path | None = None,
    mask_root: Path | None = None,
    export_dir: Path | None = None,
    export_viewer_prefix: str | None = None,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, str]]]:
    """Inspect original masks and write optional normalized exports locally."""
    if set(sources) - _PREFIXES.keys():
        raise ValueError("Unexpected dataset mask kind.")
    mount = _relative(artifact_mount)
    if len(mount.parts) != 1:
        raise ValueError("Expected one xOpat artifact mount component.")
    if (export_dir is None) != (export_viewer_prefix is None):
        raise ValueError("Supply both the export directory and its viewer path.")
    export_prefix = (
        _relative(export_viewer_prefix) if export_viewer_prefix is not None else None
    )
    if not sources:
        return {}, {}

    directories: dict[str, _AssetDirectory | str] = {}
    for location in dict.fromkeys(source.location for source in sources.values()):
        try:
            directories[location] = _directory(client, location, mount, mask_root)
        except _UnavailableError as error:
            directories[location] = str(error)

    assets: dict[str, dict[str, dict[str, str]]] = {}
    missing: dict[str, dict[str, str]] = {}
    basenames = Counter(PurePosixPath(path).stem for path in slide_paths.values())
    for slide, slide_path in slide_paths.items():
        try:
            wsi_relative = (
                _mount_relative(slide_path)
                if slide_path.startswith("/")
                else _relative(slide_path)
            )
            if basenames[wsi_relative.stem] > 1:
                raise _UnavailableError("WSIs share an ambiguous mask filename.")
            wsi = _wsi_frame((wsi_root or Path("/mnt")) / wsi_relative)
        except _UnavailableError as error:
            missing[slide] = dict.fromkeys(sources, str(error))
            continue
        except (OSError, ValueError, KeyError, ArithmeticError, ElementTree.ParseError):
            missing[slide] = dict.fromkeys(
                sources, "Original WSI geometry is unreadable."
            )
            continue

        basename = wsi_relative.with_suffix(".tiff").name
        for kind, source in sources.items():
            directory = directories[source.location]
            if isinstance(directory, str):
                missing.setdefault(slide, {})[kind] = directory
                continue
            name = _PREFIXES[kind] + basename
            try:
                with tempfile.TemporaryDirectory() as scratch:
                    path = directory.open(client, name, Path(scratch))
                    asset = _prepare_mask(
                        path,
                        wsi,
                        directory.viewer / name,
                        f"{directory.source}/{name}",
                        binary=kind != "cancer",
                        export_dir=export_dir,
                        export_prefix=export_prefix,
                    )
                assets.setdefault(slide, {})[kind] = asset
            except _UnavailableError as error:
                missing.setdefault(slide, {})[kind] = str(error)
            except FileNotFoundError:
                missing.setdefault(slide, {})[kind] = "Mask TIFF is unavailable."
            except (
                OSError,
                ValueError,
                KeyError,
                ArithmeticError,
                ElementTree.ParseError,
            ):
                missing.setdefault(slide, {})[kind] = "Mask TIFF is unreadable."
    return assets, missing
