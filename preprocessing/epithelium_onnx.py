"""Generate epithelial masks from tiles or whole slides with the ONNX model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import onnxruntime as ort
import openslide
import pandas as pd
import pyvips
from mlflow.artifacts import download_artifacts
from PIL import Image, ImageOps


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


DEFAULT_MODEL_URI = (
    "mlflow-artifacts://mlflow.rationai-mlflow:5000/10/"
    "39f821ed5b964c71a603cc6db196f9fd/artifacts/"
    "checkpoints/epoch=19-step=32020/model.onnx/model.onnx"
)
DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DEFAULT_EXPERIMENT_NAME = "MammaPrint"
DEFAULT_RUN_NAME = "MammaPrint Epithelium ONNX Masks"
MODEL_MPP = 0.468
TILE_SIZE = 512
STRIDE = 256
SUPPORTED_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".mrxs",
    ".ndpi",
    ".png",
    ".scn",
    ".svs",
    ".tif",
    ".tiff",
    ".vms",
    ".vmu",
    ".webp",
}


@dataclass(frozen=True)
class InputImage:
    path: Path
    relative_path: Path


class TileSource:
    def __init__(self, path: Path) -> None:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in {"RGBA", "LA"}:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            self.pixels = np.asarray(image, dtype=np.uint8)
        self.height, self.width = self.pixels.shape[:2]

    def read_patch(self, x: int, y: int, tile_size: int) -> tuple[np.ndarray, int, int]:
        width = min(tile_size, self.width - x)
        height = min(tile_size, self.height - y)
        return self.pixels[y : y + height, x : x + width], width, height

    def close(self) -> None:
        pass


class SlideSource:
    def __init__(self, path: Path, target_mpp: float, source_mpp: float | None) -> None:
        self.slide = openslide.OpenSlide(path)
        if source_mpp is None:
            try:
                self.mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
                self.mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            except (KeyError, TypeError, ValueError) as error:
                self.slide.close()
                raise ValueError(
                    f"{path} has no usable MPP metadata. Supply --source-mpp "
                    "with its level-0 micrometres-per-pixel value."
                ) from error
        else:
            self.mpp_x = self.mpp_y = source_mpp

        base_width, base_height = self.slide.dimensions
        self.width = max(1, round(base_width * self.mpp_x / target_mpp))
        self.height = max(1, round(base_height * self.mpp_y / target_mpp))
        self.target_mpp = target_mpp
        wanted_downsample = target_mpp / math.sqrt(self.mpp_x * self.mpp_y)
        eligible = [
            (level, downsample)
            for level, downsample in enumerate(self.slide.level_downsamples)
            if downsample <= wanted_downsample * 1.01
        ]
        self.level, self.downsample = eligible[-1] if eligible else (0, 1.0)
        print(
            f"  level {self.level}; source MPP {self.mpp_x:.4g} x "
            f"{self.mpp_y:.4g}; model MPP {target_mpp:.4g}; output "
            f"{self.width} x {self.height}"
        )

    def read_patch(self, x: int, y: int, tile_size: int) -> tuple[np.ndarray, int, int]:
        width = min(tile_size, self.width - x)
        height = min(tile_size, self.height - y)
        level_width = max(
            1,
            math.ceil(width * self.target_mpp / (self.mpp_x * self.downsample)),
        )
        level_height = max(
            1,
            math.ceil(height * self.target_mpp / (self.mpp_y * self.downsample)),
        )
        rgba = self.slide.read_region(
            (
                round(x * self.target_mpp / self.mpp_x),
                round(y * self.target_mpp / self.mpp_y),
            ),
            self.level,
            (level_width, level_height),
        )
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.uint8), width, height

    def close(self) -> None:
        self.slide.close()


class TiledSlideSource:
    """Read patches using coordinates and a pyramid level from a tiled dataset."""

    def __init__(self, path: Path, level: int, width: int, height: int) -> None:
        self.slide = openslide.OpenSlide(path)
        level_count = self.slide.level_count
        if level < 0 or level >= level_count:
            self.slide.close()
            raise ValueError(
                f"Invalid pyramid level {level} for {path}; "
                f"the slide has {level_count} levels."
            )
        self.level = level
        self.downsample = float(self.slide.level_downsamples[level])
        self.width = width
        self.height = height

    def read_patch(self, x: int, y: int, tile_size: int) -> tuple[np.ndarray, int, int]:
        width = min(tile_size, self.width - x)
        height = min(tile_size, self.height - y)
        rgba = self.slide.read_region(
            (round(x * self.downsample), round(y * self.downsample)),
            self.level,
            (width, height),
        )
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        return np.asarray(rgb, dtype=np.uint8), width, height

    def close(self) -> None:
        self.slide.close()


class OnnxModel:
    def __init__(self, model_path: Path, provider: str) -> None:
        available = ort.get_available_providers()
        if provider != "auto" and provider not in available:
            raise ValueError(
                f"Provider {provider!r} is unavailable; available: "
                f"{', '.join(available)}"
            )

        cuda_provider = "CUDAExecutionProvider"
        use_cuda = provider == cuda_provider or (
            provider == "auto" and cuda_provider in available
        )
        if use_cuda:
            try:
                ort.preload_dlls(directory="")
            except OSError as error:
                raise RuntimeError(
                    "Could not preload the CUDA and cuDNN libraries installed in "
                    "the Python environment."
                ) from error

        if provider == "auto":
            providers = (
                [cuda_provider, "CPUExecutionProvider"]
                if use_cuda
                else ["CPUExecutionProvider"]
            )
        else:
            providers = [provider]

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        active_providers = self.session.get_providers()
        if provider == cuda_provider and cuda_provider not in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider was requested but ONNX Runtime fell back "
                f"to {', '.join(active_providers)}. Check the CUDA/cuDNN versions "
                "and library paths before rerunning the job."
            )
        model_input = self.session.get_inputs()[0]
        if model_input.type != "tensor(uint8)" or len(model_input.shape) != 4:
            raise ValueError(
                "Expected the supplied uint8 NCHW epithelium model; got "
                f"{model_input.type} {model_input.shape}."
            )
        if isinstance(model_input.shape[1], int) and model_input.shape[1] != 3:
            raise ValueError(f"Expected RGB input; got {model_input.shape}.")
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Loaded {model_path.name} with {', '.join(active_providers)}")

    def predict(self, batch: np.ndarray) -> np.ndarray:
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        if output.ndim == 4 and output.shape[1] == 1:
            output = output[:, 0]
        if output.ndim != 3:
            raise ValueError(f"Expected NHW output; got {output.shape}.")
        return np.asarray(output, dtype=np.float32)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate epithelial masks with the MLflow ONNX model."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", dest="inputs", type=Path, nargs="+")
    inputs.add_argument(
        "--data-mapping",
        type=Path,
        help="CSV file whose 'path' column contains slide or tile paths.",
    )
    inputs.add_argument(
        "--tiled-dataset",
        help=(
            "Local directory or MLflow artifact URI containing slides.parquet "
            "and tiles.parquet."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_URI)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow server for the tiled dataset and output run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "mammaprint" / "episeg",
    )
    parser.add_argument("--kind", choices=("auto", "slide", "tile"), default="auto")
    parser.add_argument("--mpp", type=float, default=MODEL_MPP)
    parser.add_argument("--source-mpp", type=float)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--no-pyramid", action="store_true")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--artifact-path", default="epithelium_masks")
    parser.add_argument("--no-mlflow-log", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0 or args.batch_size <= 0:
        raise ValueError("--tile-size and --batch-size must be positive.")
    if args.stride <= 0 or args.stride > args.tile_size:
        raise ValueError("--stride must be positive and no larger than --tile-size.")
    if args.mpp <= 0 or (args.source_mpp is not None and args.source_mpp <= 0):
        raise ValueError("MPP values must be positive.")


def download_model(model: str, tracking_uri: str, cache_dir: Path) -> Path:
    local_path = Path(model).expanduser()
    if local_path.is_file():
        return local_path.resolve()
    if model.startswith("mlflow-artifacts:"):
        artifact_uri = urllib.parse.urlparse(model)
        tracking = urllib.parse.urlparse(tracking_uri)
        if not tracking.scheme or not tracking.netloc:
            raise ValueError(f"MLflow tracking URI must be HTTP(S): {tracking_uri}")
        artifact_path = artifact_uri.path.lstrip("/")
        endpoint = (
            f"{tracking.path.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/"
            f"{urllib.parse.quote(artifact_path, safe='/')}"
        )
        url = urllib.parse.urlunparse(
            (
                tracking.scheme,
                artifact_uri.netloc or tracking.netloc,
                endpoint,
                "",
                "",
                "",
            )
        )
    elif model.startswith(("http://", "https://")):
        url = model
    else:
        raise FileNotFoundError(f"Unsupported model path or URI: {model}")

    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{tracking_uri}\n{model}".encode()).hexdigest()[:16]
    destination = cache_dir / f"episeg-{key}.onnx"
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached model: {destination}")
        return destination

    partial = destination.with_suffix(".onnx.part")
    print(f"Downloading model to {destination} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "mammaprint/1.0"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as target:
            expected = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                downloaded += len(chunk)
            if expected and downloaded != expected:
                raise OSError(f"Downloaded {downloaded}; expected {expected} bytes.")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def discover_inputs(
    inputs: Sequence[Path] | None, data_mapping: Path | None
) -> list[InputImage]:
    if data_mapping is not None:
        mapping_path = data_mapping.expanduser().resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Data mapping does not exist: {mapping_path}")
        with mapping_path.open(newline="", encoding="utf-8-sig") as mapping_file:
            reader = csv.DictReader(mapping_file)
            if reader.fieldnames is None or "path" not in reader.fieldnames:
                raise ValueError(
                    f"Data mapping must contain a 'path' column: {mapping_path}"
                )
            inputs = [Path(row["path"]) for row in reader if row["path"].strip()]

    if inputs is None:
        raise ValueError("Supply either --input or --data-mapping.")

    discovered: list[InputImage] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported image: {path}")
            discovered.append(InputImage(path, Path(path.name)))
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in SUPPORTED_SUFFIXES
                ):
                    discovered.append(
                        InputImage(candidate, candidate.relative_to(path))
                    )
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    if not discovered:
        raise ValueError("No supported images were found.")
    return discovered


def openslide_format(path: Path) -> str | None:
    try:
        return openslide.OpenSlide.detect_format(path)
    except (OSError, openslide.OpenSlideError):
        return None


def positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    result = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if result[-1] != final:
        result.append(final)
    return result


def batches(values: Sequence[int], batch_size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def extract_patch(
    image: TileSource | SlideSource | TiledSlideSource,
    x: int,
    y: int,
    tile_size: int,
) -> tuple[np.ndarray, int, int]:
    pixels, width, height = image.read_patch(x, y, tile_size)
    patch = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    patch[:height, :width] = pixels
    return np.moveaxis(patch, -1, 0), width, height


def mask_values(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return np.rint(probabilities * 255).astype(np.uint8)


def predict_to_memmap(
    image: TileSource | SlideSource,
    model: OnnxModel,
    temp_path: Path,
    tile_size: int,
    stride: int,
    batch_size: int,
) -> np.memmap:
    x_positions = positions(image.width, tile_size, stride)
    y_positions = positions(image.height, tile_size, stride)
    total = len(x_positions) * len(y_positions)
    print(f"  predicting {total} patches")
    mask = np.memmap(
        temp_path, dtype=np.uint8, mode="w+", shape=(image.height, image.width)
    )
    pending_sum = np.zeros((tile_size, image.width), dtype=np.float32)
    pending_count = np.zeros((tile_size, image.width), dtype=np.uint32)
    pending_y = 0

    def flush(rows: int) -> None:
        nonlocal pending_y
        if rows <= 0:
            return
        probabilities = np.divide(
            pending_sum[:rows],
            pending_count[:rows],
            out=np.zeros_like(pending_sum[:rows]),
            where=pending_count[:rows] != 0,
        )
        mask[pending_y : pending_y + rows] = mask_values(probabilities)
        remaining = tile_size - rows
        if remaining:
            pending_sum[:remaining] = pending_sum[rows:]
            pending_count[:remaining] = pending_count[rows:]
        pending_sum[remaining:] = 0
        pending_count[remaining:] = 0
        pending_y += rows

    completed = 0
    for y in y_positions:
        flush(y - pending_y)
        valid_height = min(tile_size, image.height - y)
        for x_batch in batches(x_positions, batch_size):
            patches: list[np.ndarray] = []
            sizes: list[tuple[int, int]] = []
            for x in x_batch:
                patch, width, height = extract_patch(image, x, y, tile_size)
                patches.append(patch)
                sizes.append((width, height))
            predictions = model.predict(np.stack(patches))
            for x, prediction, (width, height) in zip(
                x_batch, predictions, sizes, strict=True
            ):
                pending_sum[:height, x : x + width] += prediction[:height, :width]
                pending_count[:height, x : x + width] += 1
            completed += len(x_batch)
        print(f"\r  predicted {completed}/{total} patches", end="", flush=True)
        if valid_height < tile_size:
            pending_sum[valid_height:] = 0
            pending_count[valid_height:] = 0
    flush(image.height - pending_y)
    mask.flush()
    print()
    return mask


def predict_coordinates_to_memmap(
    image: TiledSlideSource,
    model: OnnxModel,
    temp_path: Path,
    coordinates: Sequence[tuple[int, int]],
    tile_size: int,
    batch_size: int,
) -> np.memmap:
    """Predict a sparse tiled dataset and stitch its overlaps into a slide mask."""
    grouped: dict[int, list[int]] = {}
    for x, y in sorted(set(coordinates), key=lambda point: (point[1], point[0])):
        if x < 0 or y < 0 or x >= image.width or y >= image.height:
            raise ValueError(
                f"Tile coordinate ({x}, {y}) is outside "
                f"the {image.width} x {image.height} slide extent."
            )
        grouped.setdefault(y, []).append(x)

    print(f"  predicting {len(coordinates)} tiled-dataset patches")
    mask = np.memmap(
        temp_path, dtype=np.uint8, mode="w+", shape=(image.height, image.width)
    )
    mask[:] = 0
    pending_sum = np.zeros((tile_size, image.width), dtype=np.float32)
    pending_count = np.zeros((tile_size, image.width), dtype=np.uint32)
    pending_y = 0

    def flush(rows: int) -> None:
        nonlocal pending_y
        if rows <= 0:
            return
        probabilities = np.divide(
            pending_sum[:rows],
            pending_count[:rows],
            out=np.zeros_like(pending_sum[:rows]),
            where=pending_count[:rows] != 0,
        )
        mask[pending_y : pending_y + rows] = mask_values(probabilities)
        remaining = tile_size - rows
        if remaining:
            pending_sum[:remaining] = pending_sum[rows:]
            pending_count[:remaining] = pending_count[rows:]
        pending_sum[remaining:] = 0
        pending_count[remaining:] = 0
        pending_y += rows

    completed = 0
    for y, x_positions in grouped.items():
        distance = y - pending_y
        if distance > tile_size:
            flush(tile_size)
            pending_y = y
        else:
            flush(distance)

        for x_batch in batches(x_positions, batch_size):
            patches: list[np.ndarray] = []
            sizes: list[tuple[int, int]] = []
            for x in x_batch:
                patch, width, height = extract_patch(image, x, y, tile_size)
                patches.append(patch)
                sizes.append((width, height))
            predictions = model.predict(np.stack(patches))
            for x, prediction, (width, height) in zip(
                x_batch, predictions, sizes, strict=True
            ):
                pending_sum[:height, x : x + width] += prediction[:height, :width]
                pending_count[:height, x : x + width] += 1
            completed += len(x_batch)
        print(
            f"\r  predicted {completed}/{len(coordinates)} patches",
            end="",
            flush=True,
        )

    remaining = image.height - pending_y
    flush(min(remaining, tile_size))
    mask.flush()
    print()
    return mask


def save_mask(
    mask: np.memmap,
    destination: Path,
    is_slide: bool,
    mpp: float,
    tile_size: int,
    pyramid: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = mask.shape
    image = pyvips.Image.new_from_memory(mask.data, width, height, 1, "uchar")
    if is_slide:
        image.tiffsave(
            str(destination),
            bigtiff=True,
            compression="deflate",
            tile=True,
            tile_width=tile_size,
            tile_height=tile_size,
            xres=1000.0 / mpp,
            yres=1000.0 / mpp,
            pyramid=pyramid and (width > tile_size or height > tile_size),
        )
    else:
        image.pngsave(str(destination), compression=6)


def resolve_tiled_dataset(source: str, tracking_uri: str, cache_dir: Path) -> Path:
    local_path = Path(source).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    if not source.startswith(("mlflow-artifacts:", "runs:/", "models:/")):
        raise FileNotFoundError(f"Tiled dataset does not exist: {source}")

    destination = cache_dir.expanduser() / "tiled-datasets"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tiled dataset: {source}")
    return Path(
        download_artifacts(
            artifact_uri=source,
            dst_path=str(destination),
            tracking_uri=tracking_uri,
        )
    )


def read_tiled_table(dataset_path: Path, name: str) -> pd.DataFrame:
    single_file = dataset_path / f"{name}.parquet"
    partitioned = dataset_path / name
    if single_file.is_file():
        return pd.read_parquet(single_file)
    if partitioned.is_dir():
        return pd.read_parquet(partitioned)
    raise FileNotFoundError(
        f"Tiled dataset has neither {single_file.name} nor a {name}/ directory: "
        f"{dataset_path}"
    )


def run_tiled_dataset(args: argparse.Namespace, model: OnnxModel, source: str) -> None:
    dataset_path = resolve_tiled_dataset(source, args.tracking_uri, args.cache_dir)
    slides = read_tiled_table(dataset_path, "slides")
    tiles = read_tiled_table(dataset_path, "tiles")
    if "tile_x" in tiles.columns and "x" not in tiles.columns:
        tiles = tiles.rename(columns={"tile_x": "x", "tile_y": "y"})

    required_slide_columns = {
        "id",
        "path",
        "extent_x",
        "extent_y",
        "tile_extent_x",
        "tile_extent_y",
        "stride_x",
        "stride_y",
        "mpp_x",
        "mpp_y",
        "level",
    }
    required_tile_columns = {"slide_id", "x", "y"}
    if missing := required_slide_columns - set(slides.columns):
        raise ValueError(f"slides.parquet is missing columns: {sorted(missing)}")
    if missing := required_tile_columns - set(tiles.columns):
        raise ValueError(f"tiles.parquet is missing columns: {sorted(missing)}")

    destinations: set[Path] = set()
    records = slides.to_dict(orient="records")
    for index, slide in enumerate(records, start=1):
        path = Path(str(slide["path"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Slide does not exist: {path}")
        tile_width = int(slide["tile_extent_x"])
        tile_height = int(slide["tile_extent_y"])
        stride_x = int(slide["stride_x"])
        stride_y = int(slide["stride_y"])
        if (tile_width, tile_height) != (args.tile_size, args.tile_size):
            raise ValueError(
                f"{path} was tiled at {tile_width} x {tile_height}; expected "
                f"{args.tile_size} x {args.tile_size}."
            )
        if (stride_x, stride_y) != (args.stride, args.stride):
            raise ValueError(
                f"{path} was tiled with stride {stride_x} x {stride_y}; expected "
                f"{args.stride} x {args.stride}."
            )

        slide_tiles = tiles.loc[tiles["slide_id"] == slide["id"], ["x", "y"]]
        coordinates = [
            (int(tile.x), int(tile.y)) for tile in slide_tiles.itertuples(index=False)
        ]
        mpp_x = float(slide["mpp_x"])
        mpp_y = float(slide["mpp_y"])
        print(
            f"[{index}/{len(records)}] {path}\n"
            f"  tiled level {int(slide['level'])}; MPP {mpp_x:.4g} x "
            f"{mpp_y:.4g}; {len(coordinates)} retained tiles"
        )
        image = TiledSlideSource(
            path,
            int(slide["level"]),
            int(slide["extent_x"]),
            int(slide["extent_y"]),
        )
        destination = args.output_dir / path.with_suffix(".tiff").name
        if destination in destinations:
            raise ValueError(f"Multiple slides map to {destination}.")
        destinations.add(destination)
        try:
            with tempfile.TemporaryDirectory(prefix="episeg-mask-") as temp_dir:
                mask = predict_coordinates_to_memmap(
                    image,
                    model,
                    Path(temp_dir) / "mask.raw",
                    coordinates,
                    args.tile_size,
                    args.batch_size,
                )
                save_mask(
                    mask,
                    destination,
                    True,
                    math.sqrt(mpp_x * mpp_y),
                    args.tile_size,
                    not args.no_pyramid,
                )
                del mask
        finally:
            image.close()
        print(f"  wrote {destination}")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    model_path = download_model(args.model, args.tracking_uri, args.cache_dir)
    model = OnnxModel(model_path, args.provider)
    if args.tiled_dataset is not None:
        run_tiled_dataset(args, model, args.tiled_dataset)
        return

    inputs = discover_inputs(args.inputs, args.data_mapping)
    destinations: set[Path] = set()
    for index, input_image in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] {input_image.path}")
        detected_slide = openslide_format(input_image.path) is not None
        if args.kind == "slide" and not detected_slide:
            raise ValueError(
                f"OpenSlide cannot read {input_image.path}; use --kind tile for "
                "ordinary TIFF, PNG, or JPEG files."
            )
        is_slide = args.kind == "slide" or (args.kind == "auto" and detected_slide)
        source: TileSource | SlideSource = (
            SlideSource(input_image.path, args.mpp, args.source_mpp)
            if is_slide
            else TileSource(input_image.path)
        )
        filename = (
            f"{input_image.relative_path.stem}.tiff"
            if is_slide
            else f"{input_image.relative_path.stem}_mask.png"
        )
        destination = args.output_dir / input_image.relative_path.parent / filename
        if destination in destinations:
            raise ValueError(f"Multiple inputs map to {destination}.")
        destinations.add(destination)
        try:
            with tempfile.TemporaryDirectory(prefix="episeg-mask-") as temp_dir:
                mask = predict_to_memmap(
                    source,
                    model,
                    Path(temp_dir) / "mask.raw",
                    args.tile_size,
                    args.stride,
                    args.batch_size,
                )
                save_mask(
                    mask,
                    destination,
                    is_slide,
                    args.mpp,
                    args.tile_size,
                    not args.no_pyramid,
                )
                del mask
        finally:
            source.close()
        print(f"  wrote {destination}")


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    try:
        if args.no_mlflow_log:
            run(args)
            return

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params(
                {
                    "model": args.model,
                    "tiled_dataset": args.tiled_dataset or "",
                    "data_mapping": str(args.data_mapping or ""),
                    "output_dir": str(args.output_dir),
                    "mpp": args.mpp,
                    "tile_size": args.tile_size,
                    "stride": args.stride,
                    "batch_size": args.batch_size,
                    "provider": args.provider,
                    "mask_encoding": "uint8_probability_0_255",
                }
            )
            run(args)
            mlflow.log_artifacts(str(args.output_dir), artifact_path=args.artifact_path)
            print(f"Logged masks to {mlflow.get_artifact_uri(args.artifact_path)}")
    except (FileNotFoundError, OSError, ValueError, urllib.error.URLError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
