"""Read stored slide embeddings without changing the training datasets.

The training dataset intentionally discards tile coordinates.  Explanation output
needs those coordinates, so this module uses the instantiated dataset only to
resolve the exact split/artifact/labels and then reads the same parquet rows while
retaining ``x`` and ``y``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor

from ml.data.datasets.labels import get_label
from ml.data.datasets.single_scale import SingleScaleDataset
from ml.data.datasets.spatial_scale import SpatialScaleDataset


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True)
class EmbeddingSlide:
    """One slide's model input plus spatial/output metadata."""

    slide_id: str
    embeddings: Tensor
    x: np.ndarray
    y: np.ndarray
    label: Tensor
    slide_path: Path
    level: int


@dataclass(frozen=True)
class SlideGeometry:
    """Geometry of the model's selected WSI pyramid level."""

    width: int
    height: int
    downsample: float
    mpp_x: float
    mpp_y: float
    mpp_source_x: str = "configured"
    mpp_source_y: str = "configured"


def instantiate_embedding_dataset(config: DictConfig, split: str) -> SingleScaleDataset:
    """Instantiate one configured embedding split and reject non-v1 data shapes."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val, or test; got {split!r}")
    split_config = config.datamodule.get(split)
    if split_config is None:
        raise KeyError(f"The recovered config has no datamodule.{split} dataset.")

    dataset = instantiate(split_config)
    if isinstance(dataset, SpatialScaleDataset):
        raise TypeError(
            "SpatialScaleDataset belongs to the spatial-transformer path, which is "
            "not supported by the v1 embedding explainer."
        )
    if not isinstance(dataset, SingleScaleDataset):
        raise TypeError(
            "The v1 explainer requires SingleScaleDataset over stored embeddings; "
            f"got {type(dataset).__module__}.{type(dataset).__name__}."
        )
    return dataset


def selected_level_spec(config: DictConfig) -> tuple[int, dict[str, Any]]:
    """Return the sole configured level and its resolved physical parameters."""
    levels = config.dataset.levels
    if len(levels) != 1:
        raise ValueError(
            f"The v1 explainer is single-scale; recovered {len(levels)} levels."
        )
    level_key, level_config = next(iter(levels.items()))
    return int(level_key), dict(level_config)


def iter_embedding_slides(
    dataset: SingleScaleDataset,
    slide_ids: Iterable[str] | None = None,
) -> Iterator[EmbeddingSlide]:
    """Yield validated embeddings and coordinates in the parquet's original order."""
    requested = set(slide_ids or [])
    available = set(dataset.slides["name"].astype(str))
    missing = requested - available
    if missing:
        raise KeyError(
            "Requested slide IDs are absent from the configured split: "
            + ", ".join(sorted(missing))
        )

    for index, row in dataset.slides.iterrows():
        slide_id = str(row["name"])
        if requested and slide_id not in requested:
            continue
        yield read_embedding_slide(dataset, int(index))


def read_embedding_slide(
    dataset: SingleScaleDataset,
    index: int,
) -> EmbeddingSlide:
    """Read one dataset row, allowing a cohort runner to isolate slide failures."""
    row = dataset.slides.iloc[index]
    slide_id = str(row["name"])
    parquet_path = (dataset.embeddings_dir / slide_id).with_suffix(".parquet")
    frame = pd.read_parquet(parquet_path)
    _validate_frame(frame, parquet_path)

    embeddings_array = np.stack(frame["embedding"].to_numpy())
    embeddings = torch.from_numpy(embeddings_array).float()
    if embeddings.ndim != 2:
        raise ValueError(
            f"{parquet_path}: embeddings must form an (N, D) matrix; "
            f"got {tuple(embeddings.shape)}."
        )
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{parquet_path}: embeddings contain NaN or infinity.")

    x = frame["x"].to_numpy(dtype=np.int64, copy=True)
    y = frame["y"].to_numpy(dtype=np.int64, copy=True)
    label = get_label(row, dataset.label_mode)
    return EmbeddingSlide(
        slide_id=slide_id,
        embeddings=embeddings,
        x=x,
        y=y,
        label=label,
        slide_path=Path(str(row["path"])),
        level=dataset.level,
    )


def streaming_mean_embedding(
    dataset: SingleScaleDataset,
    *,
    expected_feature_dim: int | None = None,
) -> tuple[Tensor, int, tuple[dict[str, str], ...]]:
    """Compute a tile mean while isolating corrupt baseline slides.

    A corrupt training parquet must not abort explanations for the rest of the
    cohort. Failures are returned for the manifest; the job fails only if no valid
    baseline tiles remain.
    """
    if expected_feature_dim is not None and expected_feature_dim < 1:
        raise ValueError("expected_feature_dim must be positive when provided.")
    total: Tensor | None = (
        torch.zeros(expected_feature_dim, dtype=torch.float64)
        if expected_feature_dim is not None
        else None
    )
    count = 0
    feature_dim = expected_feature_dim
    failures: list[dict[str, str]] = []
    for index, row in dataset.slides.iterrows():
        slide_id = str(row["name"])
        try:
            slide = read_embedding_slide(dataset, int(index))
            bag = slide.embeddings
            if feature_dim is None:
                feature_dim = bag.shape[1]
                total = torch.zeros(feature_dim, dtype=torch.float64)
            elif bag.shape[1] != feature_dim:
                raise ValueError(
                    f"embedding width {bag.shape[1]} does not match expected "
                    f"width {feature_dim}"
                )
            assert total is not None
            total += bag.double().sum(dim=0)
            count += bag.shape[0]
        except Exception as error:  # noqa: BLE001 - isolate corrupt slide artifacts
            log.warning(
                "Skipping corrupt baseline slide %s: %s: %s",
                slide_id,
                type(error).__name__,
                error,
            )
            failures.append(
                {
                    "slide_id": slide_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if total is None or count == 0:
        raise ValueError("Cannot compute an IG baseline from an empty training split.")
    return (total / count).float(), count, tuple(failures)


def read_slide_geometry(
    slide_path: Path,
    level: int,
    configured_mpp: float,
) -> SlideGeometry:
    """Read exact dimensions/downsample and calibrated MPP from the original WSI."""
    if not slide_path.exists():
        raise FileNotFoundError(f"WSI does not exist: {slide_path}")

    # Imported lazily: the Python package is present in development environments,
    # while the native OpenSlide library is supplied by the inference container.
    from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
    from ratiopath.openslide import OpenSlide

    with OpenSlide(str(slide_path)) as slide:
        if not 0 <= level < slide.level_count:
            raise ValueError(
                f"WSI {slide_path} has {slide.level_count} levels; level {level} "
                "from the embedding config is unavailable."
            )
        width, height = slide.level_dimensions[level]
        downsample = float(slide.level_downsamples[level])
        base_mpp_x = _optional_positive_float(slide.properties.get(PROPERTY_NAME_MPP_X))
        base_mpp_y = _optional_positive_float(slide.properties.get(PROPERTY_NAME_MPP_Y))
        if base_mpp_x is not None and base_mpp_y is not None:
            mpp_x, mpp_y = slide.slide_resolution(level)
        else:
            mpp_x = configured_mpp
            mpp_y = configured_mpp

    return SlideGeometry(
        width=int(width),
        height=int(height),
        downsample=downsample,
        mpp_x=float(mpp_x),
        mpp_y=float(mpp_y),
        mpp_source_x=("openslide" if base_mpp_x else "configured_fallback"),
        mpp_source_y=("openslide" if base_mpp_y else "configured_fallback"),
    )


def target_names(label_mode: str, out_dim: int) -> list[str]:
    """Return stable output semantics and verify the recovered head shape."""
    expected = {
        "type": ["luminal_a_logit"],
        "index": ["mammaprint_index"],
        "both": ["luminal_a_logit", "mammaprint_index"],
    }
    try:
        names = expected[label_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported label_mode {label_mode!r}.") from error
    if len(names) != out_dim:
        raise ValueError(
            f"label_mode={label_mode!r} requires {len(names)} outputs, but the "
            f"checkpoint head exposes {out_dim}."
        )
    return names


def label_values(label: Tensor) -> list[float]:
    """Convert a scalar/vector label to JSON/Parquet-friendly values."""
    return [float(value) for value in label.detach().cpu().reshape(-1)]


def _validate_frame(frame: pd.DataFrame, path: Path) -> None:
    required = {"x", "y", "embedding"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}.")
    if frame.empty:
        raise ValueError(f"{path}: a MIL bag cannot be empty.")
    if frame[["x", "y"]].isna().any().any():
        raise ValueError(f"{path}: tile coordinates contain missing values.")
    try:
        coordinates = frame[["x", "y"]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: tile coordinates must be numeric integers."
        ) from error
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{path}: tile coordinates contain NaN or infinity.")
    if (coordinates < 0).any() or (coordinates != np.floor(coordinates)).any():
        raise ValueError(
            f"{path}: tile coordinates must be non-negative integer pixels."
        )
    if (coordinates > np.iinfo(np.int64).max).any():
        raise ValueError(f"{path}: tile coordinates exceed the int64 range.")
    if frame.duplicated(["x", "y"]).any():
        raise ValueError(f"{path}: duplicate tile coordinates are ambiguous.")


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) and parsed > 0 else None


__all__ = [
    "EmbeddingSlide",
    "SlideGeometry",
    "instantiate_embedding_dataset",
    "iter_embedding_slides",
    "label_values",
    "read_embedding_slide",
    "read_slide_geometry",
    "selected_level_spec",
    "streaming_mean_embedding",
    "target_names",
]
log = logging.getLogger(__name__)
