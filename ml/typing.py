from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, TypedDict

from torch import Tensor


class TileMetadata(TypedDict):
    """Metadata for a tile, including its slide ID and coordinates."""

    slide_id: str
    x: int
    y: int


type TilesPredictSample = tuple[Tensor, TileMetadata]


class SlideMetadata(TypedDict):
    """Metadata for a slide-level MIL sample."""

    slide_id: str


type Coordinate = tuple[int, int]
"""A tile's integer ``(x, y)`` top-left coordinate in its slide level."""


class PyramidSlideMetadata(TypedDict):
    """Metadata for a multiscale sample, aligned with its bag region dimension."""

    slide_id: str
    anchors: tuple[Coordinate, ...]


class OverlapPolicy(StrEnum):
    """How complete pyramid footprints are retained for one slide."""

    NON_OVERLAPPING = "non_overlapping"
    ALL = "all"


class LevelSpec(TypedDict):
    """One pyramid level's data card: physical facts + per-split artifact locations.

    Authored once per preprocessing artifact (a ``configs/data/tiled/*.yaml`` card).
    ``uris`` maps ``split -> embedding artifact URI`` (``train``/``val``/``test``);
    ``raw_uris`` (optional) does the same for the raw-tile artifact. Per-split URIs
    because the split is materialised into physically separate artifacts by
    ``scripts/preprocessing/split_dataset.py``.
    """

    mpp: float
    tile_extent: int
    uris: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PyramidSelectionAudit:
    """Selection outcomes for one slide before it enters a pyramid dataset."""

    slide_id: str
    split: str
    label: str
    candidates: int
    complete_before_thinning: int
    retained: int
    incomplete: int
    missing_by_level: Mapping[int, int]
    completion_rate: float
    overlap_retention_rate: float
    rejected: bool


# ── Single-level types (always shipped) ──────────────────────────────────────
Bag = NewType("Bag", Tensor)
"""A bag of per-slide items: tile features ``(N, D)`` or raw tiles ``(N, C, H, W)``.

The coordinate-aware single-scale path (``SpatialScaleDataset`` +
``SpatialTransformerMIL``) packs a wider ``(N, D + 2)`` tensor whose last two columns
are each tile's level-0 pixel ``(x, y)``; the spatial aggregator slices them off. No
other dataset or aggregator produces or consumes those extra columns.
"""

type SampleOf[BagT: Tensor, MetadataT] = tuple[BagT, Tensor, MetadataT]
"""A slide sample parameterized by its exact bag and metadata types."""

type MILSample = SampleOf[Bag, SlideMetadata]
"""A single-level slide-level training sample: ``(bag, label, metadata)``."""


# ── Multilevel types ──────────────────────────────────────────────────────────
PyramidBag = NewType("PyramidBag", Tensor)
"""Tensor ``(R, K, D)``: regions, ordered pyramid tokens, embedding features."""

type PyramidSample = SampleOf[PyramidBag, PyramidSlideMetadata]
"""A multiscale training sample: ``(pyramids, label, metadata)``."""

type Sample = MILSample | PyramidSample
"""Any slide-level sample accepted by the shared data module."""
