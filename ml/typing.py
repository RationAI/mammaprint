from collections.abc import Mapping
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


# ── Single-level types (always shipped) ──────────────────────────────────────
Bag = NewType("Bag", Tensor)
"""A bag of per-slide items: tile features ``(N, D)`` or raw tiles ``(N, C, H, W)``.

The coordinate-aware single-scale path (``SpatialScaleDataset`` +
``SpatialTransformerMIL``) packs a wider ``(N, D + 2)`` tensor whose last two columns
are each tile's level-0 pixel ``(x, y)``; the spatial aggregator slices them off. No
other dataset or aggregator produces or consumes those extra columns.
"""

type MILSample = tuple[Bag, Tensor, SlideMetadata]
"""A single-level slide-level training sample: ``(bag, label, metadata)``."""

type SampleOf[BagT: Tensor] = tuple[BagT, Tensor, SlideMetadata]
"""A slide sample parameterized by the exact bag tensor type."""


# ── Multilevel types ──────────────────────────────────────────────────────────
type Coordinate = tuple[int, int]
"""A tile's integer ``(x, y)`` top-left coordinate in its slide level."""

PyramidBag = NewType("PyramidBag", Tensor)
"""Tensor ``(R, K, D)``: regions, ordered pyramid tokens, embedding features."""

type PyramidSample = SampleOf[PyramidBag]
"""A multiscale training sample: ``(pyramids, label, metadata)``."""

type Sample = MILSample | PyramidSample
"""Any slide-level sample accepted by the shared data module."""
