from collections.abc import Mapping
from typing import TypedDict

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
type Bag = Tensor
"""A bag of per-slide items: tile features ``(N, D)`` or raw tiles ``(N, C, H, W)``."""

type MILSample = tuple[Bag, Tensor, SlideMetadata]
"""A single-level slide-level training sample: ``(bag, label, metadata)``."""


# ── Multilevel types (comment out this block to ship single-level only) ───────
# Nothing single-level references these; the multilevel dataset/aggregator are the
# only consumers, so removing/commenting them here + their imports drops multilevel.
# type Region = dict[int, Tensor]
# """One aligned multi-scale region: level -> that level's tiles ``(K_level, D)``.

# For example ``{3: (1, D), 2: (4, D)}`` is one coarse level-3 tile with the four
# level-2 tiles that zoom into the same footprint.
# """

# type MultiScaleBag = list[Region]
# """A slide as a list of aligned multi-scale regions (variable length)."""

# type MultiScaleSample = tuple[MultiScaleBag, Tensor, SlideMetadata]
# """A multi-scale slide-level training sample: ``(regions, label, metadata)``."""
