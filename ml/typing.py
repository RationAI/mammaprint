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
    """One pyramid level's data card: physical facts + artifact location.

    Authored once per preprocessing artifact (a ``configs/data/tiled/*.yaml`` card).
    ``uris`` points at the embedding artifact; ``raw_uris`` (optional) at the
    raw-tile artifact for the OpenSlide path.
    """

    mpp: float
    tile_extent: int
    uris: str


type Bag = Tensor
"""A bag of per-slide items: tile features ``(N, D)`` or raw tiles ``(N, C, H, W)``."""

type Region = dict[int, Tensor]
"""One aligned multi-scale region: level -> that level's tiles ``(K_level, D)``.

For example ``{3: (1, D), 2: (4, D)}`` is one coarse level-3 tile with the four
level-2 tiles that zoom into the same footprint.
"""

type MultiScaleBag = list[Region]
"""A slide as a list of aligned multi-scale regions (variable length)."""

type MILSample = tuple[Bag, Tensor, SlideMetadata]
"""A single-level slide-level training sample: ``(bag, label, metadata)``."""

type MultiScaleSample = tuple[MultiScaleBag, Tensor, SlideMetadata]
"""A multi-scale slide-level training sample: ``(regions, label, metadata)``."""
