from typing import TypedDict

from torch import Tensor


class TileMetadata(TypedDict):
    """Metadata for a tile, including its slide ID and coordinates."""

    slide_id: str
    x: int
    y: int


type TilesPredictSample = tuple[Tensor, TileMetadata]
