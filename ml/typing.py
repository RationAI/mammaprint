from typing import TypedDict, TypeAlias

from torch import Tensor


class TileMetadata(TypedDict):
   """Metadata for a tile, including its slide ID and coordinates."""

   slide_id: str
   tile_x: int
   tile_y: int

TilesPredictSample: TypeAlias = tuple[Tensor, TileMetadata]
