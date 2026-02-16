from collections.abc import Iterable
from typing import TypeVar

import pandas as pd
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ml.typing import TileMetadata, TilesPredictSample


T = TypeVar("T", bound=TilesPredictSample)


class _Tiles(Dataset[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        # Rename tile_x/tile_y to x/y if needed (for compatibility with tiling script output)
        tiles_renamed = tiles.rename(columns={"tile_x": "x", "tile_y": "y"})
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata["path"],
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles_renamed,
        )
        self.slide_metadata = slide_metadata
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> TilesPredictSample:
        image = self.slide_tiles[idx]
        tile_row = self.slide_tiles.tiles.iloc[idx]
        metadata: TileMetadata = {
            "slide_id": self.slide_tiles.slide_path.stem,
            "x": int(tile_row["x"]),
            "y": int(tile_row["y"]),
        }

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]
        return image, metadata


class TilesPredict(MetaTiledSlides[TilesPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Tiles[TilesPredictSample]]:
        return (
            _Tiles(
                slide_metadata=slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )
