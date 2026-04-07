from collections.abc import Iterable
from pathlib import Path

from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from ml.typing import TileMetadata, TilesPredictSample


class TileDataset(Dataset[TilesPredictSample]):
    """Wraps an OpenSlideTilesDataset with albumentations transforms and metadata.

    Each item returns a ``(tensor, metadata)`` tuple where metadata contains
    the slide id and tile coordinates.
    """

    def __init__(
        self,
        slide_path: str | Path,
        level: int,
        tile_extent_x: int,
        tile_extent_y: int,
        tiles: HFDataset,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_path = Path(slide_path)
        self.tiles = tiles
        self.inner = OpenSlideTilesDataset(
            slide_path=slide_path,
            level=level,
            tile_extent_x=tile_extent_x,
            tile_extent_y=tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> TilesPredictSample:
        image = self.inner[idx]
        tile = self.tiles[idx]

        metadata: TileMetadata = {
            "slide_id": self.slide_path.stem,
            "x": int(tile["x"]),
            "y": int(tile["y"]),
        }

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]
        return image, metadata


class SlideDataset(MetaTiledSlides[TilesPredictSample]):
    """A unified PyTorch dataset that links parent slide metadata with tile metadata.

    Inherits loading, concatenation, and filtering logic from
    :class:`rationai.mlkit.data.datasets.MetaTiledSlides`.

    Args:
        paths: Local directories containing ``slides.parquet`` and ``tiles.parquet``.
        uris: MLflow artifact URIs pointing to folders with the parquet files.
        slides_and_tiles: Pre-loaded tuple of (slides, tiles) HF Datasets.
        transforms: Optional albumentations transforms applied to each tile image.
    """

    def __init__(
        self,
        *,
        paths: Iterable[Path | str] | None = None,
        uris: Iterable[str] | None = None,
        slides_and_tiles: tuple[HFDataset, HFDataset] | None = None,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(paths=paths, uris=uris, slides_and_tiles=slides_and_tiles)

    def generate_datasets(self) -> Iterable[Dataset[TilesPredictSample]]:
        # Rename tile_x/tile_y to x/y if needed
        if "tile_x" in self.tiles.column_names:
            self.tiles = self.tiles.rename_columns({"tile_x": "x", "tile_y": "y"})

        return (
            TileDataset(
                slide_path=slide["path"],
                level=slide["level"],
                tile_extent_x=slide["tile_extent_x"],
                tile_extent_y=slide["tile_extent_y"],
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for slide in self.slides
        )
