from pathlib import Path

import numpy as np
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from datasets import load_dataset
from ratiopath.openslide import OpenSlide
from torch.utils.data import ConcatDataset, Dataset

from ml.typing import TileMetadata, TilesPredictSample


class TileDataset(Dataset[TilesPredictSample]):
    """Lazily reads pre-computed tiles from a single Whole Slide Image.

    Uses an Arrow-backed HuggingFace Dataset for O(1) random tile access
    without loading all tile metadata into RAM.
    """

    def __init__(
        self,
        slide_path: str | Path,
        level: int,
        extent_x: int,
        extent_y: int,
        tiles: HFDataset,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_path = Path(slide_path)
        self.level = level
        self.extent_x = extent_x
        self.extent_y = extent_y
        self.tiles = tiles
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> TilesPredictSample:
        tile = self.tiles[idx]

        x, y = int(tile["x"]), int(tile["y"])

        with OpenSlide(self.slide_path) as slide:
            image = slide.read_tile(x, y, self.extent_x, self.extent_y, self.level)

        metadata: TileMetadata = {
            "slide_id": self.slide_path.stem,
            "x": x,
            "y": y,
        }

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]
        return image, metadata


class SlideDataset(ConcatDataset[TilesPredictSample]):
    """A unified PyTorch dataset that links parent slide metadata with tile metadata.

    Uses HuggingFace datasets to lazily load parquet files via memory-mapped
    Arrow tables, enabling O(1) random access with near-zero RAM overhead.

    Args:
        path: Directory containing ``slides.parquet`` and ``tiles.parquet``.
        transforms: Optional albumentations transforms applied to each tile image.
    """

    def __init__(
        self,
        path: str | Path,
        transforms: TransformType | None = None,
    ) -> None:
        path = Path(path)
        slides_dataset = load_dataset(
            "parquet", data_files=str(path / "slides.parquet"), split="train"
        )
        tiles_dataset = load_dataset(
            "parquet", data_files=str(path / "tiles.parquet"), split="train"
        )

        # Rename tile_x/tile_y to x/y
        if "tile_x" in tiles_dataset.column_names:
            tiles_dataset = tiles_dataset.rename_columns({"tile_x": "x", "tile_y": "y"})

        datasets = [
            TileDataset(
                slide_path=slide["path"],
                level=slide["level"],
                extent_x=slide["tile_extent_x"],
                extent_y=slide["tile_extent_y"],
                tiles=tiles_dataset.filter(
                    lambda row: row["slide_id"] == slide["slide_id"],
                    keep_in_memory=False,
                ),
                transforms=transforms,
            )
            for slide in slides_dataset
        ]

        super().__init__(datasets)
