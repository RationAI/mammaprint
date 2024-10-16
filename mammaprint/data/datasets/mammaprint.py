from collections.abc import Iterable

import pandas as pd
import torch
from albumentations import TransformType
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from mammaprint.typing import Metadata, PredictSample, Sample


class Mammaprint(MetaTiledSlides[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        cancer_threshold: float,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        self.cancer_threshold = cancer_threshold
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        self.tiles["cancer"] = self.tiles["cancer_percentage"] > self.cancer_threshold

        return (
            _MammaprintSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_label=True,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class MammaprintPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _MammaprintSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_label=False,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _MammaprintSlideTiles(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.include_label = include_label
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        image = self.slide_tiles[idx]
        metadata = Metadata(
            slide=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label = torch.tensor([self.slide_tiles.tiles.iloc[idx]["cancer"]]).float()
            return image, label, metadata

        return image, metadata
