# Standard Imports
import math
from pathlib import Path

# Third-party Imports
import albumentations
import torch

# Local Imports
from histopipe.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from histopipe.datamodule.samplers import BaseSampler


class ContrastiveDataset(BaseDataset):
    def __init__(
        self,
        sampler: BaseSampler,
        seed: int,
        augmentations: albumentations.TemplateTransform | None = None,
        extended_selection: bool = False,
        override_tile_size: int | None = None,
        override_level: int | None = None,
    ) -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations
        self.extended_selection = extended_selection
        self.override_level = override_level
        self.override_tile_size = override_tile_size

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self._epoch_samples[index]

        # A choice to override sample level and tile size
        if self.override_level is not None:
            level = self.override_level
        else:
            level = sample["sample_level"]

        if self.override_tile_size is not None:
            tile_size = self.override_tile_size
        else:
            tile_size = sample["tile_size"]

        if self.extended_selection:
            # Compute diameter of the circumscribed circle around the patch to allow
            # 360° rotation without artifacts
            tile_size = math.ceil(math.sqrt(2 * (tile_size * tile_size)))

            # shift the patch origin by half the added area to center the original image
            offset = tile_size // 2
        else:
            offset = 0

        x = extract_tile(
            slide_fp=Path(sample.get("slide_fp")).resolve(),
            coord_x=sample["coord_x"] - offset,
            coord_y=sample["coord_y"] - offset,
            tile_size=tile_size,
            level=level,
        )

        # get 2 augmentations of sampled tile
        x_1 = self.transforms(image=x)["image"]
        x_2 = self.transforms(image=x)["image"]

        # permute to (channels, height, width)
        x_1 = torch.from_numpy(x_1).permute(2, 0, 1)
        x_2 = torch.from_numpy(x_2).permute(2, 0, 1)
        return x_1, x_2, sample
