# Copyright (c) The RationAI team.

from pathlib import Path
from typing import Any

import numpy as np
import PIL
import slidelip
import torch
from numpy.typing import NDArray

from mammaprint.datamodule.samplers import BaseSampler


class BaseDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, Any | None]]
):
    sampler: BaseSampler
    _epoch_samples: list

    def __init__(self, sampler: BaseSampler, seed: int) -> None:
        self._epoch_samples = []
        self.sampler = sampler
        self._rng = np.random.default_rng(seed)

    def generate_samples(self) -> None:
        self._epoch_samples = self.sampler.get_sample()

    def __len__(self) -> int:
        return len(self._epoch_samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
        raise NotImplementedError


def extract_tile(
    slide_fp: Path, coord_x: int, coord_y: int, tile_size: int, level: int
) -> NDArray:
    """Extracts a tile from a slide using the supplied coordinate values.

    Args:
        slide_fp (Path): Path to the slide.
        coord_x (int): Coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        coord_y (int): Coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        tile_size (int): Size of the tile to be extracted.
        level (int): Resolution level from which tile should be extracted.

    Returns:
        NDArray: RGB Tile represented as numpy array.
    """
    wsi = slidelip.open_slide(slide_path=slide_fp)
    bg_tile = PIL.Image.new(mode="RGB", size=(tile_size, tile_size), color="#FFFFFF")
    im_tile = wsi.read_region(
        location=(coord_x, coord_y), level=level, size=(tile_size, tile_size)
    )
    bg_tile.paste(im=im_tile, mask=im_tile, box=None)
    wsi.close()
    return np.array(bg_tile)
