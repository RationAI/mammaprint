from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvips
import ray
from PIL import Image
from rationai.masks import process_items, tile_mask

slides: pd.DataFrame = pd.read_parquet("data/dataset/slides.parquet")
tiles: pd.DataFrame = pd.read_parquet("data/dataset/tiles.parquet")


@ray.remote
def process_slide(slide: Any) -> None:
    slide_tiles = tiles[tiles["slide_id"] == slide.id]
    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide.tile_extent_x, slide.tile_extent_y),
        size=(slide.extent_x, slide.extent_y),
    )
    mask_path = Path("data/tile_masks") / f"{Path(slide.path).stem}.png"
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    img = pyvips.Image.new_from_file(slide.path, page=slide.level).numpy()
    mask = np.asarray(mask)
    img[mask == 255] = 0

    Image.fromarray(img).save(mask_path)


process_items(slides.itertuples(), process_slide)
