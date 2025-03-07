from pathlib import Path
from statistics import mean
from typing import Any

import pyvips
from histopath.ray.datasource import VipsTiffDatasink
from openslide import OpenSlide
from pyvips import Image
from rationai.masks import slide_resolution, tissue_mask
from ray import data
from ray.data.datasource import FilenameProvider

SLIDES_PATH = "/mnt/data/Projects/MOU/Mammaprint/Another_WSIs"
MASK_DEST = "/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tissue_masks"
LEVEL = 5


class Filename(FilenameProvider):
    def get_filename_for_row(
        self, row: dict, task_index: int, block_index: int, row_index: int
    ) -> str:
        return Path(row["item"]).with_suffix(".tiff").name


def process_slide(row: dict[str, Any]) -> dict[str, Any]:
    with OpenSlide(row["item"]) as slide:
        mpp_x, mpp_y = slide_resolution(slide, LEVEL)

    slide = Image.new_from_file(row["item"], level=LEVEL)
    row["tissue_mask"] = tissue_mask(slide, mpp=mean((mpp_x, mpp_y))).numpy()
    row["options"] = {"xres": mpp_x, "yres": mpp_y}
    return row


def main() -> None:
    slides = Path(SLIDES_PATH).rglob("*.mrxs")
    data.from_items(list(slides)).map(process_slide).write_datasink(
        VipsTiffDatasink(
            MASK_DEST,
            data_column="tissue_mask",
            options_column="options",
            filename_provider=Filename(),
            default_options={
                "bigtiff": True,
                "compression": pyvips.enums.ForeignTiffCompression.DEFLATE,
                "tile": True,
                "tile_width": 512,
                "tile_height": 512,
                "pyramid": True,
            },
        )
    )


if __name__ == "__main__":
    main()
