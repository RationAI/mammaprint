from pathlib import Path
from statistics import mean

# from typing import Any
import ray
from openslide import OpenSlide
from pyvips import Image
from rationai.masks import process_items, slide_resolution, tissue_mask, write_big_tiff

# from ray import data

SLIDES_PATH = "/mnt/data/Projects/MOU/Mammaprint/Another_WSIs"
MASK_DEST = "/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tissue_masks"
LEVEL = 5


# def process_slide(row: dict[str, Any]) -> dict[str, Any]:
#     with OpenSlide(row["item"]) as slide:
#         mpp_x, mpp_y = slide_resolution(slide, LEVEL)

#     slide = Image.new_from_file(row["item"], level=LEVEL)
#     mask = tissue_mask(slide, mpp=mean((mpp_x, mpp_y)))
#     write_big_tiff(
#         mask,
#         path=Path(MASK_DEST, Path(row["item"]).with_suffix(".tiff").name),
#         mpp_x=mpp_x,
#         mpp_y=mpp_y,
#     )
#     return row


@ray.remote
def process_slide(path) -> None:
    with OpenSlide(path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, LEVEL)

    slide = Image.new_from_file(path, level=LEVEL)
    mask = tissue_mask(slide, mpp=mean((mpp_x, mpp_y)))
    write_big_tiff(
        mask,
        path=Path(MASK_DEST, Path(path).with_suffix(".tiff").name),
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


def main() -> None:
    slides = Path(SLIDES_PATH).rglob("*.mrxs")
    Path(MASK_DEST).mkdir(exist_ok=True)
    process_items(slides, process_slide)


if __name__ == "__main__":
    main()
