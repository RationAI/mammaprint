from pathlib import Path
from statistics import mean
from typing import Any

from openslide import OpenSlide
from pyvips import Image
from rationai.masks import slide_resolution, tissue_mask, write_big_tiff
from ray import data

SLIDES_PATHS = [
    "/mnt/data/Projects/MOU/Mammaprint/Learnig_set_mamaprint",
    "/mnt/data/Projects/MOU/Mammaprint/Test_set_mamaprint",
    "/mnt/data/Projects/MOU/Mammaprint/Another_WSIs",
]
MASK_DEST = "/mnt/data/Projects/MOU/Mammaprint/tissue_masks"
LEVEL = 5


def process_slide(row: dict[str, Any]) -> dict[str, Any]:
    with OpenSlide(row["item"]) as slide:
        mpp_x, mpp_y = slide_resolution(slide, LEVEL)

    slide = Image.new_from_file(row["item"], level=LEVEL)
    mask = tissue_mask(slide, mpp=mean((mpp_x, mpp_y)))
    write_big_tiff(
        mask,
        path=Path(MASK_DEST, Path(row["item"]).with_suffix(".tiff").name),
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    return row


def main() -> None:
    Path(MASK_DEST).mkdir(exist_ok=True)
    slides = [
        path for slide_path in SLIDES_PATHS for path in Path(slide_path).rglob("*.mrxs")
    ]
    data.from_items(slides).map(process_slide).materialize()


if __name__ == "__main__":
    main()
