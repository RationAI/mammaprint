from pathlib import Path

import pyvips
import ray
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide
from rationai.masks import (
    process_items,
    tissue_mask,
    write_big_tiff,
)

SLIDES_PATH = "/mnt/data/Projects/MOU/Mammaprint/Test_set_mamaprint/"
MASK_DEST = "/mnt/data/Projects/MOU/Mammaprint/Test_set_mamaprint_tissue_masks/"
LEVEL = 1
# FILE_SIZE_LIMIT_MB = 3  # Size limit in MB


@ray.remote
def process_slide(slide_path: Path) -> None:
    mask_path = Path(MASK_DEST, f"{Path(slide_path).stem}.tiff")
    # if mask_path.exists() and mask_path.stat().st_size > FILE_SIZE_LIMIT_MB * 1024 * 1024:
    #     print(f"Mask for {slide_path} is larger than {FILE_SIZE_LIMIT_MB}MB, skipping.")
    #     return 

    with OpenSlide(slide_path) as slide:
        downsample = slide.level_downsamples[LEVEL]
        xres = 1000 / (float(slide.properties[PROPERTY_NAME_MPP_X]) * downsample)
        yres = 1000 / (float(slide.properties[PROPERTY_NAME_MPP_Y]) * downsample)

    slide = pyvips.Image.new_from_file(slide_path, level=LEVEL)
    mask = tissue_mask(slide)
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(mask, path=mask_path, xres=xres, yres=yres)


def main() -> None:
    slides = Path(SLIDES_PATH).rglob("*.mrxs")
    process_items(list(slides), process_item=process_slide)


if __name__ == "__main__":
    main()
