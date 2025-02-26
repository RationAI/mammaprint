from pathlib import Path
from typing import Any, Dict

import pyarrow
from openslide import OpenSlide
from pyvips import Image
from rationai.masks import slide_resolution, tissue_mask
from ray import data
from ray.data.datasource import FilenameProvider
from ray.data.datasource.file_datasink import RowBasedFileDatasink

SLIDES_PATH = "/mnt/data/Projects/MOU/Mammaprint/Learnig_set_mamaprint"
MASK_DEST = "/mnt/data/Projects/MOU/Mammaprint/Learnig_set_mamaprint_tissue_masks"
LEVEL = 5


class Filename(FilenameProvider):
    def get_filename_for_row(
        self, row: dict, task_index: int, block_index: int, row_index: int
    ) -> str:
        return Path(row["item"]).with_suffix("tiff").name


class BigTIFFDatasink(RowBasedFileDatasink):
    def __init__(
        self,
        path: str,
        column: str,
        tiffsave_kwargs: dict[str, Any] | None = None,
        **file_datasink_kwargs,
    ) -> None:
        super().__init__(path, file_format="tiff", **file_datasink_kwargs)
        self.column = column
        self.tiffsave_kwargs = tiffsave_kwargs or {}

    def write_row_to_file(self, row: Dict[str, Any], file: pyarrow.NativeFile) -> None:
        from pyvips import Image

        image = Image.new_from_array(row[self.column])
        buffer = image.tiffsave_buffer(**self.tiffsave_kwargs)
        file.write(buffer.getvalue())


def process_slide(row: dict[str, Any]) -> dict[str, Any]:
    with OpenSlide(row["item"]) as slide:
        mpp_x, mpp_y = slide_resolution(slide, LEVEL)

    slide = Image.new_from_file(row["item"], level=LEVEL)
    row["tissue_mask"] = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2)
    return row


def main() -> None:
    slides = Path(SLIDES_PATH).rglob("*.mrxs")
    data.from_items(list(slides)).map(process_slide).write_datasink(
        BigTIFFDatasink("data", column="tissue_mask", filename_provider=Filename())
    )


if __name__ == "__main__":
    main()
