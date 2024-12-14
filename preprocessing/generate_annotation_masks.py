# Copyright (c) The RationAI team.

from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET

import pyvips
import ray
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide
from PIL.ImageDraw import _Ink
from rationai.masks import process_items, write_big_tiff
from rationai.masks.annotations import XMLPolygonMask


SLIDES_PATH = ""
ANNOTATIONS_PATH = ""
MASK_DEST = "data/annotation_masks"


class TumorMask(XMLPolygonMask):
    @property
    def regions(self) -> Iterable[tuple[ET.Element, _Ink]]:
        regions = self.root.findall("Annotation/Regions/Region")
        return zip(regions, [255] * len(regions), strict=False)

    def get_region_coordinates(
        self, region: ET.Element
    ) -> Iterable[tuple[float, float]]:
        for vertex in region.findall("Vertices/Vertex"):
            yield float(vertex.get("X")), float(vertex.get("Y"))

    @property
    def annotation_mpp_x(self) -> float:
        return float(self.root.get("MicronsPerPixel"))

    @property
    def annotation_mpp_y(self) -> float:
        return self.annotation_mpp_x  # Both mpp_x and mpp_y are the same


LEVEL = 3


@ray.remote
def process_slide(slide_path: Path) -> None:
    annotation_file = Path(ANNOTATIONS_PATH, f"{slide_path.stem}.xml")
    with OpenSlide(slide_path) as slide:
        downsample = slide.level_downsamples[LEVEL]
        annotator = TumorMask(
            path=annotation_file,
            mask_size=slide.level_dimensions[LEVEL],
            mask_mpp_x=float(slide.properties[PROPERTY_NAME_MPP_X]) * downsample,
            mask_mpp_y=float(slide.properties[PROPERTY_NAME_MPP_Y]) * downsample,
        )

    mask = annotator()
    xres = 1000 / annotator.mask_mpp_x  # pixels/mm
    yres = 1000 / annotator.mask_mpp_y  # pixels/mm

    mask_path = Path(MASK_DEST, f"{slide_path.stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    write_big_tiff(
        pyvips.Image.new_from_array(mask),
        path=mask_path,
        xres=xres,
        yres=yres,
    )


def main() -> None:
    slides = Path(SLIDES_PATH).rglob("*.tiff")
    process_items(list(slides), process_item=process_slide)


if __name__ == "__main__":
    main()
