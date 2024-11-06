from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import ray
import pandas as pd
import polars as pl
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata, SlideMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split


SLIDES_PATH = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_tiff/"
TISSUE_MASKS_PATH = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_tissue_masks/"
ANNOTATION_MASKS_PATH = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_tissue_classification_tumor_masks/test_heatmaps/"

# @dataclass
# class CancerTileMetadata(TileMetadata):
#     cancer_percentage: float

@dataclass
class PipelineTileMetadata:
    slide_name: str
    coord_x: int
    coord_y: int
    class_id: int

@dataclass
class PipelineSlideMetadata(SlideMetadata):
    luminal_id: int
    slide_name: str
    slide_fp: str
    sample_level: int
    tile_size: int
    # from OpenSlideMetadata:
    path: str
    level: int


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        if class_overlaps.get(0, 0) > 0.5:
            return None
        return tile_labels


# class CancerMask(PyvipsMask[CancerTileMetadata]):
#     def forward_tile(
#         self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
#     ) -> CancerTileMetadata:
#         return CancerTileMetadata(
#             **asdict(tile_labels), cancer_percentage=class_overlaps.get(255, 0)
#         )


source = OpenSlideTileSource(mpp=0.48, tile_extent=512, stride=256)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)

# Determine the directory where this script resides
SCRIPT_DIR = Path(__file__).parent.resolve()

# Define the path to Learning_set.csv relative to the script's directory
LABELS_FILE = SCRIPT_DIR / 'Learning_set.csv'

# Load labels data from CSV file using pandas, then convert to Polars
LABELS_FILE = 'Learning_set.csv'
labels_df_pandas = pd.read_csv(LABELS_FILE, sep=';')
labels_df = pl.from_pandas(labels_df_pandas)
# Ensure 'luminal_id' is Int64 and 'mammaprint' is Float32
labels_df = labels_df.with_columns([
    pl.col("luminal_id").cast(pl.Int64),
    pl.col('mammaprint').str.replace(',', '.').cast(pl.Float32)
])

# cancer_mask = CancerMask(
#     tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
# )


@ray.remote
def handler(slide_path: Path) -> TiledSlideMetadata:
    print(f"Processing {slide_path}")
    slide, tiles = source(slide_path)
    assert slide.tile_extent_x == slide.tile_extent_y, "Only square tiles!"
    print(f"{len(tiles)=}")

    slide_name = slide_path.stem

    matching_row = labels_df.filter(pl.col('slide_name') == slide_name)
    if matching_row.is_empty():
        slide_label = 0  # Or any other default value indicating unlabeled
        raise ValueError(f"No matching row found for slide {slide_name}")
    else:
        slide_label = matching_row['luminal_id'].item()

    slide = PipelineSlideMetadata(
        **asdict(slide),
        luminal_id=slide_label,
        slide_name=slide_name,
        slide_fp=slide.path,
        sample_level=0,
        tile_size=slide.tile_extent_x,
    )
    print(f"{slide=}")

    tissue_mask_path = Path(TISSUE_MASKS_PATH, slide_path.name)
    cancer_mask_path = Path(ANNOTATION_MASKS_PATH, slide_path.name)
    # Check if both masks exist, if not, skip this slide
    if not cancer_mask_path.exists():
        # print(f"Skipping {slide_path} as required masks are missing.")
        tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    else:
        tiles = tissue_mask(cancer_mask_path, slide.extent, tiles)
    print("Finished tissue mask")

    tiles = [
        PipelineTileMetadata(
            **asdict(t),
            class_id=slide_label,
            slide_name=slide_name,
            coord_x=t.x,
            coord_y=t.y,
        )
        for t in tiles
    ]
    print("Finished labelling")

    return slide, tiles


def main() -> None:
    all_slides = list(Path(SLIDES_PATH).rglob("*.tiff"))
    labeled_slides_names = set(labels_df['slide_name'].to_list())
    labeled_slides = [slide for slide in all_slides if slide.stem in labeled_slides_names]

    unlabeled_slides = [slide for slide in all_slides if slide.stem not in labeled_slides_names]
    if unlabeled_slides:
        print(f"Found {len(unlabeled_slides)} slides without labels. They will be skipped.")

    # Proceed with only labeled slides
    test_slides_df, test_tiles_df = tiling(slides=labeled_slides, handler=handler)

    mlflow.set_experiment(experiment_name="Mamma-print")
    with mlflow.start_run(run_name="Tiling mammaprint train dataset from tissue classification model") as _:
        save_mlflow_dataset(
            slides=test_slides_df,
            tiles=test_tiles_df,
            dataset_name="train_tissue_classification_tumor_tiles",
        )


if __name__ == "__main__":
    main()
