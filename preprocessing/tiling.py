import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
import polars as pl
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import SlideMetadata, TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define paths
SLIDES_PATH = ""
TISSUE_MASKS_PATH = ""
ANNOTATION_MASKS_PATH = ""
LABELS_FILE = ""  # Path to the CSV file containing slide labels


@dataclass
class PipelineTileMetadata(TileMetadata):
    slide_name: str
    coord_x: int
    coord_y: int
    class_id: int
    cancer_percentage: float = 0.0


@dataclass
class PipelineSlideMetadata(SlideMetadata):
    luminal_id: int
    slide_name: str
    slide_fp: str
    sample_level: int
    tile_size: int
    slide_width: int
    slide_height: int
    step_size: int
    center_size: int
    path: str
    level: int


class TissueMask(PyvipsMask[PipelineTileMetadata]):
    def forward_tile(
        self, tile_labels: PipelineTileMetadata, class_overlaps: dict[int, float]
    ) -> PipelineTileMetadata | None:
        if class_overlaps.get(0, 0) > 0.5:
            return None
        return tile_labels


class CancerMask(PyvipsMask[PipelineTileMetadata]):
    def forward_tile(
        self, tile_labels: PipelineTileMetadata, class_overlaps: dict[int, float]
    ) -> PipelineTileMetadata:
        # Default cancer percentage
        cancer_percentage = 0.0

        # Check if background is less than 50% and apply whiteness threshold
        if class_overlaps.get(0, 0) < 0.5:
            whiteness_threshold = (
                128  # Define threshold for "white" (e.g., grayscale > 50%)
            )
            cancer_percentage = sum(
                overlap
                for value, overlap in class_overlaps.items()
                if value >= whiteness_threshold
            )

        # Log for debugging
        logging.info(
            f"Cancer mask applied. Cancer coverage for tile: {cancer_percentage}"
        )

        # Update the existing tile_labels with cancer_percentage
        tile_labels.cancer_percentage = cancer_percentage
        return tile_labels


# Initialize tile source and mask
source = OpenSlideTileSource(mpp=0.25, tile_extent=512, stride=256)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)
cancer_mask = CancerMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)

if not LABELS_FILE.exists():
    logging.error(
        f"Labels file not found at {LABELS_FILE}. Please ensure the file exists."
    )
    raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}")

# Load labels data from CSV file using pandas, then convert to Polars
labels_df_pandas = pd.read_csv(LABELS_FILE, sep=";")
labels_df = pl.from_pandas(labels_df_pandas)

labels_df = labels_df.with_columns(
    [
        pl.col("luminal_id").cast(pl.Int64),
        pl.col("mammaprint").str.replace(",", ".").cast(pl.Float32),
    ]
)

# Convert labels_df to a dictionary for efficient lookup
labels_dict = (
    labels_df.select(["slide_name", "luminal_id"])
    .to_pandas()
    .set_index("slide_name")["luminal_id"]
    .to_dict()
)


@ray.remote
def handler(slide_path: Path) -> TiledSlideMetadata | None:
    logging.info(f"Processing {slide_path}")
    try:
        slide, tiles = source(slide_path)
    except Exception as e:
        logging.error(f"Failed to load slide {slide_path}: {e}")
        return None

    if slide.tile_extent_x != slide.tile_extent_y:
        logging.error(f"Non-square tiles in slide {slide_path}. Skipping.")
        return None

    logging.info(f"Number of tiles: {len(tiles)}")

    # Extract the slide name
    slide_name = slide_path.stem

    # Efficient label lookup
    slide_label = labels_dict.get(slide_name, None)

    if slide_label is None:
        logging.warning(
            f"No label found for slide {slide_name}. Assigning default label."
        )
        slide_label = 0

    # Create PipelineSlideMetadata
    slide_metadata = PipelineSlideMetadata(
        **asdict(slide),
        luminal_id=slide_label,
        slide_name=slide_name,
        slide_fp=slide.path,
        sample_level=0,
        tile_size=slide.tile_extent_x,
        slide_width=slide.extent_x,
        slide_height=slide.extent_y,
        step_size=slide.stride_x,
        center_size=slide.stride_x,
    )
    logging.info(f"Slide metadata: {slide_metadata}")

    # Define mask paths
    tissue_mask_path = Path(TISSUE_MASKS_PATH, slide_path.name)
    cancer_mask_path = Path(ANNOTATION_MASKS_PATH, slide_path.name)

    # Validate mask paths
    if not tissue_mask_path.exists():
        logging.error(f"Tissue mask not found for slide {slide_name}. Skipping.")
        return None

    # Apply Tissue Mask
    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)

    # Apply Cancer Mask if it exists
    if cancer_mask_path.exists():
        tiles = cancer_mask(cancer_mask_path, slide.extent, tiles)
        logging.info(
            f"Applied cancer mask. Number of tiles after masking: {len(tiles)}"
        )
    else:
        logging.info("Cancer mask not found; skipping cancer mask application.")

    # Create tile metadata with `cancer_percentage` as set by the cancer mask
    tiles_metadata = [
        PipelineTileMetadata(
            **asdict(t),
            class_id=slide_label,
            slide_name=slide_name,
            coord_x=t.x,
            coord_y=t.y,
            cancer_percentage=getattr(t, "cancer_percentage", 0.0),
        )
        for t in tiles
    ]
    logging.info("Finished labeling tiles")

    return slide_metadata, tiles_metadata


def main() -> None:
    all_slides = list(Path(SLIDES_PATH).rglob("*.mrxs"))
    labeled_slides_names = set(labels_df["slide_name"].to_list())
    labeled_slides = [
        slide for slide in all_slides if slide.stem in labeled_slides_names
    ]

    unlabeled_slides = [
        slide for slide in all_slides if slide.stem not in labeled_slides_names
    ]
    if unlabeled_slides:
        logging.info(
            f"Found {len(unlabeled_slides)} slides without labels. They will be skipped."
        )

    ray.init(ignore_reinit_error=True)

    # Process labeled slides using tiling
    try:
        test_slides_df, test_tiles_df = tiling(slides=labeled_slides, handler=handler)
    except Exception as e:
        logging.error(f"Error during tiling: {e}")
        ray.shutdown()
        return

    # Save to MLflow
    try:
        mlflow.set_experiment(experiment_name="Mamma-print")
        with mlflow.start_run(
            run_name="Tiling mammaprint test dataset from tissue classification model, threshold 128"
        ) as _:
            save_mlflow_dataset(
                slides=test_slides_df,
                tiles=test_tiles_df,
                dataset_name="test_tissue_classification_tumor_tiles",
            )
    except Exception as e:
        logging.error(f"Error during MLflow logging: {e}")
    finally:
        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
