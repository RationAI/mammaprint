from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split


SLIDES_PATH = "/mnt/data/Projects/MOU/Mammaprint/Test_set_mammaprint/"
TISSUE_MASKS_PATH = "/mnt/data/Projects/MOU/Mammaprint/Test_set_mammaprint_tiff/"
ANNOTATION_MASKS_PATH = "/mnt/data/Projects/MOU/Mammaprint/Test_set_tissue_classification_tumor_masks/test_heatmaps/"

@dataclass
class CancerTileMetadata(TileMetadata):
    cancer_percentage: float


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        if class_overlaps.get(0, 0) > 0.5:
            return None
        return tile_labels


class CancerMask(PyvipsMask[CancerTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> CancerTileMetadata:
        return CancerTileMetadata(
            **asdict(tile_labels), cancer_percentage=class_overlaps.get(255, 0)
        )


source = OpenSlideTileSource(mpp=0.25, tile_extent=512, stride=256)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)
cancer_mask = CancerMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)


@ray.remote
def handler(slide_path: Path) -> TiledSlideMetadata:
    slide, tiles = source(slide_path)

    tissue_mask_path = Path(TISSUE_MASKS_PATH, slide_path.name)
    cancer_mask_path = Path(ANNOTATION_MASKS_PATH, slide_path.name)

    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    tiles = cancer_mask(cancer_mask_path, slide.extent, tiles)

    return slide, tiles


def main() -> None:
    slides, test_slides = train_test_split(
        list(Path(SLIDES_PATH).rglob("*.tiff")), test_size=0.2
    )
    train_slides, val_slides = train_test_split(slides, test_size=0.1)

    train_slides_df, train_tiles_df = tiling(slides=train_slides, handler=handler)
    val_slides_df, val_tiles_df = tiling(slides=list(val_slides), handler=handler)
    test_slides_df, test_tiles_df = tiling(slides=list(test_slides), handler=handler)

    mlflow.set_experiment(experiment_name="Mamma-print")
    with mlflow.start_run(run_name="Luminal-type Dataset") as _:
        save_mlflow_dataset(
            slides=train_slides_df,
            tiles=train_tiles_df,
            dataset_name="Luminal-type - train",
        )
        save_mlflow_dataset(
            slides=val_slides_df, tiles=val_tiles_df, dataset_name="Luminal-type - val"
        )
        save_mlflow_dataset(
            slides=test_slides_df,
            tiles=test_tiles_df,
            dataset_name="Luminal-type - test",
        )


if __name__ == "__main__":
    main()
