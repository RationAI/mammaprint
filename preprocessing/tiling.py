"""Script for creating slides and tiles datasets for mammaprint prediction using Ray Data."""

from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import ray
import ray.data as rd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, tile_overlay_overlap
from ratiopath.tiling.utils import row_hash
from ray.data import Dataset
from ray.data.block import DataBatch
from ray.data.expressions import col
from shapely.geometry import box
from shapely.geometry.polygon import Polygon

from preprocessing.monitor import ResourceMonitor

BATCH_SIZE = 256


def add_missing_epithelium_overlap(batch: DataBatch) -> DataBatch:
    batch["epithelium_overlap"] = np.nan
    return batch


def add_tile_overlap(
    tiles: Dataset,
    roi: Polygon,
    path_col: str,
    overlap_col: str,
    struct_field: str,
) -> Dataset:
    """Compute per-tile overlap score against a mask overlay.

    This helper uses `tile_overlay_overlap` to compute, for each tile,
    how much of the given ROI (tile sub-region) overlaps a binary mask.

    The mask file path is read from the column ``path_col`` in the Ray Dataset. The
    function expects the dataset to contain tile coordinates and microns-per-pixel columns
    (``tile_x``, ``tile_y``, ``mpp_x``, ``mpp_y``), as produced by the upstream tiling
    pipeline.

    Parameters
    ----------
    tiles:
        Ray Dataset with tile rows and required coordinate columns.
    roi:
        ROI (in tile pixel coordinates) used when computing the overlap.
        For example, to check tissue in the center of a tile, use a centered box.
    path_col:
        Column name containing the mask image path for each tile.
    overlap_col:
        Name of the output column to store the computed overlap score in [0, 1].
    struct_field:
        The pixel value (as string: "0" or "255") in the binary mask representing the
        undesired artifact or background. The mask contains only values 0 and 255.
        The overlap is computed as ``1 - (fraction of pixels with this value)``, yielding
        the fraction of desired (non-artifact) pixels.
        For example, when folding masks have folds as white (255), pass "255" to compute
        the amount of non-folded tissue.

    Returns:
    -------
    Dataset
        The input dataset with an additional float column ``overlap_col``.
    """
    overlap_struct_col = f"_{overlap_col}_struct"
    overlap_struct = tile_overlay_overlap(
        roi,
        col(path_col),
        col("tile_x"),
        col("tile_y"),
        col("mpp_x"),
        col("mpp_y"),
    )
    tiles = tiles.with_column(
        overlap_struct_col, overlap_struct, num_cpus=0.2, memory=512 * 1024**2
    )

    def extract_value(batch: DataBatch) -> DataBatch:
        batch[overlap_col] = np.array(
            [
                1.0 - (s.get(struct_field, 1.0) or 1.0) if isinstance(s, dict) else 0.0
                for s in batch[overlap_struct_col]
            ]
        )
        return batch

    tiles = tiles.map_batches(
        extract_value,
        batch_format="pandas",
        num_cpus=0.1,
        memory=256 * 1024**2,
    )

    return tiles.drop_columns([overlap_struct_col])


def tiling(
    row: dict[str, Any],
    tissue_masks_path: str | None,
    qc_masks_path: str | None,
    epithelium_masks_path: str | None,
) -> list[dict[str, Any]]:
    """Generate tile coordinates for a single slide."""
    slide_path = Path(row["path"])
    tiff_slide_name = slide_path.with_suffix(".tiff").name

    tissue_mask = (
        str(Path(tissue_masks_path) / tiff_slide_name) if tissue_masks_path else ""
    )

    epithelium_mask = (
        str(Path(epithelium_masks_path) / tiff_slide_name)
        if epithelium_masks_path
        else ""
    )

    if qc_masks_path:
        qc_path = Path(qc_masks_path)
        blur_mask = str(qc_path / f"Piqe_piqe_median_activity_mask_{tiff_slide_name}")
        folding_mask = str(qc_path / f"FoldingFunction_folding_test_{tiff_slide_name}")
        residual_mask = str(
            qc_path / f"ResidualArtifactsAndCoverage_coverage_mask_{tiff_slide_name}"
        )
    else:
        blur_mask = ""
        folding_mask = ""
        residual_mask = ""

    return [
        {
            "tile_x": x,
            "tile_y": y,
            "slide_id": row["id"],
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
            "tissue_mask_path": tissue_mask,
            "epithelium_mask_path": epithelium_mask,
            "blur_mask_path": blur_mask,
            "folding_mask_path": folding_mask,
            "residual_mask_path": residual_mask,
            "type": row["type"],
            "mammaprint_index": row["mammaprint_index"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if not ray.is_initialized():
        ray.init()

    monitor = ResourceMonitor("/mnt/projects/mammaprint/tiling_resource_usage.csv", interval=60)
    monitor.start()

    tissue_masks_path = (
        None
        if config.dataset.mlflow_uris.tissue_masks is None
        else download_artifacts(config.dataset.mlflow_uris.tissue_masks)
    )
    epithelium_masks_path = (
        None
        if config.dataset.mlflow_uris.epithelium_masks is None
        else download_artifacts(config.dataset.mlflow_uris.epithelium_masks)
    )
    qc_masks_path = config.dataset.paths.qc_masks or None

    # Read slide paths from data mapping
    slide_labels_df = pd.read_csv(config.dataset.paths.data_mapping)
    slide_labels_df["slide_path"] = slide_labels_df["path"] + ".mrxs"
    slide_paths = slide_labels_df["slide_path"].tolist()
    metadata_ds = rd.from_pandas(
        slide_labels_df[["slide_path", "type", "mammaprint_index"]]
    )

    print(f"\n[INFO] Processing {len(slide_paths)} slides using Ray Data pipeline\n")

    # Read slides
    monitor.log_phase("read_slides")
    slides = read_slides(
        slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    ).join(
        metadata_ds,
        on=("path",),
        right_on=("slide_path",),
        num_partitions=16,
        join_type="left_outer",
    )

    # Create unique slide IDs and save slide metadata
    slides = slides.map(row_hash, num_cpus=0.1, memory=256 * 1024**2)

    # Materialize slides so the source pipeline (read + join + hash) runs once
    # and is reused for both slides_df and the tiles pipeline.
    slides = slides.materialize()
    slides_df = slides.to_pandas()
    monitor.log_phase("slides_done", tile_count=len(slides_df))

    # Expand slides into tile coordinates
    monitor.log_phase("grid_tiling")
    tiles = slides.flat_map(
        partial(
            tiling,
            tissue_masks_path=tissue_masks_path,
            qc_masks_path=qc_masks_path,
            epithelium_masks_path=epithelium_masks_path,
        ),
        num_cpus=0.2,
        memory=256 * 1024**2,
    ).repartition(target_num_rows_per_block=1024)

    # Compute masks and filter tissue tiles
    # Tissue mask checks center 50%
    offset = config.tile_extent // 4  # 128 for 512
    size = config.tile_extent // 2  # 256 for 512
    tissue_roi = box(offset, offset, offset + size, offset + size)
    # Full tile
    full_roi = box(0, 0, config.tile_extent, config.tile_extent)

    # Add tissue overlaps
    monitor.log_phase("tissue_overlap")
    tiles = add_tile_overlap(
        tiles, tissue_roi, "tissue_mask_path", "tissue_overlap", "0"
    )
    tiles = tiles.filter(expr=col("tissue_overlap") > 0.0)

    # Materialize after tissue filter to:
    # 1. Lock in the reduction (discard background tiles before computing remaining overlaps)
    # 2. Break the long operator chain to prevent streaming backpressure starvation
    tiles = tiles.materialize()
    tissue_tile_count = tiles.count()
    monitor.log_phase("tissue_filter_done", tile_count=tissue_tile_count)

    monitor.log_phase("blur_overlap", tile_count=tissue_tile_count)
    tiles = add_tile_overlap(tiles, full_roi, "blur_mask_path", "blur_overlap", "255")

    monitor.log_phase("folding_overlap", tile_count=tissue_tile_count)
    tiles = add_tile_overlap(
        tiles, full_roi, "folding_mask_path", "folding_overlap", "255"
    )

    monitor.log_phase("residual_overlap", tile_count=tissue_tile_count)
    tiles = add_tile_overlap(
        tiles, full_roi, "residual_mask_path", "residual_overlap", "0"
    )

    if epithelium_masks_path:
        monitor.log_phase("epithelium_overlap", tile_count=tissue_tile_count)
        tiles = add_tile_overlap(
            tiles, full_roi, "epithelium_mask_path", "epithelium_overlap", "0"
        )
    else:
        print(
            "[INFO] Skipping epithelium overlap: dataset.mlflow_uris.epithelium_masks is not set"
        )

        tiles = tiles.map_batches(
            add_missing_epithelium_overlap,
            batch_format="pandas",
            num_cpus=0.1,
            memory=256 * 1024**2,
        )

    # Drop unnecessary columns
    tiles = tiles.drop_columns(
        [
            "tissue_mask_path",
            "epithelium_mask_path",
            "blur_mask_path",
            "folding_mask_path",
            "residual_mask_path",
            "mpp_x",
            "mpp_y",
        ]
    )

    # Convert tiles Ray Dataset to pandas DataFrame
    monitor.log_phase("materialize_final")
    tiles_df = tiles.to_pandas()
    monitor.log_phase("save_dataset", tile_count=len(tiles_df))

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.data_name,
    )

    monitor.log_phase("done", tile_count=len(tiles_df))
    print(
        f"\n[INFO] Generated {len(slides_df)} slide records and {len(tiles_df)} tile records\n"
    )

    monitor.stop()
    print("[INFO] Resource usage log: /mnt/projects/mammaprint/tiling_resource_usage.csv")

    ray.shutdown()


if __name__ == "__main__":
    main()
