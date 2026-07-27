"""Script for creating slides and tiles datasets for mammaprint prediction using Ray Data."""

import os
from functools import partial
from pathlib import Path
from typing import Any, Literal

import hydra
import numpy as np
import pandas as pd
import ray
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


# Rows per block after the tiling flat_map. Larger blocks amortize Ray's per-block
# scheduling overhead and the per-batch mask file-open in the overlap UDFs (one open per
# (mask, block) instead of one per 256 tiles), at the cost of more object-store memory per
# block. Kept well within the object store sized below.
BATCH_SIZE = 4096

# Fraction of the pod's memory to give Ray's object store when RAY_MEMORY_BYTES is set.
# The rest is left for Ray's heap and the Python worker processes.
RAY_OBJECT_STORE_FRACTION = 0.4


def _ray_init_kwargs() -> dict[str, Any]:
    """Build ``ray.init`` sizing kwargs from the pod's resources.

    On the K8s cluster ``ray.init()`` auto-detection was unreliable: it over-detected CPUs
    (saw the 192-core node instead of the pod's cgroup limit) and under-sized the object
    store (~9.5 GiB), which backpressured the repartition queue and collapsed the
    overlap+filter stage to a single task.

    To avoid duplicating the pod's resource numbers, the launcher
    (``scripts/preprocessing/run_tiling.py``) exports them as environment variables derived
    from its own ``submit_job(cpu=..., memory=...)`` args:

    * ``RAY_NUM_CPUS`` -- number of CPUs to give Ray.
    * ``RAY_MEMORY_BYTES`` -- the pod's total memory in bytes; the object store is sized as
      a fraction of it.

    When unset (e.g. a plain local run), we fall back to Ray's own auto-detection for both.
    """
    kwargs: dict[str, Any] = {}

    num_cpus = os.environ.get("RAY_NUM_CPUS")
    if num_cpus:
        kwargs["num_cpus"] = int(float(num_cpus))

    memory_bytes = os.environ.get("RAY_MEMORY_BYTES")
    if memory_bytes:
        # Fraction of pod memory for the object store. The rest is left for Ray's heap and
        # the Python worker processes. Capped by /dev/shm at the pod level (see run_tiling.py).
        kwargs["object_store_memory"] = int(float(memory_bytes) * RAY_OBJECT_STORE_FRACTION)

    return kwargs


def add_missing_epithelium_overlap(batch: DataBatch) -> DataBatch:
    """Fill a NaN ``epithelium_overlap`` when no epithelium mask is configured."""
    batch["epithelium_overlap"] = np.nan
    return batch


def add_tile_overlap(
    tiles: Dataset,
    roi: Polygon,
    path_col: str,
    overlap_col: str,
    keep: Literal["zero", "nonzero"],
    background: str = "0",
) -> Dataset:
    """Compute a per-tile coverage score against a mask overlay.

    This helper uses `tile_overlay_overlap` to obtain, for each tile, the fraction of the ROI
    covered by every distinct pixel value in the mask, then reduces it to a single score in
    ``[0, 1]`` where **higher = keep**.

    The masks are downsampled to a continuous ``[0, 255]`` range (they are NOT strictly binary):
    the background/clean pixel value is ``0`` and *any* non-zero value marks foreground (tissue) or
    an artifact, depending on the mask. Two reductions are therefore supported via ``keep``:

    * ``keep="nonzero"`` (tissue/epithelium): score = fraction of non-zero pixels = foreground
      coverage. Keep tiles with a lot of tissue/epithelium.
    * ``keep="zero"`` (QC artifact masks: blur/folding/residual): score = fraction of ``0`` pixels =
      clean fraction. Keep tiles that are mostly artifact-free.

    In both cases a downstream ``overlap > threshold`` filter keeps the desirable tiles.

    The mask file path is read from the column ``path_col`` in the Ray Dataset. The function expects
    the dataset to contain tile coordinates and microns-per-pixel columns (``tile_x``, ``tile_y``,
    ``mpp_x``, ``mpp_y``), as produced by the upstream tiling pipeline.

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
        Name of the output column to store the computed score in [0, 1].
    keep:
        Which pixels count as desirable: ``"zero"`` scores the clean (background) fraction,
        ``"nonzero"`` scores the foreground fraction.
    background:
        Pixel value (as string) of the clean/background class. Defaults to ``"0"``.

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
    # num_cpus=1 so Ray schedules one task per core across the pod. Without it the fused
    # overlap+filter operator ran with a single task and left 47 of 48 cores idle.
    tiles = tiles.with_column(overlap_struct_col, overlap_struct, num_cpus=1)

    def extract_value(batch: DataBatch) -> DataBatch:
        def score(s: Any) -> float:
            # No mask / no readable pixels in the ROI -> score 0 so the tile is dropped.
            if not isinstance(s, dict) or not s:
                return 0.0
            # Fraction of clean/background (== background value) pixels in the ROI.
            zero_fraction = s.get(background, 0.0) or 0.0
            # keep="zero": clean fraction; keep="nonzero": foreground (1 - clean) fraction.
            return zero_fraction if keep == "zero" else 1.0 - zero_fraction

        batch[overlap_col] = np.array([score(s) for s in batch[overlap_struct_col]])
        return batch

    tiles = tiles.map_batches(
        extract_value,
        batch_format="pandas",
        num_cpus=1,
    )

    return tiles.drop_columns([overlap_struct_col])


def count_candidate_tiles(slides_df: pd.DataFrame) -> int:
    """Number of candidate tiles across all slides, computed analytically (no mask IO).

    Mirrors ``grid_tiles(..., last="keep")``: per axis it yields
    ``ceil((extent - tile_extent) / stride) + 1`` tiles, so the per-slide count is the product
    over the x and y axes. Summed over slides this is the pre-filter tile count -- computed from
    the slide metadata alone, so we never have to materialize the ~250M-row candidate set to count
    it.
    """
    ex = np.asarray(slides_df["extent_x"], dtype=float)
    ey = np.asarray(slides_df["extent_y"], dtype=float)
    tx = np.asarray(slides_df["tile_extent_x"], dtype=float)
    ty = np.asarray(slides_df["tile_extent_y"], dtype=float)
    sx = np.asarray(slides_df["stride_x"], dtype=float)
    sy = np.asarray(slides_df["stride_y"], dtype=float)
    nx = np.ceil((ex - tx) / sx) + 1
    ny = np.ceil((ey - ty) / sy) + 1
    return int(np.sum(nx * ny))


def log_drop_funnel(
    tissue_survivors: Dataset,
    n_candidates: int,
    thresholds: dict[str, float],
    logger: MLFlowLogger,
) -> None:
    """Quantify how many tiles each QC threshold drops and log the funnel to MLflow.

    ``tissue_survivors`` must be the dataset AFTER the tissue gate but with ``blur_overlap``,
    ``folding_overlap`` and ``residual_overlap`` already computed and NOT yet filtered (so the
    dropped tiles' scores are still present). Reports two views:

    * **independent** -- how many tiles each threshold removes on its own. Tissue is measured
      against the full candidate grid (``n_candidates``); the QC gates are measured among the
      tissue survivors (the only affordable and meaningful denominator).
    * **sequential** -- the funnel in pipeline order (tissue -> blur -> folding -> residual), where
      each gate is credited only for tiles it is the first to drop. These sum to the total dropped.

    A tile is dropped by a gate when ``overlap <= threshold`` (the strict complement of the
    production ``overlap > threshold`` filter; NaN/empty overlaps already score 0.0 and so count as
    dropped, matching the run).
    """
    df = tissue_survivors.to_pandas()
    n_tissue = len(df)

    b = df["blur_overlap"] <= thresholds["blur"]
    f = df["folding_overlap"] <= thresholds["folding"]
    r = df["residual_overlap"] <= thresholds["residual"]

    independent = {
        "tissue": n_candidates - n_tissue,
        "blur": int(b.sum()),
        "folding": int(f.sum()),
        "residual": int(r.sum()),
    }
    # Sequential attribution in pipeline order: each gate credited only for tiles that survived
    # every earlier gate (tissue already applied by construction of this dataset).
    sequential = {
        "tissue": n_candidates - n_tissue,
        "blur": int(b.sum()),
        "folding": int((~b & f).sum()),
        "residual": int((~b & ~f & r).sum()),
    }
    kept_final = int((~b & ~f & ~r).sum())

    def pct(n: int) -> float:
        return round(100.0 * n / n_candidates, 3) if n_candidates else 0.0

    print("\n[DIAGNOSTICS] Tile drop funnel")
    print(f"  candidate tiles (pre-filter):        {n_candidates:,}")
    print(f"  survived tissue gate:                {n_tissue:,}")
    print("  independent drops (each gate alone; QC gates among tissue survivors):")
    for k in ("tissue", "blur", "folding", "residual"):
        print(f"    {k:9} {independent[k]:>12,}  ({pct(independent[k]):5.2f}% of candidates)")
    print("  sequential drops (pipeline order; mutually exclusive):")
    for k in ("tissue", "blur", "folding", "residual"):
        print(f"    {k:9} {sequential[k]:>12,}  ({pct(sequential[k]):5.2f}% of candidates)")
    print(f"  kept (survived all gates):           {kept_final:,}  ({pct(kept_final):5.2f}%)")

    metrics = {"diag_candidate_tiles": n_candidates, "diag_tissue_survivors": n_tissue,
               "diag_kept_final": kept_final}
    for k in ("tissue", "blur", "folding", "residual"):
        metrics[f"diag_independent_drop_{k}"] = independent[k]
        metrics[f"diag_sequential_drop_{k}"] = sequential[k]
    logger.log_metrics(metrics)


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
        ray.init(**_ray_init_kwargs())

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
    slide_paths = slide_labels_df["path"].tolist()

    print(f"\n[INFO] Processing {len(slide_paths)} slides using Ray Data pipeline\n")

    # Attach labels (type, mammaprint_index) by path. This is a tiny per-slide lookup
    # (~2k rows) that we already hold in the driver, so we broadcast it as a dict and merge
    # in a map_batches rather than a Ray hash-shuffle join. The distributed join was pure
    # overhead here -- worse, its shuffle produced a wildly inflated per-task memory estimate
    # (~1 TiB/task) that no node could satisfy, so the autoscaler rejected it forever and the
    # whole pipeline deadlocked at Join: 0/1 before row_hash/tiling ever ran.
    labels_by_path = slide_labels_df.set_index("path")[
        ["type", "mammaprint_index"]
    ].to_dict("index")

    def attach_labels(batch: pd.DataFrame) -> pd.DataFrame:
        labels = batch["path"].map(labels_by_path)
        batch["type"] = [lp["type"] if isinstance(lp, dict) else None for lp in labels]
        batch["mammaprint_index"] = [
            lp["mammaprint_index"] if isinstance(lp, dict) else None for lp in labels
        ]
        return batch

    # Read slides
    slides = read_slides(
        slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    ).map_batches(attach_labels, batch_format="pandas", num_cpus=0.1)

    # Create unique slide IDs and save slide metadata.
    # Fractional num_cpus so these cheap coordinate-generation stages don't win Ray's
    # per-operator CPU reservation. Ray's ReservationOpResourceAllocator prioritizes upstream
    # operators, so without a cap row_hash/tiling grabbed ~22 of 48 CPUs and starved the
    # expensive downstream overlap+filter stage to a single task while its input queue backed
    # up. Keeping these sub-1.0 leaves the CPU budget for the mask-reading UDFs (num_cpus=1).
    slides = slides.map(row_hash, num_cpus=0.1)

    # Expand slides into tile coordinates
    tiles = slides.flat_map(
        partial(
            tiling,
            tissue_masks_path=tissue_masks_path,
            qc_masks_path=qc_masks_path,
            epithelium_masks_path=epithelium_masks_path,
        ),
        num_cpus=0.25,
    ).repartition(target_num_rows_per_block=BATCH_SIZE)

    # Compute masks and filter tissue tiles
    # Tissue mask checks center 50%
    offset = config.tile_extent // 4  # 128 for 512
    size = config.tile_extent // 2  # 256 for 512
    tissue_roi = box(offset, offset, offset + size, offset + size)
    # Full tile
    full_roi = box(0, 0, config.tile_extent, config.tile_extent)

    # Add coverage scores (all in [0, 1], higher = keep). The masks are downsampled continuous
    # [0, 255] where 0 = clean/background and any non-zero = foreground/artifact (per the QC docs).
    # tissue/epithelium keep the non-zero (foreground) fraction; the QC artifact masks keep the
    # zero (clean) fraction. See scripts/notebooks/mask_coverage_threshold.ipynb for the evidence.
    #
    # We compute each overlap and immediately filter on it, rather than computing all overlaps up
    # front and filtering at the end. The final result is identical (the keep set is the AND of all
    # thresholds), but interleaving means a tile discarded by an early gate never pays for the
    # remaining mask reads. Only ~2.5% of tiles survive, so this roughly halves total overlap work.
    # Thresholds come from the experiment config (see configs/experiment/preprocessing/tiling/*.yaml)
    # and the coverage distributions in scripts/notebooks/mask_coverage_threshold.ipynb: a moderate
    # tissue gate, and lenient artifact-only QC cutoffs (they only remove damaged tissue).
    #
    # Order matters for speed (not for the result): the tissue gate runs first because it is both the
    # most selective and the cheapest (center-50% ROI = smallest region read).
    tiles = add_tile_overlap(
        tiles, tissue_roi, "tissue_mask_path", "tissue_overlap", keep="nonzero"
    )
    tiles = tiles.filter(expr=col("tissue_overlap") > config.tissue_threshold)

    if config.get("diagnostics", False):
        # Diagnostic path: quantify per-gate drops. Compute ALL three QC overlaps on the tissue
        # survivors WITHOUT filtering (so dropped tiles' scores are retained), materialize once so
        # the funnel scan and the subsequent production filters both reuse the same pinned blocks
        # (no re-reading of masks), log the funnel, then apply the identical QC filters below so the
        # written dataset is byte-identical to a normal run.
        tiles = add_tile_overlap(
            tiles, full_roi, "blur_mask_path", "blur_overlap", keep="zero"
        )
        tiles = add_tile_overlap(
            tiles, full_roi, "folding_mask_path", "folding_overlap", keep="zero"
        )
        tiles = add_tile_overlap(
            tiles, full_roi, "residual_mask_path", "residual_overlap", keep="zero"
        )
        tiles = tiles.materialize()

        n_candidates = count_candidate_tiles(slides.to_pandas())
        thresholds = {
            "tissue": config.tissue_threshold,
            "blur": config.blur_threshold,
            "folding": config.folding_threshold,
            "residual": config.residual_threshold,
        }
        log_drop_funnel(tiles, n_candidates, thresholds, logger)

        tiles = tiles.filter(expr=col("blur_overlap") > config.blur_threshold)
        tiles = tiles.filter(expr=col("folding_overlap") > config.folding_threshold)
        tiles = tiles.filter(expr=col("residual_overlap") > config.residual_threshold)
    else:
        tiles = add_tile_overlap(
            tiles, full_roi, "blur_mask_path", "blur_overlap", keep="zero"
        )
        tiles = tiles.filter(expr=col("blur_overlap") > config.blur_threshold)

        tiles = add_tile_overlap(
            tiles, full_roi, "folding_mask_path", "folding_overlap", keep="zero"
        )
        tiles = tiles.filter(expr=col("folding_overlap") > config.folding_threshold)

        tiles = add_tile_overlap(
            tiles, full_roi, "residual_mask_path", "residual_overlap", keep="zero"
        )
        tiles = tiles.filter(expr=col("residual_overlap") > config.residual_threshold)

    # Epithelium is computed but NOT filtered here (that mask is not always populated; add a cutoff
    # once it is, by desired tumor purity). Computed last so it only runs on the surviving tiles.
    if epithelium_masks_path:
        print("[INFO] Computing overlap: epithelium_overlap from epithelium_mask_path")
        tiles = add_tile_overlap(
            tiles,
            full_roi,
            "epithelium_mask_path",
            "epithelium_overlap",
            keep="nonzero",
        )
    else:
        print(
            "[INFO] Skipping epithelium overlap: "
            "dataset.mlflow_uris.epithelium_masks is not set"
        )
        tiles = tiles.map_batches(
            add_missing_epithelium_overlap,
            batch_format="pandas",
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

    # Rename tile coordinates to the schema mlkit's OpenSlideTilesDataset expects.
    # The ratiopath overlap helpers (tile_overlay_overlap) require "tile_x"/"tile_y", so those
    # names are kept through the whole pipeline above; mlkit's OpenSlideTilesDataset.__getitem__
    # reads tile["x"]/tile["y"], so we rename only now, right before writing.
    tiles = tiles.rename_columns({"tile_x": "x", "tile_y": "y"})

    # Convert Ray Datasets to pandas DataFrames
    slides_df = slides.to_pandas()
    tiles_df = tiles.to_pandas()

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.data_name,
    )

    print(
        f"\n[INFO] Generated {len(slides_df)} slide records and {len(tiles_df)} tile records\n"
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
