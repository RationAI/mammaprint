"""Data utilities for MammaPrint."""

from ml.data.datamodule import DataModule, mil_collate

# ── Multilevel (comment out this line + the __all__ entry to ship single-level) ──
# from ml.data.datasets.pyramid import PyramidSlideDataset
from ml.data.datasets.raw_tiles import RawTileSlideDataset
from ml.data.datasets.single_scale import SingleScaleDataset


# Grouped (single-level first, multilevel last) to make the toggle obvious, so this
# is deliberately not alphabetically sorted.
__all__ = [  # noqa: RUF022
    "DataModule",
    "RawTileSlideDataset",
    "SingleScaleDataset",
    "mil_collate",
    # ── Multilevel ──
    # "PyramidSlideDataset",
]
