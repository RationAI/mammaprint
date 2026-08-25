"""Datasets for MammaPrint.

``SlideDataset``/``TileDataset`` are imported lazily: they pull in OpenSlide via
``rationai.mlkit``, which the embeddings/alignment path does not need. Accessing
them (``from ml.data.datasets import SlideDataset``) triggers the import on demand.
"""

from typing import TYPE_CHECKING, Any

from ml.data.datasets.pyramid import PyramidScaleDataset
from ml.typing import OverlapPolicy, PyramidSelectionAudit


if TYPE_CHECKING:
    from ml.data.datasets.tiles import SlideDataset, TileDataset


__all__ = [
    "OverlapPolicy",
    "PyramidScaleDataset",
    "PyramidSelectionAudit",
    "SlideDataset",
    "TileDataset",
]


def __getattr__(name: str) -> Any:
    if name in {"SlideDataset", "TileDataset"}:
        from ml.data.datasets import tiles

        return getattr(tiles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
