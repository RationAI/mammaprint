"""Single-scale slide dataset over precomputed tile embeddings.

Each item is one slide's flat bag of tile-feature vectors ``(N, D)`` — consumed
directly by a flat aggregator (mean/max/attention) + head. One pyramid level only.

Self-contained by design: this module imports **no** multi-scale primitives
(``align_regions``/``Region``/``MultiScaleBag``), so the single-level path keeps
working even if every multilevel file is removed or commented out.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.data.datasets._sources import (
    available_slides,
    download_level_sources,
    load_labeled_slides,
    split_uri,
)
from ml.data.datasets.labels import LabelMode, get_label
from ml.typing import LevelSpec, MILSample, SlideMetadata


logger = logging.getLogger(__name__)


class SingleScaleDataset(Dataset[MILSample]):
    """Per-slide flat bags of precomputed tile embeddings (one level).

    Args:
        split: Which split to load (``train``/``val``/``test``); selects the level
            card's ``uris[split]`` (a pure, pre-split artifact).
        levels: A one-entry level map. The entry's ``uris`` maps ``split -> URI``.
        data_mapping: Label CSV path (type/index labels, joined by slide stem).
        label_mode: ``"type"`` (classification) or ``"index"`` (regression).
    """

    def __init__(
        self,
        split: str,
        levels: Mapping[str | int, LevelSpec],
        data_mapping: str | Path,
        label_mode: str = "type",
    ) -> None:
        if len(levels) != 1:
            raise ValueError(
                "SingleScaleDataset is single-scale: pass exactly one level."
            )

        self.label_mode = LabelMode(label_mode)
        (level, card) = next(iter(levels.items()))
        self.level = int(level)
        try:
            self.mpp = float(card["mpp"])
            self.tile_extent = int(card["tile_extent"])
            self.stride = int(card["stride"])
        except KeyError as error:
            raise KeyError(
                f"Level {self.level} must define mpp, tile_extent, and stride "
                "for spatial prediction heatmaps."
            ) from error

        uri = split_uri(dict(card), "uris", split, self.level)
        self.embeddings_dir = download_level_sources({self.level: uri})[self.level]

        slides = load_labeled_slides(data_mapping, self.label_mode)
        present = available_slides({self.level: self.embeddings_dir})
        self.slides = slides[slides["name"].isin(present)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> MILSample:
        row = self.slides.iloc[idx]
        frame = pd.read_parquet(
            (self.embeddings_dir / row["name"]).with_suffix(".parquet")
        )
        bag = torch.from_numpy(np.stack(frame["embedding"].to_numpy())).float()
        x = torch.from_numpy(frame["x"].to_numpy(dtype=np.int64, copy=True))
        y = torch.from_numpy(frame["y"].to_numpy(dtype=np.int64, copy=True))
        if len(x) != len(bag) or len(y) != len(bag):
            raise ValueError(
                f"Slide {row['name']!r}: coordinate and embedding counts differ."
            )
        label = get_label(row, self.label_mode)
        metadata: SlideMetadata = {
            "slide_id": str(row["name"]),
            "record_num": str(row["record_num"]),
            "slide_path": Path(row["path"]),
            "level": self.level,
            "mpp": self.mpp,
            "tile_extent": self.tile_extent,
            "stride": self.stride,
            "x": x,
            "y": y,
        }
        return bag, label, metadata


__all__ = ["SingleScaleDataset"]
