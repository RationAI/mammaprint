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
import pyarrow.parquet as pq
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


def _empty_embedding_slides(directory: Path, slide_ids: set[str]) -> tuple[str, ...]:
    """Return slide IDs whose Parquet files contain no tile rows."""
    return tuple(
        sorted(
            slide_id
            for slide_id in slide_ids
            if pq.read_metadata(directory / f"{slide_id}.parquet").num_rows == 0
        )
    )


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

        uri = split_uri(dict(card), "uris", split, self.level)
        self.embeddings_dir = download_level_sources({self.level: uri})[self.level]

        slides = load_labeled_slides(data_mapping, self.label_mode)
        present = available_slides({self.level: self.embeddings_dir})
        self.empty_slides = _empty_embedding_slides(self.embeddings_dir, present)
        if self.empty_slides:
            logger.warning(
                "Excluding %d slide(s) with empty embedding bags from the %s "
                "split at level %d: %s",
                len(self.empty_slides),
                split,
                self.level,
                ", ".join(self.empty_slides),
            )
            present.difference_update(self.empty_slides)
        self.slides = slides[slides["name"].isin(present)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> MILSample:
        row = self.slides.iloc[idx]
        frame = pd.read_parquet(
            (self.embeddings_dir / row["name"]).with_suffix(".parquet")
        )
        bag = torch.from_numpy(np.stack(frame["embedding"].to_numpy())).float()
        label = get_label(row, self.label_mode)
        metadata: SlideMetadata = {"slide_id": row["name"]}
        return bag, label, metadata


__all__ = ["SingleScaleDataset"]
