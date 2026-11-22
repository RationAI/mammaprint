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
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, get_worker_info

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
        train_tile_cap: Maximum tiles per training bag. Evaluation uses all tiles.
        tile_seed: Seed for the training tile sampler.
        validate_index_split: Verify the eligible index labels and mapped split.
    """

    def __init__(
        self,
        split: str,
        levels: Mapping[str | int, LevelSpec],
        data_mapping: str | Path,
        label_mode: str = "type",
        train_tile_cap: int | None = None,
        tile_seed: int = 0,
        validate_index_split: bool = False,
    ) -> None:
        if len(levels) != 1:
            raise ValueError(
                "SingleScaleDataset is single-scale: pass exactly one level."
            )

        self.label_mode = LabelMode(label_mode)
        if train_tile_cap is not None and train_tile_cap < 1:
            raise ValueError("train_tile_cap must be positive.")
        self.train_tile_cap = train_tile_cap if split == "train" else None
        self.tile_seed = tile_seed
        self._tile_rng: np.random.Generator | None = None
        self._tile_worker_seed: int | None = None
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
        if validate_index_split:
            self.regression_targets()
            if "split" not in self.slides:
                raise ValueError(
                    "Index split validation requires a mapping 'split' column."
                )
            unexpected = self.slides.loc[self.slides["split"].ne(split), "name"]
            if not unexpected.empty:
                raise ValueError(
                    f"The '{split}' artifact contains slides outside its mapped split: "
                    f"{', '.join(unexpected.head(5))}."
                )

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> MILSample:
        return self.load_sample(idx)

    def regression_targets(self) -> torch.Tensor:
        """Read eligible scalar targets from metadata without loading tile bags."""
        if self.label_mode is not LabelMode.INDEX:
            raise ValueError("Expected scalar index regression labels.")
        if self.slides["name"].duplicated().any():
            raise ValueError("The split contains duplicate slide IDs.")
        targets = torch.tensor(
            self.slides["mammaprint_index"].to_numpy(), dtype=torch.float32
        )
        if targets.numel() == 0 or not torch.isfinite(targets).all():
            raise ValueError("Expected nonempty, finite regression targets.")
        return targets

    def tile_sampler_state(self) -> dict[str, Any] | None:
        if self._tile_rng is None:
            return None
        return self._tile_rng.bit_generator.state

    def restore_tile_sampler(self, state: dict[str, Any] | None) -> None:
        self._tile_rng = None
        self._tile_worker_seed = None
        if state is not None:
            self._tile_rng = np.random.default_rng()
            self._tile_rng.bit_generator.state = state

    def load_sample(
        self, idx: int, rng: np.random.Generator | None = None
    ) -> MILSample:
        row = self.slides.iloc[idx]
        frame = pd.read_parquet(
            self.embeddings_dir / f"{row['name']}.parquet", columns=["embedding"]
        )
        if self.train_tile_cap is not None and len(frame) > self.train_tile_cap:
            if rng is None:
                worker = get_worker_info()
                worker_seed = worker.seed if worker is not None else None
                if self._tile_rng is None or worker_seed != self._tile_worker_seed:
                    seed = (
                        self.tile_seed
                        if worker_seed is None
                        else [self.tile_seed, worker_seed]
                    )
                    self._tile_rng = np.random.default_rng(seed)
                    self._tile_worker_seed = worker_seed
                rng = self._tile_rng
            indices = rng.choice(len(frame), size=self.train_tile_cap, replace=False)
            frame = frame.iloc[indices]
        bag = torch.from_numpy(np.stack(frame["embedding"].to_numpy())).float()
        label = get_label(row, self.label_mode)
        metadata: SlideMetadata = {"slide_id": row["name"]}
        return bag, label, metadata


__all__ = ["SingleScaleDataset"]
