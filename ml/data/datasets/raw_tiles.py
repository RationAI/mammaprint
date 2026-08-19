"""Raw-tile slide dataset for the end-to-end (image backbone) path.

Each item is one slide's bag of raw tile **images** ``(N, C, H, W)`` — fed to an
image encoder (e.g. ``VGG16Encoder``) that maps each tile to a feature vector,
then to an aggregator + head. Reads pixels via
:class:`ml.data.datasets.SlideDataset` (OpenSlide) from a *tiled* artifact
(``slides.parquet`` + ``tiles.parquet``), so ``rationai.mlkit``/OpenSlide is
imported lazily and only touched on this path.

Single-scale only (one level card, keyed by ``raw_uris[split]``). Slides can have
thousands of tiles, so ``max_tiles`` caps the bag (random-sampled, seeded) to bound
GPU memory — standard MIL practice.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data.datasets._sources import load_labeled_slides, split_uri
from ml.data.datasets.labels import LabelMode, get_label
from ml.typing import LevelSpec, MILSample, SlideMetadata


if TYPE_CHECKING:
    from ml.data.datasets.tiles import TileDataset


logger = logging.getLogger(__name__)

# ImageNet normalisation, matching preprocessing/embeddings.py.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RawTileSlideDataset(Dataset[MILSample]):
    """Per-slide bags of raw tile images from a tiled artifact.

    Args:
        split: Which split to load (``train``/``val``/``test``); selects the level
            card's ``raw_uris[split]`` (a pure, pre-split artifact).
        levels: A one-entry level map (single-scale). The entry's ``raw_uris`` maps
            ``split -> tiled artifact URI``.
        data_mapping: Label CSV path (type/index labels, joined by slide stem).
        label_mode: ``"type"`` (classification) or ``"index"`` (regression).
        max_tiles: Cap on tiles per slide; if a slide has more, a seeded random
            subset is used. ``None`` keeps all tiles.
        seed: Seed for the per-slide tile subsampling.
    """

    def __init__(
        self,
        split: str,
        levels: Mapping[str | int, LevelSpec],
        data_mapping: str | Path,
        label_mode: str = "type",
        max_tiles: int | None = 256,
        seed: int = 0,
    ) -> None:
        if len(levels) != 1:
            raise ValueError(
                "RawTileSlideDataset is single-scale: pass exactly one level."
            )

        from mlflow.artifacts import download_artifacts

        from ml.data.datasets import SlideDataset  # lazy: pulls OpenSlide

        self.label_mode = LabelMode(label_mode)
        self.max_tiles = max_tiles
        self.rng = np.random.default_rng(seed)

        (card,) = levels.values()
        uri = split_uri(dict(card), "raw_uris", split, 0)
        local_dir = Path(download_artifacts(artifact_uri=uri))

        transform = _normalize_transform()
        slide_dataset = SlideDataset(paths=[local_dir], transforms=transform)
        # One TileDataset per slide, keyed by slide stem (matches data_mapping.name).
        # generate_datasets() yields TileDataset instances (each has .slide_path).
        per_slide = cast("list[TileDataset]", list(slide_dataset.generate_datasets()))
        self.tile_datasets: dict[str, TileDataset] = {
            Path(td.slide_path).stem: td for td in per_slide
        }

        slides = load_labeled_slides(data_mapping, self.label_mode)
        self.slides = slides[slides["name"].isin(self.tile_datasets)].reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> MILSample:
        name = self.slides.iloc[idx]["name"]
        tiles = self.tile_datasets[name]

        n = len(tiles)
        indices: list[int] = list(range(n))
        if self.max_tiles is not None and n > self.max_tiles:
            indices = self.rng.choice(n, self.max_tiles, replace=False).tolist()
            logger.debug("Slide %s: sampled %d/%d tiles.", name, self.max_tiles, n)

        images = torch.stack([tiles[i][0] for i in indices])  # (N, C, H, W)
        label = get_label(self.slides.iloc[idx], self.label_mode)
        metadata: SlideMetadata = {"slide_id": name}
        return images, label, metadata


def _normalize_transform() -> Any:
    import albumentations as A

    return A.Compose([A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)])


__all__ = ["RawTileSlideDataset"]
