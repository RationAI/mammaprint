"""Coordinate-aware single-scale slide dataset.

Identical to :class:`~ml.data.datasets.single_scale.SingleScaleDataset` except each
bag is widened from ``(N, D)`` to ``(N, D + 2)`` by appending the tile's level-0
pixel ``(x, y)`` coordinates (the top-left corner) as the last two columns. Those
columns are already present in every embedding parquet (written by
``preprocessing/embeddings.py`` alongside the ``embedding`` column); this dataset
simply stops discarding them.

The wide bag is consumed only by
:class:`~ml.models.aggregators.spatial_transformer.SpatialTransformerMIL`, which
slices the coordinate columns back off inside its ``forward``. Pair the two together
(coord-aware dataset + spatial aggregator); no other aggregator understands the extra
columns, and no other dataset produces them, so the flat single-scale path is
unaffected.
"""

import numpy as np
import pandas as pd
import torch

from ml.data.datasets.labels import get_label
from ml.data.datasets.single_scale import SingleScaleDataset
from ml.typing import MILSample, SlideMetadata


class SpatialScaleDataset(SingleScaleDataset):
    """Per-slide bags of embeddings with appended ``(x, y)`` tile coordinates.

    Returns a standard :data:`~ml.typing.MILSample`; the only difference from the
    parent is the bag width — ``(N, D + 2)`` instead of ``(N, D)``, with the last two
    columns the level-0 pixel coordinates. Construction (split resolution, artifact
    download, slide filtering, label mode) is inherited unchanged.
    """

    def __getitem__(self, idx: int) -> MILSample:
        row = self.slides.iloc[idx]
        frame = pd.read_parquet(
            (self.embeddings_dir / row["name"]).with_suffix(".parquet")
        )
        features = torch.from_numpy(
            np.stack(frame["embedding"].to_numpy())
        ).float()  # (N, D)
        coords = torch.from_numpy(
            frame[["x", "y"]].to_numpy()
        ).float()  # (N, 2), level-0 pixel top-left corners
        bag = torch.cat([features, coords], dim=1)  # (N, D + 2)

        label = get_label(row, self.label_mode)
        metadata: SlideMetadata = {"slide_id": row["name"]}
        return bag, label, metadata


__all__ = ["SpatialScaleDataset"]
