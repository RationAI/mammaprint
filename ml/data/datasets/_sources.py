"""Shared setup helpers for the slide datasets.

:class:`~ml.data.datasets.pyramid.PyramidSlideDataset` needs a few setup steps:
load + label the slide table, download the per-level artifacts, and find which
slides are actually present across levels. Extracting them here keeps the dataset
free of scaffolding and leaves them independently unit-testable.

The train/val/test split is NOT applied here — splitting is materialised into
physically separate artifacts upstream by ``scripts/preprocessing/split_dataset.py``
(the ``split`` column in ``data_mapping.csv`` is the sole split authority). A card's
per-split URI already points at a pure artifact, so no split filter is needed.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from mlflow.artifacts import download_artifacts

from ml.data.datasets.labels import LabelMode, get_target_column, process_slides


logger = logging.getLogger(__name__)


def load_labeled_slides(
    data_mapping: str | Path,
    label_mode: LabelMode,
) -> pd.DataFrame:
    """Load the slide table and drop rows without a label.

    Args:
        data_mapping: Path to ``data_mapping.csv`` (``record_num``/``type``/
            ``mammaprint_index``/``path``).
        label_mode: Classification (``type``) or regression (``index``) target.

    Returns:
        A dataframe with a ``name`` (slide stem) column and the label column, keeping
        only rows that carry a label.
    """
    slides = pd.read_csv(data_mapping)
    slides = process_slides(slides, mode=label_mode)

    target_column = get_target_column(label_mode)
    keep = slides[target_column].notna()
    return slides[keep].reset_index(drop=True)


def download_level_sources(sources: Mapping[int, str]) -> dict[int, Path]:
    """Download each pyramid level's artifact and return ``level -> local dir``."""
    return {
        level: Path(download_artifacts(artifact_uri=uri))
        for level, uri in sources.items()
    }


def available_slides(dirs: Mapping[int, Path]) -> set[str]:
    """Slide stems present as ``<stem>.parquet`` in *every* level's directory."""
    per_level = ({p.stem for p in d.glob("*.parquet")} for d in dirs.values())
    return set.intersection(*per_level) if dirs else set()


__all__ = ["available_slides", "download_level_sources", "load_labeled_slides"]
