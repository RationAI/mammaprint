"""Keep exact training split membership and parquet row/coordinate alignment."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import instantiate

from ml.concept_explainability.model import SourceRun
from ml.data.datasets.single_scale import SingleScaleDataset


@dataclass
class SlideInput:
    slide_id: str
    embeddings: np.ndarray
    keys: pd.DataFrame
    label: np.ndarray
    path: Path
    level: int
    card: dict[str, Any]


def source_dataset(source: SourceRun, split: str) -> SingleScaleDataset:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if split in source.datasets:
        return source.datasets[split]
    dataset = instantiate(source.config.datamodule[split])
    if not isinstance(dataset, SingleScaleDataset):
        raise TypeError(
            "Concepts require precomputed single-scale Virchow2 embeddings."
        )
    if dataset.slides.empty:
        raise ValueError(f"Empty usable {split} embedding corpus.")
    if dataset.slides["name"].duplicated().any():
        raise ValueError("Duplicate slide identifiers in source dataset.")
    source.datasets[split] = dataset
    return dataset


def corpus_identity(
    source: SourceRun, dataset: SingleScaleDataset | None = None
) -> dict[str, Any]:
    dataset = dataset if dataset is not None else source_dataset(source, "train")
    level, card = next(iter(source.config.dataset.levels.items()))
    body = {
        "level": int(level),
        "train_uri": str(card.uris.train),
        "slides": sorted(dataset.slides["name"].astype(str)),
        "feature_dim": int(source.config.get("feature_dim", 2560)),
        "mpp": float(card.mpp),
        "tile_extent": int(card.tile_extent),
        "stride": int(card.get("stride", card.tile_extent)),
    }
    body["id"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def iter_slides(
    source: SourceRun,
    split: str,
    dataset: SingleScaleDataset | None = None,
) -> Iterator[SlideInput]:
    from ml.data.datasets.labels import get_label

    dataset = dataset if dataset is not None else source_dataset(source, split)
    level_key, card = next(iter(source.config.dataset.levels.items()))
    for _, row in dataset.slides.sort_values("name").iterrows():
        slide_id = str(row["name"])
        path = dataset.embeddings_dir / f"{slide_id}.parquet"
        frame = pd.read_parquet(path)
        if frame.empty or not {"embedding", "x", "y"}.issubset(frame.columns):
            raise ValueError(f"{path}: nonempty embedding, x and y columns required.")
        embeddings = np.stack(frame["embedding"]).astype(np.float32, copy=False)
        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != 2560
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"{path}: expected finite (N, 2560) Virchow2 embeddings.")
        coords = frame[["x", "y"]].to_numpy()
        if (
            not np.isfinite(coords).all()
            or np.any(coords < 0)
            or not np.equal(coords, np.floor(coords)).all()
        ):
            raise ValueError(f"{path}: invalid tiling-level tile coordinates.")
        keys = frame[["x", "y"]].astype(np.int64).reset_index(drop=True)
        keys.insert(0, "row_index", np.arange(len(frame)))
        keys.insert(0, "slide_id", slide_id)
        yield SlideInput(
            slide_id,
            embeddings,
            keys,
            get_label(row, dataset.label_mode).numpy(),
            Path(row["path"]),
            int(level_key),
            dict(card),
        )


def embedding_dataset(
    source: SourceRun, split: str, dataset: SingleScaleDataset | None = None, *, include_keys: bool = False
):
    import ray.data

    dataset = dataset if dataset is not None else source_dataset(source, split)
    paths = [
        str(dataset.embeddings_dir / f"{name}.parquet")
        for name in sorted(dataset.slides["name"])
    ]
    # Configure ordering before any Ray reads, including the metadata sampling pass.
    ray.data.DataContext.get_current().execution_options.preserve_order = True
    if not include_keys:
        return ray.data.read_parquet(paths, columns=["embedding"])
    return ray.data.read_parquet(
        paths, columns=["embedding", "x", "y"], include_paths=True,
    ).map_batches(_add_slide_identity, batch_format="pandas")


def _add_slide_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Ray path provenance stays attached to each embedding and its coordinates."""
    frame = frame.copy()
    frame["slide_id"] = frame["path"].map(lambda value: Path(value).stem)
    frame["x"] = frame["x"].astype(np.int64)
    frame["y"] = frame["y"].astype(np.int64)
    return frame.drop(columns="path")


def audit_splits(source: SourceRun) -> dict[str, Any]:
    """Reject overlapping slides and patients when a stable patient key exists."""
    datasets = {
        split: source_dataset(source, split) for split in ("train", "val", "test")
    }
    groups = {split: set(dataset.slides["name"]) for split, dataset in datasets.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Slide leakage between {left} and {right}.")
    common = set.intersection(
        *(set(dataset.slides.columns) for dataset in datasets.values())
    )
    patient_key = next(
        (key for key in ("patient_id", "case_id", "record_num") if key in common), None
    )
    if patient_key:
        patients = {
            split: set(dataset.slides[patient_key].dropna().astype(str))
            for split, dataset in datasets.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if patients[left] & patients[right]:
                raise ValueError(
                    f"Patient/case leakage ({patient_key}) between {left} and {right}."
                )
    return {
        "slide_disjoint": True,
        "patient_key": patient_key,
        "patient_audit": "checked"
        if patient_key
        else "unavailable: no stable patient key",
    }
