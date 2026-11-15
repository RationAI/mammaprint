"""Keep exact training split membership and parquet row/coordinate alignment."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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


def _split_audit_roster(source: SourceRun, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover usable membership without fetching any uncached embedding bags."""
    if split in source.datasets:
        dataset = source.datasets[split]
        slides = dataset.slides
        evidence = {
            "membership_source": "cached_dataset",
            "empty_bags": "checked by cached dataset"
            if hasattr(dataset, "empty_slides")
            else "unverified: cached roster has no empty-bag evidence",
        }
    else:
        from mlflow.artifacts import list_artifacts

        from ml.data.datasets._sources import load_labeled_slides, split_uri
        from ml.data.datasets.labels import LabelMode

        config = source.config.datamodule[split]
        if config.get("_target_") not in {
            "ml.data.SingleScaleDataset",
            "ml.data.datasets.single_scale.SingleScaleDataset",
            "ml.data.SpatialScaleDataset",
            "ml.data.datasets.spatial_scale.SpatialScaleDataset",
        }:
            raise ValueError("Split audit requires a known precomputed embedding dataset.")
        memberships = []
        for level, card in config.levels.items():
            uri = split_uri(dict(card), "uris", str(config.split), int(level))
            # list_artifacts lists immediate children only. Never recurse, download,
            # or inspect Parquet metadata: even zero-byte listed bags remain here.
            memberships.append({
                PurePosixPath(item.path).stem
                for item in list_artifacts(artifact_uri=uri)
                if not item.is_dir and PurePosixPath(item.path).suffix == ".parquet"
            })
        present = set.intersection(*memberships) if memberships else set()
        labeled = load_labeled_slides(config.data_mapping, LabelMode(config.get("label_mode", "type")))
        slides = labeled[labeled["name"].isin(present)].reset_index(drop=True)
        evidence = {
            "membership_source": "artifact_listing",
            "empty_bags": "unverified: listed bags retained conservatively",
        }
    if slides.empty:
        raise ValueError(f"Empty usable {split} embedding corpus.")
    if slides["name"].duplicated().any():
        raise ValueError("Duplicate slide identifiers in source dataset.")
    return slides, {**evidence, "n_slides": len(slides)}


def audit_splits(source: SourceRun) -> dict[str, Any]:
    """Reject slide/patient leakage using cached rosters or artifact metadata."""
    rosters = {
        split: _split_audit_roster(source, split) for split in ("train", "val", "test")
    }
    groups = {split: set(slides["name"]) for split, (slides, _) in rosters.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Slide leakage between {left} and {right}.")
    common = set.intersection(
        *(set(slides.columns) for slides, _ in rosters.values())
    )
    patient_key = next(
        (key for key in ("patient_id", "case_id", "record_num") if key in common), None
    )
    if patient_key:
        patients = {
            split: set(slides[patient_key].dropna().astype(str))
            for split, (slides, _) in rosters.items()
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
        "splits": {split: evidence for split, (_, evidence) in rosters.items()},
        "limitation": "Artifact listings and the current label mapping do not establish historical mapping bytes.",
    }
