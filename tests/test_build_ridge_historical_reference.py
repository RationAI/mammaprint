from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ml import build_ridge_historical_reference as reference


def test_checkpoint_metadata_must_match_best_validation_epoch():
    history = {
        reference.MONITOR: [
            SimpleNamespace(step=1427, value=0.54),
            SimpleNamespace(step=58547, value=0.0538),
        ],
        reference.MAE: [SimpleNamespace(step=58547, value=0.1864)],
        reference.F1: [
            SimpleNamespace(step=58547, value=0.77),
            SimpleNamespace(step=60299, value=0.80),
        ],
    }
    checkpoint = {
        "global_step": 58548,
        "callbacks": {
            "checkpoints": {"monitor": "valid/loss", "best_model_score": 0.0538}
        },
    }
    assert reference.select_metric_step(checkpoint, history) == {
        "logged_step": 58547,
        "validation_mse": 0.0538,
        "mae": 0.1864,
        "f1": 0.77,
    }
    with pytest.raises(ValueError, match="does not match"):
        reference.select_metric_step({**checkpoint, "global_step": 1428}, history)


def test_historical_splits_join_original_sizes_and_exclude_missing_bags(monkeypatch):
    sizes = {"train": {"a": 10, "b": 11}, "val": {"c": 12}}
    reads = []

    def listing(*, artifact_uri):
        reads.append(artifact_uri)
        return [
            SimpleNamespace(
                path=f"archive/{slide}.parquet", is_dir=False, file_size=size
            )
            for slide, size in sizes[artifact_uri].items()
        ]

    monkeypatch.setattr(reference, "list_artifacts", listing)
    inventory = {
        slide: {"name": f"{slide}.parquet", "bytes": size}
        for entries in sizes.values()
        for slide, size in entries.items()
    }
    mapping = {
        "a": ("train", "t", 0.3),
        "b": ("train", "t2", -0.5),
        "c": ("val", "v", 0.0),
        "d": ("test", "held-out", None),
    }
    uris = {"train": "train", "val": "val", "test": "test"}
    counts = {"a": 5, "c": 7}
    coverage = {"a": ("ok", 5), "b": ("empty", 0), "c": ("ok", 7)}
    assert reference.split_coverage(uris, inventory, mapping, counts, coverage) == {
        "train": ["a"],
        "val": ["c"],
    }
    assert reads == ["train", "val"]
    with pytest.raises(ValueError, match="lacks verified cache coverage"):
        reference.split_coverage(
            uris, inventory, mapping, counts, {**coverage, "b": ("unreadable", None)}
        )
    sizes["train"]["a"] = 20
    with pytest.raises(ValueError, match="differs from source inventory"):
        reference.split_coverage(uris, inventory, mapping, counts, coverage)


def test_regression_scores_include_zero_in_positive_class():
    score = reference.reference_scores(
        np.array([0.0, -0.2, 0.2]), np.array([0.0, -0.1, -0.2])
    )
    assert score == pytest.approx(
        {"mae": 0.5 / 3, "validation_mse": 0.17 / 3, "f1": 2 / 3}
    )
