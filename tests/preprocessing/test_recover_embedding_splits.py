import pytest

from preprocessing.recover_embedding_splits import (
    ArtifactFile,
    _build_recovery_plan,
)


def _file(path: str, size: int = 10) -> ArtifactFile:
    return ArtifactFile(path=path, size=size)


def test_recovery_plan_reuses_valid_files_and_finds_missing_files():
    source = {
        "train.parquet": _file("embeddings/train.parquet", 10),
        "val.parquet": _file("embeddings/val.parquet", 20),
        "test.parquet": _file("embeddings/test.parquet", 30),
    }
    destinations = {
        "train": {
            "train.parquet": _file("embeddings_train/train.parquet", 10),
        },
        "val": {},
        "test": {},
    }

    plan = _build_recovery_plan(
        source,
        destinations,
        {"train": "train", "val": "val", "test": "test"},
    )

    assert plan.missing_files == 2
    assert plan.missing_bytes == 50
    assert set(plan.missing["val"]) == {"val.parquet"}
    assert set(plan.missing["test"]) == {"test.parquet"}


def test_recovery_plan_rejects_size_mismatch():
    source = {"slide.parquet": _file("embeddings/slide.parquet", 10)}
    destinations = {
        "train": {
            "slide.parquet": _file("embeddings_train/slide.parquet", 9),
        },
        "val": {},
        "test": {},
    }

    with pytest.raises(ValueError, match="size-mismatched"):
        _build_recovery_plan(source, destinations, {"slide": "train"})
