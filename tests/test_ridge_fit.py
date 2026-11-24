from __future__ import annotations

import csv
import json
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import Ridge

from ml.ridge_experiment import signed_read


if TYPE_CHECKING:
    from pathlib import Path


def fit(*args: object, ok: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ml.run_ridge_screening",
            "fit",
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


@pytest.fixture
def external_dataset(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    source = tmp_path / "outside_embeddings"
    source.mkdir()
    mapping = tmp_path / "labels.csv"
    rng = np.random.default_rng(42)
    features, targets = [], []
    with mapping.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "record_num", "split", "response"))
        for index in range(8):
            tiles = rng.normal(size=(5, 3)).astype(np.float32)
            x = np.concatenate((tiles.mean(axis=0), tiles.std(axis=0)))
            target = 3 + 0.3 * x[0] + x[-1]
            features.append(x)
            targets.append(target)
            writer.writerow(
                (
                    f"case.{index}.svs",
                    f"record-{index // 2}",
                    "train" if index < 6 else "val",
                    target,
                )
            )
            pq.write_table(
                pa.table({"embedding": tiles.tolist()}),
                source / f"case.{index}.parquet",
            )
        writer.writerow(("held.svs", "held-out", "test", "sealed"))
    # Corrupt source tile data belongs to the holdout, where the builder needn't decode it.
    (source / "held.parquet").write_text("deliberately unreadable")
    return mapping, source, np.asarray(features, dtype=np.float64), np.asarray(targets)


def test_one_command_fits_custom_dataset_and_reuses_cache(external_dataset, tmp_path):
    mapping, source, features, targets = external_dataset
    output = tmp_path / "raw-model"
    dataset = (
        "--data-mapping",
        mapping,
        "--level",
        7,
        "--mask",
        "custom_mask",
        "--target-column",
        "response",
    )
    args = (
        *dataset,
        "--source",
        source,
        "--embedding-dim",
        3,
        "--feature",
        "raw-mean-std",
        "--alpha",
        0.1,
        "--output",
        output,
    )
    fit(*args)
    result = signed_read(output / "model.json")
    expected = Ridge(alpha=0.1, solver="svd").fit(features, targets)
    np.testing.assert_allclose(result["coefficients"], expected.coef_, atol=2e-6)
    assert result["intercept"] == pytest.approx(expected.intercept_, abs=2e-6)
    assert result["blocks"] == [
        ["l7/custom_mask", "mean_raw"],
        ["l7/custom_mask", "std_raw"],
    ]
    assert result["train_slides"] == 8
    assert result["train_records"] == 4
    assert result["target_column"] == "response"
    coverage = pq.read_table(output / "cache/coverage.parquet").to_pylist()
    assert {row["status"] for row in coverage if row["slide_id"] == "held"} == {
        "unmapped"
    }
    manifest = json.loads((output / "cache/manifest.json").read_text())
    assert set(manifest["sources"]) == {"l7/custom_mask"}
    assert (
        pq.read_metadata(output / "cache/stats/test/l7/custom_mask.parquet").num_rows
        == 0
    )

    # An identical retry is idempotent and a different alpha preserves the original model.
    saved = (output / "model.json").read_bytes()
    fit(*args)
    different = fit(*args, "--alpha", 0.2, ok=False)
    assert "A different model is already saved" in different.stderr
    assert (output / "model.json").read_bytes() == saved

    # CV also works for a continuous target with every observed value above zero.
    reused = tmp_path / "reused"
    fit(
        *dataset,
        "--cache-dir",
        output / "cache",
        "--feature",
        "ln-mean",
        "--output",
        reused,
    )
    trained = signed_read(reused / "model.json")
    assert trained["embedding_dim"] == 3
    assert trained["alpha_selection"]["method"] == "grouped_cv_mae"
    assert trained["alpha_selection"]["folds"] == 3
    assert trained["train_slides"] == 8
    assert not (reused / "cache").exists()


def test_fit_dry_run_uses_requested_card_and_skips_output(tmp_path):
    out = tmp_path / "dry"
    result = fit(
        "--data-mapping",
        tmp_path / "unmounted.csv",
        "--level",
        3,
        "--mask",
        "tissue_only",
        "--output",
        out,
        "--dry-run",
    )
    plan = json.loads(result.stdout)
    assert plan["source_key"] == "l3/tissue_only"
    assert plan["source"]["uri"].startswith("mlflow-artifacts:")
    assert not out.exists()


def test_different_cache_mapping_is_rejected(external_dataset, tmp_path):
    mapping, source, _, _ = external_dataset
    args = (
        "--data-mapping",
        mapping,
        "--level",
        7,
        "--mask",
        "custom_mask",
        "--target-column",
        "response",
        "--alpha",
        0.1,
    )
    original = tmp_path / "first"
    fit(*args, "--source", source, "--embedding-dim", 3, "--output", original)
    mapping.write_text(mapping.read_text().replace("record-0", "new-record"))
    refused = fit(
        *args,
        "--cache-dir",
        original / "cache",
        "--output",
        tmp_path / "next",
        ok=False,
    )
    assert "Data mapping does not match the cache" in refused.stderr
