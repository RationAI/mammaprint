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
import yaml
from sklearn.linear_model import Ridge

from ml.ridge_experiment import read_json, signed_read, signed_write
from ml.ridge_screening import (
    Cohort,
    InfeasibleCVError,
    Protocol,
    grouped_splits,
    nested_folds,
    paired_rows,
    ridge_path,
    score_metrics,
    summarize,
)
from preprocessing.slide_statistics import load_manifest


if TYPE_CHECKING:
    from pathlib import Path


def test_svd_path_matches_sklearn_on_wide_rank_deficient_input():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 24))
    x[:, 1] = x[:, 0]
    x[:, 3] = 4
    y = rng.normal(size=len(x))
    holdout = rng.normal(size=(5, x.shape[1]))
    alphas = (1e-6, 0.1, 100.0)
    weights, bias = ridge_path(x, y, alphas)
    for index, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha, solver="svd", fit_intercept=True).fit(x, y)
        np.testing.assert_allclose(
            holdout @ weights[:, index] + bias[index], model.predict(holdout)
        )


def test_nested_folds_keep_records_and_prefixes_together():
    ids = tuple(f"s{i:03}" for i in range(160))
    groups = np.repeat([f"r{i:03}" for i in range(80)], 2)
    y = np.repeat(np.tile([-1.0, 0.0], 40), 2)
    cohort = Cohort(ids, groups, y)
    protocol = Protocol((0, 1, 2, 3, 4), 5, 3, (0.01, 1.0))
    folds = nested_folds(cohort, protocol, learning_curve=True)
    assert len(folds) == 100
    for seed in protocol.outer_seeds:
        assert sorted(
            slide
            for fold in folds
            if fold.seed == seed and fold.fraction == 1
            for slide in fold.assess
        ) == list(range(len(y)))
        for number in range(protocol.outer_folds):
            family = [
                fold for fold in folds if fold.seed == seed and fold.number == number
            ]
            assert [fold.fraction for fold in family] == [0.25, 0.5, 0.75, 1.0]
            prev = set()
            for fold in family:
                assert prev <= set(fold.train)
                prev = set(fold.train)
                assert np.array_equal(fold.assess, family[0].assess)
                assert not (set(groups[fold.train]) & set(groups[fold.assess]))
                for fit, assess in fold.inner:
                    assert not (
                        set(groups[fold.train[fit]]) & set(groups[fold.train[assess]])
                    )
                    assert set(fold.train[fit]) | set(fold.train[assess]) == prev


def test_undefined_metrics_and_zero_boundary():
    result = score_metrics(np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5, 0.5]))
    assert result["sign_auroc"] == 1.0
    assert result["spearman"] == pytest.approx(0.8660254037844387)
    result = score_metrics(np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    assert result["spearman"] is None
    assert result["sign_auroc"] is None
    assert result["spearman_reason"] and result["sign_auroc_reason"]
    with pytest.raises(InfeasibleCVError, match="negative/nonnegative record counts"):
        grouped_splits(np.array([-1.0, 0.0, 0.0]), np.array(["a", "b", "c"]), 2, 0)


def test_paired_standard_errors_and_fold_identity():
    assert summarize([1.0, 3.0], [0.25, 0.25]) == pytest.approx((2, np.sqrt(1.5), 2))
    ref = [
        {
            "seed": 0,
            "fold": fold,
            "fit_records": 80,
            "assess_records": 20,
            "assessment_digest": str(fold),
            "mae": 3.0,
            "spearman": None,
            "sign_auroc": 0.5,
        }
        for fold in range(2)
    ]
    arms = [{**ref[0], "mae": 2.0}, {**ref[1], "mae": 3.0}]
    contrasts = paired_rows(arms, ref)
    assert contrasts[0]["delta"] == pytest.approx(-0.5)
    assert contrasts[0]["se"] == pytest.approx(np.sqrt(0.375))
    assert contrasts[1]["valid_folds"] == 0
    with pytest.raises(ValueError, match="identical outer fold"):
        paired_rows([{**arms[0], "assessment_digest": "wrong"}, arms[1]], ref)


def cli(config: Path, mode: str, output: Path, *args: str, succeeded: bool = True):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ml.run_ridge_screening",
            mode,
            "--config",
            str(config),
            "--output",
            str(output),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is succeeded, result.stdout + result.stderr
    return result


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    l3 = tmp_path / "l3"
    l3.mkdir()
    mask = tmp_path / "mask"
    mask.mkdir()
    rows = []
    rng = np.random.default_rng(12)
    means = {}
    membership = {"train": [], "val": []}
    for index in range(110):
        slide = f"s{index:03}"
        split = "train" if index < 80 else ("val" if index < 100 else "test")
        sign = 2 * (index % 2) - 1
        target = "" if index == 65 else sign * (0.5 + 0.01 * (index % 4))
        rows.append((f"{slide}.svs", f"00{index}", split, target))
        if split != "test" and target != "":
            membership[split].append(slide)
        tiles = (
            rng.normal(scale=0.1, size=(3, 4)) + np.array([sign, -sign, 0.4, -0.5])
        ).astype("float32")
        means[slide] = tiles.astype("float64").mean(axis=0).astype("float32")
        pq.write_table(
            pa.table({"embedding": tiles.tolist()}), source / f"{slide}.parquet"
        )
        if index != 2:
            (l3 / f"{slide}.parquet").hardlink_to(source / f"{slide}.parquet")
        if index != 3:
            (mask / f"{slide}.parquet").hardlink_to(source / f"{slide}.parquet")

    mapping = tmp_path / "data_mapping.csv"
    with mapping.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "record_num", "split", "mammaprint_index"))
        writer.writerows(rows)

    weights = np.array([0.3, -0.2, 0, 0])
    reference = {
        "source_uri": str(source),
        "train_uri": "original-train",
        "val_uri": "original-val",
        "transform": "raw",
        "train_slides": membership["train"],
        "val_slides": membership["val"],
        "runs": [
            {
                "run_id": "reference",
                "seed": 0,
                "code_revision": "example",
                "config": "fixture-config",
                "checkpoint": "fixture-checkpoint",
                "checkpoint_selection": "validation-mse",
                "f1": 0.95,
                "mae": 0.35,
            }
        ],
        "parity": {
            "weights": weights.tolist(),
            "intercept": 0.01,
            "predictions": {
                slide: float(means[slide] @ weights + 0.01)
                for slide in membership["val"][:4]
            },
        },
    }
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    config = {
        "data_mapping": "data_mapping.csv",
        "cache_dir": "cache",
        "sanity_dir": "sanity",
        "screen_dir": "screen",
        "lock_dir": "lock",
        "test_registry": "test-registry",
        "historical_reference": "reference.json",
        "cards_dir": None,
        "embedding_dim": 4,
        "batch_size": 2,
        "outer_seeds": [0, 1],
        "alphas": [0.01, 0.1, 1.0],
        "sources": {
            "l2/cancer_mask_5": "source",
            "l3/cancer_mask_5": "l3",
            "l2/cancer_mask_9": "mask",
        },
        "features": {
            "l2_cm5_mean": [["l2/cancer_mask_5", "mean_ln"]],
            "l1_cm5_mean": [["l1/cancer_mask_5", "mean_ln"]],
            "l3_cm5_mean": [["l3/cancer_mask_5", "mean_ln"]],
            "l2_l3_cm5_mean": [
                ["l2/cancer_mask_5", "mean_ln"],
                ["l3/cancer_mask_5", "mean_ln"],
            ],
            "l2_cm9_mean": [["l2/cancer_mask_9", "mean_ln"]],
            "l2_cm5_mean_std": [
                ["l2/cancer_mask_5", "mean_ln"],
                ["l2/cancer_mask_5", "std_ln"],
            ],
        },
        "screens": {
            "levels": ["l1_cm5_mean", "l3_cm5_mean"],
            "multilevel": ["l2_l3_cm5_mean"],
            "masks": ["l2_cm9_mean"],
            "statistics": ["l2_cm5_mean_std"],
            "learning-curve": ["l2_cm5_mean"],
        },
        "core_features": ["l2_cm5_mean", "l2_cm5_mean_std"],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_synthetic_cli_reproducibility_and_test_guard(experiment, tmp_path):
    cli(experiment, "cache", tmp_path / "cache")
    cache = tmp_path / "cache"
    manifest = load_manifest(cache)
    assert manifest["sources"]["l2/cancer_mask_5"]["status"] == "available"
    test_features = sorted((cache / "stats/test").glob("*/*.parquet"))
    assert len(test_features) == 3
    hidden = [path.rename(path.with_suffix(".hidden")) for path in test_features]

    cli(experiment, "sanity", tmp_path / "sanity")
    assert signed_read(tmp_path / "sanity/sanity.json")["passed"] is True
    reference = tmp_path / "reference.json"
    original_reference = reference.read_text()
    reference.write_text(original_reference + "\n")
    stale = cli(experiment, "screen", tmp_path / "screen-stale", succeeded=False)
    assert "matching L2 cm5 sanity check" in stale.stderr
    reference.write_text(original_reference)
    cli(experiment, "screen", tmp_path / "screen")
    cli(experiment, "screen", tmp_path / "repeat")
    for file in (
        "folds.csv",
        "metrics.csv",
        "ranking.csv",
        "summary.csv",
        "contrasts.csv",
        "run.json",
    ):
        assert (tmp_path / "screen" / file).read_bytes() == (
            tmp_path / "repeat" / file
        ).read_bytes()
    assert (tmp_path / "screen/learning_curve.png").is_file()
    assert "75% to 100%" in (tmp_path / "screen/report.md").read_text()
    assert "s065,,missing-target" in (tmp_path / "screen/exclusions.csv").read_text()
    run = signed_read(tmp_path / "screen/run.json")
    assert run["skipped"] == {"l1_cm5_mean": "source unavailable"}
    assert not run["errors"]
    with (tmp_path / "screen/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    baseline = [
        row
        for row in rows
        if row["screen"] == "ranking" and row["feature_set"] == "l2_cm5_mean"
    ]
    alternative = [
        row
        for row in rows
        if row["screen"] == "ranking" and row["feature_set"] == "l2_cm5_mean_std"
    ]
    assert len(baseline) == len(alternative) == 10
    assert (
        sum(int(row["assess_slides"]) for row in baseline if row["seed"] == "0") == 97
    )
    assert [row["assessment_digest"] for row in baseline] == [
        row["assessment_digest"] for row in alternative
    ]

    missing = cli(experiment, "test", tmp_path / "test", succeeded=False)
    assert "lock.json" in missing.stderr
    cli(experiment, "lock", tmp_path / "lock")
    for path, target in zip(hidden, test_features, strict=True):
        path.rename(target)
    cli(experiment, "test", tmp_path / "test")
    result = read_json(tmp_path / "test/test_metrics.json")
    assert result["scored_slides"] == 10
    assert result["sign_auroc"] == 1.0
    report = (tmp_path / "test/held_out_report.md").read_text()
    assert "## Development ranking" in report and "## Held-out test" in report
    assert "(../screen/learning_curve.png)" in report

    # A retry returns the saved result even when cached test features are unavailable.
    for path, target in zip(test_features, hidden, strict=True):
        path.rename(target)
    cli(experiment, "test", tmp_path / "test-repeat")
    assert result == read_json(tmp_path / "test-repeat/test_metrics.json")

    lock_path = tmp_path / "lock/lock.json"
    frozen = signed_read(lock_path)
    frozen.pop("digest")
    frozen["intercept"] += 0.01
    signed_write(lock_path, frozen)
    different = cli(experiment, "test", tmp_path / "test-other", succeeded=False)
    assert "already has a different evaluation claim" in different.stderr
