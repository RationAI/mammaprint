import csv
import json
import subprocess
import sys

import numpy as np
import yaml

from preprocessing.slide_statistics import file_digest


def test_paired_cli_uses_same_folds_and_reproduces_resumed_outputs(tmp_path):
    size = 100
    rng = np.random.default_rng(42)
    x = rng.normal(size=(size, 5))
    y = np.tile([-1.0, 1.0], size // 2) + x[:, 0] * 0.1
    other = x.copy()
    other[0] = np.nan
    source = "l2/cancer_mask_5"
    bundle = tmp_path / "development.npz"
    np.savez(
        bundle,
        ids=[f"s{i:03}" for i in range(size)],
        splits=["train"] * 80 + ["val"] * 20,
        records=[f"r{i:03}" for i in range(size)],
        targets=y,
        primary=x,
        raw=x,
        spread=other,
        metadata=json.dumps(
            {
                "cache_digest": "cache",
                "file_sha256": {source: "stats"},
                "blocks": {
                    f"{source}:mean_ln": "primary",
                    f"{source}:mean_raw": "raw",
                    f"{source}:std_ln": "spread",
                },
                "unavailable": ["l1/cancer_mask_5:mean_ln"],
            }
        ),
    )
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    sanity = tmp_path / "sanity.json"
    sanity.write_text(
        json.dumps(
            {
                "reference_sha256": file_digest(reference),
                "cache_digest": "cache",
                "development_stats_sha256": "stats",
                "ridge": {"f1": 0.77, "mae": 0.19},
                "historical_adam": {"f1": 0.77, "mae": 0.19},
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "outer_seeds": [0],
                "alphas": [1.0, 100.0],
                "features": {
                    "l1_cm5_mean": [["l1/cancer_mask_5", "mean_ln"]],
                    "l2_cm5_mean": [[source, "mean_ln"]],
                    "l2_cm5_mean_std": [[source, "mean_ln"], [source, "std_ln"]],
                },
                "screens": {
                    "levels": ["l1_cm5_mean", "l2_cm5_mean"],
                    "statistics": ["l2_cm5_mean_std"],
                    "masks": [],
                    "multilevel": ["ignored"],
                },
            }
        )
    )
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "scripts.ml.run_ridge_adam_comparison",
        "--config",
        str(config),
        "--bundle",
        str(bundle),
        "--reference",
        str(reference),
        "--sanity",
        str(sanity),
        "--output",
        str(output),
        "--workers",
        "1",
        "--steps",
        "60",
    ]
    subprocess.run(command, check=True, capture_output=True)
    run = json.loads((output / "run.json").read_text())
    assert run["slides"] == 99
    assert run["unavailable"] == ["l1_cm5_mean"]
    with (output / "fold_metrics.csv").open() as stream:
        folds = list(csv.DictReader(stream))
    assert {row["optimizer"] for row in folds} == {"ridge", "adam"}
    full = [row for row in folds if row["fraction"] == "1.0"]
    assert len(full) == 30
    for fold in range(5):
        rows = [row for row in full if int(row["fold"]) == fold]
        assert len({row["assessment_digest"] for row in rows}) == 1
        for arm in run["available"]:
            arms = [row for row in rows if row["feature_set"] == arm]
            assert len(arms) == 2 and len({row["alpha"] for row in arms}) == 1

    paths = ["fold_metrics.csv", "fold_ledger.csv", "results.json"]
    clean = {path: file_digest(output / path) for path in paths}
    (output / ".folds/l2_cm5_mean/s0-f0-0.25.json").unlink()
    subprocess.run(command, check=True, capture_output=True)
    assert {path: file_digest(output / path) for path in paths} == clean

    checkpoint = output / ".folds/l2_cm5_mean/s0-f0-0.25.json"
    swapped = checkpoint.with_name("s0-f1-0.25.json")
    checkpoint.write_bytes(swapped.read_bytes())
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode and "different run or fold" in rejected.stderr
