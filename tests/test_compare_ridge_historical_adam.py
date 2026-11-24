import hashlib
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.ridge_screening import ridge_path
from scripts.ml import compare_ridge_historical_adam as comparison


def test_adam_matches_penalized_ridge_with_unpenalized_intercept():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(28, 12)) + 2.0
    y = rng.normal(size=len(x)) + 3.0
    alpha = 4.0
    ridge_w, ridge_b = ridge_path(x, y, (alpha,))
    adam_w, adam_b = comparison.fit_adam(x, y, alpha, steps=500, learning_rate=0.03)
    np.testing.assert_allclose(
        x @ adam_w + adam_b, x @ ridge_w[:, 0] + ridge_b[0], atol=2e-6
    )


def test_export_reads_only_development_labels(tmp_path, monkeypatch):
    x = np.arange(3 * comparison.WIDTH, dtype=np.float32)
    filename = f"stats/dev/{comparison.SOURCE}.parquet"
    stats = tmp_path / filename
    stats.parent.mkdir(parents=True)
    table = pa.table(
        {
            "slide_id": ["train", "val", "test"],
            "mean_raw": pa.FixedSizeListArray.from_arrays(x, comparison.WIDTH),
        }
    )
    pq.write_table(table, stats)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "path,split,record_num,mammaprint_index\n"
        "train.svs,train,record1,-0.5\n"
        "val.svs,val,record2,0.5\n"
        "test.svs,test,record3,0.9\n"
    )
    manifest = {
        "pilot_slides": None,
        "mapping_sha256": comparison.file_sha(mapping),
        "sources": {
            comparison.SOURCE: {
                "uri": "example",
                "files": {filename: comparison.file_sha(stats)},
            }
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["digest"] = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    output = io.BytesIO()

    class Stdout:
        buffer = output

    monkeypatch.setattr(comparison.sys, "stdout", Stdout())
    comparison.export_development(tmp_path, mapping)
    output.seek(0)
    with np.load(output, allow_pickle=False) as data:
        assert data["ids"].tolist() == ["train", "val"]
        assert data["targets"].tolist() == [-0.5, 0.5]
        assert data["mean_raw"].shape == (2, comparison.WIDTH)


def test_comparison_rejects_features_from_a_different_cache_partition(tmp_path):
    reference = tmp_path / "historical_reference.json"
    reference.write_text(
        json.dumps(
            {
                "transform": "raw",
                "runs": [{}],
                "source_uri": "example",
                "verification": {
                    "mapping_sha256": "mapping",
                    "development_stats_sha256": "verified-stats",
                },
            }
        )
    )
    bundle = tmp_path / "features.npz"
    np.savez(
        bundle,
        source_uri="example",
        mapping_sha256="mapping",
        development_stats_sha256="different-stats",
    )
    with pytest.raises(ValueError, match="does not match historical development data"):
        comparison.compare(reference, bundle, steps=1, learning_rate=0.01)
