import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from preprocessing.slide_statistics import digest, file_digest, write_json
from scripts.ml import export_ridge_development as exporter


SOURCE = "l2/cancer_mask_5"
DEV = f"stats/dev/{SOURCE}.parquet"
REQUESTED = [f"{SOURCE}:mean_raw", "l1/cancer_mask_5:mean_raw", f"{SOURCE}:mean_ln"]


@pytest.fixture
def inputs(tmp_path: Path):
    mapping = tmp_path / "mapping.csv"
    with mapping.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "split", "record_num", "mammaprint_index"))
        writer.writerows(
            [
                ("z.svs", "train", "0001", -0.5),
                ("unlabeled.svs", "val", "not-labeled", "nan"),
                ("a.svs", "val", "0002", 0.6),
                ("b.svs", "train", "0002", 0.3),
                ("held-out.svs", "test", "test-hidden", "private"),
            ]
        )
    cache = tmp_path / "cache"
    path = cache / DEV
    path.parent.mkdir(parents=True)
    heldout = cache / DEV.replace("/dev/", "/test/")
    heldout.parent.mkdir(parents=True)
    heldout.write_bytes(b"intentionally inaccessible as Parquet")
    vectors = pa.list_(pa.float32(), 2)
    pq.write_table(
        pa.table(
            {
                "slide_id": ["z", "unlabeled", "a", "held-out"],
                "mean_raw": pa.array(
                    [[3, 4], [np.nan, np.nan], [1, 2], [np.nan, np.nan]], type=vectors
                ),
                "mean_ln": pa.array(
                    [[0.25, -0.25], [0, 0], [1, -1], [0, 0]], type=vectors
                ),
                "std_raw": pa.array([[2, 2]] * 4, type=vectors),
            }
        ),
        path,
    )
    manifest = {
        "schema_version": 1,
        "mapping_sha256": file_digest(mapping),
        "settings": {"embedding_dim": 2},
        "pilot_slides": None,
        "sources": {
            SOURCE: {
                "status": "available",
                "uri": "mlflow-artifacts:/example/l2",
                "files": {
                    DEV: file_digest(path),
                    DEV.replace("/dev/", "/test/"): "must-not-be-read",
                },
            }
        },
    }
    manifest = manifest | {"digest": digest(manifest)}
    write_json(cache / "manifest.json", manifest)
    return cache, mapping, manifest


def test_export_uses_only_projected_eligible_development_features(inputs):
    cache, mapping, manifest = inputs
    output = io.BytesIO()
    with (
        patch.object(exporter, "_checksum", wraps=exporter._checksum) as checksum,
        patch.object(exporter.pq, "read_table", wraps=exporter.pq.read_table) as read,
    ):
        exporter.export_development(cache, mapping, REQUESTED, output)
    assert [call.args[0] for call in checksum.call_args_list] == [mapping, cache / DEV]
    read.assert_called_once_with(
        cache / DEV,
        columns=["slide_id", "mean_raw", "mean_ln"],
        filters=[("slide_id", "in", ["a", "b", "z"])],
    )
    output.seek(0)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["ids"].tolist() == ["a", "b", "z"]
        assert archive["splits"].tolist() == ["val", "train", "train"]
        assert archive["records"].tolist() == ["0002", "0002", "0001"]
        np.testing.assert_array_equal(archive["targets"], [0.6, 0.3, -0.5])
        meta = json.loads(archive["metadata"].item())
        assert meta == {
            "cache_digest": manifest["digest"],
            "mapping_sha256": manifest["mapping_sha256"],
            "blocks": {f"{SOURCE}:mean_raw": "block_0", f"{SOURCE}:mean_ln": "block_1"},
            "unavailable": ["l1/cancer_mask_5:mean_raw"],
            "file_sha256": {SOURCE: manifest["sources"][SOURCE]["files"][DEV]},
            "source_uris": {SOURCE: "mlflow-artifacts:/example/l2"},
        }
        assert archive["block_0"].dtype == archive["block_1"].dtype == np.float32
        np.testing.assert_array_equal(
            archive["block_0"], [[1, 2], [np.nan, np.nan], [3, 4]]
        )
        np.testing.assert_array_equal(
            archive["block_1"], [[1, -1], [np.nan, np.nan], [0.25, -0.25]]
        )


def test_export_checks_manifests_mappings_and_partitions(inputs):
    cache, mapping, manifest = inputs
    manifest_path = cache / "manifest.json"
    output = io.BytesIO()
    write_json(manifest_path, manifest | {"schema_version": 99})
    with pytest.raises(ValueError, match="cache manifest"):
        exporter.export_development(cache, mapping, REQUESTED, output)
    write_json(manifest_path, manifest)

    original = mapping.read_bytes()
    mapping.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Data mapping"):
        exporter.export_development(cache, mapping, REQUESTED, output)
    mapping.write_bytes(original)

    with (cache / DEV).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Checksum or path mismatch"):
        exporter.export_development(cache, mapping, REQUESTED, output)
    assert output.getvalue() == b""


def test_stdin_cli_writes_npz_to_unseekable_stdout(inputs):
    cache, mapping, _ = inputs
    script = Path(exporter.__file__).read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-",
            "--cache",
            str(cache),
            "--data-mapping",
            str(mapping),
            "--block",
            f"{SOURCE}:mean_raw",
        ],
        input=script,
        capture_output=True,
        check=True,
    )
    with np.load(io.BytesIO(result.stdout), allow_pickle=False) as archive:
        assert archive["ids"].tolist() == ["a", "b", "z"]
        assert np.isnan(archive["block_0"][1]).all()
