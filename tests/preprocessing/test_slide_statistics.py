import csv
import fcntl
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from preprocessing import slide_statistics
from preprocessing.slide_statistics import (
    build_cache,
    file_digest,
    load_manifest,
    load_membership,
    source_inventory,
)


SOURCE = "l2/cancer_mask_5"
MAPPING_COLUMNS = ("path", "split", "record_num", "mammaprint_index")


def _mapping(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(MAPPING_COLUMNS)
        writer.writerows(rows)
    return path


def _write_embeddings(path, rows, *, fixed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = pa.list_(pa.float32(), 3) if fixed else pa.list_(pa.float32())
    pq.write_table(pa.table({"embedding": pa.array(rows, type=dtype)}), path)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    values = [[1.0, 20.0, -2.0], [1.0, 1.0, 1.0], [7.0, 1.0, 2.0]]
    _write_embeddings(source / "dev.parquet", values)
    _write_embeddings(source / "one.parquet", [[2.0, 5.0, 9.0]], fixed=True)
    _write_embeddings(source / "empty.parquet", [])
    _write_embeddings(source / "bad-width.parquet", [[1.0, 2.0]])
    _write_embeddings(source / "nonfinite.parquet", [[np.nan, 2.0, 3.0]], fixed=True)
    _write_embeddings(source / "unmapped.parquet", [[1.0, 2.0, 3.0]])
    (source / "unreadable.parquet").write_bytes(b"not parquet")
    mapping = _mapping(
        tmp_path / "mapping.csv",
        [
            ("dev.svs", "train", "0001", 0.1),
            ("one.svs", "test", "0002", "test label"),
            ("empty.svs", "val", "0001", -0.5),
            ("missing.svs", "val", "0003", ""),
            ("bad-width.svs", "val", "0004", np.nan),
            ("nonfinite.svs", "val", "0005", -np.inf),
            ("unreadable.svs", "val", "0006", -0.7),
        ],
    )
    return {
        "data_mapping": str(mapping),
        "embedding_dim": 3,
        "batch_size": 2,
        "sources": {SOURCE: str(source)},
    }, np.array(values)


def test_cache_streams_statistics_and_reconciles_every_slide(tmp_path, inputs):
    config, raw = inputs
    output = tmp_path / "cache"
    path = build_cache(config, output)

    manifest = load_manifest(output)
    assert path == output / "manifest.json"
    assert manifest["sources"][SOURCE]["status"] == "available"
    for relative, checksum in manifest["sources"][SOURCE]["files"].items():
        assert file_digest(output / relative) == checksum

    rows = pq.read_table(output / f"stats/dev/{SOURCE}.parquet").to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["slide_id"] == "dev"
    assert row["tile_count"] == 3
    centered = raw - raw.mean(axis=1, keepdims=True)
    normalized = centered / np.sqrt(raw.var(axis=1, keepdims=True) + 1e-5)
    for name, array in (("raw", raw), ("ln", normalized)):
        np.testing.assert_allclose(row[f"mean_{name}"], array.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(row[f"std_{name}"], array.std(axis=0), rtol=1e-6)

    test = pq.read_table(output / f"stats/test/{SOURCE}.parquet")
    assert test.column("slide_id").to_pylist() == ["one"]
    assert test.column("std_ln").to_pylist() == [[0.0, 0.0, 0.0]]
    assert test.column("std_raw").to_pylist() == [[0.0, 0.0, 0.0]]
    assert test.column("mean_raw").type.list_size == 3

    coverage = pq.read_table(output / "coverage.parquet").to_pylist()
    assert {row["slide_id"]: row["status"] for row in coverage} == {
        "dev": "ok",
        "one": "ok",
        "empty": "empty",
        "nonfinite": "invalid-vector",
        "bad-width": "invalid-vector",
        "missing": "absent",
        "unmapped": "unmapped",
        "unreadable": "unreadable",
    }


def test_resume_preserves_completed_source_after_interruption(tmp_path, inputs):
    config, _ = inputs
    extra = tmp_path / "l3"
    second = "l3/cancer_mask_5"
    _write_embeddings(extra / "dev.parquet", [[1.0, 4.0, 9.0]])
    config["sources"][second] = str(extra)
    output = tmp_path / "interrupted"
    build_source = slide_statistics._build_source

    def interrupt(key, *args):
        if key == second:
            raise RuntimeError("interrupted")
        return build_source(key, *args)

    with (
        patch.object(slide_statistics, "_build_source", side_effect=interrupt),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        build_cache(config, output)
    assert not (output / "manifest.json").exists()

    with patch.object(slide_statistics, "_build_source", wraps=build_source) as build:
        build_cache(config, output)
        assert [call.args[0] for call in build.call_args_list] == [second]
    clean = tmp_path / "fresh"
    build_cache(config, clean)
    assert load_manifest(output) == load_manifest(clean)
    assert build_cache(config, output) == output / "manifest.json"

    _write_embeddings(
        Path(config["sources"][SOURCE]) / "dev.parquet", [[7.0, 3.0, 9.0]]
    )
    with pytest.raises(ValueError, match="Stale or corrupt"):
        build_cache(config, output)


def test_cache_reuses_existing_sources_in_extended_version(tmp_path, inputs):
    config, _ = inputs
    first = tmp_path / "first"
    build_cache(config, first)
    version = load_manifest(first)
    l1 = tmp_path / "l1"
    _write_embeddings(l1 / "dev.parquet", [[1.0, 3.0, 9.0]])
    extra = "l1/cancer_mask_5"
    config["sources"][extra] = str(l1)
    config["reuse_cache"] = str(first)

    extended = tmp_path / "extended"
    with patch.object(
        slide_statistics, "_build_source", wraps=slide_statistics._build_source
    ) as build:
        build_cache(config, extended)
        assert [call.args[0] for call in build.call_args_list] == [extra]
    assert load_manifest(first) == version
    assert load_manifest(extended)["sources"][SOURCE] == version["sources"][SOURCE]


def test_pilot_never_reads_test_parquets(tmp_path, inputs):
    config, _ = inputs
    config["pilot_slides"] = 2
    output = tmp_path / "pilot"
    with patch.object(
        slide_statistics, "_read_slide", wraps=slide_statistics._read_slide
    ) as read:
        build_cache(config, output)
        assert [call.args[3] for call in read.call_args_list] == ["bad-width", "dev"]
    manifest = load_manifest(output)
    assert manifest["pilot_slides"] == 2
    assert manifest["pilot_diagnostics"] == {
        "attempted_slide_reads": 2,
        "usable_slide_reads": 1,
        "known_input_bytes": sum(
            manifest["sources"][SOURCE]["inventory"][slide]["bytes"]
            for slide in ("bad-width", "dev")
        ),
        "reads_without_known_size": 0,
    }
    assert pq.read_table(output / f"stats/test/{SOURCE}.parquet").num_rows == 0
    assert pq.read_table(output / "coverage.parquet").column("split").to_pylist() == [
        "val",
        "train",
    ]
    with pytest.raises(ValueError, match="Cache inputs changed"):
        build_cache(config | {"pilot_slides": None}, output)


def test_pilot_rejects_all_unusable_slides(tmp_path, inputs):
    config, _ = inputs
    output = tmp_path / "pilot"
    with pytest.raises(
        ValueError, match="no usable development embeddings after 1 slide reads"
    ):
        build_cache(config | {"pilot_slides": 1}, output)
    assert not (output / "manifest.json").exists()


def test_subsample_is_stable_across_batch_sizes(tmp_path, inputs):
    config, raw = inputs
    sample_config = config | {"sample_tiles": 2, "sample_seed": 17}
    rows = []
    for batch in (1, 2):
        output = tmp_path / f"cache-{batch}"
        build_cache(sample_config | {"batch_size": batch}, output)
        rows.append(pq.read_table(output / f"samples/dev/{SOURCE}.parquet").to_pylist())
    assert rows[0] == rows[1]
    assert len({row["row_id"] for row in rows[0]}) == 2
    for row in rows[0]:
        np.testing.assert_array_equal(row["embedding"], raw[row["row_id"]])


def test_membership_keeps_record_strings_and_reads_only_requested_labels(
    tmp_path, inputs
):
    config, _ = inputs
    path = Path(config["data_mapping"])
    with patch.object(
        slide_statistics, "_label", wraps=slide_statistics._label
    ) as label:
        assert load_membership(path)["dev"].target is None
        label.assert_not_called()
        slides = load_membership(path, labels="dev")
        assert all(call.args[0]["split"] != "test" for call in label.call_args_list)
    assert slides["dev"].record_num == "0001"
    assert slides["one"].target is None
    assert slides["dev"].target == 0.1
    assert slides["bad-width"].target is None
    assert slides["nonfinite"].target is None
    with patch.object(
        slide_statistics, "_label", wraps=slide_statistics._label
    ) as label:
        assert load_membership(path, labels="test")["dev"].target is None
        label.assert_called_once()
        assert label.call_args.args[0]["split"] == "test"

    leaked = _mapping(
        tmp_path / "leaked.csv",
        [("a.svs", "train", "001", 0.1), ("b.svs", "test", "001", -0.2)],
    )
    with pytest.raises(ValueError, match="span development and test"):
        load_membership(leaked)

    duplicate = _mapping(
        tmp_path / "duplicate.csv",
        [("x/a.svs", "train", "001", 0.1), ("y/a.svs", "val", "002", -0.2)],
    )
    with pytest.raises(ValueError, match="duplicate mapping slide"):
        load_membership(duplicate)


def test_inventory_resolves_configured_card_and_tracks_missing_l1(
    tmp_path, monkeypatch
):
    root = tmp_path / "cards"
    card = root / "cancer_mask_5" / "l1.yaml"
    card.parent.mkdir(parents=True)
    card.write_text("uris:\n  all: ${oc.env:MAMMAPRINT_L1_ALL_URI}\n")
    monkeypatch.delenv("MAMMAPRINT_L1_ALL_URI", raising=False)
    config = {"cards_dir": str(root), "sources": {SOURCE: "fixture-uri"}}
    assert source_inventory(config)["l1/cancer_mask_5"]["status"] == "unavailable"
    assert source_inventory(config)[SOURCE]["uri"] == "fixture-uri"
    monkeypatch.setenv("MAMMAPRINT_L1_ALL_URI", "mlflow-artifacts:/frozen/l1")
    assert (
        source_inventory(config)["l1/cancer_mask_5"]["uri"]
        == "mlflow-artifacts:/frozen/l1"
    )


def test_cache_requires_single_writer(tmp_path, inputs):
    config, _ = inputs
    output = tmp_path / "cache"
    output.mkdir()
    with (output / ".cache.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process is building"):
            build_cache(config, output)


def test_remote_staging_retries_and_cleans_scratch(tmp_path, inputs, monkeypatch):
    from mlflow.entities import FileInfo

    config, _ = inputs
    local = Path(config["sources"][SOURCE])
    embedding = local / "dev.parquet"
    scratch = tmp_path / "scratch"
    calls = []

    def download(*, artifact_uri, dst_path):
        calls.append(artifact_uri)
        if len(calls) == 1:
            raise OSError("Temporary read failure")
        destination = Path(dst_path) / embedding.name
        shutil.copyfile(embedding, destination)
        return str(destination)

    monkeypatch.setattr(slide_statistics.time, "sleep", lambda _: None)
    with (
        patch(
            "mlflow.artifacts.list_artifacts",
            return_value=[FileInfo(embedding.name, False, embedding.stat().st_size)],
        ),
        patch("mlflow.artifacts.download_artifacts", side_effect=download),
    ):
        output = tmp_path / "cache"
        build_cache(
            config
            | {
                "sources": {SOURCE: "mlflow-artifacts:/frozen/embeddings"},
                "scratch_dir": scratch,
            },
            output,
        )
    assert calls == ["mlflow-artifacts:/frozen/embeddings/dev.parquet"] * 2
    assert list(scratch.iterdir()) == []
    assert pq.read_table(output / f"stats/dev/{SOURCE}.parquet").num_rows == 1


@pytest.mark.parametrize("failure", ["transport", "truncated-file"])
def test_exhausted_remote_staging_leaves_source_resumable(
    tmp_path, inputs, monkeypatch, failure
):
    from mlflow.entities import FileInfo

    config, _ = inputs
    embedding = Path(config["sources"][SOURCE]) / "dev.parquet"
    remote = "l3/cancer_mask_5"
    config["sources"][remote] = "mlflow-artifacts:/frozen/embeddings"
    output = tmp_path / "cache"
    scratch = tmp_path / "scratch"
    config["scratch_dir"] = scratch
    monkeypatch.setattr(slide_statistics.time, "sleep", lambda _: None)

    def broken_download(*, artifact_uri, dst_path):
        if failure == "transport":
            raise OSError("Transport failed")
        path = Path(dst_path) / embedding.name
        path.write_bytes(b"partial")
        return str(path)

    def recovered_download(*, artifact_uri, dst_path):
        path = Path(dst_path) / embedding.name
        shutil.copyfile(embedding, path)
        return str(path)

    with patch(
        "mlflow.artifacts.list_artifacts",
        return_value=[FileInfo(embedding.name, False, embedding.stat().st_size)],
    ):
        with (
            patch(
                "mlflow.artifacts.download_artifacts", side_effect=broken_download
            ) as download,
            pytest.raises(OSError, match="could not be staged after three attempts"),
        ):
            build_cache(config, output)
        assert download.call_count == 3
        assert list(scratch.iterdir()) == []
        assert (output / f".state/complete/{SOURCE}.json").is_file()
        assert not (output / f".state/complete/{remote}.json").exists()
        assert not (output / "manifest.json").exists()
        with (
            patch(
                "mlflow.artifacts.download_artifacts", side_effect=recovered_download
            ),
            patch.object(
                slide_statistics, "_build_source", wraps=slide_statistics._build_source
            ) as build,
        ):
            build_cache(config, output)
        assert [call.args[0] for call in build.call_args_list] == [remote]
        assert pq.read_table(output / f"stats/dev/{remote}.parquet").num_rows == 1
