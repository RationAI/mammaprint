import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

import preprocessing.filter_embeddings as filtering
from preprocessing.filter_embeddings import (
    _load_filter_state,
    _load_scores,
    _prepare_output_dirs,
    _qc_thresholds,
    _selection_masks,
    _state_payload,
    _strategies,
    _validate_output_dirs,
    filter_embeddings,
)


def test_filter_embeddings_materialises_only_selected_strategies(tmp_path):
    embeddings_dir = tmp_path / "source"
    embeddings_dir.mkdir()
    pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [10, 11, 12],
            "embedding": [[1.0], [2.0], [3.0]],
        }
    ).to_parquet(embeddings_dir / "slide.parquet", index=False)
    scores = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [10, 11, 12],
            "epithelium_overlap": [0.3, 0.1, 0.4],
            "cancer_overlap": [0.1, 0.8, 0.9],
        }
    )
    strategies = _strategies(0.25, 0.5, ("or", "and"))
    output_dirs = _prepare_output_dirs(tmp_path / "outputs", strategies)

    counts, source_tiles = filter_embeddings(
        scores_by_stem={"slide": scores},
        embeddings_dir=embeddings_dir,
        output_dirs=output_dirs,
        split_map={"slide": "train"},
        epithelium_threshold=0.25,
        cancer_threshold=0.5,
        logger=logging.getLogger(__name__),
    )

    assert source_tiles == 3
    assert counts["or"]["tiles"] == 3
    assert counts["and"]["tiles"] == 1
    assert set(output_dirs) == {"or", "and"}
    assert len(pd.read_parquet(output_dirs["or"]["all"] / "slide.parquet")) == 3
    assert len(pd.read_parquet(output_dirs["and"]["all"] / "slide.parquet")) == 1
    _validate_output_dirs(output_dirs)


def test_upload_only_accepts_subset_of_saved_strategies(tmp_path):
    state_path = tmp_path / "filter_state.json"
    state_path.write_text(
        json.dumps(
            {
                "level": 2,
                "tiling_uri": "tiling",
                "embeddings_uri": "embeddings",
                "strategies": ["or", "and"],
                "epithelium_threshold": 0.25,
                "cancer_threshold": 0.5,
                "counts": {},
                "scored_tiles": 3,
                "source_embedding_tiles": 3,
            }
        )
    )

    state = _load_filter_state(
        state_path,
        level=2,
        tiling_uri="tiling",
        embeddings_uri="embeddings",
        strategies=_strategies(0.25, 0.5, ("and",)),
        epithelium_threshold=0.25,
        cancer_threshold=0.5,
    )

    assert state["strategies"] == ["or", "and"]


def test_qc_only_filters_existing_embeddings_without_epithelium_scores(tmp_path):
    tiling = tmp_path / "tiling"
    tiling.mkdir()
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "path": ["/a.mrxs", "/b.mrxs", "/c.mrxs"],
            "level": [2, 2, 2],
        }
    ).to_parquet(tiling / "slides.parquet")
    pd.DataFrame(
        {
            "slide_id": ["a"] * 4 + ["b"],
            "x": [0, 1, 2, 3, 0],
            "y": [10] * 5,
            "blur_overlap": [0.91, 0.9, 0.95, 0.95, 0.5],
            "residual_overlap": [0.95, 1.0, 0.9, 1.0, 1.0],
            "epithelium_overlap": [None] * 5,
        }
    ).to_parquet(tiling / "tiles.parquet")
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    original = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [10] * 4,
            "embedding": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        }
    )
    original.to_parquet(embeddings / "a.parquet", index=False)
    original.iloc[:1].to_parquet(embeddings / "b.parquet", index=False)
    original.iloc[:0].to_parquet(embeddings / "c.parquet", index=False)
    source_bytes = (embeddings / "a.parquet").read_bytes()
    qc = {"blur": 0.9, "residual": 0.9}
    scores, n = _load_scores(tiling, 2, ("qc_only",), qc)
    dirs = _prepare_output_dirs(
        tmp_path / "outputs", _strategies(0.25, 0.5, ("qc_only",), qc)
    )
    counts, source_tiles = filter_embeddings(
        scores,
        embeddings,
        dirs,
        {"a": "train", "b": "val", "c": "test"},
        0.25,
        0.5,
        logging.getLogger(__name__),
        qc,
    )
    assert n == 5 and source_tiles == 5
    assert counts["qc_only"] == {
        "tiles": 2,
        "empty_slides": 2,
        "slides_below_100_tiles": 3,
    }
    result = pd.read_parquet(dirs["qc_only"]["train"] / "a.parquet")
    assert result.x.tolist() == [0, 3]  # Exact equality to a cutoff is rejected.
    assert [list(x) for x in result.embedding] == [[1.0, 2.0], [7.0, 8.0]]
    assert pd.read_parquet(dirs["qc_only"]["val"] / "b.parquet").empty
    assert pd.read_parquet(dirs["qc_only"]["test"] / "c.parquet").empty
    assert source_bytes == (embeddings / "a.parquet").read_bytes()
    _validate_output_dirs(dirs)


def test_qc_gates_are_anded_with_each_epithelium_cancer_strategy():
    scores = pd.DataFrame(
        {
            "epithelium_overlap": [0.8, 0.1, 0.8],
            "cancer_overlap": [0.8, 0.8, 0.1],
            "folding_overlap": [0.4, 0.8, 0.9],
        }
    )
    masks = _selection_masks(scores, 0.25, 0.5, qc_thresholds={"folding": 0.5})
    assert masks["epithelium_only"].tolist() == [False, False, True]
    assert masks["or"].tolist() == [False, True, True]
    assert masks["and"].tolist() == [False, False, False]


@pytest.mark.parametrize("unscored_slide", [False, True])
def test_qc_only_rejects_unscored_source_tiles(tmp_path, unscored_slide):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    pd.DataFrame({"x": [0, 1], "y": [0, 0], "embedding": [[1.0], [2.0]]}).to_parquet(
        embeddings / "slide.parquet"
    )
    scores = (
        {}
        if unscored_slide
        else {"slide": pd.DataFrame({"x": [0], "y": [0], "blur_overlap": [1.0]})}
    )
    qc = {"blur": 0.9}
    dirs = _prepare_output_dirs(
        tmp_path / "outputs", _strategies(0.25, 0.5, ("qc_only",), qc)
    )
    with pytest.raises(
        ValueError, match="embedded coordinates for slide have no QC scores"
    ):
        filter_embeddings(
            scores,
            embeddings,
            dirs,
            {"slide": "train"},
            0.25,
            0.5,
            logging.getLogger(__name__),
            qc,
        )
    assert not list(dirs["qc_only"]["all"].glob("*.parquet"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_qc_thresholds_are_rejected(value):
    with pytest.raises(ValueError, match="finite and in"):
        _qc_thresholds(blur=value)


@pytest.mark.parametrize("value", [None, float("inf"), -0.1, 1.1])
def test_invalid_active_qc_scores_fail_instead_of_silently_dropping_tiles(value):
    with pytest.raises(ValueError, match="non-null, finite"):
        _selection_masks(
            pd.DataFrame({"blur_overlap": [value]}),
            0.25,
            0.5,
            ("qc_only",),
            {"blur": 0.9},
        )


def test_missing_active_mask_is_reported_and_unused_masks_are_optional():
    with pytest.raises(ValueError, match=r"Missing required.*residual_overlap"):
        _selection_masks(
            pd.DataFrame({"blur_overlap": [1.0]}),
            0.25,
            0.5,
            ("qc_only",),
            {"residual": 0.9},
        )
    with pytest.raises(ValueError, match="at least one"):
        _strategies(0.25, 0.5, ("qc_only",))
    with pytest.raises(ValueError, match="Unknown QC"):
        _qc_thresholds(typo=0.9)


def test_upload_only_rejects_changed_or_omitted_qc_thresholds(tmp_path):
    qc = {"blur": 0.9}
    strategies = _strategies(0.25, 0.5, ("qc_only",), qc)
    args = {
        "level": 2,
        "tiling_uri": "tiling",
        "embeddings_uri": "embeddings",
        "strategies": strategies,
        "epithelium_threshold": 0.25,
        "cancer_threshold": 0.5,
    }
    state = _state_payload(
        **args, counts={}, scored_tiles=1, source_embedding_tiles=1, qc_thresholds=qc
    )
    state_path = tmp_path / "filter_state.json"
    state_path.write_text(json.dumps(state))
    assert (
        _load_filter_state(state_path, **args, qc_thresholds=qc)["qc_thresholds"] == qc
    )
    for other in ({}, {"blur": 0.95}, {"residual": 0.9}):
        with pytest.raises(ValueError, match="QC thresholds"):
            _load_filter_state(state_path, **args, qc_thresholds=other)
    assert (
        strategies[0].slug
        != _strategies(0.25, 0.5, ("qc_only",), {"blur": 0.95})[0].slug
    )
    assert (
        _strategies(0.25, 0.5, ("qc_only",), {"blur": 0.9000001})[0].slug
        != _strategies(0.25, 0.5, ("qc_only",), {"blur": 0.9000002})[0].slug
    )


def test_local_only_cli_writes_filtered_data_and_state_without_mlflow_writes(
    tmp_path, monkeypatch
):
    tiling = tmp_path / "tiling"
    tiling.mkdir()
    pd.DataFrame({"id": ["slide"], "path": ["/slide.mrxs"], "level": [2]}).to_parquet(
        tiling / "slides.parquet"
    )
    pd.DataFrame(
        {
            "slide_id": ["slide", "slide"],
            "x": [0, 1],
            "y": [0, 0],
            "blur_overlap": [0.9, 0.95],
        }
    ).to_parquet(tiling / "tiles.parquet")
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    pd.DataFrame({"x": [0, 1], "y": [0, 0], "embedding": [[1.0], [2.0]]}).to_parquet(
        embeddings / "slide.parquet"
    )
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"path": ["/slide.mrxs"], "split": ["test"]}).to_csv(
        mapping, index=False
    )
    output = tmp_path / "filtered"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter",
            "--level",
            "2",
            "--tiling-uri",
            str(tiling),
            "--embeddings-uri",
            "source-uri",
            "--embeddings-path",
            str(embeddings),
            "--data-mapping",
            str(mapping),
            "--strategies",
            "qc_only",
            "--blur-threshold",
            "0.9",
            "--output-dir",
            str(output),
            "--no-upload",
        ],
    )
    monkeypatch.setattr(filtering, "download_artifacts", lambda uri, dst_path: uri)

    def forbidden(*args, **kwargs):
        pytest.fail("Local-only filtering must not create or upload MLflow runs")

    monkeypatch.setattr(filtering.mlflow, "set_experiment", forbidden)
    monkeypatch.setattr(filtering, "_log_outputs", forbidden)
    try:
        filtering.main()
    finally:
        # main configures process-wide logging; release test-local file handlers.
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()
    state = json.loads((output / "filter_state.json").read_text())
    assert state["qc_thresholds"] == {"blur": 0.9}
    result = pd.read_parquet(
        output / "outputs/qc_only_blur09/embeddings_test/slide.parquet"
    )
    assert result.x.tolist() == [1]


def test_launcher_dry_run_forwards_qc_and_does_not_substitute_source(
    tmp_path, monkeypatch, capsys
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_filter",
        Path(__file__).parents[2] / "scripts/preprocessing/run_filter_embeddings.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    def forbidden(**kwargs):
        pytest.fail("Dry run must not submit a Kubernetes job")

    monkeypatch.setattr(launcher, "submit_job", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--level",
            "2",
            "--tiling-uri",
            "tiling",
            "--embeddings-uri",
            "cm5-source",
            "--strategies",
            "qc_only",
            "--blur-threshold",
            "0.9",
            "--folding-threshold",
            "0.95",
            "--residual-threshold",
            "0.9",
            "--git-ref",
            "codex/qc-embedding-filters",
            "--output-dir",
            str(tmp_path / "filtered"),
            "--no-upload",
            "--dry-run",
        ],
    )
    launcher.main()
    output = capsys.readouterr().out
    assert "--blur-threshold 0.9" in output
    assert "--folding-threshold 0.95" in output
    assert "--residual-threshold 0.9" in output
    assert "--no-upload" in output
    assert "--embeddings-uri cm5-source" in output
    assert "--embeddings-path" not in output
    assert "Git ref: codex/qc-embedding-filters" in output
    assert len(output.splitlines()[0].removeprefix("Job: ")) <= 63
