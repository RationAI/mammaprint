import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from preprocessing.embedding_selection import (
    EMBED_DIM,
    MASK_GATES,
    prepare_selection,
    selection_digest,
)


def _tiling(tmp_path):
    directory = tmp_path / "tiling"
    directory.mkdir()
    geometry = {
        "level": 3,
        "extent_x": 53,
        "extent_y": 37,
        "tile_extent_x": 20,
        "tile_extent_y": 12,
        "stride_x": 10,
        "stride_y": 6,
        "grid_origin_x": 3,
        "grid_origin_y": 2,
        "mpp_x": 0.485,
        "mpp_y": 0.485,
    }
    pd.DataFrame(
        [
            {"id": "a-source", "path": "/slides/a.svs", **geometry},
            {"id": "zero-source", "path": "/slides/zero.svs", **geometry},
        ]
    ).to_parquet(directory / "slides.parquet", index=False)
    rows = pd.DataFrame(
        [
            {"slide_id": "a-source", "x": x, "y": y, **dict.fromkeys(MASK_GATES, 0.9)}
            for x in range(3, 34, 10)
            for y in range(2, 27, 6)
        ]
    )
    rows.to_parquet(directory / "tiles.parquet", index=False, row_group_size=4)
    mapping = tmp_path / "data_mapping.csv"
    mapping.write_text(
        "path,split\n/slides/a.svs,train\n/slides/zero.svs,val\n/slides/absent.svs,test\n"
    )
    return directory, mapping, rows


def _prepare(tmp_path, tiling, mapping, name, **kwargs):
    return prepare_selection(
        level=kwargs.pop("level", 1),
        tiling=str(tiling),
        data_mapping=mapping,
        output=tmp_path / name,
        shards=2,
        **kwargs,
    )


def test_stable_grid_selection_and_missing_score_inventory(tmp_path):
    tiling, mapping, rows = _tiling(tmp_path)
    first = _prepare(tmp_path, tiling, mapping, "first", cap=3)
    rows.sample(frac=1, random_state=8).to_parquet(
        tiling / "tiles.parquet", index=False
    )
    reordered = _prepare(
        tmp_path, tiling, mapping, "reordered", cap=3, source_gate="cancer"
    )
    uncapped = _prepare(tmp_path, tiling, mapping, "uncapped")

    assert first["digest"] == reordered["digest"]
    assert first == json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert first["mapping_without_source"] == [
        {"slide_id": "absent", "path": "/slides/absent.svs", "split": "test"}
    ]
    assert first["score_missing"]["epithelium_overlap"] == 3
    slide, zero = first["slides"]
    assert slide["slide_id"] == "a"
    assert slide["level"] == 3
    assert slide["source_tile_count"] == 20
    assert slide["grid_count"] == 6
    assert {slide["shard"], zero["shard"]} == {0, 1}
    assert zero["status"] == "zero_eligible_tiles"
    assert pq.read_table(tmp_path / "first" / "tiles" / "zero.parquet").num_rows == 0

    capped = pq.read_table(tmp_path / "first" / "tiles" / "a.parquet")
    assert selection_digest(capped) == slide["selection_digest"]
    assert uncapped["slides"][0]["tile_count"] == 6
    assert set(capped["sample_rank"].to_pylist()) == {0, 1, 2}
    all_grid = pd.read_parquet(tmp_path / "uncapped" / "tiles" / "a.parquet")
    chosen = capped.to_pandas()
    assert chosen.equals(all_grid[all_grid["sample_rank"] < 3].reset_index(drop=True))
    assert set(zip(all_grid["x"], all_grid["y"], strict=True)) == {
        (x, y) for x in (3, 23) for y in (2, 14, 26)
    }
    assert (all_grid["cancer_overlap"] == 0.9).all()
    # The final tile y=26 ends below extent_y=37, as in grid_tiles(last="keep").
    assert int(all_grid["y"].max()) + 12 > 37


def test_tissue_source_preserves_low_cancer_scores_and_requires_qc_gates(tmp_path):
    tiling, mapping, rows = _tiling(tmp_path)
    path = tiling / "tiles.parquet"
    rows["cancer_overlap"] = 0.5
    rows.loc[0, "cancer_overlap"] = 0.0
    rows.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="expected cancer_overlap source gate"):
        _prepare(tmp_path, tiling, mapping, "cancer")
    manifest = _prepare(tmp_path, tiling, mapping, "tissue", source_gate="tissue")
    assert manifest["source_gates"] == {
        key: gate for key, gate in MASK_GATES.items() if key != "cancer_overlap"
    }
    chosen = pq.read_table(tmp_path / "tissue" / "tiles" / "a.parquet")
    assert chosen["cancer_overlap"].to_pylist() == [0.0, 0.5, 0.5, 0.5, 0.5, 0.5]

    rows.loc[0, "blur_overlap"] = 0.5
    rows.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="expected blur_overlap source gate"):
        _prepare(tmp_path, tiling, mapping, "failed-qc", source_gate="tissue")


def test_tissue_source_recovers_cancer_and_epithelium_scores(tmp_path, monkeypatch):
    from ratiopath.tiling import overlays

    tiling, mapping, rows = _tiling(tmp_path)
    rows.drop(columns="cancer_overlap").to_parquet(
        tiling / "tiles.parquet", index=False
    )
    mask_sources = {}
    for score in ("cancer_overlap", "epithelium_overlap"):
        masks = tmp_path / score
        masks.mkdir()
        (masks / "a.tiff").touch()
        mask_sources[score] = str(masks)

    def fake_overlay(roi, mask_paths, x, y, mpp_x, mpp_y):
        return [np.ma.array([0, 51, 102, 153])] * len(x)

    monkeypatch.setattr(overlays, "_tile_overlay", fake_overlay)
    manifest = _prepare(
        tmp_path,
        tiling,
        mapping,
        "tissue",
        source_gate="tissue",
        cap=2,
        mask_sources=mask_sources,
    )
    chosen = pq.read_table(tmp_path / "tissue" / "tiles" / "a.parquet")
    assert chosen["cancer_overlap"].to_pylist() == [0.3, 0.3]
    assert chosen["epithelium_overlap"].to_pylist() == [0.75, 0.75]
    assert manifest["score_missing"]["cancer_overlap"] == 0
    assert manifest["score_missing"]["epithelium_overlap"] == 0


def test_scores_restored_only_for_selected_coordinates(tmp_path, monkeypatch):
    from ratiopath.tiling import overlays

    tiling, mapping, _ = _tiling(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "a.tiff").touch()
    seen = []

    def fake_overlay(roi, mask_paths, x, y, mpp_x, mpp_y):
        assert roi.bounds == (0, 0, 20, 12)
        assert all(path.as_py() == str(masks / "a.tiff") for path in mask_paths)
        assert mpp_x.to_pylist() == mpp_y.to_pylist() == [0.485, 0.485]
        seen.extend(zip(x.to_pylist(), y.to_pylist(), strict=True))
        return [np.ma.array([1, 0, 1, 1]), np.ma.array([0, 1, 0, 1])]

    monkeypatch.setattr(overlays, "_tile_overlay", fake_overlay)
    manifest = _prepare(
        tmp_path,
        tiling,
        mapping,
        "recovered",
        cap=2,
        mask_sources={"epithelium_overlap": str(masks)},
    )
    selected = pd.read_parquet(tmp_path / "recovered" / "tiles" / "a.parquet")
    assert seen == list(zip(selected["x"], selected["y"], strict=True))
    assert selected["epithelium_overlap"].tolist() == [0.75, 0.5]
    assert manifest["score_missing"]["epithelium_overlap"] == 0


def test_l2_selects_existing_features_and_detects_missing_grid_coordinates(tmp_path):
    tiling, mapping, rows = _tiling(tmp_path)
    previous = tmp_path / "previous"
    previous.mkdir()
    features = rows[["x", "y"]].copy().iloc[::-1]
    features["embedding"] = [
        np.full(EMBED_DIM, x + y, dtype=np.float32)
        for x, y in features.itertuples(index=False)
    ]
    path = previous / "a.parquet"
    features.to_parquet(path, index=False, row_group_size=4)
    good = _prepare(
        tmp_path, tiling, mapping, "good", level=2, l2_embeddings=str(previous), cap=3
    )
    selected = pd.read_parquet(tmp_path / "good" / "embeddings" / "a.parquet")
    assert good["slides"][0]["status"] == "selected"
    assert len(selected) == 3
    assert (
        np.stack(selected["embedding"])
        == (selected["x"] + selected["y"]).to_numpy()[:, None]
    ).all()

    features[~((features["x"] == 3) & (features["y"] == 2))].to_parquet(
        path, index=False
    )
    bad = _prepare(
        tmp_path, tiling, mapping, "bad", level=2, l2_embeddings=str(previous), cap=3
    )
    assert bad["slides"][0]["status"] == "l2_coordinate_mismatch"
    assert bad["slides"][0]["l2_coordinate_counts"]["missing_grid"] == 1
    assert bad["slides"][0]["tile_count"] == 0
    assert not (tmp_path / "bad" / "embeddings" / "a.parquet").exists()


def test_invalid_coordinates_and_duplicate_source_rows_fail_before_publication(
    tmp_path,
):
    tiling, mapping, rows = _tiling(tmp_path)
    path = tiling / "tiles.parquet"
    pd.concat([rows, rows.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate tile coordinates"):
        _prepare(tmp_path, tiling, mapping, "errors")
    assert not (tmp_path / "errors").exists()

    rows.loc[0, "x"] = 4
    rows.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="stored source grid"):
        _prepare(tmp_path, tiling, mapping, "errors")
    assert not (tmp_path / "errors").exists()
