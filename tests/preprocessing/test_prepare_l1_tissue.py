import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.embedding_selection import MASK_GATES, TILE_SCHEMA, selection_digest
from preprocessing.prepare_l1_tissue import _digest, build_tissue_selections


def _selection(
    root: Path,
    tile_scores: list[tuple[int, float | None, float | None]],
    *,
    cancer: bool,
    split: str = "val",
) -> None:
    (root / "tiles").mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "x": x,
                "y": 0,
                "sample_rank": rank,
                "tissue_overlap": 0.8,
                "blur_overlap": 0.8,
                "folding_overlap": 0.8,
                "residual_overlap": 0.8,
                "cancer_overlap": c,
                "epithelium_overlap": epi,
            }
            for rank, (x, c, epi) in enumerate(tile_scores)
        ],
        schema=TILE_SCHEMA,
    )
    pq.write_table(table, root / "tiles" / "slide.one.parquet")
    gates = {
        key: value
        for key, value in MASK_GATES.items()
        if cancer or key != "cancer_overlap"
    }
    fields = {
        "level": 1,
        "cap": 3000,
        "seed": 42,
        "source_gates": gates,
        "slides": [
            {
                "slide_id": "slide.one",
                "split": split,
                "path": "/input/slide.one.svs",
                "level": 1,
                "extent_x": 1000,
                "extent_y": 1000,
                "tile_extent_x": 224,
                "tile_extent_y": 224,
                "stride_x": 112,
                "stride_y": 112,
                "grid_origin_x": 0,
                "grid_origin_y": 0,
                "mpp_x": 0.485,
                "mpp_y": 0.485,
                "tile_count": len(table),
                "selection_digest": selection_digest(table),
            }
        ],
    }
    fields["digest"] = _digest(fields)
    (root / "manifest.json").write_text(json.dumps(fields))


def test_tissue_selection_reuses_scores_and_preserves_gpu_only_rows(
    tmp_path: Path,
) -> None:
    tissue, cancer, output = (
        tmp_path / name for name in ("tissue", "cancer", "output")
    )
    _selection(
        tissue, [(0, None, None), (224, None, None), (448, None, None)], cancer=False
    )
    _selection(cancer, [(0, 0.9, 0.1), (448, 0.8, 0.2)], cancer=True)
    restored = []

    def score(frame: pd.DataFrame, _slide: dict, column: str, _mask_dir: Path) -> None:
        missing = frame[column].isna()
        restored.append((column, frame.loc[missing, "x"].tolist()))
        frame.loc[missing, column] = {
            "cancer_overlap": 0.5,
            "epithelium_overlap": 0.25,
        }[column]

    with patch("preprocessing.prepare_l1_tissue._restore_scores", side_effect=score):
        audit = build_tissue_selections(
            tissue,
            cancer,
            output,
            {"cancer_overlap": tmp_path, "epithelium_overlap": tmp_path},
            {"run_id": "cancer-run"},
            workers=2,
        )
    assert restored == [("cancer_overlap", [224]), ("epithelium_overlap", [224])]
    assert audit["total"]["selected_tiles"] == 3
    assert audit["total"]["reusable_tiles"] == 2
    assert audit["total"]["missing_embeddings"] == 1
    assert audit["total"]["epithelium_or_cancer_tiles"] == 2
    assert set(audit["score_missing"].values()) == {0}
    gpu = pq.read_table(output / "missing_selection" / "tiles" / "slide.one.parquet")
    assert gpu["x"].to_pylist() == [224] and gpu["sample_rank"].to_pylist() == [1]
    full = pq.read_table(output / "selection" / "tiles" / "slide.one.parquet")
    assert full["cancer_overlap"].to_pylist() == [0.9, 0.5, 0.8]
    for folder in ("selection", "missing_selection"):
        saved = json.loads((output / folder / "manifest.json").read_text())
        assert saved["digest"] == _digest(saved)
        assert saved["reuse_source"]["run_id"] == "cancer-run"
