import json
import logging

import pandas as pd

from preprocessing.filter_embeddings import (
    _load_filter_state,
    _prepare_output_dirs,
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
