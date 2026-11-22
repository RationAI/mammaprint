from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from preprocessing.embedding_selection import TILE_SCHEMA, selection_digest
from preprocessing.resolution_embeddings import FEATURES, _infer, is_complete


def test_cross_slide_batch_keeps_tile_alignment_and_rejects_corruption(tmp_path):
    selection = tmp_path / "input"
    (selection / "tiles").mkdir(parents=True)
    output = tmp_path / "output"
    slides = []
    for stem, count in (("first.scan", 3), ("missing", 2), ("third", 4)):
        rows = [
            {
                name: index if name in {"x", "sample_rank"} else 0
                for name in TILE_SCHEMA.names
            }
            for index in range(count)
        ]
        tiles = pa.Table.from_pylist(rows, schema=TILE_SCHEMA)
        pq.write_table(tiles, selection / "tiles" / f"{stem}.parquet")
        slides.append(
            {
                "slide_id": stem,
                "path": f"{stem}.svs",
                "level": 1,
                "tile_extent_x": 2,
                "tile_extent_y": 2,
                "tile_count": count,
                "selection_digest": selection_digest(tiles),
            }
        )

    class Reader:
        def __init__(self, path):
            if path == "missing.svs":
                raise OSError("missing WSI")

        def read_tile(self, x, _y, width, height, _level):
            return np.full((height, width, 3), x, dtype=np.uint8)

        def close(self):
            pass

    class Encoder(torch.nn.Module):
        def forward(self, batch):
            return batch[:, 0, 0, 0, None].expand(-1, FEATURES)

    args = Namespace(batch_size=4, workers=0, prefetch=2, precision="fp32")
    summary = dict.fromkeys(
        (
            "committed_tiles",
            "loader_wait_seconds",
            "read_worker_seconds",
            "gpu_seconds",
            "write_worker_seconds",
            "writer_wait_seconds",
        ),
        0,
    )
    summary["failures"] = {}
    provider = SimpleNamespace(OpenSlide=Reader)
    with patch.dict("sys.modules", {"ratiopath.openslide": provider}):
        _infer(
            args, selection, output, slides, summary, torch.device("cpu"), Encoder(), {}
        )

    assert summary["committed_tiles"] == 7
    assert summary["failures"] == {"missing": "OSError: missing WSI"}
    assert is_complete(output, slides[0])
    assert not is_complete(output, slides[1])
    assert is_complete(output, slides[2])
    sample = output / "embeddings" / "third.parquet"
    table = pq.read_table(sample)
    values = table["embedding"].combine_chunks().values.to_numpy().reshape(-1, FEATURES)
    np.testing.assert_allclose(
        values[:, 0], (np.arange(4) - 0.485 * 255) / (0.229 * 255)
    )
    assert selection_digest(table) == slides[2]["selection_digest"]
    with sample.open("r+b") as stream:
        stream.seek(24)
        value = stream.read(1)
        stream.seek(24)
        stream.write(bytes([value[0] ^ 1]))
    assert not is_complete(output, slides[2])
