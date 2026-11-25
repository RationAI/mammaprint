from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir

from preprocessing.l2_augmentation import _load_manifest, card, embed, prepare
from preprocessing.resolution_embeddings import FEATURES, _infer, write_json
from preprocessing.tile_augmentation import TileView, transform_batch


def test_stain_and_rotation_compose_and_keep_background():
    cpu = torch.device("cpu")
    color = torch.tensor(
        [[[[255, 254, 253], [87, 52, 94]], [[164, 83, 91], [31, 51, 114]]]],
        dtype=torch.uint8,
    )
    stain = TileView("stain", 42, "train")
    combined = TileView("stain-rotation", 42, "train")
    turns, matrix = combined.parameters("slide", cpu)
    assert 1 <= turns <= 3
    _, stain_only = stain.parameters("slide", cpu)
    torch.testing.assert_close(matrix, stain_only)
    stained = color.clone()
    transform_batch(stained, [0], [stain.parameters("slide", cpu)])
    assert torch.equal(stained[:, 0, 0], color[:, 0, 0])
    assert not torch.equal(stained, color)
    rotated = color.clone()
    transform_batch(rotated, [0], [(turns, matrix)])
    torch.testing.assert_close(rotated, torch.rot90(stained, turns, dims=(1, 2)))
    assert TileView("rotation", 1, "val").parameters("second", cpu) == (1, None)


def _sources(tmp_path):
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    raw.mkdir()
    clean.mkdir()
    wsi = tmp_path / "slide.svs"
    wsi.touch()
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "path,split,record_num,mammaprint_index\n/data/slide.svs,train,001,0.2\n/data/unused.svs,test,002,0.1\n"
    )
    pq.write_table(
        pa.table(
            {
                "id": [1, 2],
                "path": [str(wsi), "/data/unused.svs"],
                "level": [1, 2],
                "tile_extent_x": [224, 224],
                "tile_extent_y": [224, 224],
            }
        ),
        raw / "slides.parquet",
    )
    pq.write_table(
        pa.table({"slide_id": [1, 1, 2], "tile_x": [1, 2, 3], "tile_y": [0, 0, 0]}),
        raw / "tiles.parquet",
    )
    pq.write_table(
        pa.table({"x": pa.array([2, 1], type=pa.int32()), "y": [0, 0]}),
        clean / "slide.parquet",
    )
    return Namespace(
        output=tmp_path / "selection",
        raw_tiling=str(raw),
        clean_uri=str(clean),
        data_mapping=mapping,
        split="train",
        slides=None,
        full_clean_val=False,
    )


def test_frozen_selection_rejects_split_and_coordinate_mismatch(tmp_path):
    args = _sources(tmp_path)
    mapping = args.data_mapping.read_text()
    args.data_mapping.write_text(mapping.replace(",test,002,", ",test,001,"))
    with pytest.raises(ValueError, match="records spanning"):
        prepare(args)
    args.data_mapping.write_text(mapping)
    pq.write_table(
        pa.table({"x": [2, 7], "y": [0, 0]}), Path(args.clean_uri) / "slide.parquet"
    )
    with pytest.raises(ValueError, match="unmatched historical coordinates"):
        prepare(args)
    assert not args.output.exists()


def test_training_preset_preserves_archived_fit_and_uses_clean_validation():
    import ml.train  # noqa: F401

    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        config = compose(
            config_name="ml",
            overrides=["+experiment=ml/train_mil_embeddings_l2_augmentation", "seed=2"],
        )
    assert config.test_after_fit is False
    assert config.ml.aggregator._target_ == "ml.models.aggregators.MeanPool"
    assert config.trainer.callbacks.model_checkpoint.monitor == "valid/loss"
    assert config.datamodule.train.view_seed == 2
    assert config.datamodule.train.get("train_tile_cap") is None
    assert all(
        config.datamodule[split].validate_index_split
        for split in ("train", "val", "test")
    )


def test_prepare_embed_and_verified_card(tmp_path):
    args = _sources(tmp_path)
    prepare(args)
    manifest_path = args.output / "manifest.json"
    manifest = _load_manifest(manifest_path)
    (slide,) = manifest["slides"]
    assert slide["level"] == 1
    assert slide["tile_count"] == 2
    import json

    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["pilot_slides"] == ["slide"]
    assert summary["feature_bytes_per_view"] == 2 * FEATURES * 4
    view = TileView("rotation", 42, "train")
    output = tmp_path / "view" / "production"

    class Reader:
        def __init__(self, _):
            pass

        def read_tile(self, x, _y, width, height, level):
            assert level == 1
            data = np.full((height, width, 3), x, dtype=np.uint8)
            data[0, 0, :] = 200
            return data

        def close(self):
            pass

    class Encoder(torch.nn.Module):
        def forward(self, batch):
            return batch[:, 0, 0, 0, None].expand(-1, FEATURES)

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
    worker = Namespace(batch_size=2, workers=0, prefetch=2, precision="fp32")
    with patch.dict(
        "sys.modules", {"ratiopath.openslide": SimpleNamespace(OpenSlide=Reader)}
    ):
        _infer(
            worker,
            args.output,
            output,
            [slide],
            summary,
            torch.device("cpu"),
            Encoder(),
            {},
            view=view,
        )
    table = pq.read_table(output / "embeddings/slide.parquet")
    assert table["x"].to_pylist() == [2, 1]
    values = table["embedding"].combine_chunks().values.to_numpy().reshape(-1, FEATURES)
    np.testing.assert_allclose(
        values[:, 0], (np.array([2, 1]) - 0.485 * 255) / (0.229 * 255)
    )
    run = output / "run.json"
    write_json(
        run,
        {
            "manifest_digest": manifest["digest"],
            "tile_view": view.provenance(),
            "precision": "fp32",
            "encoder_revision": "1" * 40,
        },
    )
    cfg = tmp_path / "configs"
    dataset_path = cfg / "data/embedded/rotation.yaml"
    levels = cfg / "data/levels"
    levels.mkdir(parents=True)
    (levels / "l2.yaml").write_text("tile_extent: 224\n")
    publication = Namespace(
        manifest=manifest_path,
        run_config=run,
        clean_val="/archived/val",
        clean_test="/archived/test",
        clean_train=None,
        fresh_val_manifest=None,
        augmented_uri=str(output / "embeddings"),
        output=dataset_path,
    )
    card(publication)
    saved = yaml.safe_load(dataset_path.read_text())
    assert saved["uris"]["train_augmented"] == publication.augmented_uri
    with initialize_config_dir(config_dir=str(cfg), version_base=None):
        resolved = compose(overrides=["+data/embedded=rotation"])
    assert resolved.dataset.levels["2"].tile_extent == 224
    assert resolved.dataset.levels["2"].uris.train == args.clean_uri
    (output / "complete/slide.json").write_text("{}")
    publication.output = tmp_path / "bad.yaml"
    with pytest.raises(ValueError, match="did not verify"):
        card(publication)
    assert not publication.output.exists()

    # This proves selection provenance and arguments without requiring an A40.
    dry_run = Namespace(
        manifest=manifest_path,
        output=tmp_path / "dry",
        view="stain",
        seed=42,
        encoder_revision="1" * 40,
        batch_size=1024,
        workers=8,
        prefetch=2,
        loader_timeout=300,
        pilot=None,
        pilot_slides=None,
        dry_run=True,
        mlflow_run_id=None,
        shard=0,
        precision="fp32",
    )
    assert embed(dry_run) == 0

    Path(slide["path"]).write_bytes(b"WSI changed since selection")
    with patch.dict(
        "sys.modules", {"ratiopath.openslide": SimpleNamespace(OpenSlide=Reader)}
    ):
        _infer(
            worker,
            args.output,
            tmp_path / "changed",
            [slide],
            summary,
            torch.device("cpu"),
            Encoder(),
            {},
            view=view,
        )
    assert "Source WSI changed" in summary["failures"]["slide"]
