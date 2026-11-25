import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from preprocessing.l2_augmentation import prepare
from preprocessing.resolution_embeddings import FEATURES, _publish_slide, write_json
from preprocessing.tile_augmentation import TileView
from scripts.preprocessing import score_l2_rotation_probe as probe


def _archived_config(uri: str) -> dict:
    level = {2: {"uris": {"train": "train", "val": uri}}}
    return {
        "seed": 2,
        "label_mode": "index",
        "feature_dim": FEATURES,
        "dataset": {"levels": level},
        "datamodule": {
            split: {
                "_target_": "ml.data.SingleScaleDataset",
                "split": split,
                "label_mode": "index",
                "levels": level,
                "data_mapping": "/original/data_mapping.csv",
            }
            for split in ("train", "val")
        },
        "ml": {
            "encoder": {"_target_": "ml.models.encoders.IdentityEncoder"},
            "aggregator": {"_target_": "ml.models.aggregators.MeanPool"},
            "head": {
                "_target_": "ml.models.heads.LinearHead",
                "in_dim": FEATURES,
                "out_dim": 1,
            },
            "output_activation": {"_target_": "torch.nn.Identity"},
            "loss": {"_target_": "torch.nn.MSELoss"},
            "metrics": {
                "thresholded_f1": {
                    "_target_": "ml.metrics.thresholded_f1",
                    "threshold": 0,
                }
            },
        },
        "trainer": {
            "callbacks": {
                "model_checkpoint": {
                    "monitor": "valid/loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "filename": "best",
                }
            }
        },
    }


def _probe_inputs(root: Path) -> Namespace:
    clean = root / "archive"
    raw = root / "raw"
    clean.mkdir()
    raw.mkdir()
    for stem in "ab":
        (raw / f"{stem}.svs").touch()
    mapping = root / "mapping.csv"
    mapping.write_text(
        "path,record_num,split,mammaprint_index\n"
        f"{raw / 'a.svs'},one,val,0.25\n{raw / 'b.svs'},two,val,-0.25\n"
        f"{raw / 'c.svs'},three,test,0.9\n"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": index,
                    "path": str(raw / f"{stem}.svs"),
                    "level": 1,
                    "tile_extent_x": 224,
                    "tile_extent_y": 224,
                }
                for index, stem in enumerate("ab")
            ]
        ),
        raw / "slides.parquet",
    )
    pq.write_table(
        pa.table({"slide_id": [0, 0, 1, 1], "x": [0, 112, 0, 112], "y": [0] * 4}),
        raw / "tiles.parquet",
    )
    for stem, mean in (("a", 0.25), ("b", -0.25)):
        vectors = [
            [mean - 0.125] + [0.0] * (FEATURES - 1),
            [mean + 0.125] + [0.0] * (FEATURES - 1),
        ]
        pq.write_table(
            pa.table(
                {
                    "x": pa.array([112, 0], type=pa.int32()),
                    "y": pa.array([0, 0], type=pa.int32()),
                    "embedding": pa.array(vectors, type=pa.list_(pa.float32())),
                }
            ),
            clean / f"{stem}.parquet",
            row_group_size=1,
        )
    selection = root / "selection"
    prepare(
        Namespace(
            output=selection,
            clean_uri=str(clean),
            raw_tiling=str(raw),
            data_mapping=mapping,
            split="val",
            slides=["b", "a"],
            full_clean_val=False,
        )
    )
    manifest_path = selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fresh_clean, fresh_rotation = root / "fresh", root / "rotated"
    for kind, output in (("clean", fresh_clean), ("rotation", fresh_rotation)):
        write_json(
            output / "run.json",
            {
                "tile_view": TileView(kind, 11, "val").provenance(),
                "encoder_revision": "a" * 40,
                "runtime_id": "recorded-runtime",
                "manifest_digest": manifest["digest"],
                "precision": "fp32",
                "normalization": "imagenet-rgb-fp32-v1",
            },
        )
        for slide, mean, turn_shift in zip(
            manifest["slides"], (0.25, -0.25), (0.125, -0.125), strict=True
        ):
            values = np.zeros((2, FEATURES), dtype=np.float32)
            values[:, 0] = np.array([mean - 0.125, mean + 0.125]) + 0.015625
            if kind == "rotation":
                values[:, 0] += turn_shift
            _publish_slide(selection, output, slide, values)
    weight = torch.zeros(1, FEATURES)
    weight[0, 0] = 1
    torch.save(
        {
            "state_dict": {
                "head.linear.weight": weight,
                "head.linear.bias": torch.zeros(1),
            },
            "global_step": 6,
            "callbacks": {"best": {"monitor": "valid/loss", "best_model_score": 0.1}},
        },
        root / "best.ckpt",
    )
    (root / "resolved.yaml").write_text(yaml.safe_dump(_archived_config(str(clean))))
    write_json(
        root / "history.json",
        {
            "valid/loss": [{"step": 0, "value": 0.2}, {"step": 5, "value": 0.1}],
            "valid/mae": [{"step": 5, "value": 0.25}],
            "valid/thresholded_f1": [{"step": 5, "value": 0.75}],
        },
    )
    spec = root / "head.json"
    write_json(
        spec,
        {
            "checkpoint": "best.ckpt",
            "config": "resolved.yaml",
            "metric_history": "history.json",
        },
    )
    return Namespace(
        manifest=manifest_path,
        clean_output=fresh_clean,
        rotation_output=fresh_rotation,
        data_mapping=mapping,
        local_spec=[spec],
        run_id=[],
        parity_max_abs_drift=0.02,
        parity_max_p99_relative_l2=0.15,
        rotation_min_mean_abs_shift=0.1,
    )


def test_score_reports_paired_absolute_changes_without_slide_predictions(
    tmp_path, monkeypatch
):
    args = _probe_inputs(tmp_path)
    result = tmp_path / "aggregate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_l2_rotation_probe",
            "--manifest",
            str(args.manifest),
            "--clean-output",
            str(args.clean_output),
            "--rotation-output",
            str(args.rotation_output),
            "--data-mapping",
            str(args.data_mapping),
            "--local-spec",
            str(args.local_spec[0]),
            "--parity-max-abs-drift",
            "0.02",
            "--parity-max-p99-relative-l2",
            "0.15",
            "--rotation-min-mean-abs-shift",
            "0.1",
            "--output",
            str(result),
        ],
    )
    probe.main()
    report = json.loads(result.read_text())
    assert report["slides"] == 2
    assert report["tiles"] == 4
    assert report["clean_embedding_parity"]
    assert report["clean_prediction_parity_all_seeds"]
    assert report["clean_parity_all_seeds"]
    assert report["rotation_material_all_seeds"]
    assert report["clean_embedding_relative_l2"]["p95"] == pytest.approx(0.125)
    assert report["clean_embedding_relative_l2"]["p99"] == pytest.approx(0.125)
    seed = report["seeds"][0]
    assert seed["seed"] == 2
    assert seed["validation_subset_mae"] == {
        "archived": 0.0,
        "fresh_clean": 0.015625,
        "fresh_rot90": 0.125,
    }
    assert seed["fresh_clean_to_rot90_prediction_shift"] == {
        "signed_mean": 0.0,
        "mean_absolute": 0.125,
        "median_absolute": 0.125,
        "max_absolute": 0.125,
    }
    assert '"slide_id"' not in result.read_text()
    args.parity_max_abs_drift = 0.01
    assert not probe.score(args)["rotation_material_all_seeds"]


def test_tile_vector_parity_cannot_pass_on_prediction_agreement_alone(tmp_path):
    args = _probe_inputs(tmp_path)
    slide = json.loads(args.manifest.read_text())["slides"][0]
    values = np.zeros((2, FEATURES), dtype=np.float32)
    values[:, 0] = [0.140625, 0.390625]
    values[:, 1] = 1  # The head has zero weight on this disturbed feature.
    _publish_slide(args.manifest.parent, args.clean_output, slide, values)

    report = probe.score(args)
    assert report["clean_prediction_parity_all_seeds"]
    assert (
        report["clean_embedding_relative_l2"]["p99"] > args.parity_max_p99_relative_l2
    )
    assert not report["clean_embedding_parity"]
    assert not report["clean_parity_all_seeds"]
    assert not report["rotation_material_all_seeds"]


def test_probe_rejects_mismatched_runtime_and_archived_coordinate_order(tmp_path):
    args = _probe_inputs(tmp_path)
    path = args.rotation_output / "run.json"
    config = json.loads(path.read_text())
    write_json(path, {**config, "runtime_id": "different"})
    with pytest.raises(ValueError, match="different encoder or runtime"):
        probe.score(args)
    write_json(path, config)
    original = tmp_path / "archive" / "a.parquet"
    table = pq.read_table(original)
    pq.write_table(table.take(pa.array([1, 0])), original)
    with pytest.raises(ValueError, match="Ordered tile coordinates differ"):
        probe.score(args)
