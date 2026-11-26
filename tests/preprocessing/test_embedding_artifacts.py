import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from mlflow import MlflowClient

from preprocessing.create_l1_dataset import _selection
from preprocessing.embedding_artifacts import EmbeddingArtifacts


def test_uploaded_slides_resume_from_mlflow_without_local_embeddings(tmp_path: Path):
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.create_experiment(
        "MammaPrint", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run_id = client.create_run(experiment).info.run_id
    slide = {
        "slide_id": "sample.scan",
        "split": "val",
        "tile_count": 2,
        "selection_digest": "selection-sha",
    }
    config = {"manifest_digest": "manifest-sha", "precision": "fp32"}
    stage = tmp_path / "initial"
    store = EmbeddingArtifacts(client, run_id, stage)
    store.prepare(config)
    assert not store.complete(slide)

    parquet = stage / "sample.scan.parquet"
    receipt = stage / "sample.scan.json"
    parquet.write_bytes(b"synthetic embedding")
    receipt.write_text(
        json.dumps(
            {"tiles": 2, "selection_digest": "selection-sha", "sha256": "content-sha"}
        )
    )
    store.upload(parquet, receipt)
    assert not parquet.exists()
    assert not receipt.exists()

    restart = tmp_path / "restart"
    resumed = EmbeddingArtifacts(client, run_id, restart)
    resumed.prepare(config)
    assert resumed.complete(slide)
    assert not resumed.complete(slide | {"selection_digest": "different-selection"})
    report = restart / "l1_inventory.json"
    assert resumed.report(report, [slide])
    assert json.loads(report.read_text())["complete"]["sample.scan"]["split"] == "val"

    parquet.write_bytes(b"incorrect-size")
    client.log_artifact(run_id, str(parquet), artifact_path="embeddings")
    assert not resumed.report(report, [slide])
    assert json.loads(report.read_text())["missing_or_invalid"] == ["sample.scan"]

    receipt.write_text("{")
    client.log_artifact(run_id, str(receipt), artifact_path="complete")
    resumed.prepare(config)
    assert not resumed.complete(slide)


def test_frozen_selection_is_recovered_from_mlflow(tmp_path: Path):
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.create_experiment(
        "MammaPrint", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run_id = client.create_run(experiment).info.run_id
    args = Namespace(
        tiling="source", data_mapping=tmp_path / "mapping.csv", epithelium_masks="masks"
    )
    first, restart = tmp_path / "first", tmp_path / "restart"
    first.mkdir()
    restart.mkdir()

    def create_fixture(*, output: Path, **_):
        (output / "tiles").mkdir(parents=True)
        (output / "manifest.json").write_text('{"digest":"frozen"}')
        (output / "tiles" / "slide.parquet").write_bytes(b"frozen selection")

    with patch(
        "preprocessing.create_l1_dataset.prepare_selection", side_effect=create_fixture
    ) as prepare:
        original = _selection(args, client, run_id, first)
        restored = _selection(args, client, run_id, restart)
        prepare.assert_called_once()
    assert restored.read_bytes() == original.read_bytes()
    assert (
        restored.parent / "tiles" / "slide.parquet"
    ).read_bytes() == b"frozen selection"
