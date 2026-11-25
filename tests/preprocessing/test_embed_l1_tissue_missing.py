import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from mlflow import MlflowClient

from preprocessing.embed_l1_tissue_missing import _prepare


def test_gpu_preparation_requires_finished_and_pinned_missing_selection(tmp_path: Path):
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = client.create_experiment(
        "MammaPrint", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    original = "cancer-run"
    source = client.create_run(
        experiment,
        tags={
            "workflow": "l1-prepare-tissue-selection",
            "reuse_source_run_id": original,
        },
    ).info.run_id
    encoder = "a" * 40
    root = tmp_path / "missing_selection"
    root.mkdir()
    manifest = {
        "level": 1,
        "selection_kind": "missing_tissue_embeddings",
        "shards": 1,
        "cap": 3000,
        "reuse_source": {
            "run_id": original,
            "encoder_revision": encoder,
            "precision": "fp32",
            "normalization": "imagenet-rgb-fp32-v1",
        },
        "score_missing": {"cancer_overlap": 0, "epithelium_overlap": 0},
        "slides": [
            {
                "slide_id": "pending",
                "shard": 0,
                "tile_count": 1,
                "full_tile_count": 3,
                "reused_tile_count": 2,
            },
            {
                "slide_id": "shared",
                "shard": 0,
                "tile_count": 0,
                "full_tile_count": 1,
                "reused_tile_count": 1,
            },
        ],
    }
    manifest["digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    frozen = tmp_path / "frozen.tar.gz"
    with tarfile.open(frozen, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    audit = tmp_path / "tissue_reuse_audit.json"
    audit.write_text(
        json.dumps(
            {
                "missing_selection_digest": manifest["digest"],
                "score_missing": manifest["score_missing"],
                "total": {"missing_embeddings": 1},
            }
        )
    )
    client.log_artifact(source, str(audit), artifact_path="metadata")
    client.log_artifact(source, str(frozen), artifact_path="missing_selection")
    scratch = tmp_path / "before-complete"
    scratch.mkdir()
    with pytest.raises(ValueError, match="not a finished L1 tissue preparation"):
        _prepare(client, source, experiment, encoder, scratch)

    client.set_terminated(source)
    scratch = tmp_path / "wrong-weights"
    scratch.mkdir()
    with pytest.raises(ValueError, match="saved encoder settings differ"):
        _prepare(client, source, experiment, "b" * 40, scratch)

    scratch = tmp_path / "ready"
    scratch.mkdir()
    selected, archive, digest, tiles = _prepare(
        client, source, experiment, encoder, scratch
    )
    assert selected == scratch / "missing_selection" / "manifest.json"
    assert archive.is_file()
    assert digest == manifest["digest"]
    assert tiles == 1
