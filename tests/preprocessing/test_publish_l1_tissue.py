import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mlflow import MlflowClient

from preprocessing.embedding_selection import TILE_SCHEMA, selection_digest
from preprocessing.publish_l1_tissue import assemble_slide, publish


def test_tissue_assembly_preserves_order_and_rejects_missing_or_duplicate_vectors():
    full = pa.Table.from_pylist(
        [
            {**dict.fromkeys(TILE_SCHEMA.names, 0.8), "x": x, "y": 0, "sample_rank": i}
            for i, x in enumerate((0, 224, 448))
        ],
        schema=TILE_SCHEMA,
    )
    pending = full.take(pa.array([1]))

    def vectors(indices):
        selected = full.take(pa.array(indices))
        array = np.array([[float(i)] * 2560 for i in indices], np.float32)
        return selected.append_column(
            "embedding",
            pa.FixedSizeListArray.from_arrays(pa.array(array.ravel()), 2560),
        )

    result = assemble_slide(full, pending, vectors([2, 0]), vectors([1]))
    assert result["x"].to_pylist() == [0, 224, 448]
    assert [row[0] for row in result["embedding"].to_pylist()] == [0, 1, 2]
    with pytest.raises(ValueError, match="Missing computed"):
        assemble_slide(full, pending, vectors([0, 2]), None)
    with pytest.raises(ValueError):
        assemble_slide(full, pending, vectors([0, 1, 2]), vectors([1]))


def test_tissue_publication_mlflow_and_resume(tmp_path: Path):
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = client.create_experiment(
        "MammaPrint", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    cancer = client.create_run(
        experiment, tags={"workflow": "l1-virchow2-embedding"}
    ).info.run_id
    prep = client.create_run(
        experiment,
        tags={"workflow": "l1-prepare-tissue-selection", "reuse_source_run_id": cancer},
    ).info.run_id
    new = client.create_run(
        experiment,
        tags={
            "workflow": "l1-virchow2-tissue-missing-embedding",
            "preparation_run_id": prep,
        },
    ).info.run_id

    def upload_json(run, artifact, data):
        directory = tmp_path / run / Path(artifact).parent
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / Path(artifact).name
        path.write_text(json.dumps(data))
        client.log_artifact(run, str(path), artifact_path=str(Path(artifact).parent))

    def sign(data):
        data["digest"] = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        return data

    settings = {
        "encoder_revision": "a" * 40,
        "normalization": "imagenet-rgb-fp32-v1",
        "precision": "fp32",
    }
    reuse = settings | {"run_id": cancer, "selection_digest": "original-selection"}
    full_rows = []
    pending_rows = []
    old_receipts = {}
    new_receipts = {}
    for split in ("train", "val", "test"):
        stem = f"{split}.scan"
        table = pa.Table.from_pylist(
            [
                {
                    **dict.fromkeys(TILE_SCHEMA.names, 0.8),
                    "x": 0,
                    "y": 0,
                    "sample_rank": 0,
                    "cancer_overlap": 0.8,
                    "epithelium_overlap": 0.1,
                },
                {
                    **dict.fromkeys(TILE_SCHEMA.names, 0.8),
                    "x": 224,
                    "y": 0,
                    "sample_rank": 1,
                    "cancer_overlap": 0.5,
                    "epithelium_overlap": 0.25,
                },
                {
                    **dict.fromkeys(TILE_SCHEMA.names, 0.8),
                    "x": 448,
                    "y": 0,
                    "sample_rank": 2,
                    "cancer_overlap": 0.1,
                    "epithelium_overlap": 0.3,
                },
            ],
            schema=TILE_SCHEMA,
        )
        for folder, picks, rows in (
            ("selection", [0, 1, 2], full_rows),
            ("missing_selection", [1, 2], pending_rows),
        ):
            selected = table.take(pa.array(picks))
            dest = tmp_path / folder / "tiles" / f"{stem}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(selected, dest)
            rows.append(
                {
                    "slide_id": stem,
                    "split": split,
                    "tile_count": len(selected),
                    "reused_tile_count": 1,
                    "selection_digest": selection_digest(selected),
                }
            )
        for run, picks, receipts in (
            (cancer, [0], old_receipts),
            (new, [1, 2], new_receipts),
        ):
            selected = table.take(pa.array(picks))
            values = np.array([[float(i)] * 2560 for i in picks], np.float32)
            selected = selected.append_column(
                "embedding",
                pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 2560),
            )
            path = tmp_path / f"{stem}.parquet"
            pq.write_table(selected, path)
            client.log_artifact(run, str(path), artifact_path="embeddings")
            receipts[stem] = {
                "size": path.stat().st_size,
                "tiles": len(selected),
                "split": split,
                "selection_digest": selection_digest(selected),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    full = sign(
        {
            "slides": full_rows,
            "reuse_source": reuse,
            "score_missing": dict.fromkeys(TILE_SCHEMA.names[3:], 0),
        }
    )
    missing = sign(
        {
            "slides": pending_rows,
            "reuse_source": reuse,
            "score_missing": dict.fromkeys(TILE_SCHEMA.names[3:], 0),
        }
    )
    for folder, data in (("selection", full), ("missing_selection", missing)):
        path = tmp_path / folder / "manifest.json"
        path.write_text(json.dumps(data))
        archive = tmp_path / "frozen.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(tmp_path / folder, arcname=folder)
        client.log_artifact(prep, str(archive), artifact_path=folder)
    upload_json(
        prep,
        "metadata/tissue_reuse_audit.json",
        {
            "full_selection_digest": full["digest"],
            "missing_selection_digest": missing["digest"],
            "total": {"selected_tiles": 9, "epithelium_or_cancer_tiles": 6},
            "by_split": {
                split: {"selected_tiles": 3, "epithelium_or_cancer_tiles": 2}
                for split in ("train", "val", "test")
            },
        },
    )
    upload_json(
        cancer,
        "metadata/run.json",
        settings | {"manifest_digest": "original-selection"},
    )
    upload_json(
        new, "metadata/run.json", settings | {"manifest_digest": missing["digest"]}
    )
    for run, receipts in ((cancer, old_receipts), (new, new_receipts)):
        upload_json(
            run,
            "metadata/l1_inventory.json",
            {"complete": receipts, "missing_or_invalid": [], "unexpected": []},
        )
    for run in (prep, cancer, new):
        client.set_terminated(run)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "path,split\n"
        + "".join(f"/{s}.scan.svs,{s}\n" for s in ("train", "val", "test"))
    )
    args = argparse.Namespace(
        preparation_run_id=prep,
        embedding_run_id=new,
        cancer_run_id=cancer,
        work_dir=tmp_path / "scratch",
        data_mapping=mapping,
        publication_run_id=None,
    )
    run = publish(client, args)
    assert client.get_run(run).info.status == "FINISHED"
    metadata = json.loads(
        Path(
            client.download_artifacts(
                run, "metadata/l1_dataset_cards.json", str(tmp_path)
            )
        ).read_text()
    )
    assert metadata["counts"]["tissue_only/embeddings"]["tiles"] == 9
    assert metadata["counts"]["epithel_or_cancer_5/embeddings"]["tiles"] == 6
    for variant in metadata["cards"]:
        for split in ("train", "val", "test"):
            assert len(client.list_artifacts(run, f"{variant}/embeddings_{split}")) == 1
    args.publication_run_id = run
    with patch(
        "preprocessing.publish_l1_tissue.assemble_slide",
        side_effect=AssertionError("must skip"),
    ):
        assert publish(client, args) == run
