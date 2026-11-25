import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from mlflow import MlflowClient

from preprocessing.embedded_mask_publication import BASE, EPITHELIAL, STRICT, publish_l1


def test_publish_scored_l1_masks_routes_and_resumes(tmp_path: Path) -> None:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = client.create_experiment(
        "MammaPrint", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    source = client.create_run(
        experiment, tags={"workflow": "l1-virchow2-embedding"}
    ).info.run_id
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "path,split\n/train.scan.svs,train\n/val.svs,val\n/test.svs,test\n"
    )
    scores = {
        "train.scan": ("train", [0.9, 0.6], [0.3, 0.25]),
        "val": ("val", [0.91, 0.7], [0.25, 0.9]),
        "test": ("test", [0.92, 0.6], [0.4, 0.2]),
    }
    slides = []
    complete = {}
    for stem, (split, cancer, epi) in scores.items():
        table = pa.table(
            {
                "x": [0, 224],
                "y": [0, 0],
                "sample_rank": [0, 1],
                "tissue_overlap": [0.8, 0.8],
                "blur_overlap": [0.9, 0.9],
                "folding_overlap": [0.9, 0.9],
                "residual_overlap": [0.9, 0.9],
                "cancer_overlap": cancer,
                "epithelium_overlap": epi,
                "embedding": [[1.0], [2.0]],
            }
        )
        parquet = tmp_path / f"{stem}.parquet"
        pq.write_table(table, parquet)
        client.log_artifact(source, str(parquet), artifact_path="embeddings")
        slides.append(
            {
                "slide_id": stem,
                "split": split,
                "tile_count": 2,
                "selection_digest": stem,
            }
        )
        complete[stem] = {
            "split": split,
            "tiles": 2,
            "selection_digest": stem,
            "size": parquet.stat().st_size,
            "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        }

    manifest = {
        "level": 1,
        "source_gates": {"cancer_overlap": 0.5},
        "score_missing": dict.fromkeys(
            (
                "tissue_overlap",
                "blur_overlap",
                "folding_overlap",
                "residual_overlap",
                "cancer_overlap",
                "epithelium_overlap",
            ),
            0,
        ),
        "data_mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
        "slides": slides,
        "cap": 3000,
        "seed": 42,
    }
    manifest["digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    for name, folder, data in (
        ("manifest.json", "selection", manifest),
        (
            "l1_inventory.json",
            "metadata",
            {
                "complete": complete,
                "missing_or_invalid": [],
                "unexpected": [],
                "zero_eligible_tiles": [],
            },
        ),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        client.log_artifact(source, str(path), artifact_path=folder)
    client.set_terminated(source)

    publication = publish_l1(
        client, source, None, mapping, tmp_path / "scratch", "MammaPrint"
    )
    assert client.get_run(publication).info.status == "FINISHED"
    cards = json.loads(
        Path(
            client.download_artifacts(
                publication, "metadata/l1_dataset_cards.json", str(tmp_path)
            )
        ).read_text()
    )
    assert (
        cards["cards"][BASE]["uris"]["all"]
        == client.get_run(source).info.artifact_uri + "/embeddings"
    )
    assert cards["counts"][f"{STRICT}/embeddings_train"] == {
        "tiles": 0,
        "nonempty_slides": 0,
        "empty_slides": 1,
    }
    assert cards["counts"][f"{STRICT}/embeddings"]["tiles"] == 2
    assert cards["counts"][f"{EPITHELIAL}/embeddings"]["tiles"] == 3
    for variant in (BASE, STRICT, EPITHELIAL):
        for stem, (split, _, _) in scores.items():
            path = f"{variant}/embeddings_{split}"
            assert [item.path for item in client.list_artifacts(publication, path)] == [
                f"{path}/{stem}.parquet"
            ]
    with patch(
        "preprocessing.embedded_mask_publication._publish_slide",
        side_effect=AssertionError,
    ):
        assert (
            publish_l1(
                client, source, publication, mapping, tmp_path / "scratch", "MammaPrint"
            )
            == publication
        )
