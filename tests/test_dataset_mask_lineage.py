from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from ml.explainability.dataset_mask_lineage import MaskSources, resolve_mask_sources


_REPORT = "a" * 32
_SPLIT = "b" * 32
_WHOLE = "c" * 32
_TILING = "d" * 32
_MASK = "e" * 32
_FILTER = "f" * 32


def uri(run: str, artifact: str) -> str:
    return f"mlflow-artifacts:/3/{run}/artifacts/{artifact}"


def config_for(split: str, selected: str, whole: str | None = None) -> dict[str, Any]:
    return {
        "datamodule": {
            split: {
                "split": split,
                "levels": {"2": {"uris": {split: selected, "all": whole}}},
            }
        }
    }


def digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class MockClient:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.tags: dict[str, dict[str, str]] = {}
        self.reads: list[tuple[str, str]] = []

    def config(self, run: str, value: dict[str, Any]) -> None:
        self.files[run, "configs/config-resolved.yaml"] = yaml.safe_dump(value).encode()

    def get_run(self, run: str) -> SimpleNamespace:
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=run, experiment_id="3", lifecycle_stage="active"
            ),
            data=SimpleNamespace(tags=self.tags.get(run, {})),
        )

    def list_artifacts(
        self, run: str, path: str | None = None
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                path=file, is_dir=False, file_size=len(self.files[run_id, file])
            )
            for run_id, file in self.files
            if run_id == run
            and (
                (path is None and "/" not in file)
                or (path is not None and str(Path(file).parent) == path)
            )
        ]

    def download_artifacts(self, run: str, path: str, dst: str) -> str:
        self.reads.append((run, path))
        if (run, path) not in self.files:
            raise FileNotFoundError(path)
        location = Path(dst) / "download"
        location.write_bytes(self.files[run, path])
        return str(location)


class DatasetMaskLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MockClient()
        self.tiling: dict[str, Any] = {
            "data_name": "tiles",
            "level": 2,
            "dataset": {
                "paths": {
                    "qc_masks": "/mnt/masks/qc",
                    "epithelium_masks": "/mnt/masks/epi",
                },
                "mlflow_uris": {
                    "tissue_masks": uri(_MASK, "tissue"),
                    "cancer_masks": uri(_MASK, "cancer"),
                },
            },
            "blur_threshold": 0.5,
            "folding_threshold": 0.1,
            "residual_threshold": 0.5,
            "tissue_threshold": 0.25,
            "epithelium_threshold": None,
            "cancer_threshold": 0.5,
        }

    def resolve(self, config: dict[str, Any], **overrides: str) -> MaskSources:
        provenance = {
            "explanation_run_id": _REPORT,
            "level": 2,
            "split": "test",
            "config_hash": digest(config),
            **overrides,
        }
        return resolve_mask_sources(
            self.client, provenance, yaml.safe_dump(config).encode()
        )

    def test_exact_frozen_split_card_links_export_to_tiling(self) -> None:
        selected = uri(_SPLIT, "cancer_test")
        whole = uri(_WHOLE, "embeddings")
        self.client.tags[_SPLIT] = {"mlflow.runName": "split-cancer_test"}
        self.client.files[_SPLIT, "cancer_test/mock-slide.parquet"] = b"mock-embedding"
        self.client.files[_WHOLE, "embeddings/mock-slide.parquet"] = b"mock-embedding"
        self.client.files[_WHOLE, "embeddings/other-slide.parquet"] = (
            b"mock-other-embedding"
        )
        self.client.config(_WHOLE, {"dataset": {"path": uri(_TILING, "tiles")}})
        self.client.config(_TILING, self.tiling)
        result = self.resolve(config_for("test", selected, whole))
        self.assertEqual(result.dataset_uri, selected)
        self.assertIsNone(result.unresolved)
        self.assertEqual(
            set(result.layers), {"tissue", "blur", "folding", "residual", "cancer"}
        )
        self.assertEqual(result.layers["tissue"].location, uri(_MASK, "tissue"))
        self.assertIn("> 0.25", result.layers["tissue"].selection)
        self.assertIn("clean fraction", result.layers["blur"].selection)
        self.assertIn("cancer overlap > 0.5", result.layers["cancer"].selection)
        self.assertIn(
            "matched by filenames and sizes", result.layers["cancer"].evidence
        )
        self.client.files[_SPLIT, "cancer_test/mock-slide.parquet"] = b"unexpected"
        result = self.resolve(config_for("test", selected, whole))
        self.assertEqual(result.layers, {})
        self.assertIn("inventory", result.unresolved or "")

    def test_filter_manifest_affords_both_sources_and_preserves_or(self) -> None:
        self.tiling["cancer_threshold"] = None
        self.client.config(_TILING, self.tiling)
        self.client.files[_FILTER, "filter_manifest.json"] = json.dumps(
            {
                "schema_version": 1,
                "level": 2,
                "strategy": "or",
                "comparison": ">",
                "epithelium_threshold": 0.25,
                "cancer_threshold": 0.5,
                "source_tiling_uri": uri(_TILING, "tiles"),
                "source_embeddings_uri": uri(_WHOLE, "embeddings"),
            }
        ).encode()
        result = self.resolve(config_for("test", uri(_FILTER, "embeddings_test")))
        self.assertIsNone(result.unresolved)
        for layer in ("epithelium", "cancer"):
            self.assertIn(
                "epithelium overlap > 0.25 OR cancer overlap > 0.5",
                result.layers[layer].selection,
            )

    def test_recorded_but_unapplied_scores_do_not_enable_layers(self) -> None:
        self.tiling["cancer_threshold"] = None
        self.client.config(_TILING, self.tiling)
        result = self.resolve(config_for("test", uri(_TILING, "tiles")))
        self.assertIsNone(result.unresolved)
        self.assertEqual(set(result.layers), {"tissue", "blur", "folding", "residual"})

    def test_optional_tiling_mask_needs_a_source_to_run_its_gate(self) -> None:
        self.tiling["dataset"]["mlflow_uris"]["cancer_masks"] = None
        self.client.config(_TILING, self.tiling)
        result = self.resolve(config_for("test", uri(_TILING, "tiles")))
        self.assertIsNone(result.unresolved)
        self.assertNotIn("cancer", result.layers)

    def test_resolution_inventory_binds_l1_to_original_tiling(self) -> None:
        selected = uri(_SPLIT, "embeddings_test")
        config = config_for("test", selected)
        card = config["datamodule"]["test"]["levels"].pop("2")
        config["datamodule"]["test"]["levels"]["1"] = card
        card["provenance"] = uri(_SPLIT, "metadata/l1_inventory.json")

        self.tiling["level"] = 1
        self.client.config(_TILING, self.tiling)
        source = {
            "schema_version": 1,
            "level": 1,
            "tiling_source": uri(_TILING, "tiles"),
            "source_gates": {"cancer_overlap": 0.5},
            "recovered_score_sources": {"epithelium_overlap": "/mnt/masks/epi"},
        }
        source_hash = hashlib.sha256(
            json.dumps(source, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        inventory: dict[str, Any] = {
            "source_manifest_digest": source_hash,
            "embedding_config": {"manifest_digest": source_hash},
            "source_selection": {**source, "digest": source_hash},
            "successful": {"test": {"mock-slide": "0" * 64}},
        }
        self.client.tags[_SPLIT] = {"resolution_manifest": source_hash}
        self.client.files[_SPLIT, "metadata/l1_inventory.json"] = json.dumps(
            inventory
        ).encode()
        result = self.resolve(config, level="1")
        self.assertIsNone(result.unresolved)
        self.assertIn("cancer", result.layers)
        self.assertNotIn("epithelium", result.layers)
        self.assertIn("metadata/l1_inventory.json", result.layers["cancer"].evidence)
        inventory["source_selection"]["tiling_source"] = uri(_WHOLE, "wrong")
        self.client.files[_SPLIT, "metadata/l1_inventory.json"] = json.dumps(
            inventory
        ).encode()
        bad = self.resolve(config, level="1")
        self.assertEqual(bad.layers, {})
        self.assertIn("inventory", bad.unresolved or "")
        self.assertFalse(any(run == _WHOLE for run, _ in self.client.reads))

    def test_snapshot_requires_hash_even_for_the_exact_run(
        self,
    ) -> None:
        config = config_for("test", uri(_TILING, "tiles"))
        snapshot = yaml.safe_dump(config).encode()
        provenance = {"explanation_run_id": _REPORT, "split": "test", "level": 2}
        rejected = resolve_mask_sources(self.client, provenance, snapshot)
        self.assertIn("hash", rejected.unresolved or "")
        self.assertEqual(rejected.layers, {})
        self.assertEqual(self.client.reads, [])
        self.client.files[_REPORT, "explanations/configs/config-resolved.yaml"] = (
            snapshot
        )
        self.client.config(_TILING, self.tiling)
        unbound = resolve_mask_sources(self.client, provenance, None)
        self.assertIsNotNone(unbound.unresolved)
        self.assertEqual(unbound.layers, {})
        bound = resolve_mask_sources(
            self.client, {**provenance, "config_hash": digest(config)}, None
        )
        self.assertIsNone(bound.unresolved)
        self.assertIn("cancer", bound.layers)

    def test_partial_sources_report_absence_without_discarding_others(self) -> None:
        self.tiling["dataset"]["mlflow_uris"]["tissue_masks"] = None
        self.client.config(_TILING, self.tiling)
        result = self.resolve(config_for("test", uri(_TILING, "tiles")))
        self.assertIn("tissue", result.unresolved or "")
        self.assertIn("cancer", result.layers)
        self.assertNotIn("tissue", result.layers)

    def test_bad_config_hash_is_rejected_before_mlflow_reads(self) -> None:
        result = self.resolve(
            config_for("test", uri(_TILING, "tiles")), config_hash="0" * 64
        )
        self.assertIn("hash", result.unresolved or "")
        self.assertEqual(result.layers, {})
        self.assertEqual(self.client.reads, [])

    def test_unrelated_split_run_cannot_inherit_an_all_pointer(self) -> None:
        selected = uri(_SPLIT, "cancer_test")
        result = self.resolve(config_for("test", selected, uri(_WHOLE, "embeddings")))
        self.assertEqual(result.dataset_uri, selected)
        self.assertIsNotNone(result.unresolved)
        self.assertFalse(any(run == _WHOLE for run, _ in self.client.reads))

    def test_downloads_explanation_snapshot_and_resolves_only_its_split(self) -> None:
        selected = uri(_TILING, "tiles")
        config = config_for("test", selected)
        config["datamodule"]["val"] = config_for("val", uri(_WHOLE, "wrong"))[
            "datamodule"
        ]["val"]
        artifact = "explanations/configs/config-resolved.yaml"
        self.client.files[_REPORT, artifact] = yaml.safe_dump(config).encode()
        self.client.config(_TILING, self.tiling)
        result = resolve_mask_sources(
            self.client,
            {
                "explanation_run_id": _REPORT,
                "split": "test",
                "level": 2,
                "config_hash": digest(config),
            },
            None,
        )
        self.assertIsNone(result.unresolved)
        self.assertIn((_REPORT, artifact), self.client.reads)
        self.assertFalse(any(run == _WHOLE for run, _ in self.client.reads))

    def test_training_config_fallback_checks_both_hashes(self) -> None:
        selected = uri(_TILING, "tiles")
        config = config_for("test", selected)
        config.update(
            {
                "seed": 1,
                "label_mode": "index",
                "feature_dim": 2560,
                "dataset": {"name": "example"},
                "ml": {
                    "_target_": "Example",
                    "encoder": {},
                    "aggregator": {},
                    "head": {},
                    "output_activation": {},
                },
            }
        )
        critical = {
            key: config[key]
            for key in ("seed", "label_mode", "feature_dim", "dataset", "datamodule")
        }
        critical.update({f"ml.{key}": val for key, val in config["ml"].items()})
        self.client.config(_WHOLE, config)
        self.client.config(_TILING, self.tiling)
        source_artifact = "configs/config-resolved.yaml"
        raw_digest = hashlib.sha256(
            self.client.files[_WHOLE, source_artifact]
        ).hexdigest()
        provenance = {
            "explanation_run_id": _REPORT,
            "source_run_id": _WHOLE,
            "source_config_artifact": source_artifact,
            "source_config_sha256": raw_digest,
            "source_critical_config_hash": digest(critical),
            "split": "test",
            "level": 2,
        }
        result = resolve_mask_sources(self.client, provenance, None)
        self.assertIsNone(result.unresolved)
        self.assertEqual(result.dataset_uri, selected)
        for field in ("source_config_sha256", "source_critical_config_hash"):
            result = resolve_mask_sources(
                self.client, {**provenance, field: "0" * 64}, None
            )
            self.assertEqual(result.layers, {})
            self.assertIn("verify", result.unresolved or "")

    def test_bad_artifact_pointer_is_never_downloaded(self) -> None:
        result = self.resolve(config_for("test", uri(_TILING, "../tiles")))
        self.assertIsNotNone(result.unresolved)
        self.assertEqual(self.client.reads, [])


if __name__ == "__main__":
    unittest.main()
