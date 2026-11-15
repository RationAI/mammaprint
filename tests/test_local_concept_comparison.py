"""Offline comparison contracts using synthetic local inputs only."""

import hashlib
import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.ml import run_local_concept_comparison as comparison


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


def cached_fixture(root):
    """Actual local parquet/mapping/checkpoint inputs, matching complete pilot counts."""
    config = OmegaConf.create(
        {
            "label_mode": "type",
            "feature_dim": 2560,
            "dataset": {
                "levels": {
                    "3": {
                        "mpp": 1.94,
                        "tile_extent": 224,
                        "stride": 112,
                        "uris": {
                            s: f"mlflow-artifacts:/3/embed/artifacts/{s}"
                            for s in ("train", "val", "test")
                        },
                    }
                }
            },
            "ml": {
                "_target_": "ml.models.module.MammaprintModule",
                "encoder": {
                    "_target_": "ml.models.encoders.identity.IdentityEncoder",
                    "out_dim": 2560,
                },
                "aggregator": {
                    "_target_": "ml.models.aggregators.mean.MeanPool",
                    "feature_dim": 2560,
                },
                "head": {
                    "_target_": "ml.models.heads.linear.LinearHead",
                    "in_dim": 2560,
                    "out_dim": 1,
                    "dropout": 0.5,
                },
            },
        }
    )
    OmegaConf.save(config, root / "source.yaml")
    module = instantiate(config.ml).eval()
    torch.save({"state_dict": module.state_dict()}, root / "checkpoint.ckpt")
    assets, metadata = [], []
    for split, counts in (("train", [122, 172]), ("val", [171, 168])):
        directory = root / split
        directory.mkdir()
        for i, count in enumerate(counts):
            sid = f"{split}{i}"
            path = directory / f"{sid}.parquet"
            pd.DataFrame(
                {
                    "x": np.arange(count),
                    "y": np.zeros(count, dtype=int),
                    "embedding": list(
                        np.random.default_rng(i)
                        .normal(size=(count, 2560))
                        .astype(np.float32)
                    ),
                }
            ).to_parquet(path)
            assets.append(
                {
                    "slide_id": sid,
                    "split": split,
                    "source_uri": f"mlflow-artifacts:/3/embed/artifacts/{split}/{sid}.parquet",
                    "local_path": str(path),
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                    "rows": count,
                }
            )
            metadata.append(
                {
                    "path": f"/mnt/data/{sid}.mrxs",
                    "name": sid,
                    "type": "a luminal",
                    "mammaprint_index": 0.2,
                    "level": 3,
                    "extent_x": 1000,
                    "extent_y": 1000,
                    "downsample": 8.0,
                }
            )
        pd.DataFrame([m for m in metadata if m["name"].startswith(split)]).to_csv(
            root / f"{split}-mapping.csv", index=False
        )
    pd.DataFrame(metadata).drop(columns="name").to_parquet(root / "slides.parquet")
    write_json(root / "assets.json", assets)
    baseline = {
        "status": "completed_smoke_only",
        "smoke_only": True,
        "fixture_only": True,
        "scientific_validation": False,
        "source_run_id": "source",
        "training_sha": "a" * 40,
        "checkpoint_sha256": sha(root / "checkpoint.ckpt"),
        "source_config_sha256": sha(root / "source.yaml"),
        "labels_geometry_sha256": sha(root / "slides.parquet"),
        "assets_manifest_sha256": sha(root / "assets.json"),
        "labels_geometry_uri": "mlflow-artifacts:/3/tiling/artifacts/slides.parquet",
        "selected_assets": assets,
        "original_memberships": {
            "train": ["train0", "train1"],
            "val": ["val0", "val1"],
            "test": ["test0"],
        },
        "loaded_bag_rows": {
            "train": {"train0": 122, "train1": 172},
            "val": {"val0": 171, "val1": 168},
        },
        "execution": {
            "device": "cpu",
            "log_to_mlflow": False,
            "n_components": 16,
            "use_ray": False,
        },
        "split_audit": {"patient_audit": "unverified"},
    }
    write_json(root / "baseline.json", baseline)
    wsis = []
    for sid in ("val0", "val1"):
        tree = root / f"{sid}-wsi"
        tree.mkdir()
        (tree / f"{sid}.mrxs").write_bytes(b"synthetic")
        sidecars = tree / sid
        sidecars.mkdir()
        for name in ("Index.dat", "Slidedat.ini"):
            (sidecars / name).write_bytes(b"synthetic")
        wsis.append(
            {
                "slide_id": sid,
                "local_path": str(tree / f"{sid}.mrxs"),
                "extraction_root": str(tree),
                "tree_sha256": "f" * 64,
                "n_files": 3,
            }
        )
    reuse = {
        "source_run_id": "source",
        "training_sha": "a" * 40,
        "checkpoint": str(root / "checkpoint.ckpt"),
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "source_config": str(root / "source.yaml"),
        "source_config_sha256": baseline["source_config_sha256"],
        "original_tiling_metadata": str(root / "slides.parquet"),
        "original_tiling_metadata_sha256": baseline["labels_geometry_sha256"],
        "assets_manifest": str(root / "assets.json"),
        "assets_manifest_sha256": baseline["assets_manifest_sha256"],
        "local_bags": {s: str(root / s) for s in ("train", "val")},
        "local_label_mapping": {
            s: str(root / f"{s}-mapping.csv") for s in ("train", "val")
        },
        "local_validation_wsis": wsis,
        "baseline_manifest": str(root / "baseline.json"),
        "baseline_dictionary": str(root / "baseline-dictionary"),
        "baseline_validation": str(root / "baseline-validation"),
        "baseline_dictionary_id": "d" * 64,
    }
    write_json(root / "reuse.json", reuse)
    return reuse, baseline, assets


def command(root):
    return [
        "--reuse-inputs",
        str(root / "reuse.json"),
        "--reuse-inputs-sha256",
        sha(root / "reuse.json"),
        "--baseline-manifest-sha256",
        sha(root / "baseline.json"),
        "--output-root",
        str(root / "output"),
    ]


class OfflineComparisonTests(unittest.TestCase):
    def test_dry_run_has_no_heavy_imports_network_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cached_fixture(root)
            with (
                patch.object(
                    comparison, "execute", side_effect=AssertionError("execute")
                ),
                patch("socket.socket.connect", side_effect=AssertionError("network")),
                redirect_stdout(io.StringIO()),
            ):
                result = comparison.main(command(root))
            self.assertFalse((root / "output").exists())
            self.assertEqual(result["mode"], "plan")
            self.assertEqual(result["components"], [16, 8, 32])
            self.assertFalse(result["asset_hashes_verified"])

    def test_output_overlap_and_nonempty_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cached_fixture(root)
            for output in (root, root / "train" / "results"):
                args = command(root)
                args[-1] = str(output)
                with self.assertRaises(ValueError):
                    comparison.main(args)
            (root / "output").mkdir()
            (root / "output" / "keep").write_text("preserve")
            with self.assertRaises(ValueError):
                comparison.main(command(root))
            self.assertEqual((root / "output" / "keep").read_text(), "preserve")

    def test_socket_guard_blocks_and_restores_network(self):
        original = socket.getaddrinfo
        with comparison.offline():
            with self.assertRaisesRegex(RuntimeError, "offline"):
                socket.getaddrinfo("example.invalid", 443)
            with (
                socket.socket() as client,
                self.assertRaisesRegex(RuntimeError, "offline"),
            ):
                client.connect(("127.0.0.1", 9))
        self.assertIs(socket.getaddrinfo, original)

    def test_local_loading_preserves_all_rows_original_config_and_wsi_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            reuse, baseline, assets = cached_fixture(Path(temp))
            with (
                patch("ml.concept_explainability.model.verify_training_revision"),
                comparison.offline(),
            ):
                source, evidence = comparison.prepare_source(reuse, baseline, assets)
            # Compare the adapter with existing pilot preparation, substituting
            # only its transport boundary. Everything else is the real loader.
            from ml.concept_explainability.data import iter_slides
            from ml.concept_explainability.model import SourceRun
            from scripts.ml.run_local_concept_pilot import prepare_local_sources

            chosen = pd.read_parquet(reuse["original_tiling_metadata"])
            chosen["name"] = chosen.path.map(lambda p: Path(p).stem)
            reference = SourceRun(
                source.run_id,
                source.training_sha,
                source.config,
                source.checkpoint_path,
                source.checkpoint_sha256,
            )
            cache = Path(temp) / "pilot-cache"
            cache.mkdir()
            with patch(
                "ml.data.datasets.single_scale.download_level_sources",
                side_effect=lambda levels: {k: Path(v) for k, v in levels.items()},
            ):
                prepare_local_sources(reference, cache, chosen, assets)
            for split in ("train", "val"):
                self.assertEqual(
                    source.datasets[split].level, reference.datasets[split].level
                )
                self.assertEqual(
                    source.datasets[split].label_mode,
                    reference.datasets[split].label_mode,
                )
                for adapted, original in zip(
                    iter_slides(source, split),
                    iter_slides(reference, split),
                    strict=True,
                ):
                    np.testing.assert_array_equal(
                        adapted.embeddings, original.embeddings
                    )
                    np.testing.assert_array_equal(adapted.label, original.label)
                    pd.testing.assert_frame_equal(adapted.keys, original.keys)
                bag, label, info = source.datasets[split][0]
                self.assertEqual(bag.shape[1], 2560)
                self.assertEqual(label.numel(), 1)
                self.assertIn(info["slide_id"], baseline["loaded_bag_rows"][split])
            self.assertEqual(set(source.datasets), {"train", "val"})
            self.assertEqual(evidence["bag_rows"], baseline["loaded_bag_rows"])
            self.assertTrue(
                str(source.config.dataset.levels["3"].uris.train).startswith(
                    "mlflow-artifacts:"
                )
            )
            self.assertEqual(len(evidence["wsi_inventory"]), 2)
            self.assertEqual(
                evidence["wsi_inventory"][0]["old_tree_hash_status"],
                "unverified: serialization unavailable",
            )

    def test_tampered_mapping_hash_roster_and_per_slide_rows_fail(self):
        for corruption in ("mapping", "hash", "extra", "rows", "identity"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                reuse, baseline, assets = cached_fixture(root)
                if corruption == "mapping":
                    p = root / "train-mapping.csv"
                    p.write_text(p.read_text().replace("a luminal", "b luminal"))
                elif corruption == "hash":
                    reuse["checkpoint_sha256"] = "0" * 64
                elif corruption == "extra":
                    (root / "train" / "extra.parquet").write_bytes(b"extra")
                elif corruption == "rows":
                    assets[0]["rows"] -= 1
                else:
                    reuse["source_run_id"] = "different"
                with (
                    patch("ml.concept_explainability.model.verify_training_revision"),
                    self.assertRaises(ValueError),
                ):
                    comparison.prepare_source(reuse, baseline, assets)

    def test_inventory_is_deterministic_detects_changes_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "b").write_bytes(b"b")
            (root / "a").write_bytes(b"a")
            first = comparison.file_inventory(root)
            self.assertEqual([f["path"] for f in first["files"]], ["a", "b"])
            self.assertEqual(first, comparison.file_inventory(root))
            (root / "a").write_bytes(b"changed")
            self.assertNotEqual(
                first["sha256"], comparison.file_inventory(root)["sha256"]
            )
            (root / "link").symlink_to(root / "a")
            with self.assertRaises(ValueError):
                comparison.file_inventory(root)

    def test_synthetic_comparison_preserves_settings_rows_and_exact_k16_bridge(self):
        from ml.concept_explainability.evaluation import (
            evaluate_concepts,
            fit_dictionary,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reuse, baseline, assets = cached_fixture(root)
            fit = OmegaConf.load(comparison.REPO / "configs/concepts.yaml").fit
            with (
                patch("ml.concept_explainability.model.verify_training_revision"),
                comparison.offline(),
            ):
                source, _ = comparison.prepare_source(reuse, baseline, assets)
                fit_dictionary(
                    source, fit, Path(reuse["baseline_dictionary"]), use_ray=False
                )
                evaluate_concepts(
                    source,
                    Path(reuse["baseline_dictionary"]),
                    "val",
                    Path(reuse["baseline_validation"]),
                    device="cpu",
                )
            baseline_dictionary = json.loads(
                (Path(reuse["baseline_dictionary"]) / "manifest.json").read_text()
            )
            reuse["baseline_dictionary_id"] = baseline_dictionary["dictionary_id"]
            write_json(root / "reuse.json", reuse)

            # Synthetic MRXS bytes cannot be opened: keep real numeric fitting,
            # native inference and complete coefficient verification, substituting
            # only image geometry/report IO (covered by the core report suite).
            def report(source, dictionary, evaluation, output, publication):
                k = json.loads((dictionary / "manifest.json").read_text())[
                    "n_components"
                ]
                html = output / f"naming_k{k}.html"
                html.write_text('<body class="blind">fixture</body>')
                return html

            geometry = {
                "scanner_level": 3,
                "width": 1000,
                "height": 1000,
                "downsample": 8.0,
            }
            with (
                patch("ml.concept_explainability.model.verify_training_revision"),
                patch(
                    "ml.concept_explainability.report.tile_geometry",
                    return_value=geometry,
                ),
                patch(
                    "ml.concept_explainability.report.build_report", side_effect=report
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = comparison.main([*command(root), "--execute"])
            self.assertEqual(result["status"], "completed_smoke_only")
            table = pd.read_csv(root / "output" / "comparison.csv")
            self.assertEqual(table.k.tolist(), [8, 16, 32])
            manifest = json.loads(
                (root / "output" / "comparison_manifest.json").read_text()
            )
            self.assertEqual(
                manifest["original_output_parity"], "exact native/adapter/baseline"
            )
            for k in (8, 16, 32):
                candidate = json.loads(
                    (
                        root / "output" / f"k{k}" / "dictionary" / "manifest.json"
                    ).read_text()
                )
                candidate["config"]["n_components"] = 16
                self.assertEqual(candidate["config"], baseline_dictionary["config"])
            self.assertEqual((root / "output").stat().st_mode & 0o777, 0o700)

            def changed_fit(source, config, output, use_ray):
                fit_dictionary(source, config, output, use_ray=use_ray)
                manifest = json.loads((output / "manifest.json").read_text())
                manifest["dictionary_id"] = "0" * 64
                write_json(output / "manifest.json", manifest)

            failed_args = command(root)
            failed_args[-1] = str(root / "failed-output")
            with (
                patch("ml.concept_explainability.model.verify_training_revision"),
                patch(
                    "ml.concept_explainability.report.tile_geometry",
                    return_value=geometry,
                ),
                patch(
                    "ml.concept_explainability.evaluation.fit_dictionary",
                    side_effect=changed_fit,
                ),
                self.assertRaisesRegex(ValueError, "K16 refit"),
            ):
                comparison.main([*failed_args, "--execute"])
            failed = json.loads(
                (root / "failed-output" / "comparison_manifest.json").read_text()
            )
            self.assertEqual(failed["status"], "failed")
            self.assertFalse((root / "failed-output" / "k8").exists())

    def test_original_output_parity_is_exact_and_roster_aware(self):
        original = pd.DataFrame(
            {"slide_id": ["a", "b"], "target": ["type"] * 2, "original_raw": [0.1, 0.2]}
        )
        comparison.check_original_outputs(original, original.iloc[::-1])
        for changed in (
            original.iloc[:1],
            original.assign(original_raw=[0.1, 0.20000001]),
        ):
            with self.assertRaises(ValueError):
                comparison.check_original_outputs(original, changed)


if __name__ == "__main__":
    unittest.main()
