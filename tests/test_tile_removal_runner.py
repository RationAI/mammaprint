"""Small frozen mean/attention checkpoint runs through Parquet and HTML."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import tifffile
import torch
from omegaconf import OmegaConf

from ml.explainability.tile_removal_inputs import sha256_file
from ml.explainability.tile_removal_viewer import (
    MaskCoverage,
    SlideGeometry,
    ViewerTarget,
    write_overlay,
)
from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.mean import MeanPool
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads.linear import LinearHead
from ml.models.module import MammaprintModule
from scripts.ml.run_tile_removal import run


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.TemporaryDirectory()
        self.addCleanup(self.base.cleanup)
        self.root = Path(self.base.name)
        self.labels = self.root / "mapping.csv"
        pd.DataFrame(
            [
                ("/mnt/data/slide.svs", "val", "a luminal", 0.25, "case-1"),
                ("/mnt/data/missing.svs", "val", "a luminal", 0.3, "case-2"),
                ("/mnt/data/empty.svs", "val", "b luminal", -0.3, "case-3"),
                ("/mnt/data/no_label.svs", "val", "", "", "case-4"),
                ("/mnt/data/one.svs", "val", "a luminal", 0.0, "case-5"),
                ("/mnt/data/forbidden.svs", "test", "b luminal", -0.3, "case-6"),
            ],
            columns=["path", "split", "type", "mammaprint_index", "case"],
        ).to_csv(self.labels, index=False)
        self.plan = {"split": "val", "models": [self._model(False), self._model(True)]}
        self.plan["options"] = {"control_repeats": 2, "index_band": 0.05}
        self.plan_file = self.root / "study.json"
        self._write_plan()

    def _write_plan(self):
        self.plan_file.write_text(json.dumps(self.plan), encoding="utf-8")

    def _model(self, attention):
        name = "attention" if attention else "mean"
        root = self.root / name
        root.mkdir()
        (root / "bags").mkdir()
        frame = pd.DataFrame(
            {
                "x": np.arange(20) % 5 * 2,
                "y": np.arange(20) // 5 * 2,
                "embedding": [[-8.0, 0.0, 1.0]] + [[0.35, 0.0, 0.0]] * 19,
            }
        )
        frame.to_parquet(root / "bags/slide.parquet")
        frame.iloc[:0].to_parquet(root / "bags/empty.parquet")
        frame.iloc[:2].to_parquet(root / "bags/no_label.parquet")
        single = pd.DataFrame({"x": [0], "y": [0], "embedding": [[0.0, 0.0, 0.0]]})
        single.to_parquet(root / "bags/one.parquet")
        aggregator = AttentionMIL(3, 2) if attention else MeanPool(3)
        head = LinearHead(3, dropout=0.5)
        with torch.no_grad():
            if attention:
                aggregator.attention_weights.weight.zero_()
                aggregator.attention_weights.bias.zero_()
            head.linear.weight[:] = torch.tensor([[1.0, 0.0, 0.0]])
            head.linear.bias[:] = 0.0
        model = MammaprintModule(IdentityEncoder(3), aggregator, head)
        checkpoint = root / "model.ckpt"
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        config = root / "config.yaml"
        source = f"{name}/bags"
        OmegaConf.save(
            {
                "label_mode": "index",
                "ml": {
                    "_target_": "ml.models.MammaprintModule",
                    "encoder": {
                        "_target_": "ml.models.encoders.IdentityEncoder",
                        "out_dim": 3,
                    },
                    "aggregator": {
                        "_target_": f"ml.models.aggregators.{type(aggregator).__name__}",
                        "feature_dim": 3,
                        **({"attention_dim": 2} if attention else {}),
                    },
                    "head": {"_target_": "ml.models.heads.LinearHead", "in_dim": 3},
                },
                "datamodule": {
                    "val": {
                        "_target_": "ml.data.SingleScaleDataset",
                        "split": "val",
                        "label_mode": "index",
                        "data_mapping": "mapping.csv",
                        "levels": {
                            "2": {
                                "level": 2,
                                "tile_extent": 4,
                                "stride": 2,
                                "mpp": 1.0,
                                "uris": {
                                    "val": source,
                                    "train": "/unavailable",
                                    "test": "/forbidden",
                                },
                            }
                        },
                    },
                    "test": {"_target_": "intentionally.invalid.UnavailableTestLoader"},
                },
            },
            config,
        )
        return {
            "model_id": name,
            "source_run_id": ("1" if attention else "2") * 32,
            "source_revision": "3" * 40,
            "validation_source": source,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "label_mapping": "mapping.csv",
            "label_mapping_sha256": sha256_file(self.labels),
            "patient_column": "case",
        }

    def test_end_to_end_and_retry(self):
        outputs = [self.root / "first", self.root / "second"]
        with patch(
            "ml.explainability.tile_removal_evaluate.SlideGeometry.from_wsi",
            return_value=SlideGeometry(12, 12, 48, 48, 4.0, 1.0, 1.0),
        ):
            results = [run(self.plan_file, output) for output in outputs]
        self.assertEqual(results, [results[0]] * 2)
        self.assertEqual(run(self.plan_file, outputs[0]), results[0])
        data = json.loads((outputs[0] / "report_data.json").read_bytes())
        self.assertEqual(len(data["models"]), 2)
        self.assertEqual(data["paired"][0]["common_slide_count"], 5)
        model = data["models"][0]
        rows = {row["slide_id"]: row for row in model["slides"]}
        self.assertEqual(
            {key: row["status"] for key, row in rows.items()},
            {
                "slide": "ok",
                "missing": "missing_embedding",
                "empty": "empty_bag",
                "no_label": "missing_class",
                "one": "ok",
            },
        )
        self.assertEqual(rows["one"]["intact_class"], "A")
        masks = {item["kind"]: item for item in rows["slide"]["masks"]}
        exact = masks["exact"]
        self.assertEqual(exact["removed_tile_ids"], [0])
        self.assertEqual(exact["verification_status"], "verified_minimum")
        self.assertEqual(
            exact["within_budgets"], {"1%": False, "5%": True, "10%": True}
        )
        self.assertEqual(exact["tile_footprints_level0"], [[0, 0, 16, 16]])
        self.assertEqual(exact["deleted_area_level0_px"], 256)
        self.assertEqual(masks["control"]["search_status"], "matched_control")
        self.assertTrue((outputs[0] / exact["overlay_path"]).is_file())
        self.assertIn(
            "Targeted tile removal", (outputs[0] / "mean/report.html").read_text()
        )
        self.assertEqual(len(pd.read_parquet(outputs[0] / "slides.parquet")), 10)
        self.assertEqual(
            sha256_file(outputs[0] / exact["overlay_path"]),
            sha256_file(outputs[1] / exact["overlay_path"]),
        )
        with self.assertRaisesRegex(ValueError, "different plan"):
            run(self.plan_file, outputs[0], pilot=["slide"])
        (outputs[0] / "mean/report.html").write_text("changed")
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            run(self.plan_file, outputs[0])
        bag = self.root / "mean/bags/slide.parquet"
        bag.write_bytes(bag.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Validation bag changed"):
            run(self.plan_file, outputs[1])

    def test_pilot_without_controls_writes_parquet(self):
        out = self.root / "one"
        result = run(self.plan_file, out, pilot=["one"])
        self.assertEqual(result["pilot_slide_ids"], ["one"])
        slides = pd.read_parquet(out / "slides.parquet")
        self.assertEqual(slides["slide_id"].tolist(), ["one", "one"])
        self.assertTrue(slides["exact_control_matched"].isna().all())

    def test_pilot_can_have_slides_for_only_one_model(self):
        model = self.plan["models"][1]
        other_mapping = self.root / "attention/mapping.csv"
        rows = pd.read_csv(self.labels)
        rows[~rows["path"].str.endswith("one.svs")].to_csv(other_mapping, index=False)
        config = Path(model["config"])
        resolved = OmegaConf.load(config)
        resolved.datamodule.val.data_mapping = str(other_mapping)
        OmegaConf.save(resolved, config)
        model.update(
            config_sha256=sha256_file(config),
            label_mapping=str(other_mapping),
            label_mapping_sha256=sha256_file(other_mapping),
        )
        self._write_plan()
        out = self.root / "different_rosters"
        run(self.plan_file, out, pilot=["one"])
        models = json.loads((out / "report_data.json").read_bytes())["models"]
        self.assertEqual(len(models[0]["slides"]), 1)
        self.assertEqual(models[1]["slides"], [])
        self.assertTrue((out / "mean/report.html").exists())
        self.assertFalse((out / "attention/report.html").exists())

    def test_guard_dry_run_and_manifest_mismatch(self):
        out = self.root / "dry"
        with patch("scripts.ml.run_tile_removal.prepare_model") as loader:
            preview = run(self.plan_file, out, dry_run=True)
            loader.assert_not_called()
        self.assertEqual(preview["split"], "val")
        self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError, "absent from the validation roster"):
            run(self.plan_file, out, pilot=["forbidden"])
        (self.root / "mean/bags/forbidden.parquet").touch()
        with self.assertRaisesRegex(ValueError, "non-validation"):
            run(self.plan_file, out)
        self.plan["split"] = "test"
        self._write_plan()
        with self.assertRaisesRegex(ValueError, "validation plan"):
            run(self.plan_file, out, dry_run=True)

    def test_changed_label_mapping_does_not_pass_lineage(self):
        alternative = self.root / "alternative.csv"
        alternative.write_bytes(self.labels.read_bytes().replace(b"case-1", b"case-9"))
        self.plan["models"][0]["label_mapping"] = str(alternative)
        self.plan["models"][0]["label_mapping_sha256"] = sha256_file(alternative)
        self._write_plan()
        with self.assertRaisesRegex(ValueError, "label mapping"):
            run(self.plan_file, self.root / "different")

    def test_native_roundoff_retains_later_flip_without_minimum_claim(self):
        model = self.plan["models"][0]
        checkpoint = Path(model["checkpoint"])
        state = torch.load(checkpoint, weights_only=True)
        state["state_dict"]["head.linear.bias"].fill_(-1)
        torch.save(state, checkpoint)
        model["checkpoint_sha256"] = sha256_file(checkpoint)
        self.plan["models"] = [model]
        self._write_plan()
        pd.DataFrame(
            {
                "x": [0, 2, 4, 6],
                "y": [0] * 4,
                "embedding": [
                    [x, 0.0, 0.0] for x in [4.0, 1.000000119, 0.99999988, 0.99999994]
                ],
            }
        ).to_parquet(self.root / "mean/bags/slide.parquet")
        out = self.root / "boundary"
        run(self.plan_file, out, pilot=["slide"])
        data = json.loads((out / "report_data.json").read_bytes())
        exact = data["models"][0]["slides"][0]["masks"][0]
        self.assertEqual(exact["removed_tile_ids"], [0, 1])
        self.assertTrue(exact["target_reached"])
        self.assertEqual(exact["verification_status"], "minimum_unresolved")

    def test_native_compact_boundary_verifies_smaller_window(self):
        model = self.plan["models"][0]
        checkpoint = Path(model["checkpoint"])
        state = torch.load(checkpoint, weights_only=True)
        state["state_dict"]["head.linear.bias"].fill_(-1)
        torch.save(state, checkpoint)
        config = Path(model["config"])
        resolved = OmegaConf.load(config)
        resolved.datamodule.val.levels["2"].tile_extent = 1
        resolved.datamodule.val.levels["2"].stride = 1
        OmegaConf.save(resolved, config)
        model["checkpoint_sha256"] = sha256_file(checkpoint)
        model["config_sha256"] = sha256_file(config)
        self.plan["models"] = [model]
        self._write_plan()
        pd.DataFrame(
            {
                "x": [1, 0, *[3 * x for x in range(1, 19)]],
                "y": [0, 0, *([5] * 18)],
                "embedding": [
                    [x, 0.0, 0.0] for x in [0.98, 0.99, 1.0099999, *([1.0] * 17)]
                ],
            }
        ).to_parquet(self.root / "mean/bags/slide.parquet")
        out = self.root / "compact"
        run(self.plan_file, out, pilot=["slide"])
        data = json.loads((out / "report_data.json").read_bytes())
        masks = {item["kind"]: item for item in data["models"][0]["slides"][0]["masks"]}
        compact = masks["compact"]
        self.assertEqual(compact["removed_tile_ids"], [0])
        self.assertTrue(compact["target_reached"])
        self.assertEqual(compact["verification_status"], "minimum_unresolved")

    def test_blind_overlay_publish_verifies_hash_and_private_identifiers(self):
        self.plan["models"] = self.plan["models"][:1]
        self.plan["overlay_target"] = {
            "tracking_uri": "https://tracking.example/",
            "experiment_id": "3",
            "run_id": "a" * 32,
            "artifact_root": "validation",
            "xopat_url": "https://viewer.example/",
        }
        self._write_plan()
        with self.assertRaisesRegex(ValueError, "private review_key_file"):
            run(
                self.plan_file,
                self.root / "published",
                publish_overlays=True,
                dry_run=True,
            )
        secret = bytes(range(32))
        (self.root / "review.key").write_bytes(secret)
        self.plan["review_key_file"] = "review.key"
        self._write_plan()

        class FakeStore:
            def __init__(self):
                self.artifacts = {}

            def get_run(self, run_id):
                return SimpleNamespace(
                    info=SimpleNamespace(
                        run_id=run_id,
                        experiment_id="3",
                        lifecycle_stage="active",
                        status="RUNNING",
                    ),
                    data=SimpleNamespace(
                        tags={"targeted_tile_removal.role": "validation-overlays"}
                    ),
                )

            def list_artifacts(self, run_id, directory):
                return [
                    SimpleNamespace(path=key)
                    for key in self.artifacts
                    if key.startswith(directory + "/")
                ]

            def log_artifact(self, run_id, local_path, directory):
                local = Path(local_path)
                self.artifacts[f"{directory}/{local.name}"] = local.read_bytes()

            def download_artifacts(self, run_id, artifact, directory):
                path = Path(directory) / Path(artifact).name
                path.write_bytes(self.artifacts[artifact])
                return path

        store = FakeStore()
        out = self.root / "published"
        with (
            patch("mlflow.tracking.MlflowClient", return_value=store),
            patch(
                "ml.explainability.tile_removal_evaluate.SlideGeometry.from_wsi",
                return_value=SlideGeometry(12, 12, 48, 48, 4.0, 1.0, 1.0),
            ),
        ):
            result = run(self.plan_file, out, publish_overlays=True)
        self.assertGreater(result["review_regions"], 0)
        blind = unquote((out / "blinded_review.html").read_text())
        self.assertIn("validation/", blind)
        self.assertNotIn("exact-control", blind)
        self.assertNotIn("case-1", blind)
        self.assertNotIn("review.key", str(result))
        self.assertIn("unblinding_key.csv", result["artifacts_sha256"])


class GeometryTest(unittest.TestCase):
    def test_overlapping_mask_coverage_and_pyramid(self):
        geometry = SlideGeometry(512, 512, 512, 512, 1, 0.5, 0.5)
        rectangles = np.array([[0, 0, 16, 16], [8, 0, 24, 16]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mask.ome.tif"
            write_overlay(path, rectangles, geometry)
            with tifffile.TiffFile(path) as image:
                self.assertEqual(len(image.series[0].levels), 2)
                self.assertEqual(np.count_nonzero(image.asarray()), 24 * 16)
            coverage = MaskCoverage(path, rectangles, geometry, 0.5)
            self.assertEqual(coverage.totals((0, 1)), (24 * 16, 24 * 16))

    def test_xopat_uses_generic_layer_labels(self):
        name = hashlib.sha256(b"any mask").hexdigest()
        target = ViewerTarget(
            "https://tracking.example/",
            "3",
            "a" * 32,
            "validation",
            "https://viewer.example/",
        )
        url = target.link("/mnt/data/a.svs", f"overlays/{name}.ome.tif")
        fragment = json.loads(unquote(urlsplit(url).fragment))
        self.assertEqual(fragment["data"][0], "data/a.svs")
        self.assertTrue(fragment["data"][1].startswith("secure_mlflow/3/" + "a" * 32))
        self.assertEqual(
            fragment["visualizations"][0]["shaders"]["region"]["name"], "Region"
        )
