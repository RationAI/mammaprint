"""Numerical and real artifact integration checks for exact mean evidence."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from ml.explainability.attribution import integrated_gradients, leave_one_out
from ml.explainability.checkpoint import (
    SourceConfigMismatchError,
    compose_resolved_explanation_config,
)
from ml.explainability.data import EmbeddingSlide, SlideGeometry
from ml.explainability.evidence_report import render_evidence
from ml.explainability.exact_mean import (
    attach_previews,
    decompose_mean,
    summarize_evidence,
)
from ml.explainability.reporting import write_rationai_report
from ml.explainability.runner import run_cohort, validate_module
from ml.explainability.viewer import explanation_layers
from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads.linear import LinearHead
from ml.models.module import MammaprintModule
from scripts.ml import submit_explain
from scripts.ml.submit_exact_mean_reports import job_spec


def model(outputs=1):
    head = LinearHead(2, out_dim=outputs, dropout=0.5)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[0.5, -0.25]]).repeat(outputs, 1))
        head.linear.bias.fill_(0.3)
    return MammaprintModule(IdentityEncoder(2), MeanPool(2), head).eval()


def config(mode="index"):
    settings = OmegaConf.load(
        Path(__file__).parents[1] / "configs/explain.yaml"
    ).explain
    settings.exact_mean.enabled = True
    settings.single.enabled = True
    settings.leave_one_out.enabled = True
    settings.patch_flipping.enabled = False
    settings.attention.enabled = False
    return OmegaConf.create({"feature_dim": 2, "label_mode": mode, "explain": settings})


class ExactNumericsTest(unittest.TestCase):
    def test_matches_model_reference_replacement_ig_and_removal(self):
        for outputs in (1, 2):
            module = model(outputs)
            baseline = torch.tensor([0.4, -0.8])
            for bag in (
                torch.tensor([[0.0, 0.0]]),
                torch.tensor([[1.0, -1.0], [-2.0, 1.0], [0.1, 0.0]]),
            ):
                with torch.no_grad():
                    c, reference, singleton = decompose_mean(
                        module.aggregator, module.head, bag, baseline, batch_size=2
                    )
                    prediction = module(bag)[0]
                    torch.testing.assert_close(
                        (reference + c.sum(0)).float(), prediction
                    )
                    torch.testing.assert_close(singleton.mean(0).float(), prediction)
                    for i in range(len(bag)):
                        replaced = bag.clone()
                        replaced[i] = baseline
                        torch.testing.assert_close(
                            c[i].float(), prediction - module(replaced)[0]
                        )
                self.assertFalse(c.requires_grad)
                ig = integrated_gradients(
                    module.aggregator, module.head, bag, baseline, steps=2
                )
                torch.testing.assert_close(c.float(), ig.attributions)
                if len(bag) > 1:
                    loo = leave_one_out(module.aggregator, module.head, bag)
                    torch.testing.assert_close(
                        ((singleton - prediction) / (len(bag) - 1)).float(), loo.delta
                    )

    def test_rejects_training_and_incompatible_models(self):
        module = model()
        module.train()
        with self.assertRaises(ValueError):
            decompose_mean(
                module.aggregator, module.head, torch.ones(1, 2), torch.zeros(2)
            )
        module.eval()
        module.aggregator = MaxPool(2)
        with self.assertRaises(TypeError):
            validate_module(module, config())
        module = model()
        module.output_activation = torch.nn.Sigmoid()
        with self.assertRaises(TypeError):
            validate_module(module, config())

    def test_summary_conservation_opposition_ties_and_probability(self):
        result = summarize_evidence(
            x=np.array([0, 1, 20]),
            y=np.zeros(3, dtype=int),
            contributions=np.array([0.2, -0.2, 0.2]),
            singleton=np.array([1.6, 0.4, 1.6]),
            reference=1.0,
            prediction=1.2,
            target="luminal_a_logit",
            region_size=10,
            tile_extent=2,
            top_tiles=3,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertAlmostEqual(
            sum(region["contribution"] for region in result["regions"]), 0.2
        )
        self.assertEqual(sum(region["tile_count"] for region in result["regions"]), 3)
        cancelled = next(
            region for region in result["regions"] if region["grid_x"] == 0
        )
        self.assertEqual(cancelled["contribution"], 0)
        self.assertEqual(cancelled["up"], 0.2)
        self.assertEqual(cancelled["down"], -0.2)
        self.assertEqual(
            [tile["tile_index"] for tile in result["top_tiles"]["up"]], [0, 2]
        )
        self.assertEqual(sum(result["histogram"]["counts"]), 3)
        self.assertAlmostEqual(result["probability"], 1 / (1 + np.exp(-1.2)))
        self.assertFalse(result["warning"])

    def test_zero_reference_all_equal_missing_crops_and_escaping(self):
        target = summarize_evidence(
            x=np.array([0]),
            y=np.array([0]),
            contributions=np.array([0.0]),
            singleton=np.array([0.0]),
            reference=0.0,
            prediction=1e-8,
            target="mammaprint_index",
            region_size=10,
            tile_extent=2,
            top_tiles=3,
            atol=1e-7,
            rtol=1e-5,
        )
        self.assertFalse(target["warning"])
        self.assertFalse(target["top_tiles"]["up"])
        slide = {
            "slide_id": '<script>alert("bad")</script>',
            "record_num": "record",
            "level": 3,
            "targets": {"mammaprint_index": target},
            "viewer_url": "javascript:alert(1)",
            "overlay_status": "unavailable",
        }
        with patch(
            "ml.explainability.exact_mean._open_slide",
            side_effect=FileNotFoundError("no WSI"),
        ):
            attach_previews(slide, slide_path="missing", level=3, tile_extent=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mean_evidence.json"
            path.write_text(
                json.dumps(
                    {
                        "slides": [slide],
                        "reference_tile_count": 1,
                        "reference_skipped_slides": [],
                        "visualization_scales": {},
                        "scale_percentile": 99,
                        "region_size": 10,
                    }
                )
            )
            html = render_evidence(path)
        self.assertNotIn('<script>alert("bad")</script>', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("javascript:", html)

    def test_previews_reuse_tiles_and_isolate_a_crop_failure(self):
        tiles = [
            {"tile_index": 0, "x": 16, "y": 32},
            {"tile_index": 1, "x": 48, "y": 64},
        ]
        evidence = {
            "targets": {
                name: {"top_tiles": {"up": tiles, "down": []}}
                for name in ("mammaprint_index", "luminal_a_logit")
            }
        }
        calls = []

        class WSI:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read_region_relative(self, xy, level, size):
                calls.append((xy, level, size))
                if xy == (48, 64):
                    raise OSError("one corrupt crop")
                return Image.new("RGB", size)

        with patch("ml.explainability.exact_mean._open_slide", return_value=WSI()):
            attach_previews(evidence, slide_path="slide", level=3, tile_extent=224)
        self.assertEqual(calls, [((16, 32), 3, (224, 224)), ((48, 64), 3, (224, 224))])
        self.assertEqual(evidence["previews"]["0"]["status"], "ok")
        self.assertEqual(evidence["previews"]["1"]["status"], "unavailable")


class CohortIntegrationTest(unittest.TestCase):
    def run_fixture(self, root, mode="index", images=True):
        module = model()
        cfg = config(mode)
        rows = pd.DataFrame(
            {
                "name": ["slide-a", "slide-b"],
                "record_num": ["case-1", "case-2"],
                "mammaprint_index": [0.1, -0.2],
                "type": ["a luminal", "b luminal"],
                "type_label": [1, 0],
            }
        )
        dataset = SimpleNamespace(slides=rows, level=3)
        bag = torch.tensor([[1.0, -1.0], [-2.0, 1.0], [0.1, 0.0]])
        slides = [
            EmbeddingSlide(
                name,
                bag if i == 0 else bag[:1],
                np.array([0, 16, 32]) if i == 0 else np.array([0]),
                np.array([0, 16, 32]) if i == 0 else np.array([0]),
                torch.tensor([0.1 if mode == "index" else 1]),
                Path("/mnt/data/missing.svs"),
                3,
            )
            for i, name in enumerate(rows.name)
        ]

        class WSI:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read_region_relative(self, xy, level, size):
                self.assertion = (xy, level, size)
                return Image.new("RGB", size, (120, 40, 170))

        with (
            patch(
                "ml.explainability.runner.instantiate_embedding_dataset",
                return_value=dataset,
            ),
            patch(
                "ml.explainability.runner.streaming_mean_embedding",
                return_value=(torch.tensor([0.4, -0.8]), 10, ()),
            ),
            patch(
                "ml.explainability.runner.selected_level_spec",
                return_value=(3, {"tile_extent": 16, "stride": 16, "mpp": 0.5}),
            ),
            patch(
                "ml.explainability.runner.read_embedding_slide",
                side_effect=lambda _, i: slides[i],
            ),
            patch(
                "ml.explainability.runner._read_thumbnail",
                return_value=np.zeros((48, 48, 3), dtype=np.uint8),
            ),
            patch(
                "ml.explainability.runner.read_slide_geometry",
                **(
                    {"return_value": SlideGeometry(64, 64, 8, 0.5, 0.5)}
                    if images
                    else {"side_effect": FileNotFoundError("no WSI")}
                ),
            ),
            patch(
                "ml.explainability.exact_mean._open_slide",
                **(
                    {"side_effect": lambda _: WSI()}
                    if images
                    else {"side_effect": FileNotFoundError("no WSI")}
                ),
            ),
        ):
            result = run_cohort(
                config=cfg,
                module=module,
                output_dir=root,
                provenance={
                    "explanation_run_id": "abc",
                    "explanation_experiment_id": "3",
                },
                device="cpu",
            )
        self.assertEqual(result.failed_slides, 0)
        runtimes = pd.read_parquet(result.slides_path)
        np.testing.assert_allclose(
            runtimes.runtime_seconds,
            runtimes.attribution_runtime_seconds
            + runtimes.overlay_runtime_seconds
            + runtimes.preview_runtime_seconds,
        )
        data = json.loads((root / "mean_evidence.json").read_text())
        self.assertEqual(len(data["slides"]), 2)
        self.assertTrue(
            all(not t["warning"] for s in data["slides"] for t in s["targets"].values())
        )
        tiles = pd.read_parquet(root / "tiles/slide-a.parquet")
        self.assertTrue(tiles.exact_mean.notna().all())
        self.assertEqual(tiles.integrated_gradients.notna().sum(), 0)
        manifest = json.loads(result.manifest_path.read_text())
        self.assertIn("exact_mean", manifest["methods"])
        self.assertNotIn("attention", manifest["methods"])
        layers = explanation_layers(
            "slide-a", {"mammaprint_index": manifest["methods"]}
        )
        self.assertIn("Exact contribution", layers[0].name)
        self.assertTrue(layers[0].visible)
        return result, data

    def test_real_masks_standard_report_and_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, data = self.run_fixture(root / "cohort")
            self.assertTrue(
                list((result.output_dir / "masks/exact_mean").rglob("*.tif"))
            )
            self.assertTrue(
                all(
                    preview["status"] == "ok"
                    for slide in data["slides"]
                    for preview in slide["previews"].values()
                )
            )
            old_uri = mlflow.get_tracking_uri()
            try:
                mlflow.set_tracking_uri((root / "mlruns").as_uri())
                experiment = mlflow.create_experiment("exact mean integration")
                with mlflow.start_run(experiment_id=experiment) as run:
                    mlflow.log_metric("explanations/successful_slides", 2)
                    mlflow.log_table(
                        pd.read_csv(result.slide_keys_path), "slide_keys.json"
                    )
                    path = write_rationai_report(
                        root / "report.html",
                        slides_path=result.slides_path,
                        manifest_path=result.manifest_path,
                        run_id=run.info.run_id,
                        experiment_id=experiment,
                        artifact_root="explanations",
                        slide_key_artifact="slide_keys.json",
                        xopat_ui_uri="https://example.org/",
                        mlflow_ui_uri="https://mlflow.example.org/",
                        mask_artifact_paths={
                            "explanations/"
                            + p.relative_to(result.output_dir).as_posix()
                            for p in (result.output_dir / "masks").rglob("*.tif")
                        },
                    )
                    filtered = write_rationai_report(
                        root / "single-only.html",
                        slides_path=result.slides_path,
                        manifest_path=result.manifest_path,
                        run_id=run.info.run_id,
                        experiment_id=experiment,
                        artifact_root="explanations",
                        slide_key_artifact="slide_keys.json",
                        xopat_ui_uri="https://example.org/",
                        mlflow_ui_uri="https://mlflow.example.org/",
                        included_methods=["single"],
                        mask_artifact_paths={
                            "explanations/"
                            + p.relative_to(result.output_dir).as_posix()
                            for p in (result.output_dir / "masks").rglob("*.tif")
                        },
                    )
                    self.assertNotIn("Exact tile evidence", filtered.read_text())
                document = path.read_text()
                self.assertIn("Exact tile evidence", document)
                self.assertIn("Slide Evaluation", document)
                self.assertIn("data:image/jpeg;base64,", document)
                self.assertIn("3 tiles", document)
                self.assertIn("Exact contribution vs training reference", document)
                # Retain one concrete review artifact outside the temporary fixture when requested.
                import os

                if os.environ.get("EXACT_MEAN_PREVIEW"):
                    Path(os.environ["EXACT_MEAN_PREVIEW"]).write_text(document)
            finally:
                mlflow.set_tracking_uri(old_uri)

    def test_missing_wsi_evidence_only_report_for_bce(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, data = self.run_fixture(root, mode="type", images=False)
            self.assertTrue(
                all(
                    preview["status"] == "unavailable"
                    for slide in data["slides"]
                    for preview in slide["previews"].values()
                )
            )
            path = write_rationai_report(
                root / "report.html",
                slides_path=result.slides_path,
                manifest_path=result.manifest_path,
                run_id="unused",
                experiment_id="3",
                artifact_root="explanations",
                slide_key_artifact="unused",
                xopat_ui_uri="https://example.org/",
                mlflow_ui_uri="https://example.org/",
                mask_artifact_paths=set(),
            )
            self.assertIn("Final Luminal A probability", path.read_text())
            self.assertIn("No rendered slide masks", path.read_text())


class SubmissionTest(unittest.TestCase):
    def test_fixed_model_jobs_are_pinned_cpu_only_and_previewable(self):
        for name in ("regression", "bce"):
            spec = job_spec(name, "a" * 40)
            self.assertEqual((spec.cpu, spec.memory, spec.gpu), (4, "8Gi", None))
            self.assertIn("checkout --quiet " + "a" * 40, " ".join(spec.script))
            self.assertIn("explain.exact_mean.enabled=true", spec.script[-1])
            self.assertIn("explain.patch_flipping.enabled=false", spec.script[-1])
            self.assertTrue(any("TMPDIR" in command for command in spec.script))
        with self.assertRaises(ValueError):
            job_spec("regression", "mutable-branch")

    def test_cpu_spec_through_actual_library_has_no_gpu_selector(self):
        args = submit_explain.build_parser().parse_args(
            [
                "--checkpoint-uri",
                "mlflow-artifacts:/3/abc/artifacts/checkpoints/best",
                "--gpu",
                "none",
                "--cpu",
                "4",
                "--memory",
                "8Gi",
                "--device",
                "cpu",
                "--resolved-source-config",
                "--expected-checkpoint-sha256",
                "a" * 64,
            ]
        )
        spec = submit_explain.build_job_spec(args)
        self.assertIsNone(spec.gpu)
        self.assertIn("--device cpu", spec.script[-1])
        self.assertIn("--resolved-source-config", spec.script[-1])
        package = importlib.import_module("kube_jobs.submit_job")
        captured = []
        with (
            patch.object(package, "load_kube_config"),
            patch.object(package.client, "BatchV1Api") as api,
        ):

            def capture(**kw):
                captured.append(kw["body"])
                return SimpleNamespace(metadata=SimpleNamespace(name=spec.job_name))

            api.return_value.create_namespaced_job.side_effect = capture
            submit_explain.submit(spec)
        container = captured[0].spec.template.spec.containers[0]
        self.assertEqual(container.resources.requests["cpu"], 4)
        self.assertEqual(container.resources.requests["memory"], "8Gi")
        self.assertNotIn("nvidia.com/gpu", container.resources.requests)
        self.assertFalse(captured[0].spec.template.spec.node_selector)

    def test_saved_recovery_configs_compose_without_mutating_source_graph(self):
        for mode in ("index", "type"):
            source = {
                "seed": 1,
                "feature_dim": 2,
                "label_mode": mode,
                "dataset": {"levels": {"3": {"stride": 16}}},
                "datamodule": {"test": {"path": "original-data.parquet"}},
                "ml": {
                    "_target_": "ml.models.MammaprintModule",
                    "encoder": {
                        "_target_": "ml.models.encoders.IdentityEncoder",
                        "out_dim": 2,
                    },
                    "aggregator": {
                        "_target_": "ml.models.aggregators.MeanPool",
                        "feature_dim": 2,
                    },
                    "head": {"_target_": "ml.models.heads.LinearHead", "in_dim": 2},
                    "output_activation": {"_target_": "torch.nn.Identity"},
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config-resolved.yaml"
                OmegaConf.save(OmegaConf.create(source), path)
                cfg = compose_resolved_explanation_config(
                    path, ["explain.exact_mean.enabled=true"]
                )
                self.assertTrue(cfg.explain.exact_mean.enabled)
                self.assertEqual(cfg.datamodule.test.path, "original-data.parquet")
                with self.assertRaises(SourceConfigMismatchError):
                    compose_resolved_explanation_config(path, ["ml.head.dropout=0"])


if __name__ == "__main__":
    unittest.main()
