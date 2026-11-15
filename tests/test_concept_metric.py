"""Bounded synthetic tests of the deployed classifier-only metric adapter."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml.concept_explainability import metric
from ml.concept_explainability.evaluation import (
    _FixtureDataset,
    fit_dictionary,
    load_dictionary,
)
from ml.concept_explainability.model import SourceRun
from ml.models.aggregators import MeanPool
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads import LinearHead
from ml.models.module import MammaprintModule


def metric_source(root, dim=2560):
    config = OmegaConf.create(
        {
            "label_mode": "type",
            "feature_dim": dim,
            "ml": {
                "_target_": "ml.models.MammaprintModule",
                "encoder": {
                    "_target_": "ml.models.encoders.IdentityEncoder",
                    "out_dim": dim,
                },
                "aggregator": {
                    "_target_": "ml.models.aggregators.MeanPool",
                    "feature_dim": dim,
                },
                "head": {"_target_": "ml.models.heads.LinearHead", "in_dim": dim},
                "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
            },
        }
    )
    config.dataset = {
        "levels": {
            "3": {
                "mpp": 1.94,
                "tile_extent": 224,
                "stride": 112,
                "uris": {
                    split: str(root / split) for split in ("train", "val", "test")
                },
            }
        },
        "data_mapping": str(root / "labels.csv"),
    }
    config.datamodule = {
        split: {
            "_target_": "ml.data.SingleScaleDataset",
            "split": split,
            "levels": "${dataset.levels}",
            "data_mapping": "${dataset.data_mapping}",
            "label_mode": "${label_mode}",
        }
        for split in ("train", "val", "test")
    }
    module = MammaprintModule(
        IdentityEncoder(dim),
        MeanPool(dim),
        LinearHead(dim),
        loss=torch.nn.BCEWithLogitsLoss(),
    )
    path = root / "model.ckpt"
    torch.save({"state_dict": module.state_dict()}, path)
    source = SourceRun(
        "source", "a" * 40, config, path, hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return source, module


class NativeMetricTests(unittest.TestCase):
    def test_classifier_matches_upstream_binary_class_component_and_trace_ridge(self):
        # Recovered UC output_metric classifier loop with explicit [0,z] centering.
        z = np.random.default_rng(42).normal(size=(23, 4)).astype(np.float32)
        w = np.array([[0.1, -0.7, 1.2, 0.3]], np.float32)
        b = np.array([0.8], np.float32)
        theta = np.concatenate([np.zeros_like(w), w])
        biases = np.r_[np.zeros_like(b), b]
        theta, biases = theta - theta.mean(0), biases - biases.mean()
        logits = z.astype(np.float64) @ theta.T + biases
        expected = np.zeros((4, 4))
        for c in range(len(theta)):
            expected += np.outer(theta[c], theta[c]) / logits[:, c].var()
        expected *= 4 / np.trace(expected)
        expected += 0.3 * np.eye(4)
        eigenvalues, eigenvectors = np.linalg.eigh(expected)
        expected_t = (eigenvectors * np.sqrt(eigenvalues)).T.astype(np.float32)
        transform = metric.classifier_output_metric(z, w, b, ridge=0.3)
        np.testing.assert_array_equal(transform, expected_t)
        np.testing.assert_allclose(transform.T @ transform, expected, atol=5e-7)
        self.assertAlmostEqual(
            float(np.trace(transform.T @ transform)) / 4, 1.3, places=6
        )
        self.assertGreater(np.linalg.eigvalsh(transform.T @ transform).min(), 0.299999)
        for rows in (np.ones_like(z), z[:0], np.full_like(z, np.nan)):
            with self.assertRaisesRegex(ValueError, "sample|variance"):
                metric.classifier_output_metric(rows, w, b, ridge=0.3)
        with self.assertRaisesRegex(ValueError, "variance"):
            metric.classifier_output_metric(z, np.zeros_like(w), b, ridge=0.3)

    def test_preflight_rejects_graph_before_dataset_and_keeps_weights_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            source, original = metric_source(Path(d))
            module = metric.preflight_metric_source(source)
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, module.state_dict()[key])
            cases = [
                ("label_mode", "index"),
                ("label_mode", "both"),
                ("ml.encoder._target_", "ml.models.encoders.VGG16"),
                ("ml.head._target_", "ml.models.heads.MLPHead"),
                ("ml.loss._target_", "torch.nn.MSELoss"),
            ]
            cases += [
                ("ml.aggregator._target_", f"ml.models.aggregators.{name}")
                for name in (
                    "MaxPool",
                    "AttentionMIL",
                    "TransformerMIL",
                    "SpatialTransformerMIL",
                )
            ]
            for key, value in cases:
                bad = SourceRun(
                    source.run_id,
                    source.training_sha,
                    OmegaConf.create(OmegaConf.to_container(source.config)),
                    source.checkpoint_path,
                    source.checkpoint_sha256,
                )
                OmegaConf.update(bad.config, key, value)
                with (
                    self.subTest(key=key, value=value),
                    patch(
                        "ml.concept_explainability.evaluation.source_dataset",
                        side_effect=AssertionError("downloaded"),
                    ),
                    self.assertRaisesRegex(ValueError, "metric"),
                ):
                    fit_dictionary(
                        bad,
                        OmegaConf.create(
                            {"n_components": 2, "metric": {"enabled": True}}
                        ),
                        Path(d) / "out",
                        use_ray=False,
                    )

    def test_preflight_dataset_boundary_rejects_before_source_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = metric_source(Path(d))
            cases = [
                ("datamodule.train._target_", "ml.data.SpatialScaleDataset"),
                (
                    "datamodule.val._target_",
                    "ml.data.datasets.spatial_scale.SpatialScaleDataset",
                ),
                ("datamodule.test._target_", "custom.SingleScaleSubclass"),
                ("dataset.levels.4", {"mpp": 1.0}),
                ("datamodule.train.levels", {"3": {}, "4": {}}),
                ("datamodule.val.levels", {"4": {"uris": {"val": "other"}}}),
                ("datamodule.train.split", "test"),
                (
                    "datamodule.val.levels",
                    {
                        "3": {
                            "mpp": 2.0,
                            "tile_extent": 224,
                            "stride": 112,
                            "uris": {"val": str(Path(d) / "val")},
                        }
                    },
                ),
                (
                    "datamodule.test.levels",
                    {
                        "3": {
                            "mpp": 1.94,
                            "tile_extent": 224,
                            "stride": 112,
                            "uris": {"test": "other-corpus"},
                        }
                    },
                ),
                ("datamodule.test.label_mode", "index"),
                ("feature_dim", 4),
                ("ml.encoder.out_dim", 4),
                ("ml.aggregator.feature_dim", 4),
                ("ml.head.in_dim", 4),
            ]
            for key, value in cases:
                bad = SourceRun(
                    source.run_id,
                    source.training_sha,
                    OmegaConf.create(OmegaConf.to_container(source.config)),
                    source.checkpoint_path,
                    source.checkpoint_sha256,
                )
                OmegaConf.update(bad.config, key, value, merge=False)
                with (
                    self.subTest(key=key, value=value),
                    patch(
                        "ml.concept_explainability.evaluation.source_dataset",
                        side_effect=AssertionError("DATASET_ACCESS_REACHED"),
                    ),
                    self.assertRaisesRegex(ValueError, "metric"),
                ):
                    fit_dictionary(
                        bad,
                        OmegaConf.create(
                            {"n_components": 2, "metric": {"enabled": True}}
                        ),
                        Path(d) / "out",
                        use_ray=False,
                    )

    def test_hash_sample_matches_upstream_threshold_across_batches_and_roundoff(self):
        rng = np.random.default_rng(4)
        rows = rng.normal(size=(51, 4)).astype(np.float32)
        keys = pd.DataFrame(
            {
                "slide_id": ["a"] * 19 + ["b"] * 32,
                "x": np.arange(51, dtype=np.int64),
                "y": np.zeros(51, np.int64),
            }
        )
        shift = np.array([0.321, -0.21, 1.01, 2.2], np.float32)
        hashed = pd.util.hash_pandas_object(
            keys[["slide_id", "x", "y"]], index=False, hash_key=f"{42:016d}"[-16:]
        ).to_numpy()
        selected = hashed / 2.0**64 < 17 / len(rows)
        expected = (rows - shift)[selected]
        sample1, meta1 = metric.sample_metric_rows(
            _FixtureDataset(rows, keys),
            len(rows),
            shift,
            batch_size=7,
            n_tiles=17,
            seed=42,
        )
        sample2, meta2 = metric.sample_metric_rows(
            _FixtureDataset(rows, keys),
            len(rows),
            shift,
            batch_size=30,
            n_tiles=17,
            seed=42,
        )
        np.testing.assert_array_equal(sample1, expected)
        np.testing.assert_array_equal(sample1, sample2)
        self.assertEqual(meta1, meta2)
        self.assertEqual(meta1["actual_tiles"], int(selected.sum()))
        self.assertEqual(meta1["requested_tiles"], 17)
        self.assertEqual(meta1["population_tiles"], 51)
        digest = hashlib.sha256()
        for row in keys.loc[selected].itertuples(index=False):
            digest.update(
                (
                    json.dumps([row.slide_id, row.x, row.y], separators=(",", ":"))
                    + "\n"
                ).encode()
            )
        self.assertEqual(meta1["selected_keys_sha256"], digest.hexdigest())


class MetricDictionaryTests(unittest.TestCase):
    def test_anisotropic_encoding_and_signed_raw_decode(self):
        from ml.concept_explainability.upstream import ConceptAutoencoder, nnls_batch

        h = np.array([[1, -1, 0], [0, 1, -1]], np.float32)
        shift = np.array([0.2, -0.3, 0.6], np.float32)
        t = np.diag([0.4, 2.0, 0.8]).astype(np.float32)
        rows = np.array([[1.4, -0.1, -2], [-0.5, 2, -1]], np.float32)
        autoencoder = ConceptAutoencoder(h, shift, False, 200, transform=t)
        codes = autoencoder.encode(rows)
        np.testing.assert_array_equal(
            codes, nnls_batch((rows - shift) @ t.T, h @ t.T, 200)
        )
        np.testing.assert_array_equal(autoencoder.decode(codes), shift + codes @ h)
        self.assertTrue(np.any(autoencoder.decode(codes) < 0))
        self.assertFalse(
            np.allclose(codes, ConceptAutoencoder(h, shift, False, 200).encode(rows))
        )

    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=lambda sources: {
            level: Path(uri) for level, uri in sources.items()
        },
    )
    def test_metric_fit_inverse_maps_before_gauge_and_validates_artifacts(
        self, _download
    ):
        from test_concept_evaluation import fixture_source

        from ml.concept_explainability.model import file_sha256
        from ml.concept_explainability.upstream import gauge_fix_dictionary

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = fixture_source(root)
            model_source, _ = metric_source(root, 2560)
            source.config.ml = model_source.config.ml
            source.checkpoint_path = model_source.checkpoint_path
            source.checkpoint_sha256 = model_source.checkpoint_sha256
            settings = OmegaConf.create(
                {
                    "n_components": 2,
                    "metric": {
                        "enabled": True,
                        "n_tiles": 20,
                        "ridge": 0.3,
                        "attention_weight": 1,
                    },
                    "nmf": {
                        "batch_size": 32,
                        "shuffle_buffer_size": 32,
                        "random_state": 42,
                        "epochs": 1,
                        "nnls_iter": 10,
                        "gauge_fix": True,
                    },
                    "semi_nmf": {"forget": 0.95, "ridge": 1e-6},
                }
            )
            transform = np.eye(2560, dtype=np.float32)
            transform[0, 0], transform[1, 1] = 0.5, 2
            h_fit = np.zeros((2, 2560), np.float32)
            h_fit[:, :3] = [[2, -1, 1], [1, 3, -2]]
            before = source.checkpoint_path.read_bytes()
            with (
                patch(
                    "ml.concept_explainability.evaluation.classifier_output_metric",
                    return_value=transform,
                ),
                patch(
                    "ml.concept_explainability.evaluation.fit_semi_nmf_online",
                    return_value=h_fit,
                ) as fit,
            ):
                fit_dictionary(source, settings, root / "dictionary", use_ray=False)
            np.testing.assert_array_equal(fit.call_args.kwargs["transform"], transform)
            expected = gauge_fix_dictionary(
                (h_fit @ np.linalg.inv(transform).T).astype(np.float32)
            )
            dictionary = load_dictionary(root / "dictionary", source=source)
            np.testing.assert_array_equal(dictionary.dictionary, expected)
            np.testing.assert_array_equal(dictionary.transform, transform)
            self.assertEqual(source.checkpoint_path.read_bytes(), before)
            manifest_path = root / "dictionary/manifest.json"
            original = json.loads(manifest_path.read_text())
            self.assertEqual(original["format_version"], 2)
            self.assertFalse(original["metric"]["attention_applicable"])
            self.assertEqual(original["metric"]["sample"]["split"], "train")
            self.assertEqual(original["metric"]["sample"]["population_tiles"], 32)
            self.assertEqual(
                original["metric"]["transform"]["sha256"],
                file_sha256(root / "dictionary/transform.npy"),
            )
            other = SourceRun(
                "other-model",
                source.training_sha,
                source.config,
                source.checkpoint_path,
                source.checkpoint_sha256,
            )
            with self.assertRaisesRegex(ValueError, "binding"):
                load_dictionary(root / "dictionary", source=other)
            for changed in (
                np.eye(3, dtype=np.float32),
                np.full((2560, 2560), np.nan, np.float32),
                np.eye(2560, dtype=np.float64),
            ):
                np.save(root / "dictionary/transform.npy", changed)
                broken = json.loads(json.dumps(original))
                broken["metric"]["transform"]["sha256"] = file_sha256(
                    root / "dictionary/transform.npy"
                )
                manifest_path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, "transform"):
                    load_dictionary(root / "dictionary")
            (root / "dictionary/transform.npy").unlink()
            with self.assertRaisesRegex(ValueError, "transform"):
                load_dictionary(root / "dictionary")
            np.save(root / "dictionary/transform.npy", transform)
            original["metric"]["binding"]["checkpoint_sha256"] = "bad"
            manifest_path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dictionary(root / "dictionary")

    def test_metric_identity_changes_with_transform_or_source(self):
        h = np.array([[1, -1]], np.float32)
        mu = np.zeros(2, np.float32)
        t = np.eye(2, dtype=np.float32)
        baseline = metric.metric_dictionary_id(h, mu, t, {"checkpoint": "a"})
        self.assertNotEqual(
            baseline, metric.metric_dictionary_id(h, mu, t * 2, {"checkpoint": "a"})
        )
        self.assertNotEqual(
            baseline, metric.metric_dictionary_id(h, mu, t, {"checkpoint": "b"})
        )


class MetricSmokeTests(unittest.TestCase):
    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=lambda sources: {
            level: Path(uri) for level, uri in sources.items()
        },
    )
    def test_actual_synthetic_metric_fit_load_and_native_evaluation(self, _download):
        from test_concept_evaluation import fixture_source

        from ml.concept_explainability.evaluation import evaluate_concepts

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = fixture_source(root)
            native, original = metric_source(root, 2560)
            source.config.ml = native.config.ml
            source.checkpoint_path, source.checkpoint_sha256 = (
                native.checkpoint_path,
                native.checkpoint_sha256,
            )
            settings = OmegaConf.create(
                {
                    "n_components": 2,
                    "metric": {
                        "enabled": True,
                        "n_tiles": 20000,
                        "ridge": 0.3,
                        "attention_weight": 1,
                    },
                    "nmf": {
                        "batch_size": 32,
                        "shuffle_buffer_size": 32,
                        "random_state": 42,
                        "epochs": 1,
                        "nnls_iter": 10,
                        "gauge_fix": True,
                    },
                    "semi_nmf": {"forget": 0.95, "ridge": 1e-6},
                }
            )
            fit_dictionary(source, settings, root / "dictionary", use_ray=False)
            dictionary = load_dictionary(root / "dictionary", source=source)
            self.assertIsNotNone(dictionary.transform)
            self.assertTrue(np.isfinite(dictionary.dictionary).all())
            evaluate_concepts(
                source, root / "dictionary", "val", root / "evaluation-val"
            )
            self.assertNotIn("test", source.datasets)
            manifest = json.loads((root / "evaluation-val/manifest.json").read_text())
            self.assertTrue(manifest["metric_enabled"])
            self.assertTrue(np.isfinite(manifest["embedding_rmse"]))
            loaded = metric.preflight_metric_source(source)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])

    def test_ray_key_reader_sets_order_before_read_and_keeps_rows_aligned(self):
        from types import SimpleNamespace

        import ray.data

        from ml.concept_explainability.data import (
            _add_slide_identity,
            embedding_dataset,
        )

        dataset = SimpleNamespace(
            embeddings_dir=Path("/cached"), slides=pd.DataFrame({"name": ["b", "a"]})
        )
        options = ray.data.DataContext.get_current().execution_options
        previous = options.preserve_order
        options.preserve_order = False
        observed = []

        def read(paths, **kwargs):
            observed.append((options.preserve_order, paths, kwargs))
            return SimpleNamespace(
                map_batches=lambda fn, **kw: fn(
                    pd.DataFrame(
                        {
                            "embedding": [np.array([1, 2]), np.array([3, 4])],
                            "x": [2, 8],
                            "y": [0, 1],
                            "path": ["/cached/a.parquet", "/cached/b.parquet"],
                        }
                    )
                )
            )

        try:
            with patch("ray.data.read_parquet", side_effect=read):
                result = embedding_dataset(None, "train", dataset, include_keys=True)
            self.assertTrue(observed[0][0])
            self.assertEqual(observed[0][1], ["/cached/a.parquet", "/cached/b.parquet"])
            self.assertTrue(observed[0][2]["include_paths"])
            self.assertEqual(result.slide_id.tolist(), ["a", "b"])
            np.testing.assert_array_equal(np.stack(result.embedding), [[1, 2], [3, 4]])
            self.assertEqual(result.x.tolist(), [2, 8])
            self.assertIsNotNone(_add_slide_identity)
        finally:
            options.preserve_order = previous
