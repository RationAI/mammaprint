import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml.concept_explainability.data import SlideInput, corpus_identity, iter_slides
from ml.concept_explainability.evaluation import (
    concept_push,
    displayed_outputs,
    evaluate_bag,
    fit_dictionary,
    load_dictionary,
)
from ml.concept_explainability.model import SourceRun
from ml.concept_explainability.upstream import ConceptAutoencoder, nnls_batch
from ml.models.aggregators import (
    AttentionMIL,
    MaxPool,
    MeanPool,
    SpatialTransformerMIL,
    TransformerMIL,
)
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads import LinearHead, MLPHead
from ml.models.module import MammaprintModule


def fixture_source(root, label_mode="type"):
    for split, offset in [("train", 0), ("val", 100), ("test", 10000)]:
        folder = root / split
        folder.mkdir()
        for slide in range(2):
            features = (
                np.random.default_rng(slide).normal(size=(16, 2560)).astype(np.float32)
                + offset
            )
            pd.DataFrame(
                {
                    "x": np.arange(16) * 4,
                    "y": np.zeros(16, dtype=int),
                    "embedding": list(features),
                }
            ).to_parquet(folder / f"{split}{slide}.parquet")
    pd.DataFrame(
        {
            "path": [
                str(root / f"{s}{i}.tiff")
                for s in ("train", "val", "test")
                for i in range(2)
            ],
            "type": ["a luminal", "b luminal"] * 3,
            "mammaprint_index": [0.3, -0.2] * 3,
        }
    ).to_csv(root / "labels.csv", index=False)
    config = OmegaConf.create(
        {
            "label_mode": label_mode,
            "feature_dim": 2560,
            "dataset": {
                "levels": {
                    "3": {
                        "mpp": 1.94,
                        "tile_extent": 4,
                        "stride": 2,
                        "uris": {s: str(root / s) for s in ("train", "val", "test")},
                    }
                },
                "data_mapping": str(root / "labels.csv"),
            },
            "datamodule": {
                s: {
                    "_target_": "ml.data.SingleScaleDataset",
                    "split": s,
                    "levels": "${dataset.levels}",
                    "data_mapping": "${dataset.data_mapping}",
                    "label_mode": "${label_mode}",
                }
                for s in ("train", "val", "test")
            },
        }
    )
    return SourceRun("fixture", None, config, None, None)


def local_download(sources):
    return {level: Path(uri) for level, uri in sources.items()}


class DictionaryTests(unittest.TestCase):
    def test_ibd_nnls_and_decode_parity(self):
        h = np.array([[1, -1, 0], [0, 1, -1]], dtype=np.float32)
        x = np.array([[2, -1, -1], [0, 3, -3]], dtype=np.float32)
        np.testing.assert_allclose(nnls_batch(x, h, 200), [[2, 1], [0, 3]], atol=1e-5)
        autoencoder = ConceptAutoencoder(h, np.zeros(3, dtype=np.float32), False, 200)
        np.testing.assert_allclose(
            autoencoder.decode(autoencoder.encode(x)), x, atol=1e-5
        )

    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=local_download,
    )
    def test_fit_uses_train_only(self, _download):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            training = np.concatenate(
                [slide.embeddings for slide in iter_slides(source, "train")]
            )
            settings = OmegaConf.create(
                {
                    "n_components": 16,
                    "nmf": {
                        "batch_size": 32,
                        "shuffle_buffer_size": 32,
                        "random_state": 42,
                        "epochs": 1,
                        "nnls_iter": 200,
                        "gauge_fix": True,
                    },
                    "semi_nmf": {"forget": 0.95, "ridge": 1e-6},
                }
            )
            fit_dictionary(source, settings, root / "dictionary", use_ray=False)
            dictionary = load_dictionary(root / "dictionary")
            np.testing.assert_allclose(
                dictionary.mu,
                training.astype(np.float64).mean(0).astype(np.float32),
                atol=1e-6,
            )
            self.assertEqual(dictionary.dictionary.shape, (16, 2560))
            self.assertTrue(np.any(dictionary.dictionary < 0))
            self.assertFalse((root / "dictionary" / "coefficients" / "test").exists())

    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=local_download,
    )
    def test_corpus_identity_includes_usable_slide_roster(self, _download):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            before = corpus_identity(source)
            labels = pd.read_csv(root / "labels.csv")
            labels.loc[0, "type"] = None
            labels.to_csv(root / "labels.csv", index=False)
            another_run = SourceRun("fixture", None, source.config, None, None)
            self.assertNotEqual(before, corpus_identity(another_run))

    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=local_download,
    )
    def test_source_datasets_are_resolved_once_per_split(self, download):
        with tempfile.TemporaryDirectory() as directory:
            source = fixture_source(Path(directory))
            corpus_identity(source)
            list(iter_slides(source, "train"))
            self.assertEqual(download.call_count, 1)

    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=local_download,
    )
    def test_encode_preserves_tile_keys(self, _download):
        with tempfile.TemporaryDirectory() as directory:
            source = fixture_source(Path(directory))
            slide = next(iter_slides(source, "val"))
            np.testing.assert_array_equal(slide.keys.row_index, np.arange(16))
            np.testing.assert_array_equal(slide.keys.x, np.arange(16) * 4)
            self.assertEqual(slide.embeddings.shape, (16, 2560))

    @patch(
        "ml.data.datasets.single_scale.download_level_sources",
        side_effect=local_download,
    )
    def test_empty_or_wrong_width_corpus_fails(self, _download):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            frame = pd.read_parquet(root / "train" / "train0.parquet")
            frame["embedding"] = [np.zeros(3)] * len(frame)
            frame.to_parquet(root / "train" / "train0.parquet")
            with self.assertRaisesRegex(ValueError, "2560"):
                next(iter_slides(source, "train"))


class InterventionTests(unittest.TestCase):
    def test_joint_probability_uses_sigmoid_only_on_column_zero(self):
        np.testing.assert_allclose(
            displayed_outputs(np.array([2.0, -0.7]), "both"),
            [torch.sigmoid(torch.tensor(2.0)).item(), -0.7],
        )

    def test_subtraction_stops_at_zero(self):
        codes = np.array([[0.1, 0.9], [0, 2]], dtype=np.float32)
        pushed = concept_push(codes, 0, 0.8, subtract=True)
        self.assertTrue((pushed >= 0).all())
        np.testing.assert_array_equal(pushed[:, 1], codes[:, 1])
        np.testing.assert_array_equal(codes[:, 0], np.array([0.1, 0], dtype=np.float32))

    def test_zero_push_matches_reconstructed_forward(self):
        rng = np.random.default_rng(42)
        z = rng.normal(size=(6, 4)).astype(np.float32)
        keys = pd.DataFrame(
            {
                "slide_id": ["s"] * 6,
                "row_index": np.arange(6),
                "x": np.arange(6) * 10,
                "y": np.arange(6) * 3,
            }
        )
        slide = SlideInput("s", z, keys, np.zeros(2), Path("unused"), 0, {})
        dictionary = ConceptAutoencoder(
            np.array([[1, -1, 0, 0], [0, 0, 1, -1]], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            False,
            200,
        )
        for factory in [
            lambda: MeanPool(4),
            lambda: MaxPool(4),
            lambda: AttentionMIL(4, 4),
            lambda: TransformerMIL(4, num_heads=2, num_layers=1, mlp_ratio=1),
            lambda: SpatialTransformerMIL(
                4, tile_extent=4, mpp=1, num_heads=2, num_layers=1, mlp_ratio=1
            ),
        ]:
            for head_type in (LinearHead, MLPHead):
                for mode in ("type", "index", "both"):
                    with self.subTest(aggregator=factory, head=head_type, mode=mode):
                        module = MammaprintModule(
                            IdentityEncoder(4),
                            factory(),
                            head_type(4, out_dim=2 if mode == "both" else 1),
                        )
                        original = {
                            k: v.clone() for k, v in module.state_dict().items()
                        }
                        result = evaluate_bag(
                            module, slide, dictionary, np.zeros(2), mode, "cpu"
                        )
                        np.testing.assert_allclose(result["add"], 0, atol=1e-7)
                        np.testing.assert_allclose(result["subtract"], 0, atol=1e-7)
                        for key, value in original.items():
                            torch.testing.assert_close(value, module.state_dict()[key])

    def test_spatial_coordinates_remain_unchanged(self):
        from ml.concept_explainability.evaluation import model_bag

        module = MammaprintModule(
            IdentityEncoder(4),
            SpatialTransformerMIL(4, tile_extent=4, mpp=1, num_heads=2, num_layers=1),
            LinearHead(4),
        )
        keys = pd.DataFrame({"x": [3, 70], "y": [5, 20]})
        slide = SlideInput(
            "s", np.ones((2, 4), np.float32), keys, np.zeros(1), Path("unused"), 0, {}
        )
        bag = model_bag(module, slide, np.zeros((2, 4), np.float32), "cpu")
        np.testing.assert_array_equal(bag[:, -2:].numpy(), keys[["x", "y"]].to_numpy())


if __name__ == "__main__":
    unittest.main()
