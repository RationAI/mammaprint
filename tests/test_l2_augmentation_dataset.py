import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml.data import DataModule, SingleScaleDataset


class L2AugmentationDatasetTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        self.mapping = root / "mapping.csv"
        self.mapping.write_text(
            "path,split,mammaprint_index\n"
            "/train-slide.svs,train,0.2\n"
            "/empty.svs,train,-0.5\n"
            "/unlabeled.svs,train,\n"
            "/val-slide.svs,val,-0.4\n"
            "/test-slide.svs,test,0.1\n"
        )
        self.directories = {
            split: root / split for split in ("train", "val", "test", "train_augmented")
        }
        for directory in self.directories.values():
            directory.mkdir()
        self.uris = {key: path.as_uri() for key, path in self.directories.items()}
        self.write_bag("train", "train-slide", np.arange(6, dtype=np.int32))
        self.write_bag("train", "empty", np.array([], dtype=int))
        self.write_bag("train", "unlabeled", np.arange(6))
        self.write_bag("val", "val-slide", np.arange(6))
        self.write_bag("test", "test-slide", np.arange(6))
        self.write_bag("train_augmented", "train-slide", np.arange(6), offset=100)

    def write_bag(self, split, slide, coordinates, *, offset=0):
        pd.DataFrame(
            {
                "x": coordinates,
                "y": coordinates + 10,
                "embedding": [[float(value + offset)] for value in coordinates],
            }
        ).to_parquet(self.directories[split] / f"{slide}.parquet", index=False)

    def dataset_args(self, split, *, uris=None, cap=None):
        return {
            "split": split,
            "levels": {2: {"uris": self.uris if uris is None else uris}},
            "data_mapping": str(self.mapping),
            "label_mode": "index",
            "train_tile_cap": cap,
            "tile_seed": 13,
            "validate_index_split": True,
            "view_seed": 13,
        }

    def test_selects_complete_bags_and_retains_historical_eligibility(self):
        train = SingleScaleDataset(**self.dataset_args("train"))
        self.assertEqual(len(train), 1)
        self.assertEqual(train.empty_slides, ("empty",))
        observed_bags = set()
        for _ in range(24):
            bag, label, metadata = train[0]
            observed_bags.add(tuple(bag.flatten().tolist()))
            torch.testing.assert_close(label, torch.tensor([0.2]))
            self.assertEqual(metadata, {"slide_id": "train-slide"})
        self.assertEqual(observed_bags, {tuple(range(6)), tuple(range(100, 106))})

    def test_val_test_and_prediction_ignore_unavailable_augmentation(self):
        uris = {
            **self.uris,
            "train_augmented": self.uris["train_augmented"] + "/missing",
        }
        configs = {
            split: OmegaConf.create(
                {
                    "_target_": "ml.data.SingleScaleDataset",
                    **self.dataset_args(split, uris=uris),
                }
            )
            for split in ("val", "test")
        }
        module = DataModule(**configs)
        for stage, loader in (
            ("validate", module.val_dataloader),
            ("test", module.test_dataloader),
            ("predict", module.predict_dataloader),
        ):
            with self.subTest(stage=stage):
                module.setup(stage)
                bag, _, _ = next(iter(loader()))[0]
                torch.testing.assert_close(
                    bag.flatten(), torch.arange(6, dtype=torch.float)
                )

    def test_rejects_missing_extra_reordered_and_short_augmented_bags(self):
        for coordinates, error in (
            (np.array([1, 0, 2, 3, 4, 5]), "ordered coordinates differ.*train-slide"),
            (np.arange(5), "tile count differs.*train-slide"),
        ):
            with self.subTest(error=error):
                self.write_bag("train_augmented", "train-slide", coordinates)
                with self.assertRaisesRegex(ValueError, error):
                    SingleScaleDataset(**self.dataset_args("train"))

        augmented = self.directories["train_augmented"]
        (augmented / "train-slide.parquet").unlink()
        self.write_bag("train_augmented", "val-slide", np.arange(6))
        with self.assertRaisesRegex(
            ValueError, "Missing: .*train-slide.*Unexpected: .*val-slide"
        ):
            SingleScaleDataset(**self.dataset_args("train"))

    def test_probe_does_not_consume_either_training_stream(self):
        train = SingleScaleDataset(**self.dataset_args("train", cap=3))
        twin = SingleScaleDataset(**self.dataset_args("train", cap=3))
        clean_uris = {key: self.uris[key] for key in ("train", "val", "test")}
        clean = SingleScaleDataset(**self.dataset_args("train", uris=clean_uris, cap=3))
        probe, _, _ = train.load_sample(0, rng=np.random.default_rng(42))
        self.assertTrue((probe < 100).all())
        for _ in range(12):
            bag = train[0][0]
            torch.testing.assert_close(bag, twin[0][0])
            torch.testing.assert_close(bag.remainder(100), clean[0][0])

    def test_checkpoints_replay_view_state_with_or_without_shuffle_seed(self):
        def datamodule(train_seed):
            configs = {
                split: OmegaConf.create(
                    {
                        "_target_": "ml.data.SingleScaleDataset",
                        **self.dataset_args(split, cap=3),
                    }
                )
                for split in ("train", "val")
            }
            module = DataModule(train_seed=train_seed, **configs)
            module.setup("fit")
            return module

        def draw(module):
            return next(iter(module.train_dataloader()))[0][0].flatten().tolist()

        for train_seed in (None, 42):
            with self.subTest(train_seed=train_seed):
                original = datamodule(train_seed)
                draw(original)
                state = original.state_dict()
                self.assertIn("view_sampler", state)
                if train_seed is None:
                    self.assertNotIn("train_generator", state)
                expected = [draw(original) for _ in range(12)]
                restored = datamodule(train_seed)
                restored.load_state_dict(state)
                self.assertEqual([draw(restored) for _ in range(12)], expected)


if __name__ == "__main__":
    unittest.main()
