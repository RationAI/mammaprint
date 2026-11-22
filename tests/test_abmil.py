import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ml.data import DataModule, SingleScaleDataset
from ml.models.attention_diagnostics import _removal_areas, attention_deletion


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class ABMILTest(unittest.TestCase):
    def test_presets_compose_with_width_override_and_validation_only_fit(self):
        for pooling in ("mean", "attention"):
            with self.subTest(pooling=pooling):
                with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
                    config = compose(
                        config_name="ml",
                        overrides=[
                            f"+experiment=ml/train_mil_embeddings_abmil_{pooling}_regression",
                            "seed=0",
                            "ml.encoder.out_dim=128",
                        ],
                    )
                model = instantiate(config.ml)
                self.assertFalse(config.test_after_fit)
                self.assertEqual(config.datamodule.batch_size, 4)
                self.assertEqual(config.datamodule.eval_batch_size, 1)
                self.assertEqual(
                    config.trainer.callbacks.model_checkpoint.monitor, "valid/mae"
                )
                optimizer_config = model.configure_optimizers()
                self.assertIsInstance(optimizer_config["optimizer"], torch.optim.AdamW)
                self.assertEqual(
                    optimizer_config["lr_scheduler"]["scheduler"].T_max, 50
                )
                model.eval()
                prediction, weights = model(torch.randn(9, 2560))
                self.assertEqual(prediction.shape, (1,))
                if pooling == "attention":
                    self.assertEqual(weights.shape, (9,))
                    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
                else:
                    self.assertIsNone(weights)

    def test_projector_distinguishes_bags_with_equal_means(self):
        from ml.models.encoders import TileProjector

        projector = TileProjector(1, 1, 0).eval()
        with torch.no_grad():
            projector.projection.weight.fill_(1)
            projector.projection.bias.zero_()
        spread = torch.tensor([[-2.0], [2.0]])
        zero = torch.zeros_like(spread)
        torch.testing.assert_close(spread.mean(0), zero.mean(0))
        self.assertGreater(
            projector(spread).mean().item(), projector(zero).mean().item()
        )

    def test_tile_cap_samples_fresh_reproducible_rows_in_training_only(self):
        with tempfile.TemporaryDirectory() as workdir:
            directory = Path(workdir)
            mapping = directory / "mapping.csv"
            mapping.write_text(
                "path,mammaprint_index\n/slide.svs,0.2\n/short.svs,-0.4\n"
            )
            pd.DataFrame({"embedding": [[float(i)] for i in range(32)]}).to_parquet(
                directory / "slide.parquet"
            )
            pd.DataFrame({"embedding": [[1.0]]}).to_parquet(directory / "short.parquet")
            levels = {
                2: {
                    "uris": {
                        split: directory.as_uri() for split in ("train", "val", "test")
                    }
                }
            }

            def dataset(split):
                return SingleScaleDataset(
                    split, levels, mapping, "index", train_tile_cap=8, tile_seed=13
                )

            train = dataset("train")
            first = train[0][0]
            self.assertEqual(first.unique().numel(), 8)
            torch.testing.assert_close(first, dataset("train")[0][0])
            self.assertFalse(torch.equal(first, train[0][0]))
            self.assertEqual(train[1][0].shape, (1, 1))
            self.assertEqual(dataset("val")[0][0].shape, (32, 1))
            self.assertEqual(dataset("test")[0][0].shape, (32, 1))
            torch.testing.assert_close(
                train.regression_targets(), torch.tensor([0.2, -0.4])
            )

    def test_configured_rank_metrics_use_continuous_scores_and_handle_undefined(self):
        with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
            config = compose(
                config_name="ml",
                overrides=[
                    "+experiment=ml/train_mil_embeddings_abmil_mean_regression",
                    "seed=0",
                ],
            )
        model = instantiate(config.ml)
        metrics = model.valid_metrics
        for score, target in [(101.0, -1.0), (102.0, 0.0), (102.0, 1.0), (100.0, 1.0)]:
            metrics.update(torch.tensor([[score]]), torch.tensor([[target]]))
        result = metrics.compute()
        self.assertAlmostEqual(result["valid/sign_auroc"].item(), 2 / 3)
        self.assertAlmostEqual(result["valid/spearman"].item(), -1 / 18, places=6)
        metrics.reset()
        metrics.update(torch.tensor([[1.0], [1.0]]), torch.tensor([[0.0], [2.0]]))
        result = metrics.compute()
        self.assertTrue(torch.isnan(result["valid/sign_auroc"]))
        self.assertTrue(torch.isnan(result["valid/spearman"]))

    def test_verified_split_rejects_cross_split_duplicate_and_nonfinite_labels(self):
        with tempfile.TemporaryDirectory() as workdir:
            directory = Path(workdir)
            mapping = directory / "mapping.csv"
            pd.DataFrame({"embedding": [[1.0]]}).to_parquet(directory / "slide.parquet")
            levels = {2: {"uris": {"val": directory.as_uri()}}}
            cases = [
                ("/slide.svs,train,0.2\n", "outside its mapped split"),
                ("/slide.svs,val,0.2\n/slide.svs,train,0.2\n", "duplicate slide IDs"),
                ("/slide.svs,val,inf\n", "finite regression targets"),
            ]
            for rows, reason in cases:
                with self.subTest(reason=reason):
                    mapping.write_text(f"path,split,mammaprint_index\n{rows}")
                    with self.assertRaisesRegex(ValueError, reason):
                        SingleScaleDataset(
                            "val", levels, mapping, "index", validate_index_split=True
                        )

    def test_training_data_streams_continue_after_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as workdir:
            directory = Path(workdir)
            mapping = directory / "mapping.csv"
            mapping.write_text(
                "path,mammaprint_index\n/a.svs,0.2\n/b.svs,-0.4\n/c.svs,0.1\n"
            )
            for name in ("a", "b", "c"):
                pd.DataFrame({"embedding": [[float(i)] for i in range(32)]}).to_parquet(
                    directory / f"{name}.parquet"
                )

            train_config = OmegaConf.create(
                {
                    "_target_": "ml.data.SingleScaleDataset",
                    "split": "train",
                    "levels": {2: {"uris": {"train": directory.as_uri()}}},
                    "data_mapping": str(mapping),
                    "label_mode": "index",
                    "train_tile_cap": 8,
                    "tile_seed": 13,
                }
            )

            def datamodule():
                module = DataModule(train_seed=13, train=train_config, val=train_config)
                module.setup("fit")
                return module

            def batches(module):
                return [
                    (metadata["slide_id"], tiles.flatten().tolist())
                    for batch in module.train_dataloader()
                    for tiles, _, metadata in batch
                ]

            original = datamodule()
            first = batches(original)
            state = original.state_dict()
            expected = batches(original)
            restored = datamodule()
            restored.load_state_dict(state)
            self.assertNotEqual(first, expected)
            self.assertEqual(expected, batches(restored))

    def test_attention_deletion_detects_a_high_attention_signal(self):
        generator = torch.Generator().manual_seed(7)
        logits = torch.zeros(40)
        scores = torch.zeros(40)
        uniform = attention_deletion(logits, scores, generator)
        self.assertAlmostEqual(uniform.effective_tile_fraction, 1.0, places=6)
        self.assertEqual(uniform.faithfulness, 0.0)
        logits[0] = 100.0
        scores[0] = 1.0
        signal = attention_deletion(logits, scores, generator)
        self.assertLess(signal.effective_tile_fraction, 0.1)
        self.assertGreater(signal.faithfulness, 0.0)
        self.assertLess(signal.bottom_area, signal.top_area)

    def test_deletion_area_matches_direct_projected_subset_forward(self):
        from ml.models.aggregators import AttentionMIL
        from ml.models.heads import LinearHead

        generator = torch.Generator().manual_seed(7)
        bag = torch.randn(40, 4, generator=generator)
        aggregator = AttentionMIL(4, 3, normalize_features=False).eval()
        head = LinearHead(4).eval()
        with torch.no_grad():
            head.linear.bias.fill_(0.4)
            values, logits = aggregator.attention_inputs(bag)
            order = logits.argsort(descending=True)
            fast = _removal_areas(logits, head(values).squeeze(-1), order[None])
            fractions = (0, 5, 10, 20, 30, 40, 50)
            intact = head(aggregator(bag)[0])
            direct = torch.stack(
                [
                    (head(aggregator(bag[order[len(bag) * p // 100 :]])[0]) - intact)
                    .abs()
                    .squeeze()
                    for p in fractions
                ]
            )
            expected = torch.trapezoid(direct, torch.tensor(fractions) / 100)
        torch.testing.assert_close(fast[0], expected)


if __name__ == "__main__":
    unittest.main()
