import unittest

import torch

from ml.models.aggregators import AttentionMIL, MultiScaleMIL, TransformerMIL
from ml.typing import PyramidBag


class MultiScaleMILTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MultiScaleMIL(
            feature_dim=16,
            level_ratios=(1, 2, 4),
            num_heads=4,
            num_layers=1,
            attention_dim=8,
            dropout=0.0,
        )

    def test_pools_twenty_one_token_pyramids(self) -> None:
        pooled, attention = self.model(PyramidBag(torch.randn(3, 21, 16)))

        self.assertEqual(tuple(pooled.shape), (16,))
        self.assertEqual(tuple(attention.shape), (3,))
        torch.testing.assert_close(attention.sum(), torch.tensor(1.0))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_reuses_existing_local_and_slide_aggregators(self) -> None:
        self.assertIsInstance(self.model.local_transformer, TransformerMIL)
        self.assertIsInstance(self.model.slide_attention, AttentionMIL)

    def test_deprecated_legacy_entrypoint_points_to_modular_replacement(self) -> None:
        message = getattr(MultiScaleMIL, "__deprecated__", "")

        self.assertIn("ModularMultiScaleMIL", message)

    def test_rejects_wrong_token_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "21 pyramid tokens"):
            self.model(PyramidBag(torch.randn(2, 20, 16)))

    def test_rejects_wrong_feature_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature width 16"):
            self.model(PyramidBag(torch.randn(2, 21, 12)))

    def test_rejects_empty_or_wrong_rank_bag(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one pyramid"):
            self.model(PyramidBag(torch.randn(0, 21, 16)))
        with self.assertRaisesRegex(ValueError, "rank 3"):
            self.model(PyramidBag(torch.randn(21, 16)))

    def test_rejects_non_positive_ratios(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            MultiScaleMIL(feature_dim=16, level_ratios=(1, 0, 4), num_heads=4)


if __name__ == "__main__":
    unittest.main()
