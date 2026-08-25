import unittest
from unittest.mock import patch

import torch

from ml.models.aggregators.multiscale.fusion import (
    ConcatLevelFusion,
    FlattenLevelFusion,
    GatedSumLevelFusion,
    JointTransformerLevelFusion,
    MeanLevelFusion,
    NullLevelFusion,
)


class ParameterFreeLevelFusionTest(unittest.TestCase):
    def test_null_preserves_the_tensor_object_and_semantic_axes(self) -> None:
        pyramids = torch.randn(2, 21, 4)

        fused = NullLevelFusion()(pyramids)

        self.assertIs(fused, pyramids)
        self.assertEqual(tuple(fused.shape), (2, 21, 4))

    def test_flatten_preserves_token_major_feature_order(self) -> None:
        pyramids = torch.arange(2 * 21 * 2, dtype=torch.float32).reshape(2, 21, 2)
        module = FlattenLevelFusion()

        fused = module(pyramids)

        self.assertEqual(tuple(fused.shape), (2, 42))
        torch.testing.assert_close(fused[0], torch.arange(42, dtype=torch.float32))
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 0)

    def test_concat_preserves_l5_l4_l3_order(self) -> None:
        level_vectors = torch.tensor(
            [
                [[5.0, 50.0], [4.0, 40.0], [3.0, 30.0]],
                [[15.0, 150.0], [14.0, 140.0], [13.0, 130.0]],
            ]
        )
        module = ConcatLevelFusion()

        fused = module(level_vectors)

        expected = torch.tensor(
            [
                [5.0, 50.0, 4.0, 40.0, 3.0, 30.0],
                [15.0, 150.0, 14.0, 140.0, 13.0, 130.0],
            ]
        )
        torch.testing.assert_close(fused, expected)
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 0)

    def test_mean_is_the_exact_equal_mean_of_three_levels(self) -> None:
        level_vectors = torch.tensor([[[1.0, 9.0], [4.0, 6.0], [7.0, 3.0]]])
        module = MeanLevelFusion()

        fused = module(level_vectors)

        torch.testing.assert_close(fused, torch.tensor([[4.0, 6.0]]))
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 0)


class LearnedLevelFusionTest(unittest.TestCase):
    def test_gated_sum_returns_normalized_three_level_weights(self) -> None:
        module = GatedSumLevelFusion(feature_dim=8, attention_dim=4)
        level_vectors = torch.randn(3, 3, 8, requires_grad=True)

        fused, weights = module(level_vectors)

        self.assertEqual(tuple(fused.shape), (3, 8))
        self.assertEqual(tuple(weights.shape), (3, 3))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        fused.square().sum().backward()
        self.assertIsNotNone(level_vectors.grad)
        self.assertIsNotNone(module.attention.attention_weights.weight.grad)

    def test_joint_transformer_uses_one_batched_call_for_all_pyramids(self) -> None:
        module = JointTransformerLevelFusion(
            feature_dim=8,
            level_ratios=(1, 2, 4),
            num_heads=2,
            num_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        pyramids = torch.randn(3, 21, 8, requires_grad=True)

        with patch.object(
            module.transformer,
            "forward_batched",
            wraps=module.transformer.forward_batched,
        ) as batched:
            fused, weights = module(pyramids)

        self.assertEqual(tuple(fused.shape), (3, 8))
        self.assertEqual(tuple(weights.shape), (3, 21))
        batched.assert_called_once()
        self.assertEqual(tuple(batched.call_args.args[0].shape), (3, 21, 8))
        fused.square().sum().backward()
        self.assertIsNotNone(pyramids.grad)
        self.assertIsNotNone(module.position_embedding.weight.grad)
        self.assertIsNotNone(module.level_embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
