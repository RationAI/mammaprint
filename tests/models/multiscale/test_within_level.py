import unittest
from unittest.mock import patch

import torch
from torch import nn

from ml.models.aggregators.multiscale.within_level import (
    GatedAttentionWithinLevel,
    MaxWithinLevel,
    MeanWithinLevel,
    NullWithinLevel,
    SharedTransformerWithinLevel,
)
from ml.models.aggregators.transformer import TransformerMIL


class ParameterFreeWithinLevelTest(unittest.TestCase):
    def setUp(self) -> None:
        l5 = torch.tensor([[1.0, 9.0]])
        l4 = torch.tensor([[1.0, 8.0], [3.0, 6.0], [5.0, 4.0], [7.0, 2.0]])
        l3 = torch.stack((torch.arange(16.0), torch.arange(15.0, -1.0, -1.0)), dim=1)
        self.pyramids = torch.cat((l5, l4, l3)).unsqueeze(0)

    def test_null_returns_the_same_tensor_object(self) -> None:
        self.assertIs(NullWithinLevel()(self.pyramids), self.pyramids)

    def test_mean_pools_each_level_by_hand(self) -> None:
        pooled = MeanWithinLevel(level_ratios=(1, 2, 4))(self.pyramids)

        expected = torch.tensor([[[1.0, 9.0], [4.0, 5.0], [7.5, 7.5]]])
        torch.testing.assert_close(pooled, expected)
        self.assertEqual(sum(parameter.numel() for parameter in MeanWithinLevel((1, 2, 4)).parameters()), 0)

    def test_max_pools_each_level_by_hand(self) -> None:
        pooled = MaxWithinLevel(level_ratios=(1, 2, 4))(self.pyramids)

        expected = torch.tensor([[[1.0, 9.0], [7.0, 8.0], [15.0, 15.0]]])
        torch.testing.assert_close(pooled, expected)
        self.assertEqual(sum(parameter.numel() for parameter in MaxWithinLevel((1, 2, 4)).parameters()), 0)


class LearnedWithinLevelTest(unittest.TestCase):
    def test_gated_attention_batches_once_per_level_and_backpropagates(self) -> None:
        module = GatedAttentionWithinLevel(
            feature_dim=8,
            level_ratios=(1, 2, 4),
            attention_dim=4,
        )
        pyramids = torch.randn(3, 21, 8, requires_grad=True)

        with patch.object(
            module.attention,
            "forward_batched",
            wraps=module.attention.forward_batched,
        ) as batched:
            pooled = module(pyramids)

        self.assertEqual(tuple(pooled.shape), (3, 3, 8))
        self.assertEqual(
            [tuple(call.args[0].shape) for call in batched.call_args_list],
            [(3, 1, 8), (3, 4, 8), (3, 16, 8)],
        )
        for level_slice in (slice(0, 1), slice(1, 5), slice(5, 21)):
            _, weights = module.attention.forward_batched(pyramids[:, level_slice])
            torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        pooled.square().sum().backward()
        self.assertIsNotNone(pyramids.grad)
        self.assertIsNotNone(module.attention.attention_weights.weight.grad)

    def test_shared_transformer_batches_once_per_level_and_backpropagates(self) -> None:
        module = SharedTransformerWithinLevel(
            feature_dim=8,
            level_ratios=(1, 2, 4),
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        pyramids = torch.randn(3, 21, 8, requires_grad=True)

        with patch.object(
            module.transformer,
            "forward_batched",
            wraps=module.transformer.forward_batched,
        ) as batched:
            pooled = module(pyramids)

        self.assertEqual(tuple(pooled.shape), (3, 3, 8))
        self.assertEqual(
            [tuple(call.args[0].shape) for call in batched.call_args_list],
            [(3, 1, 8), (3, 4, 8), (3, 16, 8)],
        )
        self.assertEqual(
            [tuple(table.shape) for table in module.position_embeddings],
            [(1, 8), (4, 8), (16, 8)],
        )
        self.assertEqual(tuple(module.level_embedding.weight.shape), (3, 8))
        self.assertEqual(
            sum(isinstance(child, TransformerMIL) for child in module.modules()), 1
        )
        for level_slice in (slice(0, 1), slice(1, 5), slice(5, 21)):
            _, weights = module.transformer.forward_batched(pyramids[:, level_slice])
            torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        pooled.square().sum().backward()
        self.assertIsNotNone(pyramids.grad)
        self.assertIsNotNone(module.position_embeddings[2].grad)
        self.assertIsNotNone(module.level_embedding.weight.grad)

    def test_only_transformer_mode_has_position_and_level_embeddings(self) -> None:
        modules = (
            MeanWithinLevel((1, 2, 4)),
            MaxWithinLevel((1, 2, 4)),
            GatedAttentionWithinLevel(8, (1, 2, 4), attention_dim=4),
        )

        for module in modules:
            self.assertFalse(any(isinstance(child, nn.Embedding) for child in module.modules()))
            self.assertFalse(hasattr(module, "position_embeddings"))


if __name__ == "__main__":
    unittest.main()
