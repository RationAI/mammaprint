import unittest

import torch
from torch import nn

from ml.models.aggregators.multiscale.adapter import (
    NullFeatureAdapter,
    SharedLinearAdapter,
)


class FeatureAdapterTest(unittest.TestCase):
    def test_null_adapter_returns_the_same_tensor_object(self) -> None:
        pyramids = torch.randn(2, 21, 8)

        adapted = NullFeatureAdapter()(pyramids)

        self.assertIs(adapted, pyramids)

    def test_shared_adapter_has_level_norms_and_one_projection(self) -> None:
        adapter = SharedLinearAdapter(
            feature_dim=2560,
            hidden_dim=512,
            level_ratios=(1, 2, 4),
            dropout=0.0,
        )

        norms = [
            module for module in adapter.modules() if isinstance(module, nn.LayerNorm)
        ]
        projections = [
            module for module in adapter.modules() if isinstance(module, nn.Linear)
        ]

        self.assertEqual(len(norms), 3)
        self.assertEqual(len({id(norm) for norm in norms}), 3)
        self.assertEqual(len(projections), 1)
        self.assertEqual(tuple(adapter(torch.randn(2, 21, 2560)).shape), (2, 21, 512))

    def test_shared_adapter_preserves_level_and_token_order(self) -> None:
        adapter = SharedLinearAdapter(
            feature_dim=3,
            hidden_dim=3,
            level_ratios=(1, 2, 4),
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            adapter.projection.weight.copy_(torch.eye(3))
            adapter.projection.bias.zero_()
            for level, norm in enumerate(adapter.level_norms, start=1):
                norm.weight.fill_(float(level))
                norm.bias.fill_(float(level) / 10)
        pyramids = torch.arange(2 * 21 * 3, dtype=torch.float32).reshape(2, 21, 3)
        pyramids[:, :, 1] *= -0.5
        pyramids[:, :, 2] = torch.remainder(pyramids[:, :, 2], 7)
        expected_parts = []
        for norm, level_slice in zip(
            adapter.level_norms,
            (slice(0, 1), slice(1, 5), slice(5, 21)),
            strict=True,
        ):
            expected_parts.append(
                torch.nn.functional.gelu(norm(pyramids[:, level_slice]))
            )

        adapted = adapter(pyramids)

        torch.testing.assert_close(adapted, torch.cat(expected_parts, dim=1))

    def test_shared_adapter_propagates_gradients(self) -> None:
        adapter = SharedLinearAdapter(
            feature_dim=8,
            hidden_dim=4,
            level_ratios=(1, 2, 4),
            dropout=0.0,
        )
        pyramids = torch.randn(2, 21, 8, requires_grad=True)

        adapter(pyramids).square().sum().backward()

        self.assertIsNotNone(pyramids.grad)
        self.assertIsNotNone(adapter.projection.weight.grad)
        for norm in adapter.level_norms:
            self.assertIsNotNone(norm.weight.grad)


if __name__ == "__main__":
    unittest.main()
