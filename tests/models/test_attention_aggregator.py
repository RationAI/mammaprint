import unittest

import torch

from ml.models.aggregators.attention import AttentionMIL


class AttentionMILTest(unittest.TestCase):
    def test_batched_pool_matches_independent_legacy_reference(self) -> None:
        torch.manual_seed(7)
        model = AttentionMIL(feature_dim=8, attention_dim=4).eval()
        bags = torch.randn(3, 5, 8)

        batched_pooled, batched_attention = model.forward_batched(bags)
        expected = [self._legacy_forward(model, bag) for bag in bags]
        expected_pooled = torch.stack([pooled for pooled, _ in expected])
        expected_attention = torch.stack([attention for _, attention in expected])

        self.assertEqual(tuple(batched_pooled.shape), (3, 8))
        self.assertEqual(tuple(batched_attention.shape), (3, 5))
        torch.testing.assert_close(batched_pooled, expected_pooled)
        torch.testing.assert_close(batched_attention, expected_attention)

    def test_state_dict_keeps_pre_batched_parameter_names(self) -> None:
        model = AttentionMIL(feature_dim=8, attention_dim=4)

        self.assertEqual(
            set(model.state_dict()),
            {
                "norm.weight",
                "norm.bias",
                "attention_v.weight",
                "attention_v.bias",
                "attention_u.weight",
                "attention_u.bias",
                "attention_weights.weight",
                "attention_weights.bias",
            },
        )

    def test_batched_attention_is_normalized_per_bag(self) -> None:
        model = AttentionMIL(feature_dim=8, attention_dim=4).eval()

        pooled, attention = model.forward_batched(torch.randn(3, 6, 8))

        self.assertEqual(tuple(pooled.shape), (3, 8))
        self.assertEqual(tuple(attention.shape), (3, 6))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))

    @staticmethod
    def _legacy_forward(
        model: AttentionMIL, bag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = model.norm(bag)
        gate = torch.tanh(model.attention_v(normalized)) * torch.sigmoid(
            model.attention_u(normalized)
        )
        scores = model.attention_weights(gate).squeeze(-1)
        attention = torch.softmax(scores, dim=0)
        return (attention.unsqueeze(-1) * normalized).sum(dim=0), attention


if __name__ == "__main__":
    unittest.main()
