import unittest

import torch

from ml.models.aggregators.attention import AttentionMIL


class AttentionMILTest(unittest.TestCase):
    def test_batched_pool_matches_single_bag_api(self) -> None:
        torch.manual_seed(7)
        model = AttentionMIL(feature_dim=8, attention_dim=4).eval()
        bag = torch.randn(5, 8)

        single_pooled, single_attention = model(bag)
        batched_pooled, batched_attention = model.forward_batched(bag.unsqueeze(0))

        self.assertEqual(tuple(batched_pooled.shape), (1, 8))
        self.assertEqual(tuple(batched_attention.shape), (1, 5))
        torch.testing.assert_close(batched_pooled[0], single_pooled)
        torch.testing.assert_close(batched_attention[0], single_attention)

    def test_batched_attention_is_normalized_per_bag(self) -> None:
        model = AttentionMIL(feature_dim=8, attention_dim=4).eval()

        pooled, attention = model.forward_batched(torch.randn(3, 6, 8))

        self.assertEqual(tuple(pooled.shape), (3, 8))
        self.assertEqual(tuple(attention.shape), (3, 6))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
