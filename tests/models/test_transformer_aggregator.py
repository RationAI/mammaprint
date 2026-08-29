import unittest

import torch

from ml.models.aggregators.transformer import TransformerMIL


LEGACY_TWO_LAYER_STATE_KEYS = {
    "cls_token",
    "blocks.layers.0.self_attn.in_proj_weight",
    "blocks.layers.0.self_attn.in_proj_bias",
    "blocks.layers.0.self_attn.out_proj.weight",
    "blocks.layers.0.self_attn.out_proj.bias",
    "blocks.layers.0.linear1.weight",
    "blocks.layers.0.linear1.bias",
    "blocks.layers.0.linear2.weight",
    "blocks.layers.0.linear2.bias",
    "blocks.layers.0.norm1.weight",
    "blocks.layers.0.norm1.bias",
    "blocks.layers.0.norm2.weight",
    "blocks.layers.0.norm2.bias",
    "final_norm1.weight",
    "final_norm1.bias",
    "final_attn.in_proj_weight",
    "final_attn.in_proj_bias",
    "final_attn.out_proj.weight",
    "final_attn.out_proj.bias",
    "final_norm2.weight",
    "final_norm2.bias",
    "final_ff.0.weight",
    "final_ff.0.bias",
    "final_ff.3.weight",
    "final_ff.3.bias",
}


class TransformerMILTest(unittest.TestCase):
    def test_batched_pool_matches_independent_legacy_reference(self) -> None:
        torch.manual_seed(11)
        model = TransformerMIL(
            feature_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
        ).eval()
        bags = torch.randn(3, 5, 8)

        batched_pooled, batched_attention = model.forward_batched(bags)
        expected = [self._legacy_forward(model, bag) for bag in bags]
        expected_pooled = torch.stack([pooled for pooled, _ in expected])
        expected_attention = torch.stack([attention for _, attention in expected])

        self.assertEqual(tuple(batched_pooled.shape), (3, 8))
        self.assertEqual(tuple(batched_attention.shape), (3, 5))
        torch.testing.assert_close(batched_pooled, expected_pooled)
        torch.testing.assert_close(batched_attention, expected_attention)

    def test_default_two_layer_state_dict_strictly_loads_legacy_parameter_names(
        self,
    ) -> None:
        model = TransformerMIL(
            feature_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
        )
        legacy_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

        self.assertEqual(set(legacy_state), LEGACY_TWO_LAYER_STATE_KEYS)
        restored = TransformerMIL(
            feature_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
        )
        incompatible = restored.load_state_dict(legacy_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_strict_legacy_load_rejects_a_renamed_block_projection(self) -> None:
        model = TransformerMIL(
            feature_dim=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
        )
        renamed_state = dict(model.state_dict())
        old_name = "blocks.layers.0.self_attn.in_proj_weight"
        new_name = "blocks.layers.0.self_attn.qkv_weight"
        renamed_state[new_name] = renamed_state.pop(old_name)

        with self.assertRaises(RuntimeError) as raised:
            model.load_state_dict(renamed_state, strict=True)

        self.assertIn(
            f'Missing key(s) in state_dict: "{old_name}"', str(raised.exception)
        )
        self.assertIn(
            f'Unexpected key(s) in state_dict: "{new_name}"', str(raised.exception)
        )

    def test_batched_attention_is_normalized_per_bag(self) -> None:
        model = TransformerMIL(
            feature_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ).eval()

        pooled, attention = model.forward_batched(torch.randn(3, 6, 8))

        self.assertEqual(tuple(pooled.shape), (3, 8))
        self.assertEqual(tuple(attention.shape), (3, 6))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))

    @staticmethod
    def _legacy_forward(
        model: TransformerMIL, bag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.cat([model.cls_token, bag], dim=0).unsqueeze(0)
        if model.blocks is not None:
            sequence = model.blocks(sequence)

        normed = model.final_norm1(sequence)
        attended, weights = model.final_attn(
            normed,
            normed,
            normed,
            need_weights=True,
            average_attn_weights=True,
        )
        sequence = sequence + model.final_attn_dropout(attended)
        sequence = sequence + model.final_ff_dropout(
            model.final_ff(model.final_norm2(sequence))
        )
        cls_to_tiles = weights[0, 0, 1:]
        attention = cls_to_tiles / cls_to_tiles.sum().clamp_min(1e-8)
        return sequence[0, 0], attention


if __name__ == "__main__":
    unittest.main()
