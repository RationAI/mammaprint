import unittest

import torch

from ml.models.aggregators import ModularMultiScaleMIL
from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.transformer import TransformerMIL
from ml.typing import PyramidBag


class ModularMultiScaleMILPipelineTest(unittest.TestCase):
    def test_raw_tokens_mean_over_regions_then_final_flatten(self) -> None:
        model = self._model(
            feature_dim=2,
            hidden_dim=4,
            feature_adapter=None,
            within_level=None,
            level_fusion=None,
            slide_pool="mean",
            finalizer="flatten",
        )
        bag = torch.arange(2 * 21 * 2, dtype=torch.float32).reshape(2, 21, 2)

        pooled, attention = model(PyramidBag(bag))

        torch.testing.assert_close(pooled, bag.mean(dim=0).flatten())
        self.assertEqual(tuple(pooled.shape), (42,))
        self.assertEqual(model.out_dim, 42)
        self.assertIsNone(attention)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_raw_level_mean_concat_then_slide_mean(self) -> None:
        model = self._model(
            feature_dim=1,
            hidden_dim=4,
            feature_adapter=None,
            within_level="mean",
            level_fusion="concat",
            slide_pool="mean",
            finalizer=None,
        )
        region_0 = torch.cat(
            (
                torch.full((1, 1), 1.0),
                torch.full((4, 1), 2.0),
                torch.full((16, 1), 3.0),
            )
        )
        region_1 = torch.cat(
            (
                torch.full((1, 1), 5.0),
                torch.full((4, 1), 6.0),
                torch.full((16, 1), 7.0),
            )
        )

        pooled, attention = model(PyramidBag(torch.stack((region_0, region_1))))

        torch.testing.assert_close(pooled, torch.tensor([3.0, 4.0, 5.0]))
        self.assertEqual(model.out_dim, 3)
        self.assertIsNone(attention)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_project_level_mean_concat_then_gated_slide_pool(self) -> None:
        model = self._model(
            feature_dim=8,
            hidden_dim=4,
            feature_adapter="shared_linear",
            within_level="mean",
            level_fusion="concat",
            slide_pool="gated_attention",
            finalizer=None,
        )

        pooled, attention = model(PyramidBag(torch.randn(3, 21, 8)))

        self.assertEqual(tuple(pooled.shape), (12,))
        self.assertEqual(model.out_dim, 12)
        self.assertIsNotNone(attention)
        assert attention is not None
        self.assertEqual(tuple(attention.shape), (3,))
        torch.testing.assert_close(attention.sum(), torch.tensor(1.0))
        self.assertEqual(self._module_count(model, AttentionMIL), 1)
        self.assertEqual(self._module_count(model, TransformerMIL), 0)

    def test_project_shared_transformer_concat_then_gated_slide_pool(self) -> None:
        model = self._model(
            feature_dim=8,
            hidden_dim=4,
            feature_adapter="shared_linear",
            within_level="shared_transformer",
            level_fusion="concat",
            slide_pool="gated_attention",
            finalizer=None,
            transformer_num_heads=2,
            position_embedding="learned",
            level_embedding="learned",
        )

        pooled, attention = model(PyramidBag(torch.randn(3, 21, 8)))

        self.assertEqual(tuple(pooled.shape), (12,))
        self.assertEqual(model.out_dim, 12)
        self.assertIsNotNone(attention)
        assert attention is not None
        self.assertEqual(tuple(attention.shape), (3,))
        self.assertEqual(self._module_count(model, AttentionMIL), 1)
        self.assertEqual(self._module_count(model, TransformerMIL), 1)

    def test_project_joint_transformer_then_gated_slide_pool(self) -> None:
        model = self._model(
            feature_dim=8,
            hidden_dim=4,
            feature_adapter="shared_linear",
            within_level=None,
            level_fusion="joint_transformer",
            slide_pool="gated_attention",
            finalizer=None,
            transformer_num_heads=2,
            position_embedding="learned",
            level_embedding="learned",
        )

        pooled, attention = model(PyramidBag(torch.randn(3, 21, 8)))

        self.assertEqual(tuple(pooled.shape), (4,))
        self.assertEqual(model.out_dim, 4)
        self.assertIsNotNone(attention)
        assert attention is not None
        self.assertEqual(tuple(attention.shape), (3,))
        self.assertEqual(self._module_count(model, AttentionMIL), 1)
        self.assertEqual(self._module_count(model, TransformerMIL), 1)

    def test_feature_export_with_null_slide_pool_retains_regions(self) -> None:
        model = self._model(
            feature_dim=2,
            hidden_dim=4,
            feature_adapter=None,
            within_level="mean",
            level_fusion="concat",
            slide_pool=None,
            finalizer=None,
            feature_export=True,
        )

        exported, attention = model.export_features(PyramidBag(torch.randn(3, 21, 2)))

        self.assertEqual(tuple(exported.shape), (3, 6))
        self.assertIsNone(attention)

    def test_slide_max_reduces_only_regions_and_preserves_token_axes(self) -> None:
        model = self._model(
            feature_dim=2,
            hidden_dim=4,
            feature_adapter=None,
            within_level=None,
            level_fusion=None,
            slide_pool="max",
            finalizer="flatten",
        )
        bag = torch.randn(3, 21, 2)

        pooled, attention = model(PyramidBag(bag))

        torch.testing.assert_close(pooled, bag.amax(dim=0).flatten())
        self.assertIsNone(attention)

    def test_rejects_wrong_rank_empty_tokens_and_feature_width(self) -> None:
        model = self._model(
            feature_dim=8,
            hidden_dim=4,
            feature_adapter=None,
            within_level=None,
            level_fusion="flatten",
            slide_pool="mean",
            finalizer=None,
        )

        cases = (
            (torch.randn(21, 8), "rank 3"),
            (torch.randn(0, 21, 8), "at least one pyramid"),
            (torch.randn(2, 20, 8), "21 pyramid tokens"),
            (torch.randn(2, 21, 7), "feature width 8"),
        )
        for bag, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                model(PyramidBag(bag))

    @staticmethod
    def _module_count(model: ModularMultiScaleMIL, kind: type[torch.nn.Module]) -> int:
        return sum(isinstance(module, kind) for module in model.modules())

    @staticmethod
    def _model(**overrides: object) -> ModularMultiScaleMIL:
        arguments: dict[str, object] = {
            "feature_dim": 8,
            "hidden_dim": 4,
            "level_ratios": (1, 2, 4),
            "feature_adapter": None,
            "within_level": None,
            "level_fusion": None,
            "slide_pool": "mean",
            "finalizer": "flatten",
            "projection_dropout": 0.0,
            "attention_dim": 4,
            "transformer_num_layers": 1,
            "transformer_num_heads": 2,
            "transformer_mlp_ratio": 2.0,
            "transformer_dropout": 0.0,
            "position_embedding": None,
            "level_embedding": None,
            "feature_export": False,
        }
        arguments.update(overrides)
        return ModularMultiScaleMIL(**arguments)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
