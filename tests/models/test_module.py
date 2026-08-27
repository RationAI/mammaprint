import unittest

import torch

from ml.models.aggregators import MeanPool, ModularMultiScaleMIL
from ml.models.encoders import IdentityEncoder
from ml.models.heads import LinearHead
from ml.models.module import MammaprintModule
from ml.typing import PyramidBag, PyramidSample, PyramidSlideMetadata


class MammaprintModuleTest(unittest.TestCase):
    def test_predict_step_preserves_anchors_in_slide_attention_order(self) -> None:
        anchors = ((112, 224), (336, 448), (560, 672))
        metadata: PyramidSlideMetadata = {
            "slide_id": "slide-17",
            "anchors": anchors,
        }
        sample: PyramidSample = (
            PyramidBag(torch.randn(3, 21, 2)),
            torch.tensor([1.0]),
            metadata,
        )
        module = MammaprintModule(
            encoder=IdentityEncoder[PyramidBag](out_dim=2),
            aggregator=ModularMultiScaleMIL(
                feature_dim=2,
                hidden_dim=2,
                level_ratios=(1, 2, 4),
                feature_adapter=None,
                within_level="mean",
                level_fusion="mean",
                slide_pool="gated_attention",
                finalizer=None,
                attention_dim=2,
            ),
            head=LinearHead(in_dim=2, dropout=0.0),
        )

        result = module.predict_step([sample], batch_idx=0)

        self.assertIs(result["metadata"][0], metadata)
        attention = result["attentions"][0]
        self.assertIsNotNone(attention)
        assert attention is not None
        self.assertEqual(tuple(attention.shape), (len(anchors),))
        self.assertEqual(
            tuple(
                anchor
                for anchor, _ in zip(
                    result["metadata"][0]["anchors"], attention, strict=True
                )
            ),
            anchors,
        )

    def test_rejects_aggregator_and_head_width_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Aggregator output width 4 does not match head input width 3",
        ):
            MammaprintModule(
                encoder=IdentityEncoder(out_dim=4),
                aggregator=MeanPool(feature_dim=4),
                head=LinearHead(in_dim=3, dropout=0.0),
            )

    def test_feature_export_is_explicit_and_cannot_feed_a_head(self) -> None:
        aggregator = ModularMultiScaleMIL(
            feature_dim=2,
            hidden_dim=2,
            level_ratios=(1, 2, 4),
            feature_adapter=None,
            within_level="mean",
            level_fusion="concat",
            slide_pool=None,
            finalizer=None,
            feature_export=True,
        )
        bag = PyramidBag(torch.randn(3, 21, 2))

        exported, attention = aggregator.export_features(bag)

        self.assertEqual(tuple(exported.shape), (3, 6))
        self.assertIsNone(attention)
        with self.assertRaisesRegex(RuntimeError, "export_features"):
            aggregator(bag)
        with self.assertRaisesRegex(ValueError, "explicit export-only"):
            MammaprintModule(
                encoder=IdentityEncoder[PyramidBag](out_dim=2),
                aggregator=aggregator,
                head=LinearHead(in_dim=6, dropout=0.0),
            )

    def test_flat_feature_export_remains_export_only(self) -> None:
        aggregator = ModularMultiScaleMIL(
            feature_dim=2,
            hidden_dim=2,
            level_ratios=(1, 2, 4),
            feature_adapter=None,
            within_level="mean",
            level_fusion="concat",
            slide_pool="mean",
            finalizer=None,
            feature_export=True,
        )

        exported, _ = aggregator.export_features(PyramidBag(torch.randn(3, 21, 2)))

        self.assertEqual(tuple(exported.shape), (6,))
        with self.assertRaisesRegex(ValueError, "explicit export-only"):
            MammaprintModule(
                encoder=IdentityEncoder[PyramidBag](out_dim=2),
                aggregator=aggregator,
                head=LinearHead(in_dim=6, dropout=0.0),
            )


if __name__ == "__main__":
    unittest.main()
