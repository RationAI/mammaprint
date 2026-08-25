import subprocess
import unittest
from pathlib import Path

import torch

from ml.models.aggregators import MeanPool, MultiScaleMIL
from ml.models.encoders import IdentityEncoder
from ml.models.heads import LinearHead
from ml.models.module import MammaprintModule
from ml.typing import Bag, PyramidBag


class TypedModulePathsTest(unittest.TestCase):
    def test_flat_identity_path_still_predicts(self) -> None:
        module = MammaprintModule(
            encoder=IdentityEncoder(out_dim=16),
            aggregator=MeanPool(feature_dim=16),
            head=LinearHead(in_dim=16, dropout=0.0),
        )

        prediction, attention = module(Bag(torch.randn(4, 16)))

        self.assertEqual(tuple(prediction.shape), (1,))
        self.assertIsNone(attention)

    def test_pyramid_identity_path_predicts_with_region_attention(self) -> None:
        module = MammaprintModule(
            encoder=IdentityEncoder(out_dim=16),
            aggregator=MultiScaleMIL(
                feature_dim=16,
                level_ratios=(1, 2, 4),
                num_heads=4,
                num_layers=1,
                dropout=0.0,
            ),
            head=LinearHead(in_dim=16, dropout=0.0),
        )

        prediction, attention = module(PyramidBag(torch.randn(2, 21, 16)))

        self.assertEqual(tuple(prediction.shape), (1,))
        self.assertIsNotNone(attention)
        assert attention is not None
        self.assertEqual(tuple(attention.shape), (2,))

    def test_static_types_reject_flat_aggregator_for_typed_pyramid_encoder(self) -> None:
        fixture = Path(__file__).parents[1] / "typing" / "invalid_pyramid_pair.py"
        result = subprocess.run(
            [".venv/bin/mypy", str(fixture)],
            check=False,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertRegex(output.lower(), "cannot infer|incompatible")
        self.assertNotIn("expects no type arguments", output.lower())


if __name__ == "__main__":
    unittest.main()
