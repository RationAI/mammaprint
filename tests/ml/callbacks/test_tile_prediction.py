from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.masks.mask_builders.aggregation import MeanAggregator

from ml.callbacks._targets import PredictionTarget, report_item_id
from ml.callbacks.predictions_csv import PredictionCSVCallback
from ml.callbacks.tile_probability_heatmaps import (
    _display_values,
    _validate_coordinates,
    singleton_outputs,
)
from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
from ml.models.aggregators.transformer import TransformerMIL
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads.linear import LinearHead
from ml.models.heads.mlp import MLPHead
from ml.models.module import MammaprintModule


if TYPE_CHECKING:
    from ml.typing import MILSample, SlideMetadata


class _CaptureLogger:
    def __init__(self) -> None:
        self.csv: pd.DataFrame | None = None
        self.artifact_path: str | None = None
        self.table: dict[str, Any] | None = None
        self.table_path: str | None = None
        self.background: pd.DataFrame | None = None

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        if Path(local_path).suffix == ".csv":
            self.csv = pd.read_csv(local_path, dtype={"record_num": "string"})
        elif Path(local_path).suffix == ".parquet":
            self.background = pd.read_parquet(local_path)
        self.artifact_path = artifact_path

    def log_table(self, data: dict[str, Any], artifact_file: str) -> None:
        self.table = data
        self.table_path = artifact_file


def _module(aggregator: torch.nn.Module, mlp: bool) -> MammaprintModule:
    head = (
        MLPHead(in_dim=4, hidden_dim=5, out_dim=2, dropout=0.0)
        if mlp
        else LinearHead(in_dim=4, out_dim=2, dropout=0.0)
    )
    module = MammaprintModule(
        encoder=IdentityEncoder(out_dim=4),
        aggregator=cast("Any", aggregator),
        head=head,
    )
    return module.eval()


class SingletonOutputsTest(unittest.TestCase):
    def test_matches_direct_one_tile_bags_for_all_supported_models(self) -> None:
        torch.manual_seed(7)
        bag = torch.randn(9, 4)
        aggregators = (
            MeanPool(4),
            MaxPool(4),
            AttentionMIL(4, 3),
            TransformerMIL(4, num_heads=2, num_layers=2, dropout=0.0),
        )
        for aggregator in aggregators:
            for mlp in (False, True):
                with self.subTest(aggregator=type(aggregator).__name__, mlp=mlp):
                    module = _module(aggregator, mlp)
                    actual = singleton_outputs(module, bag, batch_size=3)
                    expected = torch.stack(
                        [module(tile.unsqueeze(0))[0] for tile in bag]
                    )
                    torch.testing.assert_close(actual, expected)

    def test_rejects_raw_image_bags(self) -> None:
        module = _module(MeanPool(4), mlp=False)
        with self.assertRaisesRegex(ValueError, "stored embedding bags"):
            singleton_outputs(module, torch.randn(2, 3, 8, 8), batch_size=2)


class PredictionCSVTest(unittest.TestCase):
    def test_joint_csv_contains_record_labels_and_both_predictions(self) -> None:
        callback = PredictionCSVCallback(label_mode="both")
        logger = _CaptureLogger()
        trainer = cast(
            "Trainer",
            SimpleNamespace(
                is_global_zero=True,
                world_size=1,
                logger=logger,
            ),
        )
        metadata = cast(
            "SlideMetadata",
            {
                "slide_id": "P1",
                "record_num": "0042",
                "slide_path": Path("/mnt/data/P1.ndpi"),
            },
        )
        batch: list[MILSample] = [
            (torch.randn(2, 4), torch.tensor([1.0, -0.25]), metadata)
        ]

        callback.on_predict_start(trainer, cast("Any", None))
        callback.on_predict_batch_end(
            trainer,
            cast("Any", None),
            {"predictions": torch.tensor([[0.0, -0.5]])},
            batch,
            0,
        )
        callback.on_predict_epoch_end(trainer, cast("Any", None))

        self.assertIsNotNone(logger.csv)
        row = cast("pd.DataFrame", logger.csv).iloc[0]
        self.assertEqual(row["record_num"], "0042")
        self.assertEqual(row["type_label"], 1)
        self.assertEqual(row["luminal_a_probability"], 0.5)
        self.assertEqual(row["mammaprint_index"], -0.25)
        self.assertEqual(row["predicted_mammaprint_index"], -0.5)
        self.assertEqual(row["report_item_id"], "P1")
        self.assertIsNotNone(logger.background)
        background = cast("pd.DataFrame", logger.background).iloc[0]
        self.assertEqual(background["record_num"], "0042")
        self.assertEqual(background["slide_path"], "/mnt/data/P1.ndpi")
        self.assertEqual(
            background["ground_truth"], "Luminal A | MammaPrint index: -0.25"
        )
        self.assertEqual(logger.artifact_path, "report/background")
        self.assertEqual(logger.table_path, "predictions/predictions.json")


class SpatialValidationTest(unittest.TestCase):
    def test_ratiopath_averages_overlapping_tile_predictions(self) -> None:
        builder = MaskBuilder(
            source_extents=(4, 6),
            source_tile_extent=4,
            output_tile_extent=1,
            stride=2,
            n_channels=1,
            aggregation=MeanAggregator,
        )
        try:
            builder.update_batch(
                np.array([[0.0], [1.0]], dtype=np.float32),
                np.array([[0, 0], [0, 2]], dtype=np.int64),
            )
            mask = builder.finalize()["mask"]
            np.testing.assert_allclose(mask[0], [[0.0, 0.5, 1.0]] * 2)
        finally:
            builder.cleanup()

    def test_coordinates_are_returned_in_y_x_order(self) -> None:
        metadata = cast(
            "SlideMetadata",
            {
                "slide_id": "P1",
                "x": torch.tensor([0, 112]),
                "y": torch.tensor([224, 0]),
                "stride": 112,
            },
        )
        coords = _validate_coordinates(metadata, source_extents=(500, 500))
        self.assertEqual(coords.tolist(), [[224, 0], [0, 112]])

    def test_out_of_bounds_coordinates_fail(self) -> None:
        metadata = cast(
            "SlideMetadata",
            {
                "slide_id": "P1",
                "x": torch.tensor([560]),
                "y": torch.tensor([0]),
                "stride": 112,
            },
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            _validate_coordinates(metadata, source_extents=(500, 500))

    def test_slide_id_is_sanitized(self) -> None:
        self.assertEqual(report_item_id("../../P 1"), "P_1")

    def test_regression_display_is_centered_on_zero(self) -> None:
        target = PredictionTarget("mammaprint_index", 0, False)
        values = _display_values(
            torch.tensor([[-1.0], [0.0], [1.0]]),
            target,
            classification_outputs_are_logits=True,
            regression_display_transform="sigmoid",
        )
        np.testing.assert_allclose(
            values,
            [0.26894143, 0.5, 0.7310586],
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
