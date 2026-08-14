from __future__ import annotations

import unittest
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics import Metric

from ml.metrics import (
    ColumnMetric,
    binary_logit_summary,
    regression_summary,
    thresholded_f1,
)


METRICS_DIR = Path(__file__).resolve().parents[2] / "configs" / "ml" / "metrics"


def _configured_metric(name: str) -> Metric:
    config = OmegaConf.load(METRICS_DIR / f"{name}.yaml")
    metric = instantiate(next(iter(config.values())))
    if not isinstance(metric, Metric):
        raise TypeError(f"{name} did not instantiate a TorchMetric")
    return metric


def _stream(metric: Metric, predictions: torch.Tensor, targets: torch.Tensor) -> float:
    for prediction, target in zip(predictions, targets, strict=True):
        metric.update(prediction.unsqueeze(0), target.unsqueeze(0))
    return float(metric.compute())


class BinaryLogitMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        # At logit threshold zero this gives TN=FP=FN=TP=1. Several logits are
        # inside [0, 1], which exercises the former per-batch auto-detection bug.
        self.logits = torch.tensor([-2.0, 0.2, 0.8, -0.1])
        self.targets = torch.tensor([0, 0, 1, 1])

    def test_all_configured_binary_metrics_match_oracle_one_slide_at_a_time(
        self,
    ) -> None:
        expected = {
            "binary_accuracy": 0.5,
            "binary_auroc": 0.75,
            "binary_cohen_kappa": 0.0,
            "binary_f1": 0.5,
            "binary_precision": 0.5,
            "binary_recall": 0.5,
            "binary_specificity": 0.5,
        }
        for name, oracle in expected.items():
            with self.subTest(metric=name):
                actual = _stream(_configured_metric(name), self.logits, self.targets)
                self.assertAlmostEqual(actual, oracle)

    def test_one_slide_updates_equal_one_full_batch(self) -> None:
        for name in (
            "binary_accuracy",
            "binary_auroc",
            "binary_cohen_kappa",
            "binary_f1",
            "binary_precision",
            "binary_recall",
            "binary_specificity",
        ):
            with self.subTest(metric=name):
                streamed = _stream(_configured_metric(name), self.logits, self.targets)
                batched_metric = _configured_metric(name)
                batched_metric.update(self.logits, self.targets)
                self.assertAlmostEqual(streamed, float(batched_metric.compute()))

    def test_auditable_summary_contains_the_same_confusion_matrix(self) -> None:
        summary = binary_logit_summary(self.logits, self.targets)
        self.assertEqual(
            {key: summary[key] for key in ("tn", "fp", "fn", "tp")},
            {"tn": 1.0, "fp": 1.0, "fn": 1.0, "tp": 1.0},
        )
        self.assertEqual(summary["f1"], 0.5)
        self.assertEqual(summary["auroc"], 0.75)

    def test_zero_logit_uses_the_same_positive_boundary_as_prediction_csv(self) -> None:
        for name in (
            "binary_accuracy",
            "binary_f1",
            "binary_precision",
            "binary_recall",
        ):
            with self.subTest(metric=name):
                metric = _configured_metric(name)
                metric.update(torch.tensor([0.0]), torch.tensor([1]))
                self.assertEqual(float(metric.compute()), 1.0)


class JointMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = torch.tensor(
            [[-2.0, -0.5], [0.2, 0.2], [0.8, 0.8], [-0.1, 1.1]]
        )
        self.targets = torch.tensor(
            [[0.0, -0.3], [0.0, 0.1], [1.0, 0.7], [1.0, 1.0]]
        )

    def test_joint_f1_preserves_binary_zero_targets(self) -> None:
        self.assertAlmostEqual(
            _stream(
                _configured_metric("joint_f1"), self.predictions, self.targets
            ),
            0.5,
        )

    def test_joint_auroc_is_batch_independent(self) -> None:
        metric = _configured_metric("joint_auroc")
        streamed = _stream(metric, self.predictions, self.targets)
        metric.reset()
        metric.update(self.predictions, self.targets)
        self.assertAlmostEqual(streamed, 0.75)
        self.assertAlmostEqual(float(metric.compute()), 0.75)

    def test_all_joint_classification_metrics_match_oracle(self) -> None:
        expected = {
            "joint_accuracy": 0.5,
            "joint_auroc": 0.75,
            "joint_cohen_kappa": 0.0,
            "joint_f1": 0.5,
            "joint_precision": 0.5,
            "joint_recall": 0.5,
            "joint_specificity": 0.5,
        }
        for name, oracle in expected.items():
            with self.subTest(metric=name):
                actual = _stream(
                    _configured_metric(name), self.predictions, self.targets
                )
                self.assertAlmostEqual(actual, oracle)

    def test_regression_columns_remain_untransformed(self) -> None:
        mse = _configured_metric("joint_mse")
        mse.update(self.predictions, self.targets)
        expected = torch.mean((self.predictions[:, 1] - self.targets[:, 1]) ** 2)
        torch.testing.assert_close(mse.compute(), expected)

    def test_wrappers_clone_with_independent_state(self) -> None:
        metric = ColumnMetric(
            thresholded_f1(threshold=0.0, target_threshold=0.5), column=0
        )
        cloned = metric.clone()
        metric.update(self.predictions, self.targets)
        cloned.update(self.predictions[:1], self.targets[:1])
        self.assertAlmostEqual(float(metric.compute()), 0.5)
        self.assertEqual(float(cloned.compute()), 0.0)


class RegressionThresholdMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = torch.tensor([-0.8, -0.2, 0.9, -0.1])
        self.targets = torch.tensor([-1.0, -0.5, 0.3, 0.0])

    def test_zero_is_positive_for_continuous_index_predictions_and_targets(self) -> None:
        expected = {
            "thresholded_f1": 2 / 3,
            "thresholded_precision": 1.0,
            "thresholded_recall": 0.5,
            "thresholded_specificity": 1.0,
        }
        for name, oracle in expected.items():
            with self.subTest(metric=name):
                actual = _stream(
                    _configured_metric(name), self.predictions, self.targets
                )
                self.assertAlmostEqual(actual, oracle)

    def test_all_continuous_regression_metrics_match_auditable_summary(self) -> None:
        summary = regression_summary(self.predictions, self.targets)
        for name in ("mae", "mse", "pearson", "spearman", "r2"):
            with self.subTest(metric=name):
                metric = _configured_metric(name)
                metric.update(self.predictions, self.targets)
                self.assertAlmostEqual(float(metric.compute()), summary[name])

    def test_auditable_regression_summary(self) -> None:
        summary = regression_summary(self.predictions, self.targets)
        self.assertAlmostEqual(summary["mae"], 0.3)
        self.assertAlmostEqual(summary["mse"], 0.125)
        self.assertAlmostEqual(summary["thresholded_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
