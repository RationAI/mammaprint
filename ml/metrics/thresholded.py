"""Classification metrics computed from a regression output by thresholding.

For the MammaPrint index, the sign is the class boundary: index ``>= 0`` is
"a luminal" (class 1), ``< 0`` is "b luminal" (class 0) — matching
``_map_luminal_type`` and the notebook's LUMINAL_INTERVAL rule. This lets a
*regression* model report classification metrics (F1 / precision / recall /
sensitivity / specificity) during training: binarize both prediction and target at
the threshold, then delegate to a standard binary metric.
"""

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


class ThresholdedBinaryMetric(Metric):
    """Wraps a binary metric, binarizing continuous preds/targets at a threshold.

    Both ``preds`` and ``target`` are mapped to ``{0, 1}`` via ``value >= threshold``
    (positive class = "a luminal", index >= 0), then forwarded to the wrapped metric.

    Args:
        metric: The binary metric to delegate to (already instantiated).
        threshold: Decision boundary applied to both preds and targets (default 0.0).
    """

    def __init__(self, metric: Metric, threshold: float = 0.0) -> None:
        super().__init__()
        self.metric = metric
        self.threshold = threshold

    def update(self, preds: Tensor, target: Tensor) -> None:
        pred_cls = (preds >= self.threshold).long()
        target_cls = (target >= self.threshold).long()
        self.metric.update(pred_cls, target_cls)

    def compute(self) -> Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        super().reset()
        self.metric.reset()


def thresholded_f1(threshold: float = 0.0) -> ThresholdedBinaryMetric:
    return ThresholdedBinaryMetric(BinaryF1Score(), threshold)


def thresholded_precision(threshold: float = 0.0) -> ThresholdedBinaryMetric:
    return ThresholdedBinaryMetric(BinaryPrecision(), threshold)


def thresholded_recall(threshold: float = 0.0) -> ThresholdedBinaryMetric:
    """Recall (a.k.a. sensitivity) == TP / (TP + FN)."""
    return ThresholdedBinaryMetric(BinaryRecall(), threshold)


def thresholded_specificity(threshold: float = 0.0) -> ThresholdedBinaryMetric:
    """Specificity == TN / (TN + FP)."""
    return ThresholdedBinaryMetric(BinarySpecificity(), threshold)


__all__ = [
    "ThresholdedBinaryMetric",
    "thresholded_f1",
    "thresholded_precision",
    "thresholded_recall",
    "thresholded_specificity",
]
