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
from torchmetrics.wrappers import MetricInputTransformer


class ThresholdedBinaryMetric(MetricInputTransformer):
    """Wraps a binary metric, binarizing continuous preds/targets at a threshold.

    Both ``preds`` and ``target`` are mapped to ``{0, 1}`` via ``value >= threshold``
    (positive class = "a luminal", index >= 0), then forwarded to the wrapped metric.

    Args:
        metric: The binary metric to delegate to (already instantiated).
        threshold: Decision boundary applied to predictions (default 0.0).
        target_threshold: Decision boundary applied to continuous targets. Defaults
            to ``threshold`` for regression-derived classification. Set this to
            ``0.5`` when targets are already binary labels and predictions are
            logits thresholded at zero.
    """

    def __init__(
        self,
        metric: Metric,
        threshold: float = 0.0,
        target_threshold: float | None = None,
    ) -> None:
        super().__init__(metric)
        self.threshold = threshold
        self.target_threshold = (
            threshold if target_threshold is None else target_threshold
        )

    def transform_pred(self, pred: Tensor) -> Tensor:
        return (pred >= self.threshold).long()

    def transform_target(self, target: Tensor) -> Tensor:
        return (target >= self.target_threshold).long()


def thresholded_f1(
    threshold: float = 0.0, target_threshold: float | None = None
) -> ThresholdedBinaryMetric:
    return ThresholdedBinaryMetric(BinaryF1Score(), threshold, target_threshold)


def thresholded_precision(
    threshold: float = 0.0, target_threshold: float | None = None
) -> ThresholdedBinaryMetric:
    return ThresholdedBinaryMetric(BinaryPrecision(), threshold, target_threshold)


def thresholded_recall(
    threshold: float = 0.0, target_threshold: float | None = None
) -> ThresholdedBinaryMetric:
    """Recall (a.k.a. sensitivity) == TP / (TP + FN)."""
    return ThresholdedBinaryMetric(BinaryRecall(), threshold, target_threshold)


def thresholded_specificity(
    threshold: float = 0.0, target_threshold: float | None = None
) -> ThresholdedBinaryMetric:
    """Specificity == TN / (TN + FP)."""
    return ThresholdedBinaryMetric(BinarySpecificity(), threshold, target_threshold)


__all__ = [
    "ThresholdedBinaryMetric",
    "thresholded_f1",
    "thresholded_precision",
    "thresholded_recall",
    "thresholded_specificity",
]
