"""Select one column of a multi-task prediction/target before metricking.

In the ``both`` label mode the head emits ``(B, 2)`` and the target is ``(B, 2)``
— column 0 is the classification channel (logit / type label), column 1 the
regression channel (scalar / MammaPrint index). Standard torchmetrics expect a
single channel, so this wrapper slices out one column and delegates, letting the
existing binary and regression metrics be reused unchanged for the joint model.
"""

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.wrappers import MetricInputTransformer


class ColumnMetric(MetricInputTransformer):
    """Wraps a metric, feeding it one column of ``(B, 2)`` preds/targets.

    Args:
        metric: The metric to delegate to (already instantiated).
        column: Which column to select — ``0`` classification, ``1`` regression.
    """

    def __init__(self, metric: Metric, column: int) -> None:
        super().__init__(metric)
        self.column = column

    def transform_pred(self, pred: Tensor) -> Tensor:
        return pred[:, self.column]

    def transform_target(self, target: Tensor) -> Tensor:
        return target[:, self.column]


__all__ = ["ColumnMetric"]
