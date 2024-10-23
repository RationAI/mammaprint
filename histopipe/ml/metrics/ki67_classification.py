# Third-party Imports
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)


class Ki67MetricBase(Metric):
    """Ki67MetricBase converts continuous ki67 outputs to discrete categorical.

    Discretization enables categorical metrics for regression neural networks.

    Discretization is done by splitting outputs into n + 1 categories, where n is the
    length of thresholds. Each category represents an interval that is calculated from
    thresholds.

    TODO remove this later
    """

    def __init__(self, thresholds: list[int], metric: Metric) -> None:
        super().__init__()

        self.thresholds = thresholds
        self.metric = metric

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._relabel(preds, target)
        self.metric.update(preds, target)

    def compute(self) -> float:
        return self.metric.compute()

    def _relabel(self, preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        relabeled_preds = torch.zeros(
            preds.shape, dtype=torch.int8, device=preds.device
        )
        relabeled_target = torch.zeros(
            target.shape, dtype=torch.int8, device=target.device
        )

        i = 0  # in case for loop is skipped
        for i in range(1, len(self.thresholds)):
            low, high = self.thresholds[i - 1 : i + 1]

            relabeled_preds[(low < preds) & (preds <= high)] = i
            relabeled_target[(low < target) & (target <= high)] = i

        if self.thresholds:
            low = self.thresholds[-1]

            relabeled_preds[low < preds] = i + 1
            relabeled_target[low < target] = i + 1

        return relabeled_preds, relabeled_target


class Ki67Accuracy(Ki67MetricBase):
    def __init__(self, thresholds: list[int]) -> None:
        super().__init__(
            thresholds, MulticlassAccuracy(len(thresholds) + 1, average="weighted")
        )


class Ki67F1Score(Ki67MetricBase):
    def __init__(self, thresholds: list[int]) -> None:
        super().__init__(
            thresholds, MulticlassF1Score(len(thresholds) + 1, average="weighted")
        )


class Ki67Precision(Ki67MetricBase):
    def __init__(self, thresholds: list[int]) -> None:
        super().__init__(
            thresholds, MulticlassPrecision(len(thresholds) + 1, average="weighted")
        )


class Ki67Recall(Ki67MetricBase):
    def __init__(self, thresholds: list[int]) -> None:
        super().__init__(
            thresholds, MulticlassRecall(len(thresholds) + 1, average="weighted")
        )


class Ki67Specificity(Ki67MetricBase):
    def __init__(self, thresholds: list[int]) -> None:
        super().__init__(
            thresholds, MulticlassSpecificity(len(thresholds) + 1, average="weighted")
        )
