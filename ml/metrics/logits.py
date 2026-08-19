"""Binary metrics with explicit, batch-independent logit handling."""

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.wrappers import MetricInputTransformer


class BinaryLogitMetric(MetricInputTransformer):
    """Give a binary metric explicit, batch-independent logit semantics.

    TorchMetrics otherwise guesses whether a floating tensor contains logits or
    probabilities independently for every update. With the slide dataloader's
    batch size of one, that makes the same logit mean different things depending
    on whether it falls inside ``[0, 1]``. This wrapper makes the contract explicit:
    model predictions are logits and targets are already binary ``{0, 1}`` labels.
    Ranking metrics receive sigmoid probabilities. Confusion-matrix metrics set a
    ``decision_threshold`` and receive hard labels, using ``>=`` to match the
    prediction CSV exactly.
    """

    def __init__(self, metric: Metric, decision_threshold: float | None = None) -> None:
        super().__init__(metric)
        self.decision_threshold = decision_threshold

    def transform_pred(self, pred: Tensor) -> Tensor:
        if self.decision_threshold is not None:
            return (pred >= self.decision_threshold).long()
        return torch.sigmoid(pred)


__all__ = ["BinaryLogitMetric"]
