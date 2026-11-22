"""Cohort-level ranking metrics for scalar MammaPrint index regression."""

import torch
from torch import Tensor
from torchmetrics import Metric, SpearmanCorrCoef
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.utilities.data import dim_zero_cat


class IndexSpearman(SpearmanCorrCoef):
    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.target)
        if not (torch.isfinite(preds).all() and torch.isfinite(targets).all()):
            return preds.new_tensor(torch.nan)
        if preds.unique().numel() < 2 or targets.unique().numel() < 2:
            return preds.new_tensor(torch.nan)
        return super().compute()


class IndexSignAUROC(Metric):
    """Use raw index scores for ranking and count target zero as positive."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    raw_scores: list[Tensor]
    index_targets: list[Tensor]

    def __init__(self) -> None:
        super().__init__()
        self.add_state("raw_scores", default=[], dist_reduce_fx="cat")
        self.add_state("index_targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.raw_scores.append(preds.flatten())
        self.index_targets.append(target.flatten())

    def compute(self) -> Tensor:
        scores = dim_zero_cat(self.raw_scores)
        targets = dim_zero_cat(self.index_targets)
        if not (torch.isfinite(scores).all() and torch.isfinite(targets).all()):
            return scores.new_tensor(torch.nan)
        signs = (targets >= 0).long()
        if signs.unique().numel() < 2:
            return scores.new_tensor(torch.nan)
        # AUROC depends only on order and ties. Keep exact ties from raw scores
        # while avoiding the functional metric's auto-sigmoid on large scores.
        _, ranks = scores.unique(sorted=True, return_inverse=True)
        return binary_auroc(ranks.float() / scores.numel(), signs)
