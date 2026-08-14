"""Auditable cohort metrics computed from lossless prediction tables."""

import math

import torch
from torch import Tensor
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)


def _divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else math.nan


def binary_logit_summary(logits: Tensor, targets: Tensor) -> dict[str, float]:
    """Return binary metrics and confusion counts using logit threshold zero."""
    logits = logits.detach().float().flatten().cpu()
    targets = targets.detach().long().flatten().cpu()
    if logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError("Binary logits and targets must be non-empty equal shapes.")
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("Binary logits must be finite.")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Binary targets must contain only 0 and 1.")

    predictions = (logits >= 0).long()
    tp = int(((predictions == 1) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    total = len(targets)
    accuracy = (tp + tn) / total
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)

    true_positive_rate = (tp + fn) / total
    predicted_positive_rate = (tp + fp) / total
    expected_agreement = (
        true_positive_rate * predicted_positive_rate
        + (1 - true_positive_rate) * (1 - predicted_positive_rate)
    )
    kappa = _divide(accuracy - expected_agreement, 1 - expected_agreement)
    auroc = math.nan
    if targets.unique().numel() == 2:
        # Sigmoid is monotonic, but makes the input contract explicit and avoids
        # TorchMetrics' per-update logit/probability auto-detection.
        auroc = float(BinaryAUROC()(torch.sigmoid(logits), targets))

    return {
        "slides": float(total),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "cohen_kappa": kappa,
        "auroc": auroc,
    }


def regression_summary(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    """Return continuous and zero-threshold classification metrics."""
    predictions = predictions.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError(
            "Regression predictions and targets must be non-empty equal shapes."
        )
    if not torch.all(torch.isfinite(predictions)) or not torch.all(
        torch.isfinite(targets)
    ):
        raise ValueError("Regression predictions and targets must be finite.")

    result = {
        "mae": float(MeanAbsoluteError()(predictions, targets)),
        "mse": float(MeanSquaredError()(predictions, targets)),
        "pearson": math.nan,
        "spearman": math.nan,
        "r2": math.nan,
    }
    if predictions.numel() >= 2:
        if predictions.std() > 0 and targets.std() > 0:
            result["pearson"] = float(PearsonCorrCoef()(predictions, targets))
            result["spearman"] = float(SpearmanCorrCoef()(predictions, targets))
        if targets.var() > 0:
            result["r2"] = float(R2Score()(predictions, targets))

    thresholded = binary_logit_summary(predictions, (targets >= 0).long())
    result.update(
        {
            f"thresholded_{name}": value
            for name, value in thresholded.items()
            if name != "slides"
        }
    )
    result["slides"] = float(len(targets))
    return result


__all__ = ["binary_logit_summary", "regression_summary"]
