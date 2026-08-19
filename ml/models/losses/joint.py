"""Joint classification + regression loss for the ``both`` label mode.

The MammaPrint slide carries two aligned labels: a binary luminal type (a=1 / b=0)
and the continuous MammaPrint index. This loss trains a single model on both at
once — a standard hard-parameter-sharing multi-task setup where the encoder and
aggregator are shared and the head emits two outputs, one per task::

    total = w_cls * BCEWithLogits(logit, type) + w_reg * MSE(scalar, index)

Pair it with:

* a 2-output head (``out_dim=2``): column 0 is the classification **logit**,
  column 1 is the regression scalar;
* ``label_mode="both"``, so the target is ``(B, 2)`` = ``[type_label, index]``;
* ``output_activation=identity`` — BCE consumes raw logits, and the regression
  column must stay unbounded.

The two task losses live on different scales (BCE ~ O(1), MSE ~ the index
variance), so ``w_cls``/``w_reg`` trade them off. Defaults weight them equally;
tune them (or normalise the index) if one task dominates the gradient.
"""

import torch
from torch import Tensor, nn


class JointLoss(nn.Module):
    """Weighted sum of a binary-classification and a regression loss.

    Expects predictions and targets of shape ``(B, 2)`` where column 0 is the
    classification channel (logit / type label) and column 1 is the regression
    channel (scalar / MammaPrint index).

    Args:
        w_cls: Weight on the classification (BCE-with-logits) term.
        w_reg: Weight on the regression (MSE) term.
        pos_weight: Optional positive-class weight passed to
            :class:`~torch.nn.BCEWithLogitsLoss` to counter class imbalance.
    """

    def __init__(
        self,
        w_cls: float = 1.0,
        w_reg: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.w_cls = w_cls
        self.w_reg = w_reg
        pw: Tensor | None = None if pos_weight is None else torch.tensor([pos_weight])
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.reg_loss = nn.MSELoss()

    def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
        """Combine the per-task losses.

        Args:
            y_pred: Head output ``(B, 2)`` — ``[:, 0]`` class logits, ``[:, 1]``
                regression scalars.
            y: Targets ``(B, 2)`` — ``[:, 0]`` binary type labels, ``[:, 1]``
                MammaPrint indices.

        Returns:
            Scalar total loss ``w_cls * BCE + w_reg * MSE``.
        """
        cls = self.cls_loss(y_pred[:, 0], y[:, 0])
        reg = self.reg_loss(y_pred[:, 1], y[:, 1])
        return self.w_cls * cls + self.w_reg * reg


__all__ = ["JointLoss"]
