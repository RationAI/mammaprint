# Third-party Imports
import torch


class DiceLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1
    ) -> float:
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        # dice coefficient of two sets (A, B): (2 * |A intersect B|) / (|A| + |B|)
        # It is a number in [0, 1] interval, the bigger the number, the more similar the sets
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # invert to make it a loss function (the bigger the dice, the smaller the error)
        return 1 - dice
