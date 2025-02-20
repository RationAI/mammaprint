import torch
from torch import Tensor
from torchmetrics import Metric


class SignAgreement(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds.float()
        target = target.float()

        signs_match = torch.sign(preds) == torch.sign(target)

        self.correct += signs_match.sum()
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total
