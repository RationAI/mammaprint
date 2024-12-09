import torch
from torchmetrics import Metric

class SignAgreement(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = preds.float(), target.float()

        signs_match = torch.sign(preds) == torch.sign(target)

        self.correct += signs_match.sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
