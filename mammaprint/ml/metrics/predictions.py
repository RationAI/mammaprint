import torch
from torch import Tensor
from torchmetrics import Metric

class Predictions(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("predictions", default=torch.tensor([]), persistent=False)

    def update(self, preds: Tensor, target: Tensor):
        self.predictions = preds.detach().clone()

    def compute(self):
        return self.predictions
