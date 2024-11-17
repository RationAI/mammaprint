import torch
from torch import Tensor
from torchmetrics import Metric

class Predictions(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        # Initialize predictions as an empty tensor
        self.add_state("predictions", default=torch.tensor([]), persistent=False)

    def update(self, preds: Tensor, target: Tensor):
        # Store the latest batch of predictions
        # Since we're only storing the latest batch, we replace the contents each time
        self.predictions = preds.detach().clone()

    def compute(self):
        # Return the stored predictions
        return self.predictions
