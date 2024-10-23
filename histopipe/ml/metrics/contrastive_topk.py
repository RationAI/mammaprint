# Third-party Imports
import torch
from torch import Tensor
from torchmetrics import Metric


class ContrastiveTopK(Metric):
    """Calculates the TopK accuracy using the similarity matrix from the contrastive learning module.

    Description of this function
    """

    def __init__(self, k: int) -> None:
        super().__init__()

        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor = None) -> None:
        # Get the number of pairs
        num_pairs = preds.shape[0] // 2
        # Construct label vector - the corresponding pair is always offset by batch_size
        labels = (
            torch.concatenate(
                [torch.arange(num_pairs) + num_pairs, torch.arange(num_pairs)]
            )
            .reshape(-1, 1)
            .to(preds.device)
        )
        # Get indices of top K predictions
        topk_pred = torch.sort(preds.topk(self.k, dim=1)[1], 1)[0]
        # Calculate accuracy
        correct = (topk_pred == labels).any(1)
        self.correct += torch.sum(correct)
        self.total += torch.numel(correct)

    def compute(self) -> float:
        return self.correct.float() / self.total
