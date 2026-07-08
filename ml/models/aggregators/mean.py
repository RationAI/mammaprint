"""Mean-pooling aggregator."""

from torch import Tensor

from ml.models.aggregators.base import Aggregator
from ml.typing import Bag


class MeanPool(Aggregator[Bag]):
    """Averages tile features across the bag.

    Parameter-free; assigns no per-tile weights.

    Args:
        feature_dim: Dimensionality of the incoming tile features.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    @property
    def out_dim(self) -> int:
        return self.feature_dim

    def forward(self, bag: Tensor) -> tuple[Tensor, None]:
        return bag.mean(dim=0), None


__all__ = ["MeanPool"]
