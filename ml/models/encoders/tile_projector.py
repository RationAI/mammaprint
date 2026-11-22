"""Learn a compact nonlinear representation of each frozen tile embedding."""

from torch import Tensor, nn

from ml.models.encoders.base import Encoder


class TileProjector(Encoder):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self._out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, tiles: Tensor) -> Tensor:
        return self.dropout(self.activation(self.projection(tiles)))
