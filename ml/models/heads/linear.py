"""Linear prediction head."""

from torch import Tensor, nn

from ml.models.heads.base import Head


class LinearHead(Head):
    """Dropout followed by a single linear layer.

    Works for binary classification / regression (``out_dim=1``) and multiclass
    classification (``out_dim=C``). The output is raw (logits / unbounded
    scalar); any activation is applied by the module's ``output_activation``.

    Args:
        in_dim: Dimensionality of the incoming slide vector.
        out_dim: Number of outputs (``1`` for binary/regression, ``C`` for
            multiclass).
        dropout: Dropout probability applied before the linear layer.
        weight_init_gain: Gain for Xavier uniform weight initialization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        dropout: float = 0.5,
        weight_init_gain: float = 1.0,
    ) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_dim, out_dim)

        nn.init.xavier_uniform_(self.linear.weight, gain=weight_init_gain)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(self.dropout(features))


__all__ = ["LinearHead"]
