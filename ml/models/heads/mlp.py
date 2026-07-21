"""Two-layer MLP prediction head."""

from torch import Tensor, nn

from ml.models.heads.base import Head


class MLPHead(Head):
    """A two-layer MLP: dropout -> linear -> activation -> dropout -> linear.

    A deeper alternative to :class:`~ml.models.heads.LinearHead` for when the
    aggregated slide vector benefits from a non-linear readout (e.g. linear
    probing on frozen embeddings, where the head carries all task capacity).
    The output is raw (logits / unbounded scalar); any activation is applied by
    the module's ``output_activation``.

    Args:
        in_dim: Dimensionality of the incoming slide vector.
        out_dim: Number of outputs (``1`` for binary/regression, ``C`` for
            multiclass).
        hidden_dim: Width of the hidden layer.
        dropout: Dropout probability applied before each linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(in_dim=in_dim, out_dim=out_dim)
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


__all__ = ["MLPHead"]
