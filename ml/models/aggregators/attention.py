"""Gated-attention MIL aggregator.

Inspired by "Attention-based Deep Multiple Instance Learning" (Ilse et al., 2018,
arXiv:1802.04712) and adapted from the RationAI ``feature/mil`` reference
implementation to the encoder -> aggregator -> head contract.
"""

import torch
from torch import Tensor, nn

from ml.models.aggregators.base import Aggregator
from ml.typing import Bag


class AttentionMIL(Aggregator[Bag]):
    """Gated-attention pooling over a bag of tile features.

    Learns a scalar attention weight per tile via the gated mechanism
    ``tanh(V h) * sigmoid(U h)``, softmax-normalises the weights across the bag,
    and returns the attention-weighted sum of tile features. The per-tile weights
    are returned for interpretability (heatmaps).

    Args:
        feature_dim: Dimensionality of the incoming tile features.
        attention_dim: Width of the attention hidden layer.
    """

    def __init__(self, feature_dim: int, attention_dim: int = 512) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.attention_v = nn.Linear(feature_dim, attention_dim)
        self.attention_u = nn.Linear(feature_dim, attention_dim)
        self.attention_weights = nn.Linear(attention_dim, 1)

    @property
    def out_dim(self) -> int:
        return self.feature_dim

    def forward(self, bag: Tensor) -> tuple[Tensor, Tensor]:
        bag = self.norm(bag)  # (N, D)

        gate = torch.tanh(self.attention_v(bag)) * torch.sigmoid(
            self.attention_u(bag)
        )  # (N, attention_dim)
        scores = self.attention_weights(gate).squeeze(-1)  # (N,)
        attention = torch.softmax(scores, dim=0)  # (N,)

        pooled = (attention.unsqueeze(-1) * bag).sum(dim=0)  # (D,)
        return pooled, attention


__all__ = ["AttentionMIL"]
