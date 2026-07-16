"""Transformer-based MIL aggregator (TransMIL-style).

Treats a slide's tiles as a sequence and lets them attend to each other via
multi-head self-attention — unlike mean/gated-attention pooling, which score each
tile independently. A learnable CLS token is prepended; after ``num_layers`` of
self-attention the CLS embedding summarises the whole bag and becomes the slide
vector. The CLS->tiles attention from the final layer is returned as the per-tile
weight (interpretability / heatmaps).

Reference: Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
Learning for Whole Slide Image Classification" (NeurIPS 2021, arXiv:2106.00908).
"""

import torch
from torch import Tensor, nn

from ml.models.aggregators.base import Aggregator
from ml.typing import Bag


class TransformerMIL(Aggregator[Bag]):
    """Self-attention over the tile bag with a learned CLS-token readout.

    Args:
        feature_dim: Dimensionality of the incoming tile features ``D``.
        num_heads: Number of attention heads (must divide ``feature_dim``).
        num_layers: Number of transformer encoder layers.
        mlp_ratio: Feed-forward hidden width as a multiple of ``feature_dim``.
        dropout: Dropout inside the transformer layers.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Learnable CLS token prepended to the tile sequence; its final embedding
        # is the slide vector.
        self.cls_token = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for small/deep stacks
        )
        # All but the last layer run as a standard stack; the last layer is applied
        # manually so we can pull out CLS->tiles attention weights for the readout.
        self.blocks = (
            nn.TransformerEncoder(
                encoder_layer, num_layers - 1, enable_nested_tensor=False
            )
            if num_layers > 1
            else None
        )
        self.final_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.final_norm = nn.LayerNorm(feature_dim)

    @property
    def out_dim(self) -> int:
        return self.feature_dim

    def forward(self, bag: Tensor) -> tuple[Tensor, Tensor]:
        # bag: (N, D). Prepend CLS -> sequence (1, N+1, D) (batch dim of 1).
        sequence = torch.cat([self.cls_token, bag], dim=0).unsqueeze(0)  # (1, N+1, D)

        if self.blocks is not None:
            sequence = self.blocks(sequence)  # (1, N+1, D)

        # Final self-attention layer, keeping the CLS<-all attention map.
        normed = self.final_norm(sequence)
        attended, attn_weights = self.final_attn(
            normed, normed, normed, need_weights=True, average_attn_weights=True
        )  # attended: (1, N+1, D); attn_weights: (1, N+1, N+1)
        sequence = sequence + attended  # residual

        slide_vector = sequence[0, 0]  # CLS embedding -> (D,)

        # Per-tile weight = how much the CLS query attends to each tile (drop the
        # CLS->CLS self-weight); renormalise so the tile weights sum to 1.
        cls_to_tiles = attn_weights[0, 0, 1:]  # (N,)
        attention = cls_to_tiles / cls_to_tiles.sum().clamp_min(1e-8)
        return slide_vector, attention


__all__ = ["TransformerMIL"]
