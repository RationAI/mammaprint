"""Hierarchical MIL over fixed-layout, lazily aligned embedding pyramids."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.transformer import TransformerMIL
from ml.typing import PyramidBag


class MultiScaleMIL(Aggregator[PyramidBag]):
    """Transformer within each pyramid, then gated attention across pyramids."""

    def __init__(
        self,
        feature_dim: int,
        level_ratios: Sequence[int],
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dim: int = 512,
    ) -> None:
        super().__init__()
        ratios = tuple(int(ratio) for ratio in level_ratios)
        if not ratios or any(ratio <= 0 for ratio in ratios):
            raise ValueError("level_ratios must contain positive integers.")
        self.feature_dim = feature_dim
        self.level_ratios = ratios
        self.token_count = sum(ratio * ratio for ratio in ratios)

        self.level_embedding = nn.Embedding(len(ratios), feature_dim)
        self.position_embedding = nn.Embedding(self.token_count, feature_dim)
        nn.init.trunc_normal_(self.level_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)

        counts = torch.tensor(ratios, dtype=torch.long).square()
        token_levels = torch.repeat_interleave(
            torch.arange(len(ratios), dtype=torch.long), counts
        )
        self.register_buffer("token_levels", token_levels, persistent=False)
        self.register_buffer(
            "token_positions",
            torch.arange(self.token_count, dtype=torch.long),
            persistent=False,
        )

        self.local_transformer = TransformerMIL(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.slide_attention = AttentionMIL(
            feature_dim=feature_dim,
            attention_dim=attention_dim,
        )

    @property
    def out_dim(self) -> int:
        return self.feature_dim

    def forward(self, bag: PyramidBag) -> tuple[Tensor, Tensor]:
        if bag.ndim != 3:
            raise ValueError(
                f"MultiScaleMIL expects a rank 3 (R, K, D) bag; got {bag.ndim}."
            )
        regions, tokens, features = bag.shape
        if regions == 0:
            raise ValueError("MultiScaleMIL requires at least one pyramid.")
        if tokens != self.token_count:
            raise ValueError(
                f"MultiScaleMIL expects {self.token_count} pyramid tokens; got "
                f"{tokens}."
            )
        if features != self.feature_dim:
            raise ValueError(
                f"MultiScaleMIL expects feature width {self.feature_dim}; got "
                f"{features}."
            )

        token_context = self.level_embedding(
            self.token_levels
        ) + self.position_embedding(self.token_positions)
        enriched = bag + token_context.unsqueeze(0)
        pyramid_vectors = torch.stack(
            [self.local_transformer(pyramid)[0] for pyramid in enriched]
        )
        return self.slide_attention(pyramid_vectors)


__all__ = ["MultiScaleMIL"]
