"""Within-level reductions for ordered multi-scale pyramid tokens."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.multiscale.adapter import level_slices
from ml.models.aggregators.transformer import TransformerMIL


class NullWithinLevel(nn.Module):
    """Preserve every pyramid token without parameters or allocation."""

    def forward(self, pyramids: Tensor) -> Tensor:
        return pyramids


class _WithinLevel(nn.Module):
    def __init__(self, level_ratios: Sequence[int]) -> None:
        super().__init__()
        self.level_ratios = tuple(int(ratio) for ratio in level_ratios)
        self.slices = level_slices(self.level_ratios)
        self.token_count = sum(ratio * ratio for ratio in self.level_ratios)

    def _validate(self, pyramids: Tensor) -> None:
        if pyramids.shape[-2] != self.token_count:
            raise ValueError(
                f"{type(self).__name__} expects {self.token_count} tokens; got "
                f"{pyramids.shape[-2]}."
            )


class MeanWithinLevel(_WithinLevel):
    """Average the tokens within each pyramid level."""

    def forward(self, pyramids: Tensor) -> Tensor:
        self._validate(pyramids)
        return torch.stack(
            tuple(pyramids[:, level_slice].mean(dim=1) for level_slice in self.slices),
            dim=1,
        )


class MaxWithinLevel(_WithinLevel):
    """Take the feature-wise maximum within each pyramid level."""

    def forward(self, pyramids: Tensor) -> Tensor:
        self._validate(pyramids)
        return torch.stack(
            tuple(pyramids[:, level_slice].amax(dim=1) for level_slice in self.slices),
            dim=1,
        )


class GatedAttentionWithinLevel(_WithinLevel):
    """Pool every level with one gated-attention module shared across levels."""

    def __init__(
        self,
        feature_dim: int,
        level_ratios: Sequence[int],
        attention_dim: int = 512,
    ) -> None:
        super().__init__(level_ratios)
        self.feature_dim = feature_dim
        self.attention = AttentionMIL(feature_dim, attention_dim)

    def forward(self, pyramids: Tensor) -> Tensor:
        self._validate(pyramids)
        pooled_levels = (
            self.attention.forward_batched(pyramids[:, level_slice])[0]
            for level_slice in self.slices
        )
        return torch.stack(tuple(pooled_levels), dim=1)


class SharedTransformerWithinLevel(_WithinLevel):
    """Pool every contextualized level with one shared one-layer Transformer."""

    def __init__(
        self,
        feature_dim: int,
        level_ratios: Sequence[int],
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(level_ratios)
        self.feature_dim = feature_dim
        self.transformer = TransformerMIL(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=1,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.position_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(ratio * ratio, feature_dim))
            for ratio in self.level_ratios
        )
        self.level_embedding = nn.Embedding(len(self.level_ratios), feature_dim)
        for table in self.position_embeddings:
            nn.init.trunc_normal_(table, std=0.02)
        nn.init.trunc_normal_(self.level_embedding.weight, std=0.02)

    def forward(self, pyramids: Tensor) -> Tensor:
        self._validate(pyramids)
        pooled_levels = []
        for level, (level_slice, positions) in enumerate(
            zip(self.slices, self.position_embeddings, strict=True)
        ):
            context = positions + self.level_embedding.weight[level]
            contextualized = pyramids[:, level_slice] + context.unsqueeze(0)
            pooled, _ = self.transformer.forward_batched(contextualized)
            pooled_levels.append(pooled)
        return torch.stack(pooled_levels, dim=1)


NullWithinLevelAggregator = NullWithinLevel
MeanWithinLevelAggregator = MeanWithinLevel
MaxWithinLevelAggregator = MaxWithinLevel


__all__ = [
    "GatedAttentionWithinLevel",
    "MaxWithinLevel",
    "MaxWithinLevelAggregator",
    "MeanWithinLevel",
    "MeanWithinLevelAggregator",
    "NullWithinLevel",
    "NullWithinLevelAggregator",
    "SharedTransformerWithinLevel",
]
