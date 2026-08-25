"""Level-fusion stages for ordered multi-scale pyramid representations."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.multiscale.adapter import level_slices
from ml.models.aggregators.transformer import TransformerMIL


class NullLevelFusion(nn.Module):
    """Preserve pyramid tokens or level vectors and their semantic axes."""

    def forward(self, pyramids: Tensor) -> Tensor:
        return pyramids


class FlattenLevelFusion(nn.Module):
    """Flatten ordered token and feature axes within every pyramid."""

    def forward(self, pyramids: Tensor) -> Tensor:
        return pyramids.flatten(start_dim=1)


class ConcatLevelFusion(nn.Module):
    """Concatenate ordered L5, L4, and L3 vectors within every pyramid."""

    def forward(self, level_vectors: Tensor) -> Tensor:
        return level_vectors.flatten(start_dim=1)


class MeanLevelFusion(nn.Module):
    """Compute the equal mean of the three level vectors in every pyramid."""

    def forward(self, level_vectors: Tensor) -> Tensor:
        return level_vectors.mean(dim=1)


class GatedSumLevelFusion(nn.Module):
    """Learn normalized scalar weights over the three levels in each pyramid."""

    def __init__(self, feature_dim: int, attention_dim: int = 512) -> None:
        super().__init__()
        self.attention = AttentionMIL(feature_dim, attention_dim)

    def forward(self, level_vectors: Tensor) -> tuple[Tensor, Tensor]:
        return self.attention.forward_batched(level_vectors)


class JointTransformerLevelFusion(nn.Module):
    """Contextualize all ordered pyramid tokens in one batched Transformer call."""

    def __init__(
        self,
        feature_dim: int,
        level_ratios: Sequence[int],
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ratios = tuple(int(ratio) for ratio in level_ratios)
        slices = level_slices(ratios)
        self.token_count = sum(
            level_slice.stop - level_slice.start for level_slice in slices
        )
        self.transformer = TransformerMIL(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.position_embedding = nn.Embedding(self.token_count, feature_dim)
        self.level_embedding = nn.Embedding(len(ratios), feature_dim)
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.level_embedding.weight, std=0.02)

        counts = torch.tensor(ratios, dtype=torch.long).square()
        self.register_buffer(
            "token_levels",
            torch.repeat_interleave(torch.arange(len(ratios)), counts),
            persistent=False,
        )
        self.register_buffer(
            "token_positions",
            torch.arange(self.token_count, dtype=torch.long),
            persistent=False,
        )

    def forward(self, pyramids: Tensor) -> tuple[Tensor, Tensor]:
        if pyramids.shape[1] != self.token_count:
            raise ValueError(
                f"JointTransformerLevelFusion expects {self.token_count} tokens; "
                f"got {pyramids.shape[1]}."
            )
        context = self.position_embedding(self.token_positions)
        context = context + self.level_embedding(self.token_levels)
        return self.transformer.forward_batched(pyramids + context.unsqueeze(0))


__all__ = [
    "ConcatLevelFusion",
    "FlattenLevelFusion",
    "GatedSumLevelFusion",
    "JointTransformerLevelFusion",
    "MeanLevelFusion",
    "NullLevelFusion",
]
