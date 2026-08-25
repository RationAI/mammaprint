"""Feature adapters for ordered multi-scale pyramid tokens."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def level_slices(level_ratios: Sequence[int]) -> tuple[slice, ...]:
    """Return contiguous token slices for square grids at each pyramid level."""
    counts = tuple(int(ratio) ** 2 for ratio in level_ratios)
    if not counts or any(ratio <= 0 for ratio in level_ratios):
        raise ValueError("level_ratios must contain positive integers.")

    slices: list[slice] = []
    start = 0
    for count in counts:
        stop = start + count
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


class NullFeatureAdapter(nn.Module):
    """Preserve raw pyramid features without parameters or allocation."""

    def forward(self, pyramids: Tensor) -> Tensor:
        return pyramids


class SharedLinearAdapter(nn.Module):
    """Normalize per level and project every token through one shared linear map."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        level_ratios: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.level_ratios = tuple(int(ratio) for ratio in level_ratios)
        self.slices = level_slices(self.level_ratios)
        self.token_count = sum(ratio * ratio for ratio in self.level_ratios)

        self.level_norms = nn.ModuleList(
            nn.LayerNorm(feature_dim) for _ in self.level_ratios
        )
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, pyramids: Tensor) -> Tensor:
        if pyramids.shape[-2] != self.token_count:
            raise ValueError(
                f"SharedLinearAdapter expects {self.token_count} tokens; got "
                f"{pyramids.shape[-2]}."
            )
        projected_levels = (
            self.dropout(
                self.activation(
                    self.projection(norm(pyramids[:, level_slice]))
                )
            )
            for norm, level_slice in zip(
                self.level_norms, self.slices, strict=True
            )
        )
        return torch.cat(tuple(projected_levels), dim=1)


NullAdapter = NullFeatureAdapter


__all__ = [
    "NullAdapter",
    "NullFeatureAdapter",
    "SharedLinearAdapter",
    "level_slices",
]
