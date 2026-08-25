"""Lazy alignment of existing per-level embedding artifacts into pyramids."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ml.typing import Coordinate, LevelSpec


class PyramidDiagnostics(NamedTuple):
    """Counts recorded while complete anchors are selected for one slide."""

    candidates: int
    complete: int
    missing_by_level: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class PyramidLayout:
    """Validated level ordering and coordinate arithmetic for one pyramid shape."""

    anchor_level: int
    level_order: tuple[int, ...]
    ratios: tuple[int, ...]
    tile_extent: int

    @classmethod
    def from_levels(
        cls,
        levels: Mapping[str | int, LevelSpec],
        *,
        anchor_level: int,
        level_order: tuple[int, ...],
        tile_extent: int,
    ) -> "PyramidLayout":
        """Validate level cards and derive integer coarse-to-fine scale ratios."""
        if tile_extent <= 0:
            raise ValueError(f"tile_extent must be positive; got {tile_extent}.")
        if not level_order or level_order[0] != anchor_level:
            raise ValueError("level_order must start with anchor_level.")
        if len(set(level_order)) != len(level_order):
            raise ValueError("level_order must not contain duplicate levels.")

        cards = {int(level): card for level, card in levels.items()}
        missing = [level for level in level_order if level not in cards]
        if missing:
            raise ValueError(f"Missing level cards for levels {missing}.")

        anchor_mpp = float(cards[anchor_level]["mpp"])
        if anchor_mpp <= 0:
            raise ValueError(f"Level {anchor_level} mpp must be positive.")

        ratios: list[int] = []
        for level in level_order:
            card = cards[level]
            card_extent = int(card["tile_extent"])
            if card_extent != tile_extent:
                raise ValueError(
                    f"Level {level} tile_extent is {card_extent}, expected "
                    f"configured tile_extent {tile_extent}."
                )
            mpp = float(card["mpp"])
            if mpp <= 0:
                raise ValueError(f"Level {level} mpp must be positive.")
            raw_ratio = anchor_mpp / mpp
            ratio = round(raw_ratio)
            if ratio < 1 or not math.isclose(
                raw_ratio, ratio, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(
                    f"Level {level} must have a positive integer scale ratio to "
                    f"anchor level {anchor_level}; got {raw_ratio}."
                )
            ratios.append(ratio)

        return cls(anchor_level, level_order, tuple(ratios), tile_extent)

    @property
    def token_count(self) -> int:
        """Number of ordered tile tokens in each complete pyramid."""
        return sum(ratio * ratio for ratio in self.ratios)

    def coordinates_by_level(
        self, anchor: Coordinate
    ) -> dict[int, tuple[Coordinate, ...]]:
        """Return required coordinates, coarse-to-fine and row-major per level."""
        x, y = anchor
        return {
            level: tuple(
                (
                    ratio * x + column * self.tile_extent,
                    ratio * y + row * self.tile_extent,
                )
                for row in range(ratio)
                for column in range(ratio)
            )
            for level, ratio in zip(self.level_order, self.ratios, strict=True)
        }

    def coordinates(self, anchor: Coordinate) -> tuple[Coordinate, ...]:
        """Flatten all required coordinates in deterministic token order."""
        return tuple(
            coordinate
            for coordinates in self.coordinates_by_level(anchor).values()
            for coordinate in coordinates
        )


__all__ = ["PyramidDiagnostics", "PyramidLayout"]
