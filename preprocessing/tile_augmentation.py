"""Reproducible slide-level views of square H&E tile crops."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Iterable


VIEWS = ("clean", "stain", "rotation", "stain-rotation")
RECIPE = "he-od-residual-0.90-1.10-cardinal-v1"
# Unit optical-density directions for hematoxylin and eosin.
_STAINS = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105]])
_STAINS /= np.linalg.norm(_STAINS, axis=1, keepdims=True)
_UNMIX = np.linalg.pinv(_STAINS)


@dataclass(frozen=True)
class TileView:
    kind: str
    seed: int
    split: str

    def __post_init__(self) -> None:
        if (
            self.kind not in VIEWS
            or self.seed < 0
            or self.split not in ("train", "val")
        ):
            raise ValueError("Invalid augmentation, seed, or split")
        if self.split == "val" and self.kind not in ("clean", "rotation"):
            raise ValueError("The validation probe supports only clean and 90° views")

    def provenance(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "split": self.split,
            "recipe": RECIPE,
        }

    def parameters(
        self, slide_id: str, device: torch.device
    ) -> tuple[int, torch.Tensor | None]:
        digest = hashlib.sha256(f"{RECIPE}\0{self.seed}\0{slide_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
        scales = rng.uniform(0.90, 1.10, size=2)
        turns = int(rng.integers(1, 4)) if self.split == "train" else 1
        stain = None
        if self.kind in ("stain", "stain-rotation"):
            # Add only the change in H&E density, retaining the original residual.
            delta = (_UNMIX * (scales - 1)) @ _STAINS
            stain = torch.as_tensor(delta, dtype=torch.float32, device=device)
        return (turns if "rotation" in self.kind else 0), stain


def transform_batch(
    images: torch.Tensor,
    groups: Iterable[int],
    parameters: list[tuple[int, torch.Tensor | None]],
) -> None:
    """Modify disposable uint8 NHWC input in place before RGB normalization."""
    offset = 0
    for group, items in groupby(groups):
        count = sum(1 for _ in items)
        tiles = images[offset : offset + count]
        offset += count
        turns, delta = parameters[group]
        if delta is not None:
            for block in tiles.split(256):
                rgb = block.float().add_(1)
                density = rgb.div(256).log_().neg_()
                multiplier = (density @ delta).neg_().exp_()
                adjusted = rgb.mul_(multiplier).sub_(1).round_().clamp_(0, 255)
                background = block.amin(dim=-1, keepdim=True) >= 230
                block.copy_(torch.where(background, block, adjusted.to(torch.uint8)))
        if turns:
            tiles.copy_(torch.rot90(tiles, turns, dims=(1, 2)))
