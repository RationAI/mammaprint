"""Validation-only patch flipping for gated attention with a scalar linear head."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.trainer.states import TrainerFn
from torch import Tensor, nn
from torch.nn.functional import linear

from ml.models.aggregators.attention import AttentionMIL
from ml.models.heads.linear import LinearHead
from ml.models.module import MammaprintModule
from ml.typing import MILSample


@dataclass(frozen=True)
class AttentionDeletion:
    effective_tile_fraction: float
    top_decile_mass: float
    top_area: float
    bottom_area: float
    random_area: float

    @property
    def faithfulness(self) -> float:
        return self.top_area - self.random_area


def _removal_areas(logits: Tensor, tile_scores: Tensor, orders: Tensor) -> Tensor:
    fractions = (0, 5, 10, 20, 30, 40, 50)
    ranked_logits = logits[orders]
    ranked_scores = tile_scores[orders]
    intact_score = torch.dot(logits.softmax(dim=0), tile_scores)
    differences = [torch.zeros_like(ranked_scores[:, 0])]
    for percent in fractions[1:]:
        count = logits.numel() * percent // 100
        # Softmax over the retained suffix gives the exact attention of that bag.
        remaining_weights = ranked_logits[:, count:].softmax(dim=1)
        remaining_scores = (remaining_weights * ranked_scores[:, count:]).sum(dim=1)
        differences.append((remaining_scores - intact_score).abs())
    fractions_tensor = logits.new_tensor(fractions) / 100
    return torch.trapezoid(torch.stack(differences, dim=1), fractions_tensor, dim=1)


def attention_deletion(
    logits: Tensor,
    tile_scores: Tensor,
    generator: torch.Generator,
    random_repeats: int = 20,
) -> AttentionDeletion:
    """Measure score changes after removing top, bottom, and random tiles."""
    logits = logits.float()
    tile_scores = tile_scores.float()
    count = logits.numel()
    weights = logits.softmax(dim=0)
    if not torch.isfinite(weights).all() or not torch.isfinite(tile_scores).all():
        raise ValueError("Attention diagnostics require finite weights and scores.")
    # Permute first so tied logits do not inherit spatial ordering from the file.
    random_orders = torch.stack(
        [torch.randperm(count, generator=generator) for _ in range(random_repeats + 1)]
    ).to(logits.device)
    tie_order = random_orders[0]
    ranked_logits = logits[tie_order]
    top = tie_order[ranked_logits.argsort(descending=True, stable=True)]
    bottom = tie_order[ranked_logits.argsort(stable=True)]
    orders = torch.cat((top[None], bottom[None], random_orders[1:]), dim=0)
    areas = _removal_areas(logits, tile_scores, orders)
    decile_size = (count + 9) // 10
    return AttentionDeletion(
        effective_tile_fraction=1 / (count * weights.square().sum().item()),
        top_decile_mass=weights[top[:decile_size]].sum().item(),
        top_area=areas[0].item(),
        bottom_area=areas[1].item(),
        random_area=areas[2:].mean().item(),
    )


class AttentionValidationDiagnostics(Callback):
    """Log full-bag diagnostics for a checkpoint in an explicit validation run."""

    def __init__(
        self, seed: int, min_tiles: int = 20, random_repeats: int = 20
    ) -> None:
        super().__init__()
        if min_tiles < 2 or random_repeats < 1:
            raise ValueError(
                "Diagnostics require at least two tiles and one random repeat."
            )
        self.seed = seed
        self.min_tiles = min_tiles
        self.random_repeats = random_repeats
        self._active = False
        self._results: list[AttentionDeletion] = []
        self._slides = 0
        self._generator = torch.Generator()

    def on_validation_start(
        self, trainer: Trainer, pl_module: MammaprintModule
    ) -> None:
        self._active = trainer.state.fn == TrainerFn.VALIDATING
        if not self._active:
            return
        if trainer.world_size != 1:
            raise ValueError("Attention diagnostics require single-device validation.")
        if not (
            isinstance(pl_module.aggregator, AttentionMIL)
            and isinstance(pl_module.head, LinearHead)
            and pl_module.head.out_dim == 1
            and isinstance(pl_module.output_activation, nn.Identity)
        ):
            raise ValueError(
                "Attention diagnostics require gated attention and a scalar linear output."
            )
        self._results.clear()
        self._slides = 0
        self._generator.manual_seed(self.seed)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MammaprintModule,
        outputs: Any,
        batch: Sequence[MILSample],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._active:
            return
        aggregator = pl_module.aggregator
        head = pl_module.head
        assert isinstance(aggregator, AttentionMIL) and isinstance(head, LinearHead)
        self._slides += len(batch)
        for bag, _, _ in batch:
            if not isinstance(bag, Tensor) or len(bag) < self.min_tiles:
                continue
            values, logits = aggregator.attention_inputs(pl_module.encoder(bag))
            # A scalar linear head commutes with weighted pooling. Its constant bias
            # cancels from the differences, so projection and scoring run only once.
            tile_scores = linear(values, head.linear.weight).squeeze(-1)
            self._results.append(
                attention_deletion(
                    logits, tile_scores, self._generator, self.random_repeats
                )
            )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: MammaprintModule
    ) -> None:
        if not self._active:
            return
        metrics = {
            "valid/attention_slides": float(self._slides),
            "valid/attention_eligible_slides": float(len(self._results)),
        }
        if self._results:
            metrics.update(
                {
                    "valid/attention_effective_tile_fraction": median(
                        result.effective_tile_fraction for result in self._results
                    ),
                    "valid/attention_top_decile_mass": median(
                        result.top_decile_mass for result in self._results
                    ),
                    "valid/attention_faithfulness": mean(
                        result.faithfulness for result in self._results
                    ),
                    "valid/attention_top_deletion_area": mean(
                        result.top_area for result in self._results
                    ),
                    "valid/attention_bottom_deletion_area": mean(
                        result.bottom_area for result in self._results
                    ),
                    "valid/attention_random_deletion_area": mean(
                        result.random_area for result in self._results
                    ),
                }
            )
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)
