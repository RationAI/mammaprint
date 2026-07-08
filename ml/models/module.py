"""Generic MammaPrint LightningModule.

Wires three independently swappable pieces — an :class:`~ml.models.encoders.base.Encoder`,
an :class:`~ml.models.aggregators.base.Aggregator`, and a :class:`~ml.models.heads.base.Head` —
into a multiple-instance-learning pipeline::

    bag -> encoder (per tile) -> aggregator (bag -> vector) -> head -> prediction

The module depends only on the abstract contracts, so encoders (identity for
precomputed embeddings, VGG16/Virchow2 for raw tiles), aggregators (mean, max,
gated attention, transformer) and heads are selected purely via Hydra config.
Loss, optimizer, LR scheduler and metrics are likewise config-instantiated,
following the ``feature/mil`` reference module.
"""

import functools
import logging
from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import Tensor, nn

from ml.models.aggregators.base import Aggregator
from ml.models.encoders.base import Encoder
from ml.models.heads.base import Head
from ml.typing import AnyBag, AnySample, SlideMetadata


logger = logging.getLogger(__name__)


class MammaprintModule(pl.LightningModule):
    """MIL module composing encoder -> aggregator -> head.

    Args:
        encoder: Per-tile feature extractor. ``IdentityEncoder`` on the
            precomputed-embeddings path; an image backbone on the end-to-end path.
        aggregator: Bag pooling reducing per-tile features to a slide vector.
        head: Prediction head mapping the slide vector to logits / a scalar.
        output_activation: Applied to the head output (e.g. ``Sigmoid`` for
            binary classification, ``Identity`` for regression / when the loss
            expects logits). Defaults to identity.
        loss: Loss module. ``None`` disables training (e.g. predict-only).
        optimizer: A ``functools.partial`` of an optimizer, bound to the model
            parameters in :meth:`configure_optimizers`.
        lr_scheduler: Optional dict with a ``scheduler`` partial and an optional
            ``monitor`` key, mirroring the ``feature/mil`` convention.
        metrics: Torchmetrics keyed by name; cloned per ``train``/``valid``/``test``.
    """

    def __init__(
        self,
        encoder: Encoder,
        aggregator: Aggregator[Any],
        head: Head,
        output_activation: nn.Module | None = None,
        loss: nn.Module | None = None,
        optimizer: functools.partial[torch.optim.Optimizer] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        metrics: dict[str, torchmetrics.Metric] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.head = head
        self.output_activation = output_activation or nn.Identity()
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        base = torchmetrics.MetricCollection(dict(metrics) if metrics else {})
        self.train_metrics = base.clone(prefix="train/")
        self.valid_metrics = base.clone(prefix="valid/")
        self.test_metrics = base.clone(prefix="test/")

    def forward(self, bag: AnyBag) -> tuple[Tensor, Tensor | None]:
        """Run one slide through encoder -> aggregator -> head.

        Args:
            bag: One slide, either a flat tile tensor ``(N, D)``/``(N, C, H, W)``
                (single level) or a multi-scale bag (list of regions, each mapping
                ``level -> (K, D)``/``(K, C, H, W)``). The encoder is applied per
                tile in both cases (identity for precomputed embeddings).

        Returns:
            ``(prediction, attention)`` where ``prediction`` has shape ``(out_dim,)``
            and ``attention`` is the aggregator's weights or ``None``.
        """
        encoded = self._encode(bag)
        pooled, attention = self.aggregator(encoded)  # (aggregator.out_dim,), attn|None
        prediction = self.output_activation(self.head(pooled))  # (out_dim,)
        return prediction, attention

    def _encode(self, bag: AnyBag) -> AnyBag:
        """Apply the encoder per tile, preserving the bag's single/multi structure."""
        if isinstance(bag, list):  # multi-scale: encode each level within each region
            return [
                {level: self.encoder(tiles) for level, tiles in region.items()}
                for region in bag
            ]
        return self.encoder(bag)  # flat bag: (N, ...) -> (N, encoder.out_dim)

    def _forward_batch(
        self, bags: Sequence[AnyBag]
    ) -> tuple[Tensor, list[Tensor | None]]:
        """Run a batch of variable-length bags, stacking per-slide predictions."""
        predictions: list[Tensor] = []
        attentions: list[Tensor | None] = []
        for bag in bags:
            prediction, attention = self(bag)
            predictions.append(prediction)
            attentions.append(attention)
        return torch.stack(predictions), attentions

    def _step(
        self,
        batch: list[AnySample],
        stage: str,
        metrics: torchmetrics.MetricCollection,
    ) -> Tensor:
        bags, labels, _ = _unpack_batch(batch)
        y = torch.stack(labels).to(self.device)
        y_pred, _ = self._forward_batch(bags)

        if self.loss is None:
            raise RuntimeError("A loss is required for training/validation steps.")

        loss = self.loss(y_pred, y)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True)

        metrics.update(y_pred, y)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: list[AnySample], batch_idx: int) -> Tensor:
        return self._step(batch, "train", self.train_metrics)

    def validation_step(self, batch: list[AnySample], batch_idx: int) -> Tensor:
        return self._step(batch, "valid", self.valid_metrics)

    def test_step(self, batch: list[AnySample], batch_idx: int) -> Tensor:
        return self._step(batch, "test", self.test_metrics)

    def predict_step(
        self, batch: list[AnySample], batch_idx: int
    ) -> dict[str, Any]:
        bags, _, metadata = _unpack_batch(batch)
        predictions, attentions = self._forward_batch(bags)
        return {
            "predictions": predictions,
            "attentions": attentions,
            "metadata": metadata,
        }

    def configure_optimizers(self) -> Any:
        if self.optimizer is None:
            return None

        optimizer = self.optimizer(self.parameters())
        config: dict[str, Any] = {"optimizer": optimizer}

        if self.lr_scheduler:
            scheduler = {"scheduler": self.lr_scheduler["scheduler"](optimizer)}
            if monitor := self.lr_scheduler.get("monitor"):
                scheduler["monitor"] = monitor
            config["lr_scheduler"] = scheduler

        return config


def _unpack_batch(
    batch: list[AnySample],
) -> tuple[list[AnyBag], list[Tensor], list[SlideMetadata]]:
    """Split a list-of-samples batch into parallel lists of bags/labels/metadata."""
    bags = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    metadata = [sample[2] for sample in batch]
    return bags, labels, metadata


__all__ = ["MammaprintModule"]
