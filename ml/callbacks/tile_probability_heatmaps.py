"""Prostate-style local tile prediction heatmaps for embedding MIL models."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.masks.mask_builders.aggregation import MeanAggregator
from torch import Tensor

from ml.callbacks._targets import (
    PredictionTarget,
    prediction_targets,
    report_item_id,
)
from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
from ml.models.aggregators.transformer import TransformerMIL
from ml.models.module import MammaprintModule
from ml.typing import MILSample, SlideMetadata


logger = logging.getLogger(__name__)


class _ArtifactLogger(Protocol):
    def log_artifact(self, local_path: str, artifact_path: str) -> None: ...


def _slide_level_extents(metadata: SlideMetadata) -> tuple[int, int]:
    """Read exact ``(height, width)`` for the embedding pyramid level."""
    try:
        from openslide import OpenSlide
    except ImportError as error:
        raise RuntimeError(
            "OpenSlide is required to align prediction heatmaps to the WSI."
        ) from error

    slide_path = Path(metadata["slide_path"])
    if not slide_path.is_file():
        raise FileNotFoundError(f"WSI not found: {slide_path}")
    with OpenSlide(str(slide_path)) as slide:
        level = metadata["level"]
        if level < 0 or level >= slide.level_count:
            raise ValueError(
                f"Slide {metadata['slide_id']!r} has {slide.level_count} levels; "
                f"requested level {level}."
            )
        width, height = slide.level_dimensions[level]
    return int(height), int(width)


def _validate_coordinates(
    metadata: SlideMetadata, source_extents: tuple[int, int]
) -> np.ndarray:
    """Return validated MaskBuilder coordinates in ``(y, x)`` order."""
    x = metadata["x"].detach().cpu().numpy().astype(np.int64, copy=False)
    y = metadata["y"].detach().cpu().numpy().astype(np.int64, copy=False)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Tile x/y coordinates must be equal-length vectors.")
    if len(x) == 0:
        raise ValueError(f"Slide {metadata['slide_id']!r} contains no tiles.")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Tile coordinates must be non-negative.")
    height, width = source_extents
    if np.any(x >= width) or np.any(y >= height):
        raise ValueError(
            "Tile coordinates fall outside the selected WSI level; check that the "
            "embedding card and WSI correspond."
        )
    stride = metadata["stride"]
    if np.any(x % stride) or np.any(y % stride):
        raise ValueError(
            "Tile coordinates must align to the configured stride for MaskBuilder."
        )
    return np.stack([y, x], axis=-1)


@torch.inference_mode()
def singleton_outputs(module: MammaprintModule, bag: Tensor, batch_size: int) -> Tensor:
    """Predict every tile as a one-instance bag, batched efficiently.

    Mean and max pooling are identities for singleton bags. Gated attention first
    applies its learned LayerNorm, while its singleton softmax weight is exactly
    one. Transformer MIL is evaluated on batched two-token sequences (CLS + one
    tile). These paths reproduce direct one-tile model calls without a Python
    forward pass per tile.
    """
    if bag.ndim != 2:
        raise ValueError(
            "TileProbabilityHeatmapCallback supports stored embedding bags shaped "
            "(tiles, features), not raw image bags."
        )
    encoded = module.encoder(bag)
    if isinstance(module.aggregator, (MeanPool, MaxPool)):
        local_features = encoded
    elif isinstance(module.aggregator, AttentionMIL):
        local_features = module.aggregator.norm(encoded)
    elif isinstance(module.aggregator, TransformerMIL):
        local_features = _transformer_singleton_features(
            module.aggregator, encoded, batch_size
        )
    else:
        raise TypeError(
            "Tile probability heatmaps support mean, max, gated-attention, and "
            f"transformer aggregators; got {type(module.aggregator).__name__}."
        )

    predictions = []
    for chunk in local_features.split(batch_size):
        predictions.append(module.output_activation(module.head(chunk)))
    return torch.cat(predictions, dim=0)


def _transformer_singleton_features(
    aggregator: TransformerMIL, encoded: Tensor, batch_size: int
) -> Tensor:
    """Return exact TransformerMIL pooled features for one-tile bags in batches."""
    pooled = []
    for chunk in encoded.split(batch_size):
        cls = aggregator.cls_token.expand(len(chunk), 1, -1)
        sequence = torch.cat((cls, chunk.unsqueeze(1)), dim=1)
        if aggregator.blocks is not None:
            sequence = aggregator.blocks(sequence)

        normed = aggregator.final_norm(sequence)
        attended, _ = aggregator.final_attn(
            normed,
            normed,
            normed,
            need_weights=False,
        )
        pooled.append((sequence + attended)[:, 0])
    return torch.cat(pooled, dim=0)


def _display_values(
    raw: Tensor,
    target: PredictionTarget,
    classification_outputs_are_logits: bool,
    regression_display_transform: str,
) -> np.ndarray:
    """Transform one raw output channel into a viewer-safe ``[0, 1]`` value."""
    values = raw[:, target.output_index]
    if target.is_classification and classification_outputs_are_logits:
        values = torch.sigmoid(values)
    elif not target.is_classification:
        if regression_display_transform == "sigmoid":
            # MammaPrint's decision boundary is zero. Sigmoid maps it to the
            # viewer midpoint (0.5), while retaining direction and ordering.
            values = torch.sigmoid(values)
        else:
            raise ValueError(
                "regression_display_transform must be 'sigmoid'; got "
                f"{regression_display_transform!r}."
            )
    return values.detach().float().cpu().numpy()


class TileProbabilityHeatmapCallback(Callback):
    """Create per-tile local-prediction BigTIFFs during ``Trainer.predict``.

    This intentionally mirrors the prostate heatmap callback: scalar tile outputs
    are expanded over their tile footprints by ``ratiopath.MaskBuilder`` and
    overlapping regions are averaged. Unlike a causal MIL attribution, each value
    answers what the complete head predicts when that tile is the only bag member.
    """

    def __init__(
        self,
        label_mode: str,
        artifact_path: str = "heatmaps/local_tile_prediction",
        batch_size: int = 8192,
        classification_outputs_are_logits: bool = True,
        regression_display_transform: str = "sigmoid",
        save_dir: str | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.targets = prediction_targets(label_mode)
        self.artifact_path = artifact_path.strip("/")
        self.batch_size = batch_size
        self.classification_outputs_are_logits = classification_outputs_are_logits
        self.regression_display_transform = regression_display_transform
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: list[MILSample],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Build and log every slide heatmap from one prediction batch."""
        if not trainer.is_global_zero:
            return
        if trainer.world_size != 1:
            raise RuntimeError(
                "TileProbabilityHeatmapCallback requires a single prediction "
                "process; use one H100 device."
            )
        if not isinstance(pl_module, MammaprintModule):
            raise TypeError("Expected MammaprintModule.")
        if not hasattr(trainer.logger, "log_artifact"):
            raise TypeError("Tile heatmaps require an MLflow logger.")

        for bag, _, metadata in batch:
            self._render_slide(trainer, pl_module, bag, metadata)

    def _render_slide(
        self,
        trainer: Trainer,
        module: MammaprintModule,
        bag: Tensor,
        metadata: SlideMetadata,
    ) -> None:
        source_extents = _slide_level_extents(metadata)
        coords = _validate_coordinates(metadata, source_extents)
        raw_outputs = singleton_outputs(module, bag, self.batch_size)
        if raw_outputs.shape != (len(coords), len(self.targets)):
            raise ValueError(
                f"Expected local output shape {(len(coords), len(self.targets))}, "
                f"got {tuple(raw_outputs.shape)}."
            )

        values = np.stack(
            [
                _display_values(
                    raw_outputs,
                    target,
                    self.classification_outputs_are_logits,
                    self.regression_display_transform,
                )
                for target in self.targets
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        builder = MaskBuilder(
            source_extents=source_extents,
            source_tile_extent=metadata["tile_extent"],
            output_tile_extent=1,
            stride=metadata["stride"],
            n_channels=len(self.targets),
            storage="inmemory",
            aggregation=MeanAggregator,
        )
        try:
            builder.update_batch(values, coords)
            compact_mask = cast("dict[str, np.ndarray]", builder.finalize())["mask"]
            self._write_targets(
                trainer,
                builder,
                compact_mask,
                metadata,
                report_item_id(metadata["slide_id"]),
            )
        finally:
            builder.cleanup()

    def _write_targets(
        self,
        trainer: Trainer,
        builder: MaskBuilder,
        mask: np.ndarray,
        metadata: SlideMetadata,
        slide_name: str,
    ) -> None:
        if self.save_dir is not None:
            self._write_to_dir(
                trainer, builder, mask, metadata, slide_name, self.save_dir
            )
            return
        with TemporaryDirectory() as tmp_dir:
            self._write_to_dir(
                trainer, builder, mask, metadata, slide_name, Path(tmp_dir)
            )

    def _write_to_dir(
        self,
        trainer: Trainer,
        builder: MaskBuilder,
        mask: np.ndarray,
        metadata: SlideMetadata,
        slide_name: str,
        output_dir: Path,
    ) -> None:
        for channel, target in enumerate(self.targets):
            target_dir = output_dir / target.name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_mask = (
                (mask[channel : channel + 1] * 255).clip(0, 255).astype(np.uint8)
            )
            image = builder.resize_to_source(target_mask, kernel="nearest")
            path = target_dir / f"{slide_name}.tiff"
            write_big_tiff(
                image,
                path,
                mpp_x=metadata["mpp"],
                mpp_y=metadata["mpp"],
            )
            artifact_dir = "/".join(
                part for part in (self.artifact_path, target.name) if part
            )
            mlflow_logger = cast("_ArtifactLogger", trainer.logger)
            mlflow_logger.log_artifact(str(path), artifact_path=artifact_dir)
        logger.info(
            "Logged %d local tile-prediction heatmap(s) for %s (record_num=%s).",
            len(self.targets),
            metadata["slide_id"],
            metadata["record_num"],
        )


__all__ = ["TileProbabilityHeatmapCallback", "singleton_outputs"]
