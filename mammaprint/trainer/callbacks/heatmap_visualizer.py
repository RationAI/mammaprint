# Copyright (c) The RationAI team.

import functools
import logging
from typing import Any

import lightning
import mlflow

from mammaprint.trainer.callbacks.dataloader_agnostic import (
    DataloaderAgnosticCallback,
)
from mammaprint.trainer.callbacks.image_builders import ImageBuilder


logger = logging.getLogger("callbacks/heatmap_visualizer")


class HeatmapVisualizer(DataloaderAgnosticCallback):
    image_builder: ImageBuilder | None = None
    partial_image_builder: functools.partial

    def __init__(self, image_builder: functools.partial, save_dir: str) -> None:
        super().__init__()
        self.partial_image_builder = image_builder
        self.save_dir = save_dir
        self.image_builder = None

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        logger.debug("Creating new Heatmap visualizer.")
        logger.debug(f"Metadata for dataloader {dataloader_idx}: {metadata}")
        self.image_builder = self.partial_image_builder(
            metadata=metadata, save_dir=self.save_dir
        )

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        logger.debug("Saving heatmap.")
        save_path = self.image_builder.save()
        mlflow.log_artifact(local_path=save_path, artifact_path=self.save_dir)
        artifact_uri = mlflow.get_artifact_uri(str(save_path))
        logger.debug(f"Heatmap saved to: {artifact_uri}")
        stripped_uri = artifact_uri.removeprefix("mlflow-artifacts:/")
        logger.debug(f"Saving heatmap URI to the cache as {stripped_uri}")

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Call the parent implementation for standard handling
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        # Unpack batch to get metadata
        _, _, metadata = batch

        # Debug metadata structure
        logger.debug(
            f"Batch metadata structure: {type(metadata)}, content: {metadata[:1] if isinstance(metadata, list) else metadata}"
        )

        # Extract and reshape attention weights
        attention_weights = outputs["attention_weights"].detach().cpu().numpy()
        logger.debug(f"Original attention weights shape: {attention_weights.shape}")

        # Reshape attention weights to align with metadata
        if attention_weights.ndim == 2 and attention_weights.shape[0] == 1:
            # Squeeze the batch dimension to match metadata
            attention_weights = attention_weights.squeeze(0)  # Shape: [2000]
            logger.debug(f"Reshaped attention weights shape: {attention_weights.shape}")

        # Validate metadata and attention weights match
        assert len(metadata) == attention_weights.shape[0], (
            f"Metadata and attention weights mismatch: {len(metadata)} metadata entries, "
            f"{attention_weights.shape[0]} attention weights."
        )

        # Update the image builder with attention weights and metadata
        self.image_builder.update(data=attention_weights, metadata=metadata)
        logger.debug("Image builder updated with attention weights and metadata.")
