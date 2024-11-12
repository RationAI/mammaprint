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
        logger.debug(f"heatmap saved to: {artifact_uri}")
        stripped_uri = artifact_uri.removeprefix("mlflow-artifacts:/")
        logger.debug(f"saving heatmap URI to the cache as {stripped_uri}")

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        logger.debug("Starting on_test_batch_end.")

        # Extract attention weights and metadata
        attention_weights = outputs["attention_weights"]  # shape [batch_size, num_tiles]
        logger.debug(f"Extracted attention_weights with shape: {attention_weights.shape}")

        metadata_list = batch[2]  # list of metadata dicts per batch element
        logger.debug(f"Extracted metadata_list with length: {len(metadata_list)}")

        batch_size = attention_weights.shape[0]
        num_tiles = attention_weights.shape[1]
        logger.debug(f"Batch size: {batch_size}, Number of tiles: {num_tiles}")

        for i in range(batch_size):
            data = attention_weights[i]  # shape [num_tiles]
            logger.debug(f"Processing batch {i+1}/{batch_size}. Data shape: {data.shape}")

            # Reshape data to [num_tiles, 1]
            data = data.unsqueeze(1)  # Now data shape is [num_tiles, 1]
            logger.debug(f"Reshaped data to: {data.shape}")

            metadata = metadata_list[i]  # dict containing per-tile metadata
            logger.debug(f"Extracted metadata for batch {i+1}: {metadata}")

            # Loop over each tile and construct per-tile metadata
            for tile_idx in range(num_tiles):
                tile_metadata = {
                    'coord_x': metadata['coord_x'][tile_idx].item(),  # Get per-tile coord_x
                    'coord_y': metadata['coord_y'][tile_idx].item(),  # Get per-tile coord_y
                    'tile_size': metadata['tile_size'].item(),
                    'sample_level': metadata['sample_level'].item(),
                    'slide_name': metadata['slide_name'][0],  # Assuming slide_name is the same
                    'slide_width': metadata['slide_width'].item(),
                    'slide_height': metadata['slide_height'].item(),
                    'slide_channels': 1,  # Since attention weights are scalar
                }

                logger.debug(
                    f"Constructed tile_metadata for tile {tile_idx+1}/{num_tiles}: {tile_metadata}"
                )

                # Update the image builder with per-tile data and metadata
                self.image_builder.update(data=data[tile_idx], metadata=tile_metadata)

        logger.debug("Completed on_test_batch_end.")

