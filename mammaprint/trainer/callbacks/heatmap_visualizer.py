# Copyright (c) The RationAI team.

import functools
import logging
from typing import Any

import lightning
import mlflow
import torch

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

        metadata_list = batch[2]  # list of per-tile metadata dicts
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

            # Aggregate per-tile metadata
            per_tile_metadata_list = metadata_list
            logger.debug(f"Type of per_tile_metadata_list: {type(per_tile_metadata_list)}")
            logger.debug(f"First per-tile metadata: {per_tile_metadata_list[0]}")

            # Initialize lists to collect per-tile metadata
            coord_x_list = []
            coord_y_list = []
            # Add other per-tile metadata fields as needed

            for tile_meta in per_tile_metadata_list:
                coord_x_list.append(tile_meta['coord_x'].item())
                coord_y_list.append(tile_meta['coord_y'].item())
                # Extract other fields if necessary

            # Convert lists to tensors
            coord_x = torch.tensor(coord_x_list, device=data.device)
            coord_y = torch.tensor(coord_y_list, device=data.device)

            # Use the first tile's metadata for shared fields
            tile_size = per_tile_metadata_list[0]['tile_size']
            sample_level = per_tile_metadata_list[0]['sample_level']
            slide_name = per_tile_metadata_list[0]['slide_name']
            slide_width = per_tile_metadata_list[0]['slide_width']
            slide_height = per_tile_metadata_list[0]['slide_height']
            slide_channels = 1

            # Construct the aggregated metadata dictionary
            tile_metadata = {
                'coord_x': coord_x,
                'coord_y': coord_y,
                'tile_size': tile_size,
                'sample_level': sample_level,
                'slide_name': slide_name,
                'slide_width': slide_width,
                'slide_height': slide_height,
                'slide_channels': slide_channels,
            }
            logger.debug(f"Constructed tile_metadata for batch {i+1}: {tile_metadata}")

            # Apply logarithmic scaling and normalization
            epsilon = 1e-8
            data = torch.log(data + epsilon)
            data_min = data.min()
            data_max = data.max()
            data = (data - data_min) / (data_max - data_min + epsilon)
            logger.debug(f"Data after log scaling and normalization: min={data.min()}, max={data.max()}")
            logger.debug(f"Data shape after log scaling and normalization: {data.shape}")

            # Update the image builder with arrays of data and metadata
            logger.debug(f"Updating image builder for batch {i+1}.")
            self.image_builder.update(data=data, metadata=tile_metadata)

        logger.debug("Completed on_test_batch_end.")
