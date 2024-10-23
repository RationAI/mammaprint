# Standard Imports
import functools
import logging
from typing import Any

# Third-Party Imports
import lightning
import mlflow

# Local Imports
from histopipe.trainer.callbacks.dataloader_agnostic import DataloaderAgnosticCallback
from histopipe.trainer.callbacks.image_builders import ImageBuilder


logger = logging.getLogger("callbacks/heatmap_visualizer")


class HeatmapVisualizer(DataloaderAgnosticCallback):
    image_builder: ImageBuilder | None = None
    partial_image_builder: functools.partial

    def __init__(self, image_builder: functools.partial, save_dir: str) -> None:
        super().__init__()
        self.partial_image_builder = image_builder
        self.save_dir = save_dir
        self.image_builder = None

    def _init_image_builder(self, metadata: dict):
        logger.debug("Creating new Heatmap visualizer.")
        self.image_builder = self.partial_image_builder(
            metadata=metadata, save_dir=self.save_dir
        )

    def _save_image_builder(self):
        logger.debug("Saving heatmap.")
        save_path = self.image_builder.save()
        mlflow.log_artifact(local_path=save_path, artifact_path=self.save_dir)
        artifact_uri = mlflow.get_artifact_uri(str(save_path))
        logger.debug(f"heatmap saved to: {artifact_uri}")
        stripped_uri = artifact_uri.removeprefix("mlflow-artifacts:/")
        stripped_uri = artifact_uri.removeprefix("mlflow-artifacts:/")
        logger.debug(f"saving heatmap URI to the cache as {stripped_uri}")

    def _update_heatmap_step(self, batch, outputs):
        _, _, metadata = batch
        self.image_builder.update(data=outputs["outputs"], metadata=metadata)

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        self._init_image_builder(metadata)

    def on_predict_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        self._init_image_builder(metadata)

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        self._save_image_builder()

    def on_predict_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        self._save_image_builder()

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
        self._update_heatmap_step(batch, outputs)

    def on_predict_batch_end(
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
        self._update_heatmap_step(batch, outputs)
