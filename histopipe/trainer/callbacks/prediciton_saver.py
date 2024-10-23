# Standard Imports
import logging
import os
from typing import Any

import lightning

# Third-Party Imports
import mlflow
import pyarrow as pa
import torch
from nptyping import NDArray
from pyarrow.parquet import ParquetWriter

# Local Imports
from histopipe.trainer.callbacks import DataloaderAgnosticCallback


logger = logging.getLogger("callbacks/prediction_saver")


class ParquetPredictionSaver(DataloaderAgnosticCallback):
    writer: ParquetWriter

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir)

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        schema = pa.schema(
            [
                ("slide_name", pa.string()),
                ("coord_x", pa.int64()),
                ("coord_y", pa.int64()),
                ("model_output", pa.list_(pa.float32())),
            ]
        )
        self.writer = ParquetWriter(self.save_dir + "/predicitons.parquet", schema)

    @staticmethod
    def _preprocess_data(data: torch.Tensor) -> list[NDArray]:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return list(data)

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        self.writer.close()
        mlflow.log_artifact(local_path=self.save_dir, artifact_path="")
        _ = mlflow.get_artifact_uri(str(self.save_dir))

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
        _, _, metadata = batch

        batch = pa.record_batch(
            [
                metadata["slide_name"],
                self._preprocess_data(metadata["coord_x"]),
                self._preprocess_data(metadata["coord_y"]),
                self._preprocess_data(outputs["outputs"]),
            ],
            names=["slide_name", "coord_x", "coord_y", "model_output"],
        )
        self.writer.write(batch)
