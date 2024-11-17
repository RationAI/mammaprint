# Standard Imports
import logging
from typing import Any
import os

# Third-Party Imports
import mlflow
import lightning
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
from pyarrow.parquet import read_table, write_table
import torch
import pandas as pd
import numpy as np

# Local Imports
from mammaprint.trainer.callbacks import DataloaderAgnosticCallback

logger = logging.getLogger("callbacks/prediction_saver")


class BagPredictionSaver(DataloaderAgnosticCallback):
    writer: ParquetWriter

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir)

        schema = pa.schema(
            [
                ("slide_name", pa.string()),
                ("model_output", pa.list_(pa.float32())),
            ]
        )
        self.writer = ParquetWriter(self.save_dir + "/predictions.parquet", schema)
   
    @staticmethod
    def _preprocess_data(data: Any, target_dtype=np.float32) -> list:
        # Convert to numpy array if data is a list
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # Ensure the target dtype
        data = data.astype(target_dtype)
        # Convert to list format expected by PyArrow
        return data.tolist()


    def on_test_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        if self.writer:
            self.writer.close()

        mlflow.log_artifact(local_path=self.save_dir, artifact_path="")
        artifact_uri = mlflow.get_artifact_uri(str(self.save_dir))

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

        model_output_processed = self._preprocess_data(outputs["outputs"], target_dtype=np.float32)

        # Proceed with pyarrow conversion after ensuring it’s not a tensor
        batch = pa.record_batch(
            [
                pa.array(metadata["slide_name"]),
                pa.array(model_output_processed, pa.list_(pa.float32())),  # Ensure float32 for model_output
            ],
            names=["slide_name", "model_output"]
        )

        self.writer.write(batch)

        logger.info("Batch data and slide metadata successfully written.")