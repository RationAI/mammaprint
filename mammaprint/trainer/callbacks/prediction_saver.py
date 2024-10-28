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
from nptyping import NDArray
import pandas as pd

# Local Imports
from mammaprint.trainer.callbacks import DataloaderAgnosticCallback

logger = logging.getLogger("callbacks/prediction_saver")


class ParquetPredictionSaver(DataloaderAgnosticCallback):
    writer: ParquetWriter
    writer2: ParquetWriter

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir)

        schema = pa.schema(
            [
                ("slide_name", pa.string()),
                ("coord_x", pa.int64()),
                ("coord_y", pa.int64()),
                ("model_output", pa.list_(pa.float32())),
                ("class_id", pa.int64()),
            ]
        )
        self.writer = ParquetWriter(self.save_dir + "/tiles.parquet", schema)

        schema_slides = pa.schema(
            [
                ("slide_name", pa.string()),
                ("slide_width", pa.int64()),
                ("slide_height", pa.int64()),
                ("sample_level", pa.int64()),
                ("slide_fp", pa.string()),
                ("tile_size", pa.int64()),
                ("step_size", pa.int64()),
                ("center_size", pa.int64()),
                ("year", pa.string()),
                ("patient_id", pa.string()),
                ("luminal_id", pa.string()),
            ]
        )
        self.writer2 = ParquetWriter(self.save_dir + "/slides_batch.parquet", schema_slides)

    @staticmethod
    def _preprocess_data(data: Any) -> list[NDArray]:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):
            data = [d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data]
        if isinstance(data, np.ndarray) and len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return list(data)

    def on_test_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        if self.writer:
            self.writer.close()
        if self.writer2:
            self.writer2.close()

        # Clean up duplicate slides
        slides_file_path = os.path.join(self.save_dir, "slides_batch.parquet")
        clean_slides_path = os.path.join(self.save_dir, "slides.parquet")

        if os.path.exists(slides_file_path):
            table = read_table(slides_file_path)
            df = table.to_pandas()
            df_clean = df.drop_duplicates()
            table_clean = pa.Table.from_pandas(df_clean)
            write_table(table_clean, clean_slides_path, compression='snappy')
        
        print(f"Cleaned slides data saved to: {clean_slides_path}")
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

        batch = pa.record_batch(
            [
                metadata["slide_name"],
                self._preprocess_data(metadata["coord_x"]),
                self._preprocess_data(metadata["coord_y"]),
                self._preprocess_data(outputs["outputs"]),
                self._preprocess_data(metadata["class_id"]),
            ],
            names=["slide_name", "coord_x", "coord_y", "model_output", "class_id"],
        )
        self.writer.write(batch)
        # Prepare DataFrame for new slide metadata
        new_slide_record = pd.DataFrame({
            "slide_name": metadata["slide_name"],
            "slide_width": self._preprocess_data(metadata["slide_width"]),
            "slide_height": self._preprocess_data(metadata["slide_height"]),
            "sample_level": self._preprocess_data(metadata["sample_level"]),
            "slide_fp": metadata["slide_fp"],
            "tile_size": self._preprocess_data(metadata["tile_size"]),
            "step_size": self._preprocess_data(metadata["step_size"]),
            "center_size": self._preprocess_data(metadata["center_size"]),
            "year": metadata["year"],
            "patient_id": metadata["patient_id"],
            "luminal_id": metadata["luminal_id"],
        })
        table_slide = pa.Table.from_pandas(new_slide_record)
        self.writer2.write_table(table_slide)
        logger.info("Batch data and slide metadata successfully written.")
