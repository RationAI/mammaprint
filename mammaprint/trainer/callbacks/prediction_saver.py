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
                ("mammaprint_value", pa.float32()),
            ]
        )
        self.writer = ParquetWriter(self.save_dir + "/tiles.parquet", schema)

        schema_slides = pa.schema(
            [
                ("slide_name", pa.string()),
                ("slide_width", pa.float64()),
                ("slide_height", pa.float64()),
                ("sample_level", pa.float64()),
                ("slide_fp", pa.string()),
                ("tile_size", pa.float64()),
                ("step_size", pa.float64()),
                ("center_size", pa.float64()),
                ("year", pa.string()),
                ("patient_id", pa.string()),
                ("luminal_id", pa.float64()),
                ("mammaprint", pa.float64()),
            ]
        )
        self.writer2 = ParquetWriter(self.save_dir + "/slides_batch.parquet", schema_slides)
    
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

        # Write actual tiles with enforced float32 type for model_output
        model_output_processed = self._preprocess_data(outputs["outputs"], target_dtype=np.float32)
        
        mammaprint_value = metadata["mammaprint_value"]

        if isinstance(mammaprint_value, torch.Tensor):
            # Convert tensor to a list or float
            mammaprint_value = mammaprint_value.cpu().tolist() if mammaprint_value.dim() > 0 else mammaprint_value.item()

        batch = pa.record_batch(
            [
                pa.array(metadata["slide_name"]),
                pa.array(metadata["coord_x"], pa.int64()),
                pa.array(metadata["coord_y"], pa.int64()),
                pa.array(model_output_processed, pa.list_(pa.float32())),
                pa.array(metadata["class_id"], pa.int64()),
                pa.array(mammaprint_value, pa.float32()),
            ],
            names=["slide_name", "coord_x", "coord_y", "model_output", "class_id", "mammaprint_value"],
        )

        self.writer.write(batch)
        
        # Save slide metadata
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
            "luminal_id": self._preprocess_data(metadata["luminal_id"]),
            "mammaprint": self._preprocess_data(metadata["mammaprint"]),
        })
        table_slide = pa.Table.from_pandas(new_slide_record)
        self.writer2.write_table(table_slide)
        logger.info("Batch data and slide metadata successfully written.")
