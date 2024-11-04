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
import numpy as np

# Local Imports
from mammaprint.trainer.callbacks import DataloaderAgnosticCallback

logger = logging.getLogger("callbacks/prediction_saver")


class ParquetPredictionSaver(DataloaderAgnosticCallback):
    writer: ParquetWriter
    writer2: ParquetWriter
    min_tiles_per_slide: int = 3000  # Minimum number of tiles per slide

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir)
        self.tile_counts = {}  # Track tile counts per slide

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
                ("luminal_id", pa.int64()),
                ("mammaprint", pa.float32()),
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

        # Handling pooled features
        #self.pool_features_by_slide("mean")

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

        slide_name = metadata["slide_name"][0]
        # Update tile count for the current slide
        tile_count = self.tile_counts.get(slide_name, 0)
        new_tile_count = tile_count + len(metadata["coord_x"])
        self.tile_counts[slide_name] = new_tile_count

        # Write actual tiles with enforced float32 type for model_output
        model_output_processed = self._preprocess_data(outputs["outputs"], target_dtype=np.float32)
        
        # Debug print statement to verify data type consistency
        print("Data type of model_output_processed:", type(model_output_processed[0][0]))
        
        # Extract mammaprint_value and ensure it is not a tensor
        mammaprint_value = metadata["mammaprint_value"]

        if isinstance(mammaprint_value, torch.Tensor):
            # Convert tensor to a list or float
            mammaprint_value = mammaprint_value.cpu().tolist() if mammaprint_value.dim() > 0 else mammaprint_value.item()

        # Proceed with pyarrow conversion after ensuring it’s not a tensor
        batch = pa.record_batch(
            [
                pa.array(metadata["slide_name"]),
                pa.array(metadata["coord_x"], pa.int64()),
                pa.array(metadata["coord_y"], pa.int64()),
                pa.array(model_output_processed, pa.list_(pa.float32())),  # Ensure float32 for model_output
                pa.array(metadata["class_id"], pa.int64()),
                pa.array(mammaprint_value, pa.float32()),  # Now mammaprint_value should be converted
            ],
            names=["slide_name", "coord_x", "coord_y", "model_output", "class_id", "mammaprint_value"],
        )

        self.writer.write(batch)
        
        # If the slide has reached its final batch, add padding if necessary
        if new_tile_count < self.min_tiles_per_slide:
            self._add_padding(slide_name, self.min_tiles_per_slide - new_tile_count)

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

    def _add_padding(self, slide_name: str, padding_needed: int) -> None:
        """Add empty tiles to reach the minimum tile count."""
        logger.info(f"Adding {padding_needed} empty tiles for slide {slide_name}.")

        slide_names = pa.array([slide_name] * padding_needed, pa.string())
        coord_x = pa.array([0] * padding_needed, pa.int64())
        coord_y = pa.array([0] * padding_needed, pa.int64())
        model_output = pa.array([[0.0] * 512] * padding_needed, pa.list_(pa.float32()))  # Explicitly float32
        class_id = pa.array([0] * padding_needed, pa.int64())
        mammaprint_value = pa.array([0] * padding_needed, pa.float32())

        # Create the record batch directly from arrays
        batch = pa.RecordBatch.from_arrays(
            [slide_names, coord_x, coord_y, model_output, class_id, mammaprint_value],
            names=["slide_name", "coord_x", "coord_y", "model_output", "class_id", "mammaprint_value"]
        )

        self.writer.write(batch)
        logger.info(f"{padding_needed} empty tiles added for slide {slide_name}.")


    def pool_features_by_slide(self, aggregation_function="max"):
        """
        Reads data from a Parquet file, pools features by slide name using the specified aggregation function,
        and saves the pooled data back to a new Parquet file.
        :param aggregation_function: str or dict, aggregation function to use ('mean', 'max', 'min', or a dictionary for custom aggregation).
        """
        # Read the input Parquet file into a DataFrame
        df = pd.read_parquet(self.save_dir + "/tiles.parquet")

        # Check the aggregation function
        if isinstance(aggregation_function, str):
            if aggregation_function == "mean":
                # Group by slide name and compute the mean of numeric columns
                pooled_df = df.groupby("slide_name").mean()
            elif aggregation_function == "max":
                pooled_df = df.groupby("slide_name").max()
            elif aggregation_function == "min":
                pooled_df = df.groupby("slide_name").min()
            else:
                raise ValueError(f"Unsupported aggregation function: {aggregation_function}")
        elif isinstance(aggregation_function, dict):
            # Group by slide name and apply the custom aggregation
            pooled_df = df.groupby("slide_name").agg(aggregation_function)
        else:
            raise ValueError("Aggregation function must be a string or dictionary.")
        # Reset the index to make 'slide_name' a regular column

        # Reset the index to make 'slide_name' a regular column
        pooled_df = pooled_df.reset_index()

        # Ensure the output file path is correct
        pooled_file_path = os.path.join(self.save_dir, "pooled_predictions.parquet")

        # Save the pooled DataFrame to the output Parquet file
        pooled_df.to_parquet(pooled_file_path)

        print(f"Pooled data saved to: {pooled_file_path}")
