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
                # ("mammaprint_value", pa.float64()), for mammaprint dataset
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
                ("is_cancer", pa.int64()),
                # ("luminal_id", pa.int64()),
                # ("mammaprint", pa.float64()),
            ]
        )
        self.writer2 = ParquetWriter(self.save_dir + "/slides_batch.parquet", schema_slides)
    
    @staticmethod
    def _preprocess_data(data: torch.Tensor) -> list[NDArray]:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if len(data.shape) > 2:
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

        # Handling pooled features
        self.pool_features_by_slide("mean")

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

        # Write actual tiles
        batch = pa.record_batch(
            [
                metadata["slide_name"],
                self._preprocess_data(metadata["coord_x"]),
                self._preprocess_data(metadata["coord_y"]),
                self._preprocess_data(outputs["outputs"]),
                self._preprocess_data(metadata["class_id"]),
                # self._preprocess_data(metadata["mammaprint_value"]),
            ],
            names=["slide_name", "coord_x", "coord_y", "model_output", "class_id"],
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
            "is_cancer": self._preprocess_data(metadata["is_cancer"]),
            # "luminal_id": self._preprocess_data(metadata["luminal_id"]),
            # "mammaprint": self._preprocess_data(metadata["mammaprint"]),
        })
        table_slide = pa.Table.from_pandas(new_slide_record)
        self.writer2.write_table(table_slide)
        logger.info("Batch data and slide metadata successfully written.")

    def _add_padding(self, slide_name: str, padding_needed: int) -> None:
        """Add empty tiles to reach the minimum tile count."""
        logger.info(f"Adding {padding_needed} empty tiles for slide {slide_name}.")

        # Create an empty tile with the same structure as regular data entries
        empty_tile = {
            "slide_name": slide_name,
            "coord_x": 0,
            "coord_y": 0,
            "model_output": [0.0] * 512,  # Adjust the size as per model_output length
            "class_id": 0,
            # "mammaprint_value": 0.0,
        }

        # Convert to a list of empty tiles
        empty_tiles = [empty_tile] * padding_needed

        # Convert to PyArrow format
        batch = pa.Table.from_pandas(pd.DataFrame(empty_tiles))
        self.writer.write(batch)
        logger.info(f"{padding_needed} empty tiles added for slide {slide_name}.")

    def pool_features_by_slide(self, aggregation_function="mean"):
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
