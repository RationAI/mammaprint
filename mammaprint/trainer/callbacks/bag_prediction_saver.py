# Standard Imports
import logging
from typing import Any
import os

# Third-Party Imports
import mlflow
import lightning
import torch
import pandas as pd
import numpy as np

# Local Imports
from mammaprint.trainer.callbacks import DataloaderAgnosticCallback

logger = logging.getLogger("callbacks/prediction_saver")


class BagPredictionSaver(DataloaderAgnosticCallback):
    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.predictions = []

    @staticmethod
    def _preprocess_data(data: Any, target_dtype=np.float32) -> np.ndarray:
        # Convert to numpy array if data is a list
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # Ensure the target dtype
        data = data.astype(target_dtype)
        return data

    def on_test_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        # Convert collected predictions into a DataFrame and save to CSV
        df = pd.DataFrame(self.predictions, columns=["slide_name", "prediction"])
        csv_path = os.path.join(self.save_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        
        # Log artifact using MLflow
        mlflow.log_artifact(local_path=csv_path, artifact_path="")
        artifact_uri = mlflow.get_artifact_uri(str(self.save_dir))
        logger.info(f"Predictions CSV saved to {csv_path} and logged as MLflow artifact.")

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

        metadata_list = batch[2]  # list of per-tile metadata dicts
        slide_name = metadata_list[0]['slide_name']

        model_output = self._preprocess_data(outputs["outputs"], target_dtype=np.float32)

        # Assume model_output is a single prediction per slide. If it’s multiple,
        # you may need to iterate or summarize. Here we assume a 1D array.
        for slidename, pred in zip(slide_name, model_output):
            # If pred is more than one value, handle accordingly.
            # For this example, we assume a single value per slide.
            self.predictions.append((slidename, pred.item() if hasattr(pred, 'item') else pred))

        logger.info("Batch predictions successfully appended to internal list.")
