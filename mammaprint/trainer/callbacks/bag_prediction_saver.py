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
        logger.info(f"Predictions saved to {csv_path} and logged to MLflow.")

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

        for slidename, pred in zip(slide_name, model_output):
            self.predictions.append((slidename, pred.item() if hasattr(pred, 'item') else pred))
            logger.debug(f"Prediction for slide {slidename}: {pred}")
