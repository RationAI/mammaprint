"""Log one lossless, labeled prediction table for a Lightning predict run."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import pandas as pd
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import Tensor

from ml.callbacks._targets import (
    PredictionTarget,
    prediction_targets,
    report_item_id,
)
from ml.typing import MILSample


logger = logging.getLogger(__name__)


class _PredictionLogger(Protocol):
    def log_artifact(self, local_path: str, artifact_path: str) -> None: ...

    def log_table(self, data: dict[str, Any], artifact_file: str) -> None: ...


class PredictionCSVCallback(Callback):
    """Collect slide labels and predictions and log CSV plus an MLflow table.

    Classification model outputs are assumed to be logits by default and are
    converted to Luminal A probabilities for display. The raw logit remains in
    the CSV. Regression outputs are never transformed.
    """

    def __init__(
        self,
        label_mode: str,
        artifact_path: str = "predictions",
        filename: str = "predictions.csv",
        table_filename: str = "predictions.json",
        report_background_artifact_path: str = "report/background",
        report_background_filename: str = "slides.parquet",
        classification_outputs_are_logits: bool = True,
        threshold: float = 0.5,
        save_dir: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        self.targets = prediction_targets(label_mode)
        self.artifact_path = artifact_path.strip("/")
        self.filename = filename
        self.table_filename = table_filename
        self.report_background_artifact_path = report_background_artifact_path.strip(
            "/"
        )
        self.report_background_filename = report_background_filename
        self.classification_outputs_are_logits = classification_outputs_are_logits
        self.threshold = threshold
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self._rows: list[dict[str, Any]] = []
        self._background_rows: list[dict[str, Any]] = []

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset state when the same callback instance is reused."""
        self._rows.clear()
        self._background_rows.clear()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: list[MILSample],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Append slide rows from one predict batch."""
        predictions = cast("Tensor", outputs["predictions"]).detach().cpu()
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        if predictions.shape[0] != len(batch):
            raise ValueError(
                "Prediction batch size does not match the number of slide samples."
            )
        expected_outputs = len(self.targets)
        if predictions.shape[1] != expected_outputs:
            raise ValueError(
                f"Configured task expects {expected_outputs} outputs, got "
                f"{predictions.shape[1]}."
            )

        for prediction, (_, label, metadata) in zip(predictions, batch, strict=True):
            item_id = report_item_id(metadata["slide_id"])
            row: dict[str, Any] = {
                "record_num": metadata["record_num"],
                "slide_id": metadata["slide_id"],
                "report_item_id": item_id,
            }
            for target in self.targets:
                raw = float(prediction[target.output_index])
                truth = float(label[target.output_index])
                if target.is_classification:
                    probability = (
                        float(torch.sigmoid(torch.tensor(raw)))
                        if self.classification_outputs_are_logits
                        else raw
                    )
                    predicted_label = int(probability >= self.threshold)
                    row.update(
                        {
                            "type_label": int(truth),
                            "type": "a luminal" if truth >= 0.5 else "b luminal",
                            "luminal_a_logit": raw,
                            "luminal_a_probability": probability,
                            "predicted_type_label": predicted_label,
                            "predicted_type": (
                                "a luminal" if predicted_label else "b luminal"
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "mammaprint_index": truth,
                            "predicted_mammaprint_index": raw,
                        }
                    )
            self._rows.append(row)
            self._background_rows.append(
                {
                    "report_item_id": item_id,
                    "slide_id": metadata["slide_id"],
                    "record_num": metadata["record_num"],
                    "slide_path": str(metadata["slide_path"]),
                    "ground_truth": _ground_truth(label, self.targets),
                }
            )

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Write the complete table once and log it to the prediction run."""
        if not self._rows:
            logger.warning("Prediction run produced no slide rows; no CSV was logged.")
            return
        if not trainer.is_global_zero:
            return
        if trainer.world_size != 1:
            raise RuntimeError(
                "PredictionCSVCallback currently requires a single process so its "
                "CSV cannot silently omit slides processed by other ranks."
            )

        frame = pd.DataFrame(self._rows)
        background_frame = pd.DataFrame(self._background_rows)
        raw_logger = trainer.logger
        if not hasattr(raw_logger, "log_artifact") or not hasattr(
            raw_logger, "log_table"
        ):
            raise TypeError(
                "PredictionCSVCallback requires an MLflow logger with log_artifact "
                "and log_table methods."
            )
        mlflow_logger = cast("_PredictionLogger", raw_logger)

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / self.filename
            frame.to_csv(path, index=False)
            mlflow_logger.log_artifact(str(path), artifact_path=self.artifact_path)
            background_path = self.save_dir / self.report_background_filename
            background_frame.to_parquet(background_path, index=False)
            mlflow_logger.log_artifact(
                str(background_path),
                artifact_path=self.report_background_artifact_path,
            )
        else:
            with TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / self.filename
                frame.to_csv(path, index=False)
                mlflow_logger.log_artifact(str(path), artifact_path=self.artifact_path)
                background_path = Path(tmp_dir) / self.report_background_filename
                background_frame.to_parquet(background_path, index=False)
                mlflow_logger.log_artifact(
                    str(background_path),
                    artifact_path=self.report_background_artifact_path,
                )

        table_path = "/".join(
            part for part in (self.artifact_path, self.table_filename) if part
        )
        mlflow_logger.log_table(frame.to_dict(orient="list"), artifact_file=table_path)
        logger.info(
            "Logged %d labeled slide predictions to %s/%s.",
            len(frame),
            self.artifact_path,
            self.filename,
        )


def _ground_truth(label: Tensor, targets: tuple[PredictionTarget, ...]) -> str:
    """Format the configured truth values for the report's slide badge."""
    parts: list[str] = []
    for target in targets:
        value = float(label[target.output_index])
        if target.is_classification:
            parts.append("Luminal A" if value >= 0.5 else "Luminal B")
        else:
            parts.append(f"MammaPrint index: {value:.4g}")
    return " | ".join(parts)


__all__ = ["PredictionCSVCallback"]
