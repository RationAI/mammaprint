"""Test-only CPMP comparison metrics; never imported by the training model.

Capture outputs of the existing test forward calls, without another inference
pass. A=1 is low risk (index >= 0); B=0 is high risk. Scores remain continuous
for ranking metrics. sklearn/SciPy implement every statistical metric.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch import Callback
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def comparison_metrics(scores, targets, *, regression: bool) -> dict:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not len(scores) or scores.shape != targets.shape:
        raise ValueError("Test scores/targets require nonempty equal shapes.")
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("Test scores/targets must be finite.")
    if not regression and not np.isin(targets, [0, 1]).all():
        raise ValueError("Classification targets must be A=1 or B=0.")
    labels = (targets >= 0).astype(int) if regression else targets.astype(int)
    predicted = (scores >= 0).astype(int)  # index or A-positive logit
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        labels, predicted, average="weighted", zero_division=0
    )
    metrics = {
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf),
        "f1_a": float(f1[1]),
        "f1_b": float(f1[0]),
        "precision_a": float(precision[1]),
        "precision_b": float(precision[0]),
        "recall_a": float(recall[1]),
        "recall_b": float(recall[0]),
        "auroc": None,
        "balanced_accuracy": None,
        "auprc_a_trapezoid": None,
        "auprc_b_trapezoid": None,
        "average_precision_a": None,
        "average_precision_b": None,
    }
    undefined = {}
    if len(np.unique(labels)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predicted))
        for name, truth, ranking in (("a", labels, scores), ("b", 1 - labels, -scores)):
            p, r, _ = precision_recall_curve(truth, ranking)
            metrics[f"auprc_{name}_trapezoid"] = float(auc(r, p))
            metrics[f"average_precision_{name}"] = float(
                average_precision_score(truth, ranking)
            )
    else:
        undefined.update(
            {
                key: "Both true classes are required."
                for key, value in metrics.items()
                if value is None
            }
        )
    if regression:
        for name, function in (("spearman", spearmanr), ("pearson", pearsonr)):
            key = f"{name}_index_correlation"
            if len(scores) < 2 or np.ptp(scores) == 0 or np.ptp(targets) == 0:
                metrics[key] = None
                undefined[key] = "Correlation requires nonconstant scores and targets."
            else:
                value = float(function(scores, targets).statistic)
                metrics[key] = value if np.isfinite(value) else None
                if metrics[key] is None:
                    undefined[key] = "Correlation statistic is undefined."
    return {
        "n": len(scores),
        "n_a": int(labels.sum()),
        "n_b": int((1 - labels).sum()),
        "metrics": metrics,
        "undefined": undefined,
    }


class ComparisonTestCollector(Callback):
    """Attach only after fit; retain scalar predictions from native test calls."""

    def __init__(self, config):
        self.regression = config["label_mode"] == "index"
        if config["label_mode"] not in {"index", "type"}:
            raise ValueError(
                "Comparison test metrics support single-output index/type models."
            )
        activation = (
            config["ml"]
            .get("output_activation", {})
            .get("_target_", "torch.nn.Identity")
        )
        if activation != "torch.nn.Identity":
            raise ValueError(
                "Comparison test metrics require index outputs or raw A-positive logits."
            )
        self.mapping_path = Path(config["datamodule"]["test"]["data_mapping"])
        self.rows = []
        self._predictions = []
        self._hook = None

    def on_test_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError("Comparison test collection requires one process.")
        self.rows = []
        self._hook = pl_module.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        prediction, _ = output
        if prediction.numel() != 1:
            raise ValueError(
                "Comparison test collection requires one scalar per slide."
            )
        self._predictions.append(float(prediction.detach().cpu().item()))

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._predictions = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if dataloader_idx != 0 or len(self._predictions) != len(batch):
            raise ValueError(
                "Native test predictions do not match one test dataloader/batch."
            )
        for prediction, (_, label, metadata) in zip(
            self._predictions, batch, strict=True
        ):
            if label.numel() != 1:
                raise ValueError("Comparison test collection requires scalar targets.")
            self.rows.append(
                {
                    "slide_id": metadata["slide_id"],
                    "target": float(label.detach().cpu().item()),
                    "score": prediction,
                }
            )

    def on_test_end(self, trainer, pl_module):
        self.close()

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def report(self) -> dict:
        frame = pd.DataFrame(self.rows)
        if frame.empty or frame.slide_id.duplicated().any():
            raise ValueError("Test collection is empty or contains duplicate slides.")
        report = {
            "format_version": 1,
            "label_mode": "index" if self.regression else "type",
            "class_definition": {
                "a": "low risk, target=1 or index>=0",
                "b": "high risk, target=0 or index<0",
            },
            "decision_threshold": 0.0,
            "auprc_definition": "trapezoidal area of sklearn precision_recall_curve; average precision logged separately",
            "zero_division": 0,
            "repeated_evaluation": {
                "n": 1,
                "standard_deviation": None,
                "protocol": "single existing held-out split; no bootstrap or repeated cross-validation",
            },
            "agendia_comparison": "Pearson and Spearman reported separately; Agendia r=0.954 statistic unconfirmed.",
            "slide": comparison_metrics(
                frame.score, frame.target, regression=self.regression
            ),
            "patient": None,
        }
        mapping = pd.read_csv(self.mapping_path, dtype=str)
        mapping["slide_id"] = mapping["path"].map(lambda value: Path(value).stem)
        if mapping.slide_id.duplicated().any():
            raise ValueError("Duplicate slide IDs in test label mapping.")
        key = next(
            (key for key in ("patient_id", "case_id", "record_num") if key in mapping),
            None,
        )
        report["patient_key"] = key
        if key is None:
            report["patient_status"] = (
                "Unavailable: mapping has no stable patient/case key; slide IDs are not patient IDs."
            )
            return report
        frame = frame.merge(
            mapping[["slide_id", key]], on="slide_id", how="left", validate="one_to_one"
        )
        if frame[key].isna().any() or frame[key].str.strip().eq("").any():
            raise ValueError("Test slides have missing patient/case identifiers.")
        grouped = frame.groupby(key, sort=True)
        if (grouped.target.nunique() != 1).any():
            raise ValueError(
                "Inconsistent targets within a patient/case; cannot combine slides."
            )
        patients = grouped.agg(score=("score", "mean"), target=("target", "first"))
        report["patient_status"] = (
            "Grouped by mapping key; mean score per patient/case, identical targets required."
        )
        report["patient"] = comparison_metrics(
            patients.score, patients.target, regression=self.regression
        )
        return report


def metric_scalars(report: dict) -> dict[str, float]:
    """Keep undefined values in the JSON, out of finite-only MLflow metrics."""
    return {
        f"test/comparison_{unit}/{key}": value
        for unit in ("slide", "patient")
        if report[unit] is not None
        for key, value in report[unit]["metrics"].items()
        if value is not None
    }
