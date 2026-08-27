from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ml.logging.pyramid_audit import log_pyramid_audit
from ml.train import _log_pyramid_audits


AUDIT_COLUMNS = [
    "slide_id",
    "split",
    "label",
    "candidates",
    "complete_before_thinning",
    "retained",
    "incomplete",
    "missing_by_level",
    "completion_rate",
    "overlap_retention_rate",
    "rejected",
]
SUMMARY_COLUMNS = [
    "split",
    "label",
    "slides",
    "rejected_slides",
    "candidates",
    "complete_before_thinning",
    "retained",
    "incomplete",
    "completion_rate",
    "overlap_retention_rate",
    "retained_p25",
    "retained_p50",
    "retained_p75",
]


class FakeAuditDataset:
    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "slide_id": "slide-a",
                    "split": "train",
                    "label": "a luminal",
                    "candidates": 4,
                    "complete_before_thinning": 3,
                    "retained": 2,
                    "incomplete": 1,
                    "missing_by_level": {3: 1},
                    "completion_rate": 0.75,
                    "overlap_retention_rate": 2 / 3,
                    "rejected": False,
                },
                {
                    "slide_id": "slide-b",
                    "split": "train",
                    "label": "b luminal",
                    "candidates": 2,
                    "complete_before_thinning": 0,
                    "retained": 0,
                    "incomplete": 2,
                    "missing_by_level": {4: 2},
                    "completion_rate": 0.0,
                    "overlap_retention_rate": 0.0,
                    "rejected": True,
                },
            ],
            columns=AUDIT_COLUMNS,
        )

    def audit_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "split": "train",
                    "label": "a luminal",
                    "slides": 1,
                    "rejected_slides": 0,
                    "candidates": 4,
                    "complete_before_thinning": 3,
                    "retained": 2,
                    "incomplete": 1,
                    "completion_rate": 0.75,
                    "overlap_retention_rate": 2 / 3,
                    "retained_p25": 2.0,
                    "retained_p50": 2.0,
                    "retained_p75": 2.0,
                },
                {
                    "split": "train",
                    "label": "b luminal",
                    "slides": 1,
                    "rejected_slides": 1,
                    "candidates": 2,
                    "complete_before_thinning": 0,
                    "retained": 0,
                    "incomplete": 2,
                    "completion_rate": 0.0,
                    "overlap_retention_rate": 0.0,
                    "retained_p25": 0.0,
                    "retained_p50": 0.0,
                    "retained_p75": 0.0,
                },
            ],
            columns=SUMMARY_COLUMNS,
        )


class FakeLogger:
    def __init__(self) -> None:
        self.artifact_calls: list[tuple[str, str | None]] = []

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str | None = None,
    ) -> None:
        self.artifact_calls.append((local_dir, artifact_path))


class FakeDataModule:
    def __init__(self) -> None:
        self.setup_stages: list[str | None] = []
        self.datasets = {
            split: FakeAuditDataset() for split in ("train", "val", "test")
        }

    def setup(self, stage: str | None = None) -> None:
        self.setup_stages.append(stage)

    def train_dataloader(self) -> SimpleNamespace:
        return SimpleNamespace(dataset=self.datasets["train"])

    def val_dataloader(self) -> SimpleNamespace:
        return SimpleNamespace(dataset=self.datasets["val"])

    def test_dataloader(self) -> SimpleNamespace:
        return SimpleNamespace(dataset=self.datasets["test"])


class PyramidAuditLoggerTest(unittest.TestCase):
    def test_writes_exact_csv_and_json_tables_then_logs_the_managed_directory(
        self,
    ) -> None:
        logger = FakeLogger()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "train"

            logged = log_pyramid_audit(
                FakeAuditDataset(),
                logger,
                output_dir,
                artifact_path="pyramid_selection/train",
            )

            self.assertTrue(logged)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                [
                    "pyramid_selection_audit.csv",
                    "pyramid_selection_audit.json",
                    "pyramid_selection_summary_by_split_label.csv",
                    "pyramid_selection_summary_by_split_label.json",
                ],
            )
            audit_csv = pd.read_csv(output_dir / "pyramid_selection_audit.csv")
            summary_csv = pd.read_csv(
                output_dir / "pyramid_selection_summary_by_split_label.csv"
            )
            self.assertEqual(list(audit_csv.columns), AUDIT_COLUMNS)
            self.assertEqual(list(summary_csv.columns), SUMMARY_COLUMNS)
            self.assertEqual(
                summary_csv[["split", "label"]].to_dict(orient="records"),
                [
                    {"split": "train", "label": "a luminal"},
                    {"split": "train", "label": "b luminal"},
                ],
            )

            audit_json = json.loads(
                (output_dir / "pyramid_selection_audit.json").read_text()
            )
            summary_json = json.loads(
                (
                    output_dir
                    / "pyramid_selection_summary_by_split_label.json"
                ).read_text()
            )
            self.assertEqual(list(audit_json[0]), AUDIT_COLUMNS)
            self.assertEqual(list(summary_json[0]), SUMMARY_COLUMNS)
            self.assertEqual(audit_json[0]["missing_by_level"], {"3": 1})
            self.assertEqual(
                [(row["split"], row["label"]) for row in summary_json],
                [("train", "a luminal"), ("train", "b luminal")],
            )
            self.assertEqual(
                logger.artifact_calls,
                [(str(output_dir), "pyramid_selection/train")],
            )

    def test_single_scale_dataset_is_a_no_op(self) -> None:
        logger = FakeLogger()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "single-scale"

            logged = log_pyramid_audit(object(), logger, output_dir)

            self.assertFalse(logged)
            self.assertFalse(output_dir.exists())
            self.assertEqual(logger.artifact_calls, [])

    def test_fit_logs_train_and_validation_audits_from_managed_output(self) -> None:
        datamodule = FakeDataModule()
        logger = FakeLogger()

        _log_pyramid_audits(datamodule, logger, "fit")

        self.assertEqual(datamodule.setup_stages, ["fit"])
        self.assertEqual(
            [artifact_path for _, artifact_path in logger.artifact_calls],
            ["pyramid_selection/train", "pyramid_selection/val"],
        )
        self.assertTrue(
            all(not Path(local_dir).exists() for local_dir, _ in logger.artifact_calls)
        )


if __name__ == "__main__":
    unittest.main()
