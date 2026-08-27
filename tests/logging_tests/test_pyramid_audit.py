"""Regression tests for pyramid-audit logging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from ml.logging.pyramid_audit import log_pyramid_audit
from ml.train import _log_pyramid_audits
from ml.train import main as train_main


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
        self.finalize_calls: list[str] = []

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str | None = None,
    ) -> None:
        self.artifact_calls.append((local_dir, artifact_path))

    def finalize(self, status: str = "success") -> None:
        self.finalize_calls.append(status)


class EmptyAuditDataset(FakeAuditDataset):
    def __len__(self) -> int:
        return 0


class FakeDataModule:
    def __init__(self, datasets: dict[str, object] | None = None) -> None:
        self.setup_stages: list[str | None] = []
        self.dataset_calls: list[str] = []
        self.loader_calls: list[str] = []
        self.datasets = datasets or {
            split: FakeAuditDataset() for split in ("train", "val", "test")
        }

    def setup(self, stage: str | None = None) -> None:
        self.setup_stages.append(stage)

    def get_dataset(self, split: str) -> object:
        self.dataset_calls.append(split)
        return self.datasets[split]

    def train_dataloader(self) -> None:
        self.loader_calls.append("train")
        raise AssertionError("audit logging must not construct a train loader")

    def val_dataloader(self) -> None:
        self.loader_calls.append("val")
        raise AssertionError("audit logging must not construct a validation loader")

    def test_dataloader(self) -> None:
        self.loader_calls.append("test")
        raise AssertionError("audit logging must not construct a test loader")


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
                    output_dir / "pyramid_selection_summary_by_split_label.json"
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
        self.assertEqual(datamodule.dataset_calls, ["train", "val"])
        self.assertEqual(datamodule.loader_calls, [])
        self.assertEqual(
            [artifact_path for _, artifact_path in logger.artifact_calls],
            ["pyramid_selection/train", "pyramid_selection/val"],
        )
        self.assertTrue(
            all(not Path(local_dir).exists() for local_dir, _ in logger.artifact_calls)
        )

    def test_single_scale_setup_does_not_construct_loaders_or_log(self) -> None:
        datamodule = FakeDataModule(
            {split: object() for split in ("train", "val", "test")}
        )
        logger = FakeLogger()

        _log_pyramid_audits(datamodule, logger, "fit")

        self.assertEqual(datamodule.setup_stages, ["fit"])
        self.assertEqual(datamodule.dataset_calls, ["train", "val"])
        self.assertEqual(datamodule.loader_calls, [])
        self.assertEqual(logger.artifact_calls, [])

    def test_zero_length_audited_train_dataset_still_logs_without_a_loader(
        self,
    ) -> None:
        datamodule = FakeDataModule(
            {
                "train": EmptyAuditDataset(),
                "val": object(),
                "test": object(),
            }
        )
        logger = FakeLogger()

        _log_pyramid_audits(datamodule, logger, "fit")

        self.assertEqual(datamodule.loader_calls, [])
        self.assertEqual(
            [artifact_path for _, artifact_path in logger.artifact_calls],
            ["pyramid_selection/train"],
        )

    def test_non_fit_modes_audit_only_their_stage_split(self) -> None:
        cases = {
            "validate": "val",
            "test": "test",
            "predict": "test",
        }
        for mode, split in cases.items():
            with self.subTest(mode=mode):
                datamodule = FakeDataModule()
                logger = FakeLogger()

                _log_pyramid_audits(datamodule, logger, mode)

                self.assertEqual(datamodule.setup_stages, [mode])
                self.assertEqual(datamodule.dataset_calls, [split])
                self.assertEqual(datamodule.loader_calls, [])
                self.assertEqual(
                    [path for _, path in logger.artifact_calls],
                    [f"pyramid_selection/{split}"],
                )

    def test_main_finalizes_failed_run_and_reraises_audit_error(self) -> None:
        config = OmegaConf.create(
            {
                "seed": 1729,
                "datamodule": {},
                "ml": {},
                "trainer": {},
                "mode": "fit",
                "checkpoint": None,
                "test_after_fit": False,
            }
        )
        datamodule = FakeDataModule()
        logger = FakeLogger()
        audit_error = RuntimeError("audit upload failed")
        trainer = object()
        undecorated_main = train_main.__wrapped__.__wrapped__

        with (
            patch(
                "ml.train.hydra.utils.instantiate",
                side_effect=[datamodule, object(), trainer],
            ),
            patch("ml.train.seed_everything"),
            patch("ml.train._log_pyramid_audits", side_effect=audit_error),
            self.assertRaises(RuntimeError) as raised,
        ):
            undecorated_main(config, logger)

        self.assertIs(raised.exception, audit_error)
        self.assertEqual(logger.finalize_calls, ["failed"])


if __name__ == "__main__":
    unittest.main()
