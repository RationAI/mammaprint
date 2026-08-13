import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

import ml.train
from ml.pathologist_report import build_report_command, prediction_run_url
from ml.predict_heatmaps import build_command, merge_overrides, source_run_id


class PredictHeatmapsLauncherTest(unittest.TestCase):
    def test_source_run_id(self) -> None:
        self.assertEqual(
            source_run_id("mlflow-artifacts:/3/abc123/artifacts/checkpoints/best"),
            "abc123",
        )

    def test_build_command_creates_new_labeled_prediction_run(self) -> None:
        command = build_command(
            ["+experiment=ml/train_mil_embeddings_joint"],
            Path("/tmp/checkpoint.ckpt"),
            "mlflow-artifacts:/3/abc/artifacts/checkpoints/best",
            "abc",
            "source model",
            "test",
        )
        joined = " ".join(command)
        self.assertIn("mode=predict", command)
        self.assertIn("predict_split=test", command)
        self.assertIn("callbacks@trainer.callbacks.predictions_csv", joined)
        self.assertIn("tile_probability_heatmaps", joined)
        self.assertIn("source_run_id=abc", joined)
        self.assertNotIn("logger.run_id", joined)

    def test_prediction_command_can_tag_the_new_run(self) -> None:
        command = build_command(
            [],
            Path("/tmp/checkpoint.ckpt"),
            "mlflow-artifacts:/3/source/artifacts/checkpoints/best",
            "source",
            "source model",
            "test",
            invocation_id="unique123",
        )
        self.assertIn("+logger.tags.prediction_invocation_id=unique123", command)

    def test_caller_override_replaces_original_key(self) -> None:
        self.assertEqual(
            merge_overrides(
                ["+experiment=old", "data/embedded=l3", "feature_dim=2560"],
                ["+experiment=new", "feature_dim=1280"],
            ),
            ["data/embedded=l3", "+experiment=new", "feature_dim=1280"],
        )


class TrainPredictTest(unittest.TestCase):
    @patch("ml.train.hydra.utils.instantiate")
    def test_predict_does_not_retain_all_batch_outputs(self, instantiate: Mock) -> None:
        datamodule = object()
        module = object()
        trainer = Mock()
        instantiate.side_effect = [datamodule, module, trainer]
        config = OmegaConf.create(
            {
                "datamodule": {"_target_": "unused.DataModule"},
                "ml": {"_target_": "unused.Module"},
                "trainer": {"_target_": "unused.Trainer"},
                "mode": "predict",
                "checkpoint": "/tmp/model.ckpt",
                "test_after_fit": False,
            }
        )

        ml.train.main.__wrapped__.__wrapped__(config, Mock())

        trainer.predict.assert_called_once_with(
            model=module,
            datamodule=datamodule,
            ckpt_path="/tmp/model.ckpt",
            return_predictions=False,
        )


class PathologistReportTest(unittest.TestCase):
    def test_external_report_command_uses_shared_reporter(self) -> None:
        command = build_report_command(
            config_dir=Path("/repo/configs/report"),
            prediction_run_id="run123",
            tracking_uri="http://mlflow.internal",
            user="pathologist",
        )
        self.assertIn("+reporter=mammaprint", command)
        self.assertIn(
            "++reporter.evaluation_runs={run123:"
            "{artifact_file:predictions/predictions.json,"
            "slide_item_key:report_item_id}}",
            command,
        )

    def test_prediction_run_url(self) -> None:
        self.assertEqual(
            prediction_run_url("https://mlflow.example/", "3", "run123"),
            "https://mlflow.example/#/experiments/3/runs/run123",
        )


if __name__ == "__main__":
    unittest.main()
