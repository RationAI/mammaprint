"""Regression coverage for model-derived trainable parameter logging."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from ml.train import main as train_main


CONFIGS = Path(__file__).parents[2] / "configs"


class ModelParameterLoggingTest(unittest.TestCase):
    def test_logs_actual_counts_after_architecture_overrides(self) -> None:
        for attention_dim, expected in ((256, 5283586), (128, 4889986)):
            with self.subTest(attention_dim=attention_dim):
                with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
                    config = compose(
                        config_name="ml",
                        overrides=[
                            "+experiment=ml/multiscale/multiscale_transformer",
                            f"ml.aggregator.attention_dim={attention_dim}",
                        ],
                    )
                logger = self._run_main(instantiate(config.ml))
                logger.log_hyperparams.assert_called_once_with(
                    {"trainable_parameters": expected}
                )

    def test_excludes_frozen_parameters(self) -> None:
        module = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 1))
        module[0].requires_grad_(False)

        logger = self._run_main(module)

        logger.log_hyperparams.assert_called_once_with({"trainable_parameters": 4})

    @staticmethod
    def _run_main(module: nn.Module) -> Mock:
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
        datamodule = object()
        trainer = Mock()
        logger = Mock(spec=["log_hyperparams"])
        undecorated_main = train_main.__wrapped__.__wrapped__
        with (
            patch(
                "ml.train.hydra.utils.instantiate",
                side_effect=[datamodule, module, trainer],
            ),
            patch("ml.train.seed_everything"),
            patch("ml.train._log_pyramid_audits"),
        ):
            undecorated_main(config, logger)
        trainer.fit.assert_called_once_with(
            module, datamodule=datamodule, ckpt_path=None
        )
        return logger


if __name__ == "__main__":
    unittest.main()
