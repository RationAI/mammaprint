import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import lightning
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities.types import STEP_OUTPUT

import histopipe


if TYPE_CHECKING:
    from histopipe.trainer import Trainer


class TestVGG16(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            histopipe.ml.nets.VGG16Features(),
            histopipe.ml.nets.GMaxPool(),
            torch.nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, *args: any, **kwargs: any) -> any:
        return self.model(*args, **kwargs)

    def test_step(self, *args: any, **kwargs: any) -> STEP_OUTPUT | None:
        return self.model(*args, **kwargs)


def test_callback_simple(f_datamodule):
    logging.basicConfig(level=logging.DEBUG)
    trainer_conf = OmegaConf.load(Path("tests/conf/trainer/trainer_basic.yaml"))
    report_callback_conf = OmegaConf.load(
        Path("tests/conf/trainer/callbacks/reporter.yaml")
    )
    trainer_conf.callbacks = {"reporter": report_callback_conf.reporter}
    trainer: Trainer = hydra.utils.instantiate(trainer_conf, _convert_="partial")
    trainer.test(model=TestVGG16(), datamodule=f_datamodule)
    os.listdir(report_callback_conf.reporter.save_dir)
