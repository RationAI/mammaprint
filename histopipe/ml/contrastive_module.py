# Standard Imports
import logging

# Local Imports
from histopipe.ml.histopipemodule import HistoPipeModule


logger = logging.getLogger("contrastive_module")


class SimCLRModule(HistoPipeModule):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_1, x_2, _ = batch

        proj_1 = self(x_1)
        proj_2 = self(x_2)

        loss, sim_matrix = self.loss(proj_1, proj_2)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        self.metrics.update("train", sim_matrix, None)
        self.log_dict(self.metrics.get("train"), on_step=True)
        return {
            "loss": loss,
            "metrics": self.metrics.compute("train"),
            "outputs": proj_1,
        }

    def validation_step(self, batch, batch_idx):
        x_1, x_2, _ = batch

        proj_1 = self(x_1)
        proj_2 = self(x_2)

        loss, sim_matrix = self.loss(proj_1, proj_2)
        self.log("valid/loss", loss, on_step=True, on_epoch=True)

        self.metrics.update("valid", sim_matrix, None)
        self.log_dict(self.metrics.get("valid"), add_dataloader_idx=False)

        return {
            "loss": loss,
            "metrics": self.metrics.compute("valid"),
            "outputs": proj_1,
        }
