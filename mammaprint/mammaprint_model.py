from lightning import LightningModule
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from mammaprint.modeling.binary_classifier import BinaryClassifier
from mammaprint.typing import Input, Outputs


class MammaprintModel(LightningModule):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = BinaryClassifier()
        self.criterion = nn.BCELoss()

        self.val_metrics = MetricCollection(
            {
                "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary"),
                "precision": Precision("binary"),
                "recall": Recall("binary"),
            }
        )
        self.test_metrics = LazyMetricDict(self.val_metrics.clone())
        self.val_metrics.prefix = "validation/"

    def forward(self, x: Tensor) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log(
            "train/loss", loss, batch_size=len(inputs), on_step=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch: Input) -> None:
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log(
            "validation/loss",
            loss,
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def test_step(self, batch: Input) -> None:
        inputs, targets, metadata = batch
        outputs = self(inputs)
        for output, target, slide in zip(
            outputs, targets, metadata["slide"], strict=False
        ):
            self.test_metrics.update(output, target, key=slide)

    def on_test_epoch_end(self) -> None:
        for key, metrics in self.test_metrics.compute().items():
            table = {k: v.item() for k, v in metrics.items()}
            self.logger.log_table({"slide": key, **table}, "test_metrics.json")
        self.test_metrics.reset()

    def predict_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Outputs:
        inputs, _ = batch
        return self(inputs)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=0.00001)
