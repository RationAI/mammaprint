import lightning.pytorch
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from histopipe.datamodule.datasets import GraphDataset


class GraphDataModule(lightning.pytorch.LightningDataModule):
    def __init__(
        self,
        dataset: GraphDataset,
        dataloaders_partial: dict[str, torch.utils.data.DataLoader],
        samplers: dict[str, torch.utils.data.Sampler] | None = None,
    ) -> None:
        super().__init__()
        if samplers is None:
            samplers = {}
        self.dataloaders_partial = dataloaders_partial
        self.datasets = dataset.split()
        self.samplers = samplers

    def _get_dataloader(self, name):
        dataset = self.datasets[name]
        if name in self.samplers:
            sampler_partial = self.samplers[name]
            dataset, sampler = sampler_partial(dataset)
        else:
            # default dataloader sampler
            sampler = None
        dataloader = self.dataloaders_partial[name]
        return dataloader(dataset, sampler=sampler)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_dataloader("train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("valid")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = self.datasets["test"]
        # fixed one graph per datalodader
        # not nice, but it testing callbacks work with this assumption.
        partial_loader = self.dataloaders_partial["test"]
        return [partial_loader(dataset=[graph], batch_size=1) for graph in dataset]
