# Standard Imports
import logging
import sys
from collections import defaultdict
from copy import copy

# Third-party Imports
import lightning.pytorch
from albumentations import TemplateTransform
from humanize import naturalsize
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch.utils.data import DataLoader

# Local Imports
from histopipe.datamodule.datasets import Ki67Dataset
from histopipe.datamodule.datasets.base_wsi import BaseDataset
from histopipe.datamodule.datasources import BaseDataSource, JpgDataSource


log = logging.getLogger("datamodule")


class WSIDataModule(lightning.pytorch.LightningDataModule):
    """WSIDataModule.

    Attributes:
        datasets (dict[str, BaseDataset]): Dict of datasets, keys must be in ['train', 'valid', 'test', 'predict'].
        data_sources (dict[str, BaseDataSource]):
        dataloaders_kwargs (dict[str, dict]): See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
    """

    datasets: dict[str, BaseDataset]
    data_sources: dict[str, BaseDataSource]
    dataloaders_kwargs: dict[str, dict]

    def __init__(
        self,
        datasets: dict[str, BaseDataset],
        data_sources: dict[str, BaseDataSource],
        dataloaders_kwargs: dict[str, dict],
        multi_dataloaders: bool = True,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.data_sources = data_sources
        self.dataloaders_kwargs = dataloaders_kwargs
        self._built = False
        self._val_dataloader = None
        self._val_dataloader_built = False
        self.multi_dataloaders = multi_dataloaders

    def setup(self, stage: str | None = None) -> None:
        if self._built:
            return

        splits = self._split_data_sources()
        self._build_datasets(splits)
        self._built = True

    def _split_data_sources(self) -> dict[str, dict[str, BaseDataSource]]:
        """Splits data sources into train, valid, test, and predict data sources.

        The original data sources are deleted after splitting.
        """
        all_ds_splits = defaultdict(dict)

        while self.data_sources:
            ds_name, ds = self.data_sources.popitem()
            ds_splits = ds.split()

            for stage, ds_split in ds_splits.items():
                all_ds_splits[stage][ds_name] = ds_split

        del self.data_sources
        log.debug(f"Data sources split between {', '.join(all_ds_splits.keys())}.")

        return dict(all_ds_splits)

    def _build_datasets(
        self, data_sources: dict[str, dict[str, BaseDataSource]]
    ) -> None:
        for stage, dataset in self.datasets.items():
            splits = data_sources.get(stage)
            if splits is None:
                raise ValueError(f"{stage} split not found in data sources.")

            if len(splits) > 1:
                raise NotImplementedError(
                    f"Multiple data sources found for {stage} split."
                )
                raise NotImplementedError(
                    f"Multiple data sources found for {stage} split."
                )

            ds_name, ds = splits.popitem()
            dataset.sampler.build_inner_structure(ds)
            log.debug(f"{stage} dataset built")

    def _build_val_dataloader(self):
        """We typically rebuild the dataloaders after each epoch (in trainer by `reload_dataloaders_every_n_epochs=1`), but we want the validation dataloader unchanged."""
        self.datasets["valid"].generate_samples()
        kwargs = self.dataloaders_kwargs.get("valid", {})
        self._val_dataloader = DataLoader(self.datasets["valid"], **kwargs)

    def train_dataloader(self):
        self.datasets["train"].generate_samples()
        return DataLoader(
            self.datasets["train"], **self.dataloaders_kwargs.get("train", {})
        )

    def val_dataloader(self):
        if not self._val_dataloader_built:
            self._build_val_dataloader()
            self._val_dataloader_built = True
        return self._val_dataloader

    def _prepare_multi_dataloaders(self, stage: str):
        """On test and predict, we assume sequential sampling. One dataloader contains the tiles of one slide. We thus create a list of dataloaders, for the whole test set. todo: Current implementation requires large shm."""
        log.debug(f"Generating {stage} DataLoaders")
        dataset = self.datasets[stage]
        dataloaders = []
        there_is_more = True
        while there_is_more:
            try:
                dataset.generate_samples()
                dataset_size = sys.getsizeof(dataset)
                dataset_size = naturalsize(dataset_size, binary=True)
                dl = DataLoader(copy(dataset), **self.dataloaders_kwargs.get(stage, {}))
                dataloaders.append(dl)
                log.debug(
                    f"Dataloader copied (sampler.active.node={dataset.sampler.active_node})."
                )
            except StopIteration:
                there_is_more = False
        return dataloaders

    def predict_dataloader(self):
        if self.multi_dataloaders:
            return self._prepare_multi_dataloaders("predict")
        self.datasets["predict"].generate_samples()
        return DataLoader(
            self.datasets["predict"], **self.dataloaders_kwargs.get("predict", {})
        )

    def test_dataloader(self):
        if self.multi_dataloaders:
            return self._prepare_multi_dataloaders("test")
        self.datasets["test"].generate_samples()
        return DataLoader(
            self.datasets["test"], **self.dataloaders_kwargs.get("test", {})
        )


class Ki67DataModule(lightning.pytorch.LightningDataModule):
    """DataModule for Ki67 scans.

    Scans are stored as jpgs, therefore no tilling is required.
    """

    def __init__(
        self,
        data_sources: JpgDataSource,
        dataloaders_kwargs: dict[str, dict],
        transform: TemplateTransform | None = None,
    ) -> None:
        super().__init__()
        self.dataloaders_kwargs = dataloaders_kwargs
        self.data_sources = data_sources
        self.transform = transform
        self.datasets: dict[str, Ki67Dataset] = {}
        self.setup("")  # W0201 hack

    def setup(self, stage: str) -> None:
        for stage, ds in self.data_sources.split().items():
            self.datasets[stage] = Ki67Dataset(
                ds.get_path(), ds.get_table(), self.transform
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.datasets["train"],
            **self.dataloaders_kwargs.get("train", {}),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.datasets["valid"],
            **self.dataloaders_kwargs.get("valid", {}),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.datasets["test"],
            **self.dataloaders_kwargs.get("test", {}),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()
