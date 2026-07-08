"""Thin Lightning datamodule wiring three slide datasets.

Splits are owned by the datasets: each of ``train``/``val``/``test`` is a fully
constructed :class:`~ml.data.datasets.pyramid.PyramidSlideDataset` (built by Hydra
with its own ``split=...``). This module only wraps them in dataloaders — no
artifact download or split logic lives here.

Bags have variable tile/region counts, so batches are collated as a list of
samples (:func:`mil_collate`) rather than a stacked tensor.
"""

from collections.abc import Sequence

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from ml.typing import MILSample  #, MultiScaleSample


type Sample = MILSample # | MultiScaleSample


def mil_collate[S](batch: Sequence[S]) -> list[S]:
    """Collate variable-length bags by keeping the batch as a list of samples."""
    return list(batch)


class DataModule(pl.LightningDataModule):
    """Wraps pre-built train/val/test datasets in dataloaders.

    Args:
        train: Training dataset (a slide dataset yielding MIL samples).
        val: Validation dataset.
        test: Test dataset.
        batch_size: Slides (bags) per batch.
        num_workers: Dataloader worker processes.
        pin_memory: Pin host memory for faster host->device copies.
        persistent_workers: Keep workers alive between epochs.
    """

    def __init__(
        self,
        train: Dataset[Sample],
        val: Dataset[Sample],
        test: Dataset[Sample],
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.datasets = {"train": train, "val": val, "test": test}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def _dataloader(self, split: str, shuffle: bool) -> DataLoader[Sample]:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=mil_collate,
        )

    def train_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("test", shuffle=False)


__all__ = ["DataModule", "mil_collate"]
