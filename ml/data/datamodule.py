"""Thin Lightning datamodule that builds its split datasets lazily.

Each split (``train``/``val``/``test``) is passed as an *unresolved* Hydra config
node and instantiated on demand in :meth:`setup`, only for the running stage — so a
test-only run never downloads the train/val artifacts, and vice versa. The
entrypoint must instantiate this with ``_recursive_=False`` so the split nodes
arrive as ``DictConfig`` rather than pre-built datasets.

Pattern follows the sibling ``ulcerative_colitis.data.DataModule``. Bags have
variable tile/region counts, so batches stay a list of samples (:func:`mil_collate`).
"""

from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from ml.data.datasets.single_scale import SingleScaleDataset
from ml.typing import MILSample


type Sample = MILSample


def mil_collate[S](batch: Sequence[S]) -> list[S]:
    """Collate variable-length bags by keeping the batch as a list of samples."""
    return list(batch)


class DataModule(pl.LightningDataModule):
    """Builds train/val/test datasets lazily from unresolved config nodes.

    Args:
        batch_size: Slides (bags) per batch.
        num_workers: Dataloader worker processes.
        pin_memory: Pin host memory for faster host->device copies.
        persistent_workers: Keep workers alive between epochs.
        eval_batch_size: Slides per validation, test, or prediction batch.
        train_seed: Optional dedicated seed for shuffling and training workers.
        **datasets: One entry per split (``train``/``val``/``test``), each an
            unresolved dataset ``DictConfig`` instantiated in :meth:`setup`.
            Prediction reuses the test split.
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        eval_batch_size: int | None = None,
        train_seed: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.eval_batch_size = (
            batch_size if eval_batch_size is None else eval_batch_size
        )
        self._train_generator = (
            torch.Generator().manual_seed(train_seed)
            if train_seed is not None
            else None
        )
        self._dataset_cfgs = datasets
        self._datasets: dict[str, Dataset[Sample]] = {}

    def _build(self, split: str) -> None:
        if split in self._datasets:
            return
        cfg = self._dataset_cfgs.get(split)
        if cfg is None:
            raise KeyError(f"No dataset config provided for split '{split}'.")
        self._datasets[split] = instantiate(cfg)

    def setup(self, stage: str | None = None) -> None:
        needed = {
            "fit": ("train", "val"),
            "validate": ("val",),
            "test": ("test",),
            "predict": ("test",),
        }.get(stage or "", ("train", "val", "test"))
        for split in needed:
            self._build(split)

    def _dataloader(self, split: str, shuffle: bool) -> DataLoader[Sample]:
        training = split == "train"
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size if training else self.eval_batch_size,
            shuffle=shuffle,
            generator=self._train_generator if training else None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=mil_collate,
        )

    @property
    def train_dataset(self) -> Dataset[Sample]:
        return self._datasets["train"]

    def state_dict(self) -> dict[str, Any]:
        if self._train_generator is None:
            return {}
        state: dict[str, Any] = {"train_generator": self._train_generator.get_state()}
        dataset = self._datasets.get("train")
        if self.num_workers == 0 and isinstance(dataset, SingleScaleDataset):
            state["tile_sampler"] = dataset.tile_sampler_state()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._train_generator is None or "train_generator" not in state_dict:
            return
        self._train_generator.set_state(state_dict["train_generator"].cpu())
        dataset = self._datasets.get("train")
        if self.num_workers == 0 and isinstance(dataset, SingleScaleDataset):
            dataset.restore_tile_sampler(state_dict.get("tile_sampler"))

    def train_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader[Sample]:
        return self._dataloader("test", shuffle=False)


__all__ = ["DataModule", "mil_collate"]
