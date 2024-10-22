# Copyright (c) The RationAI team.

import random
from pathlib import Path
import torchvision.transforms.functional as F

import albumentations
import torch
from torchvision import transforms

from mammaprint.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from mammaprint.datamodule.samplers import BaseSampler


class ClassificationDataset(BaseDataset):
    transforms: albumentations.TemplateTransform | None

    def __init__(
        self,
        sampler: BaseSampler,
        seed: int,
        augmentations: albumentations.TemplateTransform | None = None,
    ) -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations

        self.preprocess = transforms.Compose([
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]

        image = extract_tile(
            slide_fp=Path(sample.get("slide_fp")).resolve(),
            coord_x=sample["coord_x"],
            coord_y=sample["coord_y"],
            tile_size=sample["tile_size"],
            level=sample["sample_level"],
        )

        if self.transforms:
            random.seed(int(self._rng.integers(0, 2**63 - 1)))
            image = self.transforms(image=image)["image"]

        image = F.to_tensor(image)
        image = self.preprocess(image)
        label = torch.FloatTensor([sample["class_id"]])
        return image, label, sample
