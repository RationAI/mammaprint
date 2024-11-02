import torch
from pathlib import Path
import numpy
import logging
import albumentations
from mammaprint.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from mammaprint.datamodule.samplers import BaseSampler
from collections import defaultdict
import torch
import pyarrow as pa

class MILDataset(BaseDataset):
    def __init__(self, sampler: BaseSampler, seed, augmentations: albumentations.TemplateTransform | None = None, label: str = "class_id") -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations
        self.label = label
        self.slide_groups = defaultdict(list)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]
        # logging.debug(f"Prepared samples{len(sample)}")
        # Debug to check what's actually in sample
        images = []
        for s in sample:            
            model_output = s['model_output']
            model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
            images.append(model_output_tensor)

        if not images:
            logging.error("No valid images to process.")
            return None

        images_tensor = torch.stack(images)
        # logging.debug(f"Torch stack of images: {images_tensor.shape}")
        label = torch.tensor([float(sample[0][self.label])])  # Encapsulate the float in a list
        # logging.debug(f"Label: {label}")

        return images_tensor, label, sample
