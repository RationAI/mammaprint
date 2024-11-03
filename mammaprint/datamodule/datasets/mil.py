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
        self.tiles_per_bag = 2000
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]
        # logging.debug(f"Prepared samples{len(sample)}")
        # Debug to check what's actually in sample
        images = []
        default_tensor = torch.zeros(2048, dtype=torch.float32)
        for s in sample:            
            model_output = s['model_output']
            model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
            images.append(model_output_tensor)
        
        # Ensure exactly 3000 tiles by padding or truncating
        num_images = len(images)
        if num_images < self.tiles_per_bag:
            # Pad with default tensors if fewer than `tiles_per_bag`
            images += [default_tensor] * (self.tiles_per_bag - num_images)
            logging.info(f"Padded images to {self.tiles_per_bag} tiles at index {index}.")
            logging.info(f"Number of tiles in slide {len(images)} tiles at index {index}.")
        elif num_images > self.tiles_per_bag:
            # Truncate if more than `tiles_per_bag`
            images = images[:self.tiles_per_bag]
            logging.info(f"Truncated images to {self.tiles_per_bag} tiles at index {index}.")
            logging.info(f"Number of tiles in slide {len(images)} tiles at index {index}.")


        if not images:
            logging.error("No valid images to process.")
            return None

        images_tensor = torch.stack(images)
        # logging.debug(f"Torch stack of images: {images_tensor.shape}")
        label = torch.tensor([float(sample[0][self.label])])  # Encapsulate the float in a list
        # logging.debug(f"Label: {label}")

        return images_tensor, label, sample
