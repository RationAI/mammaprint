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
        images = []

        # Determine the shape of the first model_output tensor to create a default tensor of matching shape
        if 'model_output' in sample[0]:
            sample_shape = len(sample[0]['model_output'])  # Assuming model_output is a 1D array
            default_tensor = torch.zeros(sample_shape, dtype=torch.float32)
        else:
            logging.error(f"No 'model_output' key found in sample at index {index}")
            return None

        for s in sample:
            model_output = s.get('model_output', None)
            if model_output is not None:
                model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
                images.append(model_output_tensor)
            else:
                # Log any missing or empty model outputs in samples
                logging.warning(f"Empty or None model_output in sample at index {index}, adding default tensor as padding.")
                images.append(default_tensor)

        # Ensure exactly `tiles_per_bag` tiles by padding or truncating
        num_images = len(images)
        if num_images < self.tiles_per_bag:
            images += [default_tensor] * (self.tiles_per_bag - num_images)
            logging.info(f"Padded images to {self.tiles_per_bag} tiles at index {index}.")
        elif num_images > self.tiles_per_bag:
            images = images[:self.tiles_per_bag]
            logging.info(f"Truncated images to {self.tiles_per_bag} tiles at index {index}.")

        if not images:
            logging.error("No valid images to process.")
            return None

        # Stack images into a tensor
        images_tensor = torch.stack(images)
        label = torch.tensor([float(sample[0][self.label])])  # Encapsulate the float in a list

        return images_tensor, label, sample
