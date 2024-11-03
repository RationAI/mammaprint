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
        self.default_feature_size = 2048  # Set expected feature size here (adjust if different)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]
        images = []

        # Define a default tensor with a known shape to ensure consistency
        default_tensor = torch.zeros(self.default_feature_size, dtype=torch.float32)

        for s in sample:
            model_output = s.get('model_output', None)
            
            # Ensure model_output is valid and matches the expected shape, otherwise use default tensor
            if isinstance(model_output, (list, tuple, torch.Tensor)):
                model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
                if model_output_tensor.shape == default_tensor.shape:
                    images.append(model_output_tensor)
                else:
                    logging.warning(f"Shape mismatch in model_output at index {index}, using default tensor.")
                    images.append(default_tensor)
            else:
                # Log if model_output is missing or invalid, and use the default tensor
                logging.warning(f"Invalid model_output in sample at index {index}, using default tensor as padding.")
                images.append(default_tensor)

        # Ensure exactly `tiles_per_bag` tiles by padding or truncating
        num_images = len(images)
        if num_images < self.tiles_per_bag:
            images += [default_tensor] * (self.tiles_per_bag - num_images)
            logging.info(f"Padded images to {self.tiles_per_bag} tiles at index {index}.")
        elif num_images > self.tiles_per_bag:
            images = images[:self.tiles_per_bag]
            logging.info(f"Truncated images to {self.tiles_per_bag} tiles at index {index}.")

        # Stack images into a tensor
        try:
            images_tensor = torch.stack(images)
        except RuntimeError as e:
            logging.error(f"Failed to stack images at index {index}. Error: {e}")
            # Return a fully padded tensor if stacking fails
            images_tensor = default_tensor.repeat(self.tiles_per_bag, 1)
        
        # Ensure label is provided correctly, fallback to zero if missing
        label_value = float(sample[0].get(self.label, 0.0))  # Default label if missing
        label = torch.tensor([label_value])

        return images_tensor, label, sample