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

        # Attempt to find the shape of a valid model_output, falling back to a default shape if needed
        sample_shape = None
        for s in sample:
            model_output = s.get('model_output', None)
            if isinstance(model_output, (list, tuple, torch.Tensor)):
                sample_shape = model_output.shape if hasattr(model_output, 'shape') else (len(model_output),)
                break

        # If no valid shape is found, use a default shape (e.g., [2048] for a feature vector)
        if sample_shape is None:
            logging.warning(f"No valid 'model_output' shape found in sample at index {index}. Using default shape [2048].")
            sample_shape = (2048,)  # Update this to match the expected feature size

        # Create a default tensor with the determined or fallback shape
        default_tensor = torch.zeros(sample_shape, dtype=torch.float32)

        for s in sample:
            model_output = s.get('model_output', None)
            
            # Verify model_output is correctly shaped, or use the default tensor if not
            if isinstance(model_output, (list, tuple, torch.Tensor)) and (hasattr(model_output, 'shape') and model_output.shape == sample_shape):
                model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
                images.append(model_output_tensor)
            else:
                logging.warning(f"Invalid or mismatched model_output in sample at index {index}. Adding default tensor as padding.")
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
            return default_tensor.repeat(self.tiles_per_bag, 1), torch.tensor([0.0]), {}

        label = torch.tensor([float(sample[0][self.label])])  # Encapsulate the float in a list

        return images_tensor, label, sample