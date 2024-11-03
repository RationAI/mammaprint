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

        # Find the shape of the first valid model_output to create default_tensor with the correct shape
        sample_shape = None
        for s in sample:
            model_output = s.get('model_output', None)
            if isinstance(model_output, (list, tuple, torch.Tensor)) and hasattr(model_output, 'shape'):
                sample_shape = model_output.shape  # Use full shape, not just length
                break

        # If we couldn't determine a valid shape, log an error and exit
        if sample_shape is None:
            logging.error(f"Could not determine a valid shape for 'model_output' in sample at index {index}")
            return None

        # Create default tensor with the determined shape
        default_tensor = torch.zeros(sample_shape, dtype=torch.float32)

        for s in sample:
            model_output = s.get('model_output', None)
            
            # Check if model_output has a shape attribute and is non-empty
            if isinstance(model_output, (list, tuple, torch.Tensor)) and hasattr(model_output, 'shape') and model_output.shape == sample_shape:
                model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
                images.append(model_output_tensor)
            else:
                # Log any invalid or mismatched model outputs and add the default tensor as padding
                logging.warning(f"Invalid or mismatched model_output in sample at index {index}, adding default tensor as padding.")
                images.append(default_tensor)

        # Ensure exactly `tiles_per_bag` tiles by padding or truncating
        num_images = len(images)
        if num_images < self.tiles_per_bag:
            images += [default_tensor] * (self.tiles_per_bag - num_images)
            logging.info(f"Padded images to {self.tiles_per_bag} tiles at index {index}.")
        elif num_images > self.tiles_per_bag:
            images = images[:self.tiles_per_bag]
            logging.info(f"Truncated images to {self.tiles_per_bag} tiles at index {index}.")

        # Verify that all tensors in `images` have the correct shape before stacking
        for i, img in enumerate(images):
            if img.shape != default_tensor.shape:
                logging.error(f"Image at position {i} has shape {img.shape} which does not match expected shape {default_tensor.shape}.")
                return None  # Optionally handle this more gracefully if needed

        # Stack images into a tensor
        images_tensor = torch.stack(images)
        label = torch.tensor([float(sample[0][self.label])])  # Encapsulate the float in a list

        return images_tensor, label, sample
