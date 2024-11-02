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
        images = []
        default_tensor = torch.zeros(2048, dtype=torch.float32)

        for s in sample:
            model_output = s.get('model_output')
            if isinstance(model_output, list) and len(model_output) == 2048:
                # Create a tensor and ensure the shape is correct
                model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
                if model_output_tensor.shape == (2048,):
                    images.append(model_output_tensor)
                else:
                    logging.warning(f"Unexpected tensor shape {model_output_tensor.shape} at index {index}. Using default tensor.")
                    images.append(default_tensor)
            else:
                logging.warning(f"Invalid or missing model_output at index {index}. Using default tensor.")
                images.append(default_tensor)  # Use default zero tensor if model_output is invalid

        # Check if images is empty or has inconsistent shapes
        if not images:
            logging.error("No valid images to process.")
            return None

        try:
            images_tensor = torch.stack(images)
        except RuntimeError as e:
            logging.error(f"Error stacking images: {e}. Shapes: {[img.shape for img in images]}")
            return None

        label = torch.tensor([float(sample[0][self.label])])  # Encapsulate the float in a list
        return images_tensor, label, sample
