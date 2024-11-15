import torch
import logging
import albumentations
from mammaprint.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from mammaprint.datamodule.samplers import BaseSampler
import torch

class MILDataset(BaseDataset):
    def __init__(self, sampler: BaseSampler, seed, augmentations: albumentations.TemplateTransform | None = None, label: str = "luminal_id") -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations
        self.label = label
        self.max_tiles = 2000
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]
        images = []

        for s in sample:
            model_output = s.get('model_output', None)
            model_output_tensor = torch.tensor(model_output, dtype=torch.float32)
            images.append(model_output_tensor)

        # Pad or truncate the number of tiles
        num_tiles = len(images)
        if num_tiles < self.max_tiles:
            # Add zero-padding for missing tiles
            padding = [torch.zeros_like(images[0]) for _ in range(self.max_tiles - num_tiles)]
            images.extend(padding)

        # Stack images into a tensor
        images_tensor = torch.stack(images)
        
        # Ensure label is provided correctly, fallback to zero if missing
        # label_value = float(sample[0].get(self.label, 0.0))  # Default label if missing
        # Check if any tile in `sample` has `self.label` not 0.0
        label_value = next((tile.get(self.label) for tile in sample if tile.get(self.label, 0.0) != 0.0), 0.0)
        logging.info(f"label value found: {label_value}")
        label = torch.tensor([label_value])

        return images_tensor, label, sample