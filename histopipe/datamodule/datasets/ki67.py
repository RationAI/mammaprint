# Standard Imports
from pathlib import Path

# Third-party Imports
import albumentations
import numpy as np
import torch
from pandas import DataFrame
from PIL import Image


class Ki67Dataset(torch.utils.data.Dataset):
    """Dataset for Ki67 scans.

    `dir_path`: path to the dir with images in format YY_CASE_N.jpg
    `labels`:   dataframe with labels stored in `percentage` column and images
                name (with .jpg) in `index` column
    """

    def __init__(
        self,
        dir_path: Path,
        labels: DataFrame,
        augmentations: albumentations.TemplateTransform | None = None,
    ) -> None:
        self.dir_path = dir_path
        self.labels = labels
        self.transform = augmentations

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        name = str(self.labels.iloc[index].name)
        image = np.array(Image.open(self.dir_path / name))
        target = torch.tensor(
            self.labels.iloc[index]["percentage"], dtype=torch.float32
        )

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, target.unsqueeze(0), {"filename": name}
