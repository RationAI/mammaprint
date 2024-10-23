# Third-party Imports
import albumentations
import mlflow
import torch

# Local Imports
from histopipe.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from histopipe.datamodule.samplers import BaseSampler


class SegmentationDataset(BaseDataset):
    transforms: albumentations.TemplateTransform | None
    _downloaded: dict[str, str]  # mapping URI -> fp
    grayscale: bool

    def __init__(
        self,
        sampler: BaseSampler,
        seed: int,
        grayscale: bool,
        augmentations: albumentations.TemplateTransform | None = None,
    ) -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations
        self._downloaded: dict[str, str] = {}
        self.grayscale = grayscale

    def _download_sample_artifacts(self, uri: str, dst: str) -> str:
        """Downloads artifacts from mlflow URI and stores them in `./slides/{dst}/` directory.

        Avoids repeated downloads by caching paths of downloaded artifacts in `_downloaded` dict.
        """
        if uri in self._downloaded:
            slide_fp = self._downloaded[uri]
        else:
            slide_fp = mlflow.artifacts.download_artifacts(
                artifact_uri=uri, dst_path=f"./slides/{dst}/"
            )
            self._downloaded[uri] = slide_fp
        return slide_fp

    def generate_samples(self) -> None:
        """This method gathers samples from sampler and also downloads all slides for current epoch.

        It modifies samples: adds `mask_fp` and `slide_fp` keys.
        """
        super().generate_samples()

        # Downloading slides and masks from mlflow URI and storing fp in sample
        for sample in self._epoch_samples:
            sample["slide_fp"] = self._download_sample_artifacts(
                uri=sample["mlflow_slide_uri"], dst="slide"
            )
            sample["mask_fp"] = self._download_sample_artifacts(
                uri=sample["mask_path"], dst="mask"
            )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """__getitem__.

        Returns:
            x: input RGB image as tensor with values in range [-1, 1]
            y: desired output as B&W mask with values in range [0, 1]
            sample: dict object representing tile properties
        """
        sample = self._epoch_samples[index]

        kwargs = {
            "coord_x": sample["coord_x"],
            "coord_y": sample["coord_y"],
            "tile_size": sample["tile_size"],
            "level": sample["sample_level"],
        }

        image = extract_tile(slide_fp=sample["slide_fp"], **kwargs)
        mask = extract_tile(slide_fp=sample["mask_fp"], **kwargs)

        # from 3 Color channels to 1
        if self.grayscale:
            image = image[:, :, 0].reshape(sample["tile_size"], sample["tile_size"], 1)

        mask = mask[:, :, 0].reshape(sample["tile_size"], sample["tile_size"], 1)

        transformed = {"image": image, "mask": mask}
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)

        # permute to (channels, height, width)
        image = torch.from_numpy(transformed["image"]).float().permute(2, 0, 1)
        mask = torch.from_numpy(transformed["mask"]).float().permute(2, 0, 1) / 255
        return image, mask, sample
