import albumentations
import numpy as np
from numpy.typing import NDArray
from skimage.color import rgb2hsv


class SaturationSeparation(albumentations.ImageOnlyTransform):
    """Image augmentation performing saturation channel separation."""

    def __init__(self, p: float = 0.5, always_apply: bool = False) -> None:
        super().__init__(p=p, always_apply=always_apply)

    def apply(self, img: NDArray, **params) -> NDArray:
        """Assumes img to be rgb numpy array with values in [0, 255].

        Returns:
            Array with values in [0, 255] representing saturation.
        """
        hsv_array = rgb2hsv(img)
        s_channel = hsv_array[:, :, 1]

        s_channel = (s_channel * 255).astype(np.uint8)
        return s_channel
