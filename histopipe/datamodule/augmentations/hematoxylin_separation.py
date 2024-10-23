import numpy as np
from numpy.typing import NDArray
from skimage.color import separate_stains

from histopipe.datamodule.augmentations.her_separation import HERSeparation


class HematoxylinSeparation(HERSeparation):
    """Custom image augmentation performing Hematoxylin staining separation on given HE image."""

    def __init__(
        self,
        h_vector: list[float],
        e_vector: list[float],
        r_vector: list[float],
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(
            h_vector=h_vector,
            e_vector=e_vector,
            r_vector=r_vector,
            p=p,
            always_apply=always_apply,
        )

    def _get_hematoxylin(self, her: NDArray) -> NDArray:
        hematoxylin_channel = her[:, :, 0]
        h_img = np.stack(
            (
                hematoxylin_channel,
                np.zeros_like(hematoxylin_channel),
                np.zeros_like(hematoxylin_channel),
            ),
            axis=-1,
        )

        return h_img

    def apply(self, img: NDArray, **params) -> NDArray:
        her_img = separate_stains(rgb=img, conv_matrix=self.her_from_rgb)
        h_img = self._get_hematoxylin(her_img)
        return self.convert_to_rgb(h_img)
