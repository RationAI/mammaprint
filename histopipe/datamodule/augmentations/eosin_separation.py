import numpy as np
from numpy.typing import NDArray
from skimage.color import separate_stains

from histopipe.datamodule.augmentations.her_separation import HERSeparation


class EosinSeparation(HERSeparation):
    """Custom image augmentation performing Eosin staining separation on given HE image."""

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

    def _get_eosin(self, her: NDArray) -> NDArray:
        eosin_channel = her[:, :, 1]

        e_img = np.stack(
            (np.zeros_like(eosin_channel), eosin_channel, np.zeros_like(eosin_channel)),
            axis=-1,
        )

        return e_img

    def apply(self, img: NDArray, **params) -> NDArray:
        her_img = separate_stains(rgb=img, conv_matrix=self.her_from_rgb)
        e_img = self._get_eosin(her_img)
        return self.convert_to_rgb(e_img)
