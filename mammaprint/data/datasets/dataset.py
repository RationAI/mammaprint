import random
from pathlib import Path

import albumentations
import numpy as np
import PIL
import slidelip  # used for reading whole-slide images
import torch

from mammaprint.data.datasets.base_wsi import BaseDataset, extract_tile
from mammaprint.data.samplers import BaseSampler

# === MACENKO NORMALIZATION FUNCTIONS ===


def convert_to_OD(I: np.ndarray, Io: float = 240) -> np.ndarray:
    """
    Convert an RGB image to optical density (OD) space.
    Adding 1 to avoid log(0); Io is the transmitted light intensity.
    """
    # Ensure we are working in float and avoid zeros
    I = I.astype(np.float32)
    I[I == 0] = 1
    return -np.log((I + 1) / Io)


def get_stain_matrix(
    OD: np.ndarray, beta: float = 0.15, alpha: float = 1
) -> np.ndarray:
    """
    Estimate the stain matrix from the OD values.
    This follows the procedure described in Macenko et al.

    Args:
        OD: Optical density values reshaped as (num_pixels, 3).
        beta: Threshold to remove low OD pixels.
        alpha: Percentile used for robust estimation.

    Returns:
        stain_matrix: A (3x2) matrix whose columns are the estimated stain vectors.
    """
    # Remove pixels with low OD in all channels (background)
    OD_hat = OD[np.any(OD > beta, axis=1)]
    if OD_hat.size == 0:
        # If no pixels pass the threshold, return an identity-like matrix.
        return np.array([[1, 0], [0, 1], [0, 0]])

    # Compute covariance and eigen-decomposition
    cov = np.cov(OD_hat.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvectors descending (largest eigenvalues first)
    eigvecs = eigvecs[:, ::-1]

    # Project data onto the plane spanned by the first two eigenvectors
    That = np.dot(OD_hat, eigvecs[:, :2])
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(eigvecs[:, :2], np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(eigvecs[:, :2], np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Order the stain vectors: for example, ensure the first channel is higher for Hematoxylin
    if v1[0] > v2[0]:
        HE = np.array([v1, v2]).T  # shape (3,2)
    else:
        HE = np.array([v2, v1]).T
    return HE


def get_concentrations(OD: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """
    Compute stain concentrations by solving a least-squares problem.

    Args:
        OD: Optical density values reshaped as (num_pixels, 3).
        stain_matrix: (3x2) stain matrix.

    Returns:
        concentrations: A (2 x num_pixels) array.
    """
    # Solve for C in: stain_matrix * C = OD.T for each pixel.
    C, _, _, _ = np.linalg.lstsq(stain_matrix, OD.T, rcond=None)
    return C


def macenko_normalize(
    I: np.ndarray,
    Io: float = 240,
    beta: float = 0.15,
    alpha: float = 1,
) -> np.ndarray:
    """
    Normalize an image I using the Macenko method.

    This function first sets a target stain matrix (hard-coded from literature),
    then computes the optical density (OD) representation of the image, estimates the
    source stain matrix, and finally reconstructs a normalized image.

    Args:
        I: Input RGB image as a NumPy array (H x W x 3).
        Io: Transmitted light intensity (default: 240).
        beta: OD threshold for background pixels.
        alpha: Percentile for robust estimation.

    Returns:
        I_normalized: The stain-normalized image.
    """
    # === SET A TARGET STAIN MATRIX ===
    target_stain_matrix = np.array([[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]])
    target_stain_matrix = normalize_columns(target_stain_matrix)

    # Step 1: Convert image to optical density (OD) space.
    OD = convert_to_OD(I, Io)
    OD_reshaped = OD.reshape((-1, 3))

    # Step 2: Estimate stain matrix for the input image.
    source_stain_matrix = get_stain_matrix(OD_reshaped, beta, alpha)

    # Step 3: Get stain concentrations for the image.
    C = get_concentrations(OD_reshaped, source_stain_matrix)

    # Step 4: Normalize concentrations by the 99th percentile per stain.
    maxC = np.percentile(C, 99, axis=1)
    maxC[maxC == 0] = 1  # Prevent division by zero
    C_norm = C / maxC[:, None]

    # Step 5: Reconstruct OD using the target stain matrix and normalized concentrations.
    OD_normalized = np.dot(target_stain_matrix, C_norm)
    OD_normalized = OD_normalized.T.reshape(I.shape)

    # Step 6: Convert normalized OD back to RGB space.
    I_normalized = Io * np.exp(-OD_normalized) - 1
    I_normalized = np.clip(I_normalized, 0, 255).astype(np.uint8)
    return I_normalized


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Helper to normalize each column of a matrix to unit length."""
    return matrix / np.linalg.norm(matrix, axis=0)


class Dataset(BaseDataset):
    transforms: albumentations.TemplateTransform | None

    def __init__(
        self,
        sampler: BaseSampler,
        seed: int,
        augmentations: albumentations.TemplateTransform | None = None,
        label: str = "class_id",
    ) -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations
        self.label = label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]

        image = extract_tile(
            slide_fp=Path(sample.get("slide_fp")).resolve(),
            coord_x=sample["coord_x"],
            coord_y=sample["coord_y"],
            tile_size=sample["tile_size"],
            level=sample["sample_level"],
        )

        if self.transforms:
            random.seed(int(self._rng.integers(0, 2**63 - 1)))
            image = self.transforms(image=image)["image"]

        # Apply Macenko normalization.
        image = macenko_normalize(image)

        # Convert the normalized image to a torch.Tensor (with channels first).
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.FloatTensor([sample[self.label]])
        return image, label, sample
