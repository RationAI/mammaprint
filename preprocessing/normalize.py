import numpy as np
import pyvips
import logging
from pathlib import Path
from datetime import datetime

# Hardcoded directories
INPUT_DIR = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tiff/'
OUTPUT_DIR = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_normalized/'

# Ensure the output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def macenko_normalization(image_np, Io=240, alpha=1, beta=0.15):
    """
    Perform Macenko stain normalization on a NumPy image array.

    Parameters:
    - image_np: The input RGB image as a NumPy array.
    - Io: Transmitted light intensity.
    - alpha: Percentile for robust extremes.
    - beta: Threshold for removing low-intensity pixels.

    Returns:
    - Inorm: Normalized image.
    """
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = image_np.shape
    if c != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Reshape image
    image_np = image_np.reshape((-1, 3))

    # Convert RGB to optical density (OD)
    OD = -np.log((image_np.astype(np.float32) + 1) / Io)

    # Remove transparent pixels (where any channel is less than beta)
    ODhat = OD[~np.any(OD < beta, axis=1)]

    if ODhat.size == 0:
        logging.warning("Insufficient data for normalization, returning original image.")
        return image_np.reshape((h, w, 3))

    # Compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Project onto the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
    That = ODhat.dot(eigvecs[:, :2])

    # Calculate angle and robust extremes
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, :2].dot([np.cos(minPhi), np.sin(minPhi)])
    vMax = eigvecs[:, :2].dot([np.cos(maxPhi), np.sin(maxPhi)])

    # Create the stain matrix
    if vMin[0] > vMax[0]:
        HE = np.array((vMin, vMax)).T
    else:
        HE = np.array((vMax, vMin)).T

    # Compute concentrations of the stains
    Y = np.dot(OD, np.linalg.pinv(HE))
    C = np.maximum(Y, 0)

    # Normalize stain concentrations
    maxC = np.array([np.percentile(C[:, 0], 99), np.percentile(C[:, 1], 99)])
    C2 = (C / maxC) * maxCRef

    # Recreate the normalized image
    Inorm = Io * np.exp(-np.dot(C2, HERef.T))
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8).reshape((h, w, 3))

    return Inorm

def process_image(input_file, output_file):
    """
    Read a TIFF file, apply Macenko normalization, and save the result as a normalized TIFF image.

    Parameters:
    - input_file: Path to the input TIFF file.
    - output_file: Path to the output normalized TIFF file.
    """
    start_time = datetime.now()
    logging.info(f"Processing {input_file}...")

    # Read the image using pyvips
    image = pyvips.Image.new_from_file(input_file, access='sequential')

    # Convert pyvips image to numpy array
    memory_array = image.write_to_memory()
    image_np = np.frombuffer(memory_array, dtype=np.uint8).reshape(image.height, image.width, image.bands)

    # Check if image has 3 bands (RGB)
    if image_np.shape[2] != 3:
        logging.warning(f"Image {input_file} does not have 3 bands (RGB). Skipping.")
        return

    # Apply Macenko normalization
    Inorm = macenko_normalization(image_np)

    # Convert normalized numpy array back to pyvips image
    normalized_image = pyvips.Image.new_from_memory(Inorm.tobytes(),
                                                    Inorm.shape[1],
                                                    Inorm.shape[0],
                                                    Inorm.shape[2],
                                                    'uchar')

    # Save the normalized image as a TIFF
    normalized_image.tiffsave(output_file, compression='deflate', bigtiff=True)
    logging.info(f"Saved normalized image to {output_file}")
    logging.info(f"Processing completed in {datetime.now() - start_time}")

def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_paths = list(input_dir.glob('*.tif*'))  # Match .tif and .tiff
    total_files = len(file_paths)

    for index, path in enumerate(file_paths, start=1):
        output_file = output_dir / (path.stem + '_normalized.tiff')

        if not output_file.exists():
            logging.info(f"Processing file {index} of {total_files} ({(index/total_files)*100:.2f}%)")
            process_image(str(path), str(output_file))
        else:
            logging.info(f"File {output_file} already exists. Skipping conversion.")

if __name__ == '__main__':
    main()
