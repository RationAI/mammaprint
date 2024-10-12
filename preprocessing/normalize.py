import numpy as np
import pyvips
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Hardcoded directories
INPUT_DIR = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tiff/'
OUTPUT_DIR = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_normalized/'

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
    
    # Ensure the input is in the expected shape
    h, w, c = image_np.shape
    if c != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Reshape image
    image_np = image_np.reshape((-1, 3))

    # Convert RGB to optical density (OD)
    OD = -np.log((image_np.astype(np.float32) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Remove low-intensity pixels

    if ODhat.size == 0:
        logging.warning("Insufficient data for normalization, returning original image.")
        return image_np.reshape((h, w, 3))

    # Compute eigenvectors
    _, V = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(V[:, 1:3])

    # Compute robust extremes
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha * 100)
    maxPhi = np.percentile(phi, (1 - alpha) * 100)

    vMin = V[:, 1:3].dot(np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = V[:, 1:3].dot(np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Create the hematoxylin and eosin (HE) mixing matrix
    HE = np.array((vMin, vMax)).T if vMin[0] > vMax[0] else np.array((vMax, vMin)).T

    # Convert OD back to RGB
    Y = np.dot(OD, np.linalg.inv(HE))
    C = np.maximum(Y, 0)

    maxC = np.array([np.percentile(C[:, 0], 99), np.percentile(C[:, 1], 99)])
    C2 = np.divide(C, maxC) * maxCRef

    Inorm = Io * np.exp(-np.dot(C2, HERef.T))
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8).reshape((h, w, 3))

    return Inorm

def process_tile(input_file, tile_size, x, y):
    """
    Extract and normalize a tile of the image.

    Parameters:
    - input_file: Path to the input TIFF file.
    - tile_size: Size of the tile to be processed.
    - x: X-coordinate of the top-left corner of the tile.
    - y: Y-coordinate of the top-left corner of the tile.

    Returns:
    - Tuple of (x, y, normalized_tile_np).
    """
    image = pyvips.Image.new_from_file(input_file)
    
    # Handle boundary tiles
    tile_width = min(tile_size, image.width - x)
    tile_height = min(tile_size, image.height - y)
    
    # Crop the tile
    tile = image.crop(x, y, tile_width, tile_height)

    # Convert tile to NumPy array
    tile_np = np.ndarray(buffer=tile.write_to_memory(),
                         dtype=np.uint8,
                         shape=[tile_height, tile_width, 3])

    # Apply Macenko normalization
    normalized_tile_np = macenko_normalization(tile_np)

    return x, y, normalized_tile_np

def convert_tiff_to_tiff_parallel(input_file, output_file, tile_size=256, num_threads=4):
    """
    Read a TIFF file, process it tile-by-tile using Macenko normalization,
    and save the result as a normalized TIFF image.

    Parameters:
    - input_file: Path to the input TIFF file.
    - output_file: Path to the output normalized TIFF file.
    - tile_size: Size of the tiles for processing.
    - num_threads: Number of processes to use for parallel processing.
    """
    start_time = datetime.now()
    logging.info(f"Converting {input_file} to {output_file}")

    image = pyvips.Image.new_from_file(input_file)
    width, height = image.width, image.height
    output_image = pyvips.Image.black(width, height, bands=3)

    # Generate all tile coordinates
    all_tiles = [(x, y) for y in range(0, height, tile_size) for x in range(0, width, tile_size)]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_tile, input_file, tile_size, x, y): (x, y) for (x, y) in all_tiles}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                x, y, normalized_tile_np = result

                # Convert the normalized NumPy tile back to pyvips and insert into output image
                normalized_tile = pyvips.Image.new_from_memory(
                    normalized_tile_np.tobytes(),
                    normalized_tile_np.shape[1],  # width
                    normalized_tile_np.shape[0],  # height
                    normalized_tile_np.shape[2],  # bands (channels)
                    'uchar'
                )
                output_image = output_image.insert(normalized_tile, x, y)

    # Save the final normalized image as a TIFF
    output_image.tiffsave(output_file, bigtiff=True, compression=pyvips.enums.ForeignTiffCompression.DEFLATE, tile=True)
    logging.info(f"Processing completed in {datetime.now() - start_time}")

def main():
    file_paths = list(Path(INPUT_DIR).glob('*.tiff'))
    total_files = len(file_paths)
    
    for index, path in enumerate(file_paths, start=1):
        output_file = Path(OUTPUT_DIR) / (path.stem + '_normalized.tiff')
        
        if not output_file.exists():
            logging.info(f"Processing file {index} of {total_files} ({(index/total_files)*100:.2f}%)")
            convert_tiff_to_tiff_parallel(str(path), str(output_file), tile_size=256, num_threads=4)
        else:
            logging.info(f"File {output_file} already exists. Skipping conversion.")

if __name__ == '__main__':
    main()
