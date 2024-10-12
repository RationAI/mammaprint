# Customized original code taken from https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

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

# Macenko Normalization Function
def macenko_normalization(image_np, Io=240, alpha=1, beta=0.15):
    """
    Perform Macenko stain normalization on a NumPy image array.
    """
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    
    h, w, c = image_np.shape
    image_np = image_np.reshape((-1, 3))

    # Convert RGB to OD
    OD = -np.log((image_np.astype(np.float32) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]


    # Check if the tile has sufficient valid pixels to perform normalization
    
    # Check if the tile has sufficient valid pixels to perform normalization
    if ODhat.size == 0:
        logging.warning("Insufficient data for normalization, returning original image.")
        return image_np

    _, V = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(V[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha * 100)
    maxPhi = np.percentile(phi, (1 - alpha) * 100)

    vMin = V[:, 1:3].dot(np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = V[:, 1:3].dot(np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    HE = np.array((vMin, vMax)).T if vMin[0] > vMax[0] else np.array((vMax, vMin)).T

    Y = np.dot(OD, np.linalg.inv(HE))
    C = np.maximum(Y, 0)
    maxC = np.array([np.percentile(C[:, 0], 99), np.percentile(C[:, 1], 99)])
    C2 = np.divide(C, maxC) * maxCRef

    Inorm = Io * np.exp(-np.dot(C2, HERef.T))
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8).reshape((h, w, 3))

    return Inorm

# Function to process and normalize a single tile
def process_tile(input_file, tile_size, x, y):
    image = pyvips.Image.new_from_file(input_file)
    tile = image.crop(x, y, min(tile_size, image.width - x), min(tile_size, image.height - y))

    tile_np = np.ndarray(buffer=tile.write_to_memory(), dtype=np.uint8, shape=[tile.height, tile.width, 3])
    normalized_tile_np = macenko_normalization(tile_np)

    return x, y, normalized_tile_np

# Convert TIFF to normalized TIFF with batch processing and parallelization
def convert_tiff_to_tiff_parallel(input_file, output_file, tile_size=256, num_threads=4):
    start_time = datetime.now()
    logging.info(f"Converting {input_file} to {output_file}")

    image = pyvips.Image.new_from_file(input_file)
    width, height = image.width, image.height
    output_image = pyvips.Image.black(width, height, bands=3)

    all_tiles = [(x, y) for y in range(0, height, tile_size) for x in range(0, width, tile_size)]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_tile, input_file, tile_size, x, y): (x, y) for (x, y) in all_tiles}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                x, y, normalized_tile_np = result
                normalized_tile = pyvips.Image.new_from_memory(
                    normalized_tile_np.tobytes(),
                    normalized_tile_np.shape[1],
                    normalized_tile_np.shape[0],
                    normalized_tile_np.shape[2],
                    'uchar'
                )
                output_image = output_image.insert(normalized_tile, x, y)

    output_image.tiffsave(output_file, bigtiff=True, compression=pyvips.enums.ForeignTiffCompression.DEFLATE, tile=True)
    logging.info(f"Processing completed in {datetime.now() - start_time}")

# Main function to process TIFF slides from the specified directory
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
