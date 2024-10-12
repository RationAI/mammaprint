# Customized original code taken from https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py

import pyvips
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directories
BG_DIR_TIFF_INPUT = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tiff/'
BG_DIR_TIFF_OUTPUT = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_normalized_tiff/'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Macenko Normalization Function
def macenko_normalization(image_np, Io=240, alpha=1, beta=0.15):
    """
    Perform Macenko stain normalization on a NumPy image array.
    """
    # Reference matrix for H&E
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = image_np.shape
    image_np = image_np.reshape((-1, 3))

    # Convert RGB to OD
    OD = -np.log((image_np.astype(np.float32) + 1) / Io)
    
    # Remove transparent or low-intensity pixels (OD < beta)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    # Check if the tile has sufficient valid pixels to perform normalization
    if ODhat.size == 0:
        logging.warning("Tile contains insufficient data for normalization, skipping.")
        return image_np.reshape((h, w, 3))  # Return the original image

    # Compute eigenvectors
    _, V = np.linalg.eigh(np.cov(ODhat.T))

    # Project onto the plane spanned by the two largest eigenvectors
    That = ODhat.dot(V[:, 1:3])

    # Compute robust extremes
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha * 100)
    maxPhi = np.percentile(phi, (1 - alpha) * 100)

    vMin = V[:, 1:3].dot(np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = V[:, 1:3].dot(np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    if vMin[0] > vMax[0]:
        HE = np.array((vMin, vMax)).T
    else:
        HE = np.array((vMax, vMin)).T

    # Convert back to RGB
    Y = np.dot(OD, np.linalg.inv(HE))
    C = np.maximum(Y, 0)

    maxC = np.array([np.percentile(C[:, 0], 99), np.percentile(C[:, 1], 99)])
    C2 = np.divide(C, maxC) * maxCRef

    Inorm = Io * np.exp(-np.dot(C2, HERef.T))
    Inorm[Inorm > 255] = 255
    Inorm = np.reshape(Inorm, (h, w, 3)).astype(np.uint8)

    return Inorm

# Function to process and normalize a single tile
def process_tile(image, tile_size, x, y):
    """
    Extract and process a tile of size tile_size at coordinates (x, y).
    Applies Macenko normalization to the tile.
    """
    tile = image.crop(x, y, tile_size, tile_size)
    
    # Convert tile to NumPy array
    tile_np = np.ndarray(buffer=tile.write_to_memory(),
                         dtype=np.uint8,
                         shape=[tile.height, tile.width, 3])
    
    # Apply Macenko normalization
    normalized_tile_np = macenko_normalization(tile_np)

    # Convert back to pyvips image
    normalized_tile = pyvips.Image.new_from_memory(
        normalized_tile_np.tobytes(),
        normalized_tile_np.shape[1],  # width
        normalized_tile_np.shape[0],  # height
        normalized_tile_np.shape[2],  # bands (channels)
        'uchar'
    )
    
    return x, y, normalized_tile

# Function to process tiles in batches
def process_batch(tiles, image, tile_size):
    """
    Process a batch of tiles by applying normalization to each.
    """
    results = []
    for (x, y) in tiles:
        results.append(process_tile(image, tile_size, x, y))
    return results

# Function to convert tiff to normalized tiff with batch processing and parallelization
def convert_tiff_to_tiff_parallel(input_file, output_file, tile_size=256, num_threads=4, batch_size=16):
    """
    Reads a TIFF file, processes the image tile-by-tile using Macenko normalization,
    and saves the result as a tiled .tiff image. Processing is done in parallel using threads.
    
    Parameters:
    - input_file: Path to the input TIFF file.
    - output_file: Path to the output normalized TIFF file.
    - tile_size: Size of the tiles for processing.
    - num_threads: Number of threads to use for parallel processing.
    - batch_size: Number of tiles to process in one batch.
    """
    start_time = datetime.now()
    logging.info(f"Converting {input_file} to {output_file} using batch processing and parallel threads")

    # Load the image from .tiff using pyvips
    image = pyvips.Image.new_from_file(input_file)
    width, height = image.width, image.height

    # Create an empty VIPS image for the output
    output_image = pyvips.Image.black(width, height, bands=3)

    # Generate all tile coordinates
    all_tiles = [(x, y) for y in range(0, height, tile_size) for x in range(0, width, tile_size)]
    total_tiles = len(all_tiles)

    # Process in batches
    processed_tiles = 0
    for i in range(0, total_tiles, batch_size):
        batch_tiles = all_tiles[i:i + batch_size]

        # Parallelize the processing of each tile in the batch
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_tile, image, tile_size, x, y) for (x, y) in batch_tiles]

            for future in as_completed(futures):
                x, y, normalized_tile = future.result()
                output_image = output_image.insert(normalized_tile, x, y)

        # Update the number of processed tiles
        processed_tiles += len(batch_tiles)

        # Calculate and log percentage complete
        percentage_complete = (processed_tiles / total_tiles) * 100
        logging.info(f"Processed {processed_tiles} / {total_tiles} tiles ({percentage_complete:.2f}%)")

    # Save the final normalized image as a tiled .tiff
    output_image.tiffsave(
        output_file,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True
    )

    end_time = datetime.now()
    logging.info(f"Batch and parallel processing completed in {end_time - start_time}")

# Main function to process the first 3 TIFF slides
def main():
    # Get the first 3 .tiff files
    file_paths = list(Path(BG_DIR_TIFF_INPUT).glob('*.tiff'))[:3]
    total_files = len(file_paths)
    
    # Process each file
    for index, path in enumerate(file_paths, start=1):
        file_name = path.stem

        bg_in_path = Path(BG_DIR_TIFF_INPUT) / (file_name + '.tiff')
        bg_out_path = Path(BG_DIR_TIFF_OUTPUT) / (file_name + '_normalized.tiff')
        
        # Check if the output file already exists
        if not bg_out_path.exists():
            logging.info(f"Processing file {index} of {total_files} ({(index/total_files)*100:.2f}%)")
            convert_tiff_to_tiff_parallel(bg_in_path, bg_out_path, tile_size=256, num_threads=4, batch_size=16)
        else:
            logging.info(f"File {bg_out_path} already exists. Skipping conversion. ({(index/total_files)*100:.2f}%)")

if __name__ == '__main__':
    main()
