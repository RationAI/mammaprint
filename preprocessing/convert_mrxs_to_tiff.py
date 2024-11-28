import os
import numpy as np
import pyvips
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

BG_DIR_MRXS = '/mnt/data/Projects/MOU/Mammaprint/Test_set_mamaprint/'
BG_DIR_TIFF = '/mnt/data/Projects/MOU/Mammaprint/Test_set_mamaprint_tiff/'

# Set file size threshold to 500 MB
SIZE_THRESHOLD_MB = 500 * 1024 * 1024  # Convert MB to bytes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_mrxs_to_tiff(input_file, output_file):
    """Convert a single MRXS file to a TIFF file."""
    start_time = datetime.now()
    logging.info(f"Converting {input_file} to {output_file}")
    
    try:
        image = pyvips.Image.new_from_file(input_file, level=0)
        image.tiffsave(
            output_file,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=512,
            tile_height=512,
            pyramid=True
        )
    except Exception as e:
        logging.error(f"Error converting {input_file} to TIFF: {e}")
    finally:
        end_time = datetime.now()
        logging.info(f"Conversion done in {end_time - start_time} for {input_file}.")


def process_file(file_paths, index, total_files):
    """Wrapper to process a single file and handle logging."""
    file_name = file_paths.stem

    bg_in_path = os.path.join(BG_DIR_MRXS, file_name + '.mrxs')
    bg_out_path = os.path.join(BG_DIR_TIFF, file_name + '.tiff')

    # Check if the output file already exists
    if Path(bg_out_path).exists():
        file_size = os.path.getsize(bg_out_path)
        if file_size > SIZE_THRESHOLD_MB:
            logging.info(f"File {bg_out_path} already exists and is larger than 500 MB. Skipping conversion. ({(index/total_files)*100:.2f}%)")
            return
        else:
            logging.info(f"File {bg_out_path} already exists but is smaller than 500 MB. Proceeding with conversion.")
    else:
        logging.info(f"Processing file {index} of {total_files} ({(index/total_files)*100:.2f}%)")
    
    # Convert the MRXS to TIFF if the conditions are met
    convert_mrxs_to_tiff(bg_in_path, bg_out_path)


def main():
    file_paths = list(Path(BG_DIR_MRXS).glob('*.mrxs'))
    total_files = len(file_paths)

    if total_files == 0:
        logging.info("No MRXS files found for conversion.")
        return

    max_workers = min(4, os.cpu_count())  # Use up to the available number of CPUs
    logging.info(f"Starting conversion of {total_files} files with {max_workers} workers.")

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, path, index, total_files)
            for index, path in enumerate(file_paths, start=1)
        ]
        for future in futures:
            future.result()  # Ensure any exceptions are raised

    logging.info("All conversions completed.")


if __name__ == '__main__':
    main()
