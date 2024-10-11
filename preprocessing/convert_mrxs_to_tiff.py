import os
import numpy as np
import pyvips
import logging
from pathlib import Path
from datetime import datetime

BG_DIR_MRXS = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs/'
BG_DIR_TIFF = '/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tiff/'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_mrxs_to_tiff(input_file, output_file):
    start_time = datetime.now()
    logging.info(f"Converting {input_file} to {output_file}")
    
    image = pyvips.Image.new_from_file(input_file, level=1)
    image.tiffsave(
        output_file,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=512,
        tile_height=512,
        pyramid=True
    )
    
    end_time = datetime.now()
    logging.info(f"Conversion done in {end_time - start_time}.")

def main():
    file_paths = list(Path(BG_DIR_MRXS).glob('*.mrxs'))
    total_files = len(file_paths)
    
    for index, path in enumerate(file_paths, start=1):
        file_name = path.stem

        bg_in_path = os.path.join(BG_DIR_MRXS, file_name + '.mrxs')
        bg_out_path = os.path.join(BG_DIR_TIFF, file_name + '.tiff')
        
        # Check if the output file already exists
        if not Path(bg_out_path).exists():
            logging.info(f"Processing file {index} of {total_files} ({(index/total_files)*100:.2f}%)")
            convert_mrxs_to_tiff(bg_in_path, bg_out_path)
        else:
            logging.info(f"File {bg_out_path} already exists. Skipping conversion. ({(index/total_files)*100:.2f}%)")

if __name__ == '__main__':
    main()
