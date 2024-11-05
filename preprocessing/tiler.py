import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import pyarrow.parquet as pq
import pyarrow as pa
import mlflow
from tqdm import tqdm

# Allow processing of very large images
Image.MAX_IMAGE_PIXELS = None  

def calculate_annot_coverage(mask_tile, center_size):
    """Calculate the annotation coverage for the central region of a given tile."""
    center_x = mask_tile.shape[1] // 2
    center_y = mask_tile.shape[0] // 2
    half_center_size = center_size // 2
    
    center_tile = mask_tile[center_y - half_center_size:center_y + half_center_size, 
                            center_x - half_center_size:center_x + half_center_size]
    
    return np.mean(center_tile > 0)

def calculate_tissue_coverage(tile, min_tissue=0.5, max_tissue=1.0):
    """Calculate the tissue coverage for a tile and filter based on min and max thresholds."""
    tissue_coverage = np.mean(tile > 0)  # Assuming that tissue is represented by non-zero values
    return min_tissue <= tissue_coverage <= max_tissue

def process_single_tile(mask_array, x, y, tile_size, slide_name, global_x, global_y, center_size, scale_factor=2.0, min_tissue=0.5, max_tissue=1.0, min_annot_coverage=0.5):
    """Process a single tile and return its global coordinates and annotation coverage."""
    # Adjust coordinates and tile size to match the original slide's scale
    scaled_x = int(x / scale_factor)
    scaled_y = int(y / scale_factor)
    scaled_tile_size = int(tile_size / scale_factor)
    
    # Extract the corresponding tile from the mask
    mask_tile = mask_array[y:y + tile_size, x:x + tile_size]
    
    # Skip tile if tissue coverage doesn't meet requirements
    if not calculate_tissue_coverage(mask_tile, min_tissue, max_tissue):
        return None
    
    # Calculate annotation coverage
    annot_coverage = calculate_annot_coverage(mask_tile, center_size)
    if annot_coverage < min_annot_coverage:
        return None
    
    # Compute global coordinates in the slide space
    global_coord_x = global_x + scaled_x
    global_coord_y = global_y + scaled_y
    
    return {
        'coord_x': global_coord_x,
        'coord_y': global_coord_y,
        'slide_name': slide_name,
        'annot_coverage': annot_coverage
    }

def process_mask_and_save(mask_path, slide_name, tile_size=512, step_size=256, center_size=256, mask_level=0, global_x=0, global_y=0, scale_factor=2.0, min_tissue=0.5, max_tissue=1.0, min_annot_coverage=0.5):
    """Process a mask file and return tile data."""
    try:
        mask = Image.open(mask_path)
        mask.seek(mask_level)
        mask_array = np.array(mask)
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return []

    # Calculate step size in mask coordinates
    adjusted_step_size = int(step_size * scale_factor)
    
    coordinates = [
        (x, y)
        for y in range(0, mask_array.shape[0], adjusted_step_size)
        for x in range(0, mask_array.shape[1], adjusted_step_size)
    ]

    tile_data = []
    for x, y in coordinates:
        data = process_single_tile(mask_array, x, y, tile_size, slide_name, global_x, global_y, center_size, scale_factor, min_tissue, max_tissue, min_annot_coverage)
        if data:
            tile_data.append(data)

    return tile_data

def create_parquet_from_selected_masks(mask_dir, slide_dir, output_path, tile_size=512, step_size=256, center_size=256, mask_level=0, global_x=0, global_y=0, scale_factor=2.0, min_tissue=0.5, max_tissue=1.0, min_annot_coverage=0.5):
    """
    Process all mask files that have a corresponding slide in the slide directory,
    and save the combined tile data to a single Parquet file.
    """
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.tiff', '.tif'))]
    if not mask_files:
        print("No mask files found in the directory.")
        return None

    all_tile_data = []  # To store data from all slides

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        slide_name = os.path.splitext(mask_file)[0]  # Remove extension for slide name
        slide_path = os.path.join(slide_dir, slide_name + '.mrxs')  # Assuming slides have .mrxs extension

        # Check if corresponding slide exists
        if os.path.exists(slide_path):
            mask_path = os.path.join(mask_dir, mask_file)
            print(f"Processing mask: {mask_path} and corresponding slide: {slide_path}")
            
            # Process mask and collect tile data
            tile_data = process_mask_and_save(
                mask_path, slide_name, tile_size, step_size, center_size, mask_level, 
                global_x, global_y, scale_factor, min_tissue, max_tissue, min_annot_coverage
            )

            if tile_data:
                all_tile_data.extend(tile_data)
            else:
                print(f"No data was processed for {mask_file}.")
        else:
            print(f"Skipping {mask_file} because corresponding slide {slide_path} does not exist.")

    # Convert the collected data to a DataFrame and write it to a single Parquet file
    if all_tile_data:
        df = pd.DataFrame(all_tile_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
        print(f"Combined Parquet file saved to {output_path}")
        return output_path
    else:
        print("No data was processed for any mask-slide pairs.")
        return None

def main():
    # Define directories and MLflow parameters
    mask_directory = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_tissue_classification_tumor_masks/test_heatmaps"
    slide_directory = "/mnt/data/Projects/MOU/Mammaprint/Learnig_set_mamaprint/"
    output_parquet = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_tissue_classification_tumor_masks/tiles.parquet"
    
    mlflow_uri = "http://mlflow.rationai-mlflow:5000/"
    experiment_name = "Mamma-print"
    run_name = "mammaprint train set - Tissue classification tiling"
    description = "Tiling for mammaprint train set using tissue classification tumor masks"

    scale_factor = 2.0  # Use scale factor of 2 to adjust from mask to slide coordinates

    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, description=description):
        # Process all masks and save the combined data to a single Parquet file
        parquet_path = create_parquet_from_selected_masks(
            mask_directory, slide_directory, output_parquet, 
            tile_size=512, step_size=256, center_size=256, 
            mask_level=0, global_x=0, global_y=0, 
            scale_factor=scale_factor, min_tissue=0.5, 
            max_tissue=1.0, min_annot_coverage=0.5
        )

        if parquet_path:
            # Log the Parquet file as an artifact to MLflow
            mlflow.log_artifact(parquet_path)
            print(f"Combined Parquet file logged to MLflow: {parquet_path}")
        else:
            print("No Parquet file was created to log to MLflow.")

if __name__ == "__main__":
    main()
