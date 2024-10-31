import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import pyarrow.parquet as pq
import pyarrow as pa
import mlflow
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # Handle very large images

def calculate_annot_coverage(mask_tile, center_size):
    """Calculate the annotation coverage for the central region of a given tile."""
    center_x = mask_tile.shape[1] // 2
    center_y = mask_tile.shape[0] // 2
    half_center_size = center_size // 2
    
    center_tile = mask_tile[center_y - half_center_size:center_y + half_center_size, 
                            center_x - half_center_size:center_x + half_center_size]
    
    if np.sum(center_tile) == 0:
        return 0.0
    else:
        return np.mean(center_tile > 0)

def calculate_tissue_coverage(tile, min_tissue=0.5, max_tissue=1.0):
    """Calculate the tissue coverage for a tile and filter based on min and max thresholds."""
    tissue_coverage = np.mean(tile > 0)  # Assuming that tissue is represented by non-zero values
    return min_tissue <= tissue_coverage <= max_tissue

def process_single_tile(mask_array, x, y, tile_size, slide_name, global_x, global_y, center_size, scale_factor=2.0, min_tissue=0.5, max_tissue=1.0, min_annot_coverage=0.5):
    """Process a single tile and return its global coordinates and annotation coverage."""
    mask_tile = mask_array[y:y + tile_size, x:x + tile_size]
    
    if not calculate_tissue_coverage(mask_tile, min_tissue, max_tissue):
        print(f"Tile at (x, y): ({x}, {y}) excluded due to tissue coverage.")
        return None
    
    annot_coverage = calculate_annot_coverage(mask_tile, center_size)
    if annot_coverage < min_annot_coverage:
        print(f"Tile at (x, y): ({x}, {y}) excluded due to low annotation coverage ({annot_coverage}).")
        return None
    
    global_coord_x = int((global_x + x) * scale_factor)
    global_coord_y = int((global_y + y) * scale_factor)
    
    print(f"Processing tile at (x, y): ({x}, {y}), Global coordinates: ({global_coord_x}, {global_coord_y}), Scale factor: {scale_factor}")
    
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

    coordinates = [
        (x, y)
        for y in range(0, mask_array.shape[0], step_size)
        for x in range(0, mask_array.shape[1], step_size)
    ]

    tile_data = []
    for x, y in coordinates:
        data = process_single_tile(mask_array, x, y, tile_size, slide_name, global_x, global_y, center_size, scale_factor, min_tissue, max_tissue, min_annot_coverage)
        if data:
            tile_data.append(data)

    return tile_data

def create_parquet_from_selected_masks(mask_dir, slide_dir, output_path, tile_size=512, step_size=256, center_size=256, mask_level=0, global_x=0, global_y=0, scale_factor=2.0, min_tissue=0.5, max_tissue=1.0, min_annot_coverage=0.5):
    """Generate a Parquet file from mask files that have corresponding slides in the slide directory."""
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.tiff', '.tif'))]
    if not mask_files:
        print("No mask files found in the directory.")
        return

    all_tile_data = []

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        slide_name = os.path.splitext(mask_file)[0]
        slide_path = os.path.join(slide_dir, slide_name + '.mrxs')  # Assuming slides have .mrxs extension

        if not os.path.exists(slide_path):
            print(f"Skipping {mask_file} because corresponding slide {slide_path} does not exist.")
            continue

        mask_path = os.path.join(mask_dir, mask_file)

        print(f"Processing mask: {mask_path}")
        tile_data = process_mask_and_save(mask_path, slide_name, tile_size, step_size, center_size, mask_level, global_x, global_y, scale_factor, min_tissue, max_tissue, min_annot_coverage)

        if tile_data:
            all_tile_data.extend(tile_data)

    # Convert the collected data to a DataFrame and write it to Parquet
    if all_tile_data:
        df = pd.DataFrame(all_tile_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
        print(f"Parquet file saved to {output_path}")
    else:
        print("No data was processed and saved.")

def main():
    mask_directory = "/mnt/data/Projects/MOU/Mammaprint/Test_set_tissue_classification_tumor_masks/test_heatmaps"
    slide_directory = "/mnt/data/Projects/MOU/Mammaprint/Test_set_mamaprint/"
    output_parquet = "/mnt/data/Projects/MOU/Mammaprint/Test_set_tissue_classification_tumor_masks/output_tiles.parquet"
    mlflow_uri = "http://mlflow.rationai-mlflow:5000/"
    experiment_name = "Mamma-print"
    run_name = "mammaprint test set - Tissue classification tiling"
    description = "Tiling for mammaprint, coverage 0.5"

    scale_factor = 2.0  # Apply this scale factor to match the original Parquet file

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, description=description):
        create_parquet_from_selected_masks(mask_directory, slide_directory, output_parquet, scale_factor=scale_factor)

        mlflow.log_artifact(output_parquet)
        print(f"Parquet file logged to MLflow: {output_parquet}")

if __name__ == "__main__":
    main()
