import pyvips
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Input and output directories
input_dir = Path("/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tissue_classification_tumor_masks/test_heatmaps")
output_dir = Path("/mnt/data/Projects/MOU/Mammaprint/Another_WSIs_tissue_classification_tumor_masks_resized/")
output_dir.mkdir(parents=True, exist_ok=True)
scaling_factor = 0.5
size_limit_mb = 40  # Size limit in MB

def downsize_tiff(input_path: Path, scale: float) -> None:
    """Downsize, create a pyramid, and save TIFF files with LZW compression."""
    # Output path
    output_path = output_dir / (input_path.stem + ".tiff")
    
    # Check if the output file exists and is larger than the size limit
    if output_path.exists():
        print(f"Skipping {output_path}: already exists and is larger than {size_limit_mb}MB")
        return

    # Load the image, resize, and save
    img = pyvips.Image.new_from_file(str(input_path), access="sequential")
    img_resized = img.resize(scale)
    
    # Save with LZW compression and pyramid structure
    img_resized.tiffsave(
        str(output_path), 
        compression="lzw",
        pyramid=True,  # Enable pyramid structure for multi-resolution
        tile=True,     # Enables tiled TIFF for efficient loading
        tile_width=256,  # Set tile size, 256x256 is common for zoomable images
        tile_height=256
    )
    print(f"Downsized, pyramid, and saved: {output_path}")

def process_files_in_parallel(input_dir: Path, scale: float, max_workers: int = 32) -> None:
    """Process all TIFF files in input_dir using up to max_workers in parallel."""
    tiff_files = list(input_dir.glob("*.tiff"))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each file to be processed in parallel
        futures = [
            executor.submit(downsize_tiff, tiff_file, scale)
            for tiff_file in tiff_files
        ]
        # Wait for all futures to complete
        for future in futures:
            future.result()

# Run the parallel processing
if __name__ == "__main__":
    process_files_in_parallel(input_dir, scaling_factor, max_workers=32)
