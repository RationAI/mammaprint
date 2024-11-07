import pyvips
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Input and output directories
input_dir = Path("/mnt/data/Projects/MOU/Mammaprint/Learning_set_tissue_classification_tumor_masks/test_heatmaps")
output_dir = Path("/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_tissue_masks_resized/")
output_dir.mkdir(parents=True, exist_ok=True)
scaling_factor = 0.5

def downsize_tiff(input_path: Path, scale: float) -> None:
    """Downsize and save TIFF files in two versions: LZW and JPEG compression."""
    img = pyvips.Image.new_from_file(str(input_path), access="sequential")
    img_resized = img.resize(scale)
    
    # Output paths
    output_path_lzw = output_dir / (input_path.stem + ".tiff")

    # Save with LZW compression (lossless, often efficient for binary masks)
    img_resized.tiffsave(
        str(output_path_lzw), 
        compression="lzw"
    )
    print(f"downsized and saved: {output_path_lzw}")

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
