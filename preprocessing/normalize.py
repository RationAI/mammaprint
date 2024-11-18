import os
from pathlib import Path
import pyvips
import numpy as np

def normalizeStaining_pyvips(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearance of H&E stained images using pyvips and numpy '''
    # Convert pyvips image to numpy array for normalization
    img = np.ndarray(buffer=img.write_to_memory(),
                     dtype=np.uint8,
                     shape=(img.height, img.width, img.bands))
    
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    
    h, w, c = img.shape
    img = img.reshape((-1, 3))
    
    # Compute optical density
    OD = -np.log((img.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    # Compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T
    
    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    
    # Recreate normalized image
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
    
    # Return normalized numpy image
    return Inorm


def save_image_with_pyvips(img: np.ndarray, output_path: str):
    ''' Save normalized image as tiled and pyramidal TIFF using pyvips '''
    vips_img = pyvips.Image.new_from_memory(
        img.tobytes(), img.shape[1], img.shape[0], img.shape[2], "uchar"
    )
    vips_img.tiffsave(
        str(output_path),
        pyramid=True,
        tile=True,
        tile_width=256,
        tile_height=256
    )


def process_file(input_path: Path, output_path: Path, Io=240, alpha=1, beta=0.15):
    ''' Process a single TIFF file '''
    print(f"Processing: {input_path}")
    image = pyvips.Image.new_from_file(str(input_path), access="sequential")
    normalized_img = normalizeStaining_pyvips(image, Io=Io, alpha=alpha, beta=beta)
    save_image_with_pyvips(normalized_img, str(output_path))


def process_files_sequentially(input_dir: Path, output_dir: Path):
    ''' Process all TIFF files sequentially '''
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_dir.glob("*.tiff"))
    for tiff_file in tiff_files:
        try:
            output_path = output_dir / f"{tiff_file.stem}.tiff"
            process_file(tiff_file, output_path)
        except Exception as e:
            print(f"Error processing file {tiff_file}: {e}")


if __name__ == '__main__':
    input_dir = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_tiff/"
    output_dir = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_normalized_tiff/"
    process_files_sequentially(input_dir, output_dir)
