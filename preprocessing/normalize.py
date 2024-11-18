import os
import argparse
import numpy as np
import pyvips
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearance of H&E stained images '''
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log((img.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
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
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)
    if saveFile is not None:
        save_image_with_pyvips(Inorm, saveFile + ".tiff")
    return Inorm


def load_large_image_pyvips(img_path):
    ''' Load large images using pyvips '''
    image = pyvips.Image.new_from_file(img_path, access="sequential")
    return np.ndarray(buffer=image.write_to_memory(),
                      dtype=np.uint8,
                      shape=(image.height, image.width, image.bands))


def save_image_with_pyvips(img: np.ndarray, output_path: str):
    '''Save normalized image as tiled and pyramidal TIFF using pyvips'''
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
    '''Process a single TIFF file'''
    img = load_large_image_pyvips(str(input_path))
    normalizeStaining(img, saveFile=str(output_path), Io=Io, alpha=alpha, beta=beta)


def process_files_in_parallel(input_dir: Path, output_dir: Path, max_workers: int = 16):
    '''Process all TIFF files in the input directory using parallel workers'''
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_dir.glob("*.tiff"))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                tiff_file,
                output_dir / tiff_file.stem
            )
            for tiff_file in tiff_files
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")


if __name__ == '__main__':
    input_dir = Path("/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_tiff/")
    output_dir = Path("/mnt/data/Projects/MOU/Mamaprint/Learning_set_mamaprint_normalized_tiff/")
    process_files_in_parallel(input_dir, output_dir, max_workers=32)

