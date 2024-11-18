import os
import argparse
import numpy as np
import pyvips


def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearance of H&E stained images '''
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log((img.astype(np.float) + 1) / Io)
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
        from PIL import Image
        Image.fromarray(Inorm).save(saveFile + '.png')
    return Inorm


def load_large_image_pyvips(img_path):
    ''' Load large images using pyvips '''
    image = pyvips.Image.new_from_file(img_path, access="sequential")
    return np.ndarray(buffer=image.write_to_memory(),
                      dtype=np.uint8,
                      shape=(image.height, image.width, image.bands))


def batch_normalize(input_dir, output_dir, Io=240, alpha=1, beta=0.15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".tiff"):
            img_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, os.path.splitext(file_name)[0])
            print(f"Processing {file_name}...")
            img = load_large_image_pyvips(img_path)  # Load using pyvips
            normalizeStaining(img, saveFile=output_path, Io=Io, alpha=alpha, beta=beta)
    print("Batch normalization complete!")



if __name__ == '__main__':
    input_dir = "/mnt/data/Projects/MOU/Mammaprint/Learning_set_mamaprint_tiff/"
    output_dir = "/mnt/data/Projects/MOU/Mammaprint/Learnig_set_mamaprint_normalized_tiff/"
    batch_normalize(input_dir, output_dir, Io=240, alpha=1, beta=0.15)
