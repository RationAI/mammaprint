PATH = '/mnt/data/MOU/breast/mammaprint/P2023_0946.mrxs'

import openslide
from rationai.staining.estimate_stain_vectors import estimate_stain_vectors
from rationai.staining.utils.residual import residual

image = openslide.OpenSlide(PATH)
region = image.read_region((0, 0), 0, image.dimensions).convert("RGB")

print(estimate_stain_vectors(region))
