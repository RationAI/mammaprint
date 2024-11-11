# Copyright (c) The RationAI team.

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pyvips
import scipy
import torch
from numpy.typing import NDArray

from mammaprint.trainer.callbacks.vis_mode import VisualisationMode


logger = logging.getLogger("callbacks/image_builder")


class ImageBuilder(ABC):
    visualization_mode: VisualisationMode
    save_dir: Path | str
    filename: str

    def __init__(
        self, vis_mode: VisualisationMode, save_dir: Path | str, filename: str
    ) -> None:
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.visualization_mode = vis_mode

        self.save_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def update(self, data: torch.Tensor, metadata: dict) -> None: ...

    @abstractmethod
    def save(self) -> str: ...

    @staticmethod
    def _to_numpy(data: torch.Tensor) -> NDArray:
        if isinstance(data, np.ndarray):
            return data

        if isinstance(data, torch.Tensor):
            logger.debug("Converting tiles.")
            return data.detach().cpu().numpy()

        raise TypeError(f"Type '{type(data)}' is not supported.")

    @staticmethod
    def _to_nwhc(data: NDArray) -> NDArray:
        """Converts data to [N, W, H, C] format.

        [N, W, H, C]:
            N - batch size
            W - width in pixels
            H - height in pixels
            C - number of channels
        """
        if data.ndim == 4:  # tile is image
            logger.debug("Shuffling dimensions.")
            return data.transpose(
                (0, 2, 3, 1)
            )  # [N, C, W, H] into sensible [N, W, H, C]
        if data.ndim == 2:  # tile is classification
            logger.debug("Adding dimensions.")
            return data[:, None, None, :]  # [N, C] into [N, W, H, C]
        raise ValueError(
            f"Incorrect data format. Expected 4-dim [N, C, W, H] or 2-dim [N, C]"
            f" but found {data.ndim}-dim {data.shape}."
        )

    def _preprocess_data(self, data: torch.Tensor) -> NDArray:
        """Converts data to numpy, scales it to [0,255] and resizes it to tile_size if segmentation task is visualized.

        Args:
            data: A tensor of shape [N, C, W, H] or [N, C].
        """
        data = self._to_numpy(data)
        data = self._to_nwhc(data)
        data = self.visualization_mode.scale(data)  # Scale from [0,1] to [0,255]
        return data

    def _finalize_image(self, vips_im: pyvips.Image) -> pyvips.Image:
        # Restore default value
        if self.visualization_mode.zero_offset != 0:
            vips_im += self.visualization_mode.zero_offset
        return vips_im


class ImageAssembler(ImageBuilder, ABC):
    image_size: tuple[int, int, int]
    tile_size: int
    interpolation: str

    def __init__(
        self,
        image_size: tuple[int, int, int],
        tile_size: int,
        vis_mode: VisualisationMode,
        save_dir: Path | str,
        filename: str,
        interpolation: str,
    ) -> None:
        super().__init__(vis_mode=vis_mode, save_dir=save_dir, filename=filename)
        self.image_size = image_size
        self.tile_size = tile_size
        self.interpolation = interpolation

    def _resize_to_tile_size(self, data: NDArray) -> NDArray:
        interpolation_map = {"nearest": 0, "bilinear": 1, "cubic": 2}
        width, height = data.shape[1], data.shape[2]

        if self.tile_size != width or self.tile_size != height:
            logger.debug("Resizing the tiles to fit tile_size.")
            zoom = (1, self.tile_size / width, self.tile_size / height, 1)
            data = scipy.ndimage.zoom(
                input=data, zoom=zoom, order=interpolation_map[self.interpolation]
            )

        return data

    def _save_xopat_compatible(self, vips_im: pyvips.Image) -> str:
        vips_im = vips_im.cast("uchar")
        save_path = (self.save_dir / self.filename).with_suffix(".tiff")

        vips_im.tiffsave(
            save_path,
            bigtiff=True,
            compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
        )

        return save_path


class DiskMappedPatchAssembler(ImageAssembler):
    """A class to compose any masks into a segmentation overlay matching the input image.

    Stores data in a disk-mapped arrays to prevent RAM overflow.
    Is generally slower than in-memory patch assemblers.
    """

    scale_factor: int
    image: np.memmap
    count: np.memmap

    def __init__(
        self,
        metadata: dict,
        vis_mode: VisualisationMode,
        save_dir: Path | str,
        interpolation: str = "nearest",
    ) -> None:
        filename = metadata["slide_name"]
        image_size = (
            int(metadata["slide_width"]),
            int(metadata["slide_height"]),
            int(metadata["slide_channels"]),
        )
        tile_size = int(metadata["tile_size"])
        super().__init__(
            image_size=image_size,
            tile_size=tile_size,
            vis_mode=vis_mode,
            save_dir=save_dir,
            filename=filename,
            interpolation=interpolation,
        )

        self.scale_factor = 2 ** int(metadata["sample_level"])

        save_path = (self.save_dir / self.filename).as_posix()

        self.image = np.memmap(
            save_path + "_mask.nmp",
            dtype=np.float32,
            mode="w+",
            shape=(
                self.image_size[1],
                self.image_size[0],
                self.image_size[2],
            ),  # row-first format (H, W, C)
        )
        self.count = np.memmap(
            save_path + "_count.nmp",
            dtype=np.int32,
            mode="w+",
            shape=(
                self.image_size[1],
                self.image_size[0],
                1,
            ),  # row-first format (H, W, C)
        )

    def update(self, data: torch.Tensor, metadata: dict) -> None:
        logger.debug("Pasting tiles.")
        xs, ys = (
            metadata["coord_x"] // self.scale_factor,
            metadata["coord_y"] // self.scale_factor,
        )
        data = self._preprocess_data(data)
        data = self._resize_to_tile_size(data)

        # Paste tiles onto mask
        for x, y, tile in zip(xs, ys, data, strict=False):
            mm_h, mm_w, mm_c = self.image[
                y : y + self.tile_size, x : x + self.tile_size, :
            ].shape
            self.image[y : y + self.tile_size, x : x + self.tile_size, :] += tile[
                :mm_h, :mm_w, :mm_c
            ]
            self.count[y : y + self.tile_size, x : x + self.tile_size, :] += 1

        self.image.flush()
        self.count.flush()

    def save(self) -> str:
        # Converting to pyVips
        vips_im = pyvips.Image.new_from_array(self.image)
        count_im = pyvips.Image.new_from_array(self.count)

        # Resolve overlaps
        vips_im /= count_im  # invokes zero-safe division from vips

        # TODO: Add options to save each band as separate image?
        # Sub: [vips_im_i.tiffsave() for vips_im_i in vips_im.bandsplit()]

        vips_im = self._finalize_image(vips_im)
        save_path = self._save_xopat_compatible(vips_im)

        return save_path


class InMemoryHeatmapAssembler(ImageAssembler):
    """A class to accumulate predictions into a prediction heatmap.

    Minimizes required RAM by calculating minimum loss-less resolution
    from tile size and step size and accumulates predictions into compressed arrays.
    Compression is meaningful only for scalar predictions,
    To control compression, use `compress_accumulator_array` parameter.
    """

    heatmap_accumulator: np.ndarray
    patch_overlap_counter: np.ndarray
    gcd_size_factor: int
    accumulator_tile_size: int
    overlap_counter_tile_size: int
    compress_accumulator_array: bool

    def __init__(
        self,
        metadata: dict,
        vis_mode: VisualisationMode,
        save_dir: Path | str,
        interpolation: str,
        compress_accumulator_array: bool,
    ) -> None:
        filename = metadata["slide_name"]
        image_size = (
            int(metadata["slide_width"]),
            int(metadata["slide_height"]),
            int(metadata["slide_channels"]),
        )
        tile_size = int(metadata["tile_size"])
        step_size = int(metadata["step_size"])
        super().__init__(
            image_size=image_size,
            tile_size=tile_size,
            vis_mode=vis_mode,
            save_dir=save_dir,
            filename=filename,
            interpolation=interpolation,
        )
        # setup compression parameters
        self.compress_accumulator_array = compress_accumulator_array
        self.gcd_size_factor = math.gcd(tile_size, step_size)
        self.accumulator_tile_size = tile_size
        self.overlap_counter_tile_size = tile_size // self.gcd_size_factor

        # set accumulator tile size to compressed size if enabled
        if self.compress_accumulator_array:
            self.accumulator_tile_size = self.overlap_counter_tile_size
        self.level_coord_multiplier = 2 ** int(metadata["sample_level"])

        # Calculate sizes for accumulator and overlap counter
        self.w, self.h, self.c = self.image_size
        compressed_w = self.w // self.gcd_size_factor
        compressed_h = self.h // self.gcd_size_factor

        # set accumulator size to compressed size if enabled
        if self.compress_accumulator_array:
            accum_h, accum_w = compressed_h, compressed_w
        else:
            accum_h, accum_w = self.h, self.w

        self.heatmap_accumulator = np.zeros(
            shape=(accum_h, accum_w, self.c),  # row-first format (H, W, C)
            dtype=np.float32,
        )
        # overlap counter is always compressed
        self.patch_overlap_counter = np.zeros(
            shape=(compressed_h, compressed_w, 1),  # row-first format (H, W, C)
            dtype=np.uint8,
        )

    def update(self, data: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Update the heatmap and patch overlap counter with new data and metadata.

        Args:
            data: A numpy array of attention weights.
            metadata: A list of dictionaries containing metadata about the tiles.
        """
        # Debug metadata and attention weights structure
        logger.debug(f"Metadata: {metadata}")
        logger.debug(f"Data shape: {data.shape}")

        # Extract coordinates and other relevant information from metadata
        xs_accum = []
        ys_accum = []
        xs_count = []
        ys_count = []

        for md in metadata:
            coord_x = md.get("coord_x", 0)
            coord_y = md.get("coord_y", 0)

            # Validate coordinates
            if torch.is_tensor(coord_x):
                coord_x = coord_x.item()
            if torch.is_tensor(coord_y):
                coord_y = coord_y.item()

            xs_accum.append(coord_x)
            ys_accum.append(coord_y)

            xs_count.append(coord_x // self.overlap_counter_tile_size)
            ys_count.append(coord_y // self.overlap_counter_tile_size)

        # Log the extracted coordinates
        logger.debug(f"Transformed coordinates xs_accum: {xs_accum}, ys_accum: {ys_accum}")
        logger.debug(f"Compressed coordinates xs_count: {xs_count}, ys_count: {ys_count}")

        # Paste tiles onto heatmap accumulator and count overlaps
        for xa, ya, xc, yc, tile in zip(xs_accum, ys_accum, xs_count, ys_count, data):
            # Validate tile type and shape
            if not isinstance(tile, np.ndarray):
                logger.error(f"Tile at ({ya}, {xa}) is not a numpy array. Type: {type(tile)}")
                continue

            if len(tile.shape) != 3:
                logger.error(f"Tile at ({ya}, {xa}) has an invalid shape: {tile.shape}")
                continue

            # Determine the max possible shape for the heatmap slice
            mm_h, mm_w, mm_c = self.heatmap_accumulator[
                ya : ya + self.accumulator_tile_size,
                xa : xa + self.accumulator_tile_size,
                :
            ].shape

            # Log the current operation
            logger.debug(f"Adding tile to heatmap at position ({ya}, {xa}) with size ({mm_h}, {mm_w}, {mm_c}).")

            # Safeguard against indexing issues
            try:
                self.heatmap_accumulator[
                    ya : ya + self.accumulator_tile_size,
                    xa : xa + self.accumulator_tile_size,
                    :
                ] += tile[:mm_h, :mm_w, :mm_c]
            except IndexError as e:
                logger.error(f"IndexError while processing tile at ({ya}, {xa}): {e}")
                continue

            # Update overlap counter for averaging
            try:
                self.patch_overlap_counter[
                    yc : yc + self.overlap_counter_tile_size,
                    xc : xc + self.overlap_counter_tile_size,
                    :
                ] += 1
            except IndexError as e:
                logger.error(f"IndexError while updating overlap counter at ({yc}, {xc}): {e}")
                continue



    def save(self) -> str:
        logger.info("Starting heatmap save process.")
        
        # Convert the accumulator and overlap counter to pyVips images
        vips_im = pyvips.Image.new_from_array(self.heatmap_accumulator)
        count_im = pyvips.Image.new_from_array(self.patch_overlap_counter)
        
        logger.debug("Created pyVips images for accumulator and overlap counter.")

        # Normalize by overlap counter, with zero-safe division
        vips_im = vips_im / (count_im + 1e-5)  # Add epsilon to prevent division by zero
        logger.info(f"Normalized heatmap with overlap counter. Range after normalization: min {vips_im.min()}, max {vips_im.max()}.")

        # Finalize image and save
        vips_im = self._finalize_image(vips_im)
        save_path = self._save_xopat_compatible(vips_im)
        logger.info(f"Saved heatmap to {save_path}.")
        
        return save_path
