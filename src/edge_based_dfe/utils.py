"""Image loading and ground-truth helpers."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as float in [0, 1] (MATLAB ``im2double(imread(path))``), dropping alpha."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(float) / np.iinfo(arr.dtype).max
    return arr.astype(float)


def disk_to_gaussian(blur_map: np.ndarray, conversion: np.ndarray) -> np.ndarray:
    """Map disk-kernel blur radii to Gaussian std via nearest entry of a lookup table.

    Args:
        blur_map: Blur map expressed as disk radii.
        conversion: (N, 2) table with Gaussian std in column 0 and disk radius in column 1.
    """
    nearest = np.abs(conversion[:, 1][None, :] - blur_map.reshape(-1, 1)).argmin(axis=1)
    return conversion[nearest, 0].reshape(blur_map.shape)
