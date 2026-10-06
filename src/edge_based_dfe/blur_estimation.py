"""Defocus blur estimation (port of blur_estimate_our.m)."""

from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .canny import canny
from .interpolation import edge_aware_interpolation
from .kernels import g1x, g1y

SIGMAS = np.arange(1, 5.01, 0.5)  # Canny scales 1:0.5:5
STD1 = 1.0
MAX_BLUR = 5.0
GAMMA = 10.0


class BlurEstimate(NamedTuple):
    blur_map: np.ndarray
    """Dense defocus blur map."""
    sparse_blur_map: np.ndarray
    """Edge-consistency filtered blur values on the reliable edges."""
    edge_map: np.ndarray
    """Boolean map of reliable edges."""


def rgb2ycbcr_luma(image: np.ndarray) -> np.ndarray:
    """Y channel of MATLAB's ``rgb2ycbcr`` for a double RGB image in [0, 1]."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return (16 + 65.481 * r + 128.553 * g + 24.966 * b) / 255


def _multiscale_edges(gray: np.ndarray) -> np.ndarray:
    return np.stack([canny(gray, s) for s in SIGMAS], axis=2)


def _select_scales(edges: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Re-blur std for each edge point: half the scale at which the edge first vanishes."""
    ep = edges[rows, cols, :].astype(int)
    drops = np.diff(ep, axis=1) == -1
    chosen = np.where(drops.any(axis=1), drops.argmax(axis=1), len(SIGMAS) - 1)
    return 0.5 * SIGMAS[chosen]


def _sparse_blur(
    gray: np.ndarray, rows: np.ndarray, cols: np.ndarray, std2: np.ndarray
) -> np.ndarray:
    """Blur amount at each edge point from the gradient ratio of two re-blurred versions."""
    pad = 2 * int(np.ceil(2 * MAX_BLUR)) + 1
    padded = np.pad(gray, pad, mode="symmetric")

    blur = np.zeros(len(rows))
    for s2 in np.unique(std2):
        sel = std2 == s2
        size = 2 * int(np.ceil(2 * s2)) + 1
        offs = np.arange(-size, size + 1)
        x, y = np.meshgrid(offs, offs)
        r = rows[sel, None, None] + pad + offs[None, :, None]
        c = cols[sel, None, None] + pad + offs[None, None, :]
        windows = padded[r, c]

        def magnitude(s):
            gx = np.einsum("nij,ij->n", windows, g1x(x, y, s))
            gy = np.einsum("nij,ij->n", windows, g1y(x, y, s))
            return np.sqrt(gx**2 + gy**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = magnitude(STD1) / magnitude(s2)
            val = (ratio**2 * STD1**2 - s2**2) / (1 - ratio**2)
        # MATLAB yields complex/NaN values here, which are reset to 0
        blur[sel] = np.where(val >= 0, np.sqrt(np.where(val >= 0, val, 0)), 0)

    return np.minimum(blur, MAX_BLUR)


def _edge_consistency_filter(
    edge_map: np.ndarray, edge_count: np.ndarray, blur: np.ndarray
) -> np.ndarray:
    """Smooth blur values along connected edges.

    Solves ``(diag(1 + et) + 2*gamma*L) x = (1 + et) * blur`` where ``L`` is the
    8-neighbour graph Laplacian of the edge pixels and ``et`` the number of
    scales each pixel was detected at. Connected edges are independent blocks,
    so this equals solving each edge separately as the MATLAB code does.
    """
    h, w = edge_map.shape
    rows, cols = np.nonzero(edge_map)
    n = len(rows)
    index = -np.ones((h, w), dtype=int)
    index[rows, cols] = np.arange(n)

    padded = np.pad(index, 1, constant_values=-1)
    src, dst = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = padded[rows + 1 + dr, cols + 1 + dc]
            ok = nb >= 0
            src.append(np.flatnonzero(ok))
            dst.append(nb[ok])
    src = np.concatenate(src)
    dst = np.concatenate(dst)

    degree = np.bincount(src, minlength=n)
    et = edge_count[rows, cols]
    a = sparse.coo_matrix((np.full(len(src), -2 * GAMMA), (src, dst)), shape=(n, n))
    a = (a + sparse.diags(degree * 2 * GAMMA + 1 + et)).tocsc()
    b = blur * (1 + et)

    out = np.zeros((h, w))
    if n:
        out[rows, cols] = spsolve(a, b)
    return out


def estimate_blur(image: np.ndarray) -> BlurEstimate:
    """Estimate the defocus blur map of an image.

    Args:
        image: RGB image (H, W, 3) or grayscale image (H, W) as floats in [0, 1].

    Returns:
        ``BlurEstimate(blur_map, sparse_blur_map, edge_map)``.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim == 3 and image.shape[2] == 3:
        gray = rgb2ycbcr_luma(image) ** 2.4
    else:
        gray = image

    edges = _multiscale_edges(gray)
    edge_map = edges[:, :, :4].sum(axis=2) == 4
    edge_count = edges.sum(axis=2)

    rows, cols = np.nonzero(edge_map)
    std2 = _select_scales(edges, rows, cols)
    blur = _sparse_blur(gray, rows, cols, std2)

    sparse_filtered = _edge_consistency_filter(edge_map, edge_count, blur)
    dense = edge_aware_interpolation(image, sparse_filtered, edge_map)
    return BlurEstimate(dense, sparse_filtered, edge_map)
