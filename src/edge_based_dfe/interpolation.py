"""Edge-aware propagation of the sparse blur map (port of EdgeAwareInterpolation.m).

Uses the recursive-filter variant of the domain transform from
E. Gastal, M. Oliveira, "Domain Transform for Edge-Aware Image and Video
Processing", ACM TOG (SIGGRAPH 2011).
"""

import numpy as np


def _recursive_filter_horizontal(img: np.ndarray, d: np.ndarray, sigma: float) -> np.ndarray:
    a = np.exp(-np.sqrt(2) / sigma)
    f = img.copy()
    v = a**d
    if f.ndim == 3:
        v = v[:, :, None]
    w = f.shape[1]
    for i in range(1, w):  # left -> right
        f[:, i] += v[:, i] * (f[:, i - 1] - f[:, i])
    for i in range(w - 2, -1, -1):  # right -> left
        f[:, i] += v[:, i + 1] * (f[:, i + 1] - f[:, i])
    return f


def _transpose(img: np.ndarray) -> np.ndarray:
    return np.swapaxes(img, 0, 1)


def recursive_filter(
    img: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    num_iterations: int = 3,
    joint_image: np.ndarray | None = None,
) -> np.ndarray:
    """Domain transform recursive filter (``RF.m``), optionally guided by ``joint_image``."""
    img = np.asarray(img, dtype=float)
    joint = img if joint_image is None else np.asarray(joint_image, dtype=float)
    if joint.shape[:2] != img.shape[:2]:
        raise ValueError("Input and joint images must have equal width and height.")
    if joint.ndim == 2:
        joint = joint[:, :, None]

    h, w = joint.shape[:2]
    didx = np.zeros((h, w))
    didy = np.zeros((h, w))
    didx[:, 1:] = np.abs(np.diff(joint, axis=1)).sum(axis=2)
    didy[1:, :] = np.abs(np.diff(joint, axis=0)).sum(axis=2)

    dhdx = 1 + sigma_s / sigma_r * didx
    dvdy = (1 + sigma_s / sigma_r * didy).T

    n = num_iterations
    f = img.copy()
    for i in range(n):
        sigma_i = sigma_s * np.sqrt(3) * 2 ** (n - (i + 1)) / np.sqrt(4**n - 1)
        f = _recursive_filter_horizontal(f, dhdx, sigma_i)
        f = _transpose(f)
        f = _recursive_filter_horizontal(np.ascontiguousarray(f), dvdy, sigma_i)
        f = np.ascontiguousarray(_transpose(f))
    return f


def edge_aware_interpolation(
    image: np.ndarray, sparse_map: np.ndarray, edge_map: np.ndarray
) -> np.ndarray:
    """Propagate a sparse defocus blur map to every pixel.

    Args:
        image: Reference RGB image in [0, 1].
        sparse_map: Sparse defocus blur map (non-zero on edges).
        edge_map: Boolean map of reliable edges.

    Returns:
        Dense defocus blur map.
    """
    h, w = edge_map.shape
    mask = edge_map.astype(float)

    sigma_s = min(h, w) / 8
    sigma_r = 3.75
    niter = 5

    ref = recursive_filter(image, 7, 0.5, niter)
    f_ic = recursive_filter(sparse_map * mask, sigma_s, sigma_r, niter, ref)
    mask_ic = recursive_filter(mask, sigma_s, sigma_r, niter, ref)
    return f_ic / mask_ic
