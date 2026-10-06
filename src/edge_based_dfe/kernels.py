"""First-order derivative of Gaussian kernels (port of g1x.m / g1y.m)."""

import numpy as np


def g1x(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    """Horizontal derivative of a 2-D Gaussian with standard deviation ``s1``."""
    s1sq = s1**2
    return -(x / (2 * np.pi * s1sq**2)) * np.exp(-(x**2 + y**2) / (2 * s1sq))


def g1y(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    """Vertical derivative of a 2-D Gaussian with standard deviation ``s1``."""
    s1sq = s1**2
    return -(y / (2 * np.pi * s1sq**2)) * np.exp(-(x**2 + y**2) / (2 * s1sq))
