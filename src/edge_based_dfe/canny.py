"""Canny edge detector replicating MATLAB's ``edge(I, 'Canny', [], sigma)``.

The method counts how many scales an edge survives, so the detector must
behave like MATLAB's: same derivative-of-Gaussian kernel, automatic
thresholds, interpolated non-maximum suppression and final thinning.
``skimage.feature.canny`` differs in all of these.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import thin

PERCENT_OF_PIXELS_NOT_EDGES = 0.7
THRESHOLD_RATIO = 0.4


def _smooth_gradient(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    extent = int(np.ceil(4 * sigma))
    x = np.arange(-extent, extent + 1, dtype=float)
    gauss = np.exp(-(x**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    gauss /= gauss.sum()

    deriv = np.gradient(gauss)
    neg, pos = deriv < 0, deriv > 0
    deriv[neg] /= abs(deriv[neg].sum())
    deriv[pos] /= abs(deriv[pos].sum())

    # imfilter(..., 'conv', 'replicate')
    gx = ndimage.convolve1d(image, gauss, axis=0, mode="nearest")
    gx = ndimage.convolve1d(gx, deriv, axis=1, mode="nearest")
    gy = ndimage.convolve1d(image, gauss, axis=1, mode="nearest")
    gy = ndimage.convolve1d(gy, deriv, axis=0, mode="nearest")
    return gx, gy


def _select_thresholds(mag: np.ndarray) -> tuple[float, float]:
    # imhist(mag, 64) for a double image in [0, 1]
    counts = np.bincount(np.round(mag.ravel() * 63).astype(int), minlength=64)
    first = np.flatnonzero(np.cumsum(counts) > PERCENT_OF_PIXELS_NOT_EDGES * mag.size)[0]
    high = (first + 1) / 64
    return THRESHOLD_RATIO * high, high


def _local_maxima(direction: int, ix: np.ndarray, iy: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Boolean mask of non-maximum-suppressed pixels for one of 4 gradient sectors."""
    if direction == 1:
        sel = ((iy <= 0) & (ix > -iy)) | ((iy >= 0) & (ix < -iy))
    elif direction == 2:
        sel = ((ix > 0) & (-iy >= ix)) | ((ix < 0) & (-iy <= ix))
    elif direction == 3:
        sel = ((ix <= 0) & (ix > iy)) | ((ix >= 0) & (ix < iy))
    else:
        sel = ((iy < 0) & (ix <= iy)) | ((iy > 0) & (ix >= iy))

    # exclude exterior pixels
    sel[0, :] = sel[-1, :] = sel[:, 0] = sel[:, -1] = False
    r, c = np.nonzero(sel)
    ixv, iyv, m = ix[r, c], iy[r, c], mag[r, c]

    if direction == 1:
        d = np.abs(iyv / ixv)
        m1 = mag[r, c + 1] * (1 - d) + mag[r - 1, c + 1] * d
        m2 = mag[r, c - 1] * (1 - d) + mag[r + 1, c - 1] * d
    elif direction == 2:
        d = np.abs(ixv / iyv)
        m1 = mag[r - 1, c] * (1 - d) + mag[r - 1, c + 1] * d
        m2 = mag[r + 1, c] * (1 - d) + mag[r + 1, c - 1] * d
    elif direction == 3:
        d = np.abs(ixv / iyv)
        m1 = mag[r - 1, c] * (1 - d) + mag[r - 1, c - 1] * d
        m2 = mag[r + 1, c] * (1 - d) + mag[r + 1, c + 1] * d
    else:
        d = np.abs(iyv / ixv)
        m1 = mag[r, c - 1] * (1 - d) + mag[r - 1, c - 1] * d
        m2 = mag[r, c + 1] * (1 - d) + mag[r + 1, c + 1] * d

    keep = (m >= m1) & (m >= m2)
    out = np.zeros(mag.shape, dtype=bool)
    out[r[keep], c[keep]] = True
    return out


def canny(image: np.ndarray, sigma: float) -> np.ndarray:
    """Binary edge map of a 2-D float image, as MATLAB ``edge(image, 'Canny', [], sigma)``."""
    image = np.asarray(image, dtype=float)
    gx, gy = _smooth_gradient(image, sigma)
    mag = np.hypot(gx, gy)
    magmax = mag.max()
    if magmax > 0:
        mag = mag / magmax

    low, high = _select_thresholds(mag)

    weak = np.zeros(mag.shape, dtype=bool)
    strong = np.zeros(mag.shape, dtype=bool)
    for direction in range(1, 5):
        local_max = _local_maxima(direction, gx, gy, mag)
        w = local_max & (mag > low)
        weak |= w
        strong |= w & (mag > high)

    if not strong.any():
        return np.zeros(mag.shape, dtype=bool)

    # bwselect(E, strong, 8): keep 8-connected weak components touching a strong pixel
    labels, _ = ndimage.label(weak, structure=np.ones((3, 3), dtype=bool))
    keep = np.unique(labels[strong])
    edges = np.isin(labels, keep[keep > 0])
    # bwmorph(E, 'thin', 1)
    return thin(edges, max_num_iter=1)
