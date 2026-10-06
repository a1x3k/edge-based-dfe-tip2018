from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.io import loadmat

from edge_based_dfe import canny, disk_to_gaussian, estimate_blur, load_image
from edge_based_dfe.blur_estimation import _sparse_blur, _edge_consistency_filter
from edge_based_dfe.kernels import g1x, g1y

DATA = Path(__file__).resolve().parent.parent / "data"


def step_image(blur: float, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size))
    img[:, size // 2 :] = 1.0
    return ndimage.gaussian_filter(img, blur) if blur > 0 else img


def test_canny_finds_vertical_step():
    edges = canny(step_image(1.0), 1.0)
    cols = np.unique(np.nonzero(edges)[1])
    assert set(cols) <= {31, 32}
    assert edges[5:-5].any(axis=1).all()


def test_sparse_blur_recovers_known_sigma():
    true_sigma = 1.5
    img = step_image(true_sigma)
    rows = np.arange(20, 44)
    cols = np.full_like(rows, 32)
    est = _sparse_blur(img, rows, cols, np.full(len(rows), 2.0))
    np.testing.assert_allclose(est, true_sigma, atol=0.15)


def test_sparse_blur_matches_matlab_loop():
    rng = np.random.default_rng(0)
    img = ndimage.gaussian_filter(rng.random((40, 40)), 1.5)
    rows = rng.integers(0, 40, 30)
    cols = rng.integers(0, 40, 30)
    std2 = rng.choice([1.25, 1.5, 2.0, 2.5], 30)

    pad = 11
    padded = np.pad(img, pad, mode="symmetric")
    expected = np.zeros(30)
    for k, (r, c, s2) in enumerate(zip(rows, cols, std2)):
        size = 2 * int(np.ceil(2 * s2)) + 1
        x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))
        win = padded[r + pad - size : r + pad + size + 1, c + pad - size : c + pad + size + 1]
        m1 = np.hypot((win * g1x(x, y, 1)).sum(), (win * g1y(x, y, 1)).sum())
        m2 = np.hypot((win * g1x(x, y, s2)).sum(), (win * g1y(x, y, s2)).sum())
        rr = m1 / m2
        v = (rr**2 - s2**2) / (1 - rr**2)
        expected[k] = min(np.sqrt(v), 5) if v >= 0 else 0

    np.testing.assert_allclose(_sparse_blur(img, rows, cols, std2), expected)


def test_edge_consistency_keeps_constant_values():
    edge_map = np.zeros((10, 10), dtype=bool)
    edge_map[5, 2:8] = True
    count = np.full((10, 10), 9)
    out = _edge_consistency_filter(edge_map, count, np.full(6, 2.0))
    np.testing.assert_allclose(out[edge_map], 2.0)
    assert (out[~edge_map] == 0).all()


def test_estimate_blur_on_sample_image():
    image = load_image(DATA / "image_05.png")[:, :-20]
    result = estimate_blur(image)
    assert result.blur_map.shape == image.shape[:2]
    assert np.isfinite(result.blur_map).all()
    assert result.edge_map.sum() > 100

    gt = loadmat(DATA / "map_05.mat")["blurMap"][:, :-20]
    conversion = loadmat(DATA / "convertion.mat")["convertion"]
    gt_gauss = disk_to_gaussian(gt, conversion)
    assert np.corrcoef(result.blur_map.ravel(), gt_gauss.ravel())[0, 1] > 0.5
