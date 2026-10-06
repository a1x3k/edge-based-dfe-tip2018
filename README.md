# TIP2018-Edge-Based-Defocus-Blur-Estimation-With-Adaptive-Scale-Selection
A. Karaali, CR. Jung, "Edge-Based Defocus Blur Estimation with Adaptive Scale Selection", 
IEEE Transactions on Image Processing (TIP 2018), 2018

Any papers using this code should cite the paper accordingly.

## Python implementation

A Python port of the original MATLAB code, built on NumPy, SciPy and scikit-image (no OpenCV).
The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                          # create the environment
uv run --extra demo examples/demo.py --image 05  # run a demo (01, 05 or 22)
uv run pytest                                    # run the tests
```

```python
from edge_based_dfe import estimate_blur, load_image

image = load_image("data/image_05.png")  # RGB in [0, 1]
blur_map, sparse_blur_map, edge_map = estimate_blur(image)
```

| Module | MATLAB source |
|---|---|
| `edge_based_dfe.blur_estimation` | `blur_estimate_our.m` |
| `edge_based_dfe.interpolation` | `EdgeAwareInterpolation.m` (+ domain transform `RF.m`) |
| `edge_based_dfe.kernels` | `g1x.m`, `g1y.m` |
| `edge_based_dfe.canny` | MATLAB `edge(I, 'Canny', [], sigma)` |
| `examples/demo.py` | `demo_one_image_{1,2,3}.m` |

The sample images and ground-truth blur maps in `data/` are from
"Non-parametric blur map regression for depth of field extension".
