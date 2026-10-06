"""Estimate the blur map of a sample image and compare it with the ground truth.

Port of demo_one_image_{1,2,3}.m. Ground-truth maps are from "Non-parametric blur
map regression for depth of field extension" and use a disk kernel, so they are
converted to Gaussian std before display.

    uv run --extra demo examples/demo.py --image 05
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from edge_based_dfe import disk_to_gaussian, estimate_blur, load_image

DATA = Path(__file__).resolve().parent.parent / "data"
NUM_CROP_COLUMNS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", choices=["01", "05", "22"], default="05")
    parser.add_argument("--save", type=Path, help="save the figure here instead of showing it")
    args = parser.parse_args()

    image = load_image(DATA / f"image_{args.image}.png")[:, :-NUM_CROP_COLUMNS]
    result = estimate_blur(image)

    gt = loadmat(DATA / f"map_{args.image}.mat")["blurMap"][:, :-NUM_CROP_COLUMNS]
    conversion = loadmat(DATA / "convertion.mat")["convertion"]
    gt_gauss = disk_to_gaussian(gt, conversion)
    mae = np.abs(result.blur_map - gt_gauss).mean()
    print(f"image_{args.image}: {result.edge_map.sum()} reliable edge pixels, MAE vs GT = {mae:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    panels = [(image, "Input"), (gt_gauss, "Ground truth"), (result.blur_map, "Estimated")]
    for ax, (data, title) in zip(axes, panels):
        if data.ndim == 2:
            ax.imshow(data, vmin=0.5, vmax=4)
        else:
            ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
