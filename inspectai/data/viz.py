"""Matplotlib helpers for looking at the data (requires matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layout import index_category
from .loading import load_image, load_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a red mask onto an RGB uint8 image. ``mask`` is (H, W) bool."""
    out = image.astype(np.float32).copy()
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * red
    return out.clip(0, 255).astype(np.uint8)


def plot_category_samples(root: str | Path, category: str, n_normal: int = 4, per_defect: int = 1):
    """Figure: a row of normal train images, then one row per defect type
    (defective image | ground-truth mask | overlay). Uses the original images."""
    import matplotlib.pyplot as plt

    train = index_category(root, category, "train")
    test = index_category(root, category, "test")
    # evenly spaced picks => deterministic and spread over the folder
    idx = np.linspace(0, len(train) - 1, num=min(n_normal, len(train))).round().astype(int)
    normals = [train[i] for i in idx]
    defect_types = sorted({s.defect_type for s in test if s.label == 1})

    n_cols = max(len(normals), 3 * per_defect)
    n_rows = 1 + len(defect_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, s in enumerate(normals):
        axes[0, j].imshow(load_image(s.image_path))
        axes[0, j].set_title(f"train/good #{j}", fontsize=9)
    for i, d in enumerate(defect_types, start=1):
        picks = [s for s in test if s.defect_type == d][:per_defect]
        for k, s in enumerate(picks):
            img = np.asarray(load_image(s.image_path))
            msk = np.asarray(load_mask(s.mask_path)) > 0
            axes[i, 3 * k].imshow(img)
            axes[i, 3 * k].set_title(f"{d}: image", fontsize=9)
            axes[i, 3 * k + 1].imshow(msk, cmap="gray", vmin=0, vmax=1)
            axes[i, 3 * k + 1].set_title("ground-truth mask", fontsize=9)
            axes[i, 3 * k + 2].imshow(overlay_mask(img, msk))
            axes[i, 3 * k + 2].set_title("overlay", fontsize=9)
    fig.suptitle(f"MVTec AD / {category}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_batch(batch: dict, n: int = 8, mean=None, std=None):
    """Figure of a preprocessed batch: de-normalised image on top, mask below."""
    import matplotlib.pyplot as plt

    from .transforms import IMAGENET_MEAN, IMAGENET_STD, denormalize

    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    n = min(n, batch["image"].shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)
    for j in range(n):
        img = denormalize(batch["image"][j], mean, std).permute(1, 2, 0).numpy()
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{batch['defect_type'][j]}\nlabel={int(batch['label'][j])}", fontsize=8)
        axes[1, j].imshow(batch["mask"][j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    fig.tight_layout()
    return fig
