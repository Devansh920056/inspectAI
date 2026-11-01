"""Tiny synthetic dataset in the MVTec AD folder layout, for tests only.

Images are random-noise textures; defective images contain a bright square and
a matching mask, so expected counts and mask areas are known exactly. This is
NOT MVTec data and must never be used for reporting dataset statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SIZES = {"bottle": 64, "hazelnut": 80, "carpet": 96}
BOX = 16  # defect square side in pixels


def _noise(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(0, 120, size=(size, size, 3), dtype=np.uint8)


def make_synthetic_mvtec(
    root: Path,
    categories=("bottle", "hazelnut", "carpet"),
    sizes: dict[str, int] | None = None,
    n_train: int = 6,
    n_test_good: int = 2,
    defects: dict[str, int] | None = None,
    seed: int = 0,
) -> dict:
    """Create the dataset under ``root``; return expected counts per category."""
    sizes = sizes or DEFAULT_SIZES
    defects = defects or {"broken": 3, "stain": 2}
    rng = np.random.default_rng(seed)
    expected = {}
    for cat in categories:
        size = sizes[cat]
        for i in range(n_train):
            d = root / cat / "train" / "good"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_noise(rng, size)).save(d / f"{i:03d}.png")
        for i in range(n_test_good):
            d = root / cat / "test" / "good"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_noise(rng, size)).save(d / f"{i:03d}.png")
        for defect, n in defects.items():
            img_dir = root / cat / "test" / defect
            gt_dir = root / cat / "ground_truth" / defect
            img_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = _noise(rng, size)
                y = x = 8 + 4 * i
                arr[y:y + BOX, x:x + BOX] = 250
                mask = np.zeros((size, size), dtype=np.uint8)
                mask[y:y + BOX, x:x + BOX] = 255
                Image.fromarray(arr).save(img_dir / f"{i:03d}.png")
                Image.fromarray(mask).save(gt_dir / f"{i:03d}_mask.png")
        expected[cat] = {
            "size": size,
            "n_train": n_train,
            "n_test_good": n_test_good,
            "defects": dict(defects),
            "n_defective": sum(defects.values()),
            "box_fraction": BOX * BOX / (size * size),
        }
    return expected
