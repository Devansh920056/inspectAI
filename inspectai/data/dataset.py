"""PyTorch Dataset for MVTec AD."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from .layout import DatasetError, Sample, find_missing_masks, index_category
from .loading import CorruptImageError, load_image, load_mask, read_image_info
from .transforms import DEFAULT_IMAGE_SIZE, build_image_transform, build_mask_transform


class MVTecDataset(Dataset):
    """One category, one split of MVTec AD.

    Each item is a dict, with image-level label and pixel-level mask kept apart::

        image        float32 (3, H, W)  preprocessed image
        label        int64 scalar       image-level: 0 normal, 1 defective
        mask         float32 (1, H, W)  pixel-level ground truth in {0, 1} (all zeros for normal)
        defect_type  str                "good" or the defect name
        path         str                source image path

    ``split="train"`` reads only ``<category>/train/good``. The test folder is
    never opened for a train dataset, so test data cannot leak into training.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
        image_transform: Callable | None = None,
        mask_transform: Callable | None = None,
        verify_images: bool = False,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.samples: list[Sample] = index_category(self.root, category, split)

        missing = find_missing_masks(self.samples)
        if missing:
            raise DatasetError(f"{len(missing)} ground-truth mask(s) missing, e.g. {missing[0]}")
        if verify_images:
            bad = []
            for s in self.samples:
                try:
                    read_image_info(s.image_path, deep=True)
                except CorruptImageError as exc:
                    bad.append(str(exc))
            if bad:
                raise DatasetError(f"{len(bad)} corrupted image(s), e.g. {bad[0]}")

        self.image_transform = image_transform or build_image_transform(image_size)
        self.mask_transform = mask_transform or build_mask_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        s = self.samples[index]
        pil_image = load_image(s.image_path, "RGB")
        image = self.image_transform(pil_image)
        if s.mask_path is None:
            mask = torch.zeros(1, *image.shape[-2:], dtype=torch.float32)
        else:
            pil_mask = load_mask(s.mask_path)
            if pil_mask.size != pil_image.size:
                raise DatasetError(f"mask/image size mismatch: {s.mask_path} vs {s.image_path}")
            mask = self.mask_transform(pil_mask)
        return {
            "image": image,
            "label": torch.tensor(s.label, dtype=torch.int64),
            "mask": mask,
            "defect_type": s.defect_type,
            "path": str(s.image_path),
        }

    @property
    def labels(self) -> list[int]:
        return [s.label for s in self.samples]

    @property
    def num_normal(self) -> int:
        return sum(1 for s in self.samples if s.label == 0)

    @property
    def num_defective(self) -> int:
        return len(self.samples) - self.num_normal

    def __repr__(self) -> str:
        return (f"MVTecDataset(category={self.category!r}, split={self.split!r}, "
                f"n={len(self)}, normal={self.num_normal}, defective={self.num_defective})")
