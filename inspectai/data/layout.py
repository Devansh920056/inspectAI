"""MVTec AD folder layout: sample indexing and validation (torch-free).

Expected layout::

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<image_stem>_mask.png

Image-level labels (0 normal / 1 defective) and pixel-level masks are separate
fields on :class:`Sample`; normal images have no mask file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .loading import CorruptImageError, read_image_info

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
FOCUS_CATEGORIES = ("bottle", "hazelnut", "carpet")
SPLITS = ("train", "test")
GOOD = "good"
GROUND_TRUTH_DIR = "ground_truth"
MASK_SUFFIX = "_mask"
MASK_EXTENSION = ".png"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

_DOWNLOAD_HINT = "See docs/dataset.md for how to obtain MVTec AD and where to place it."


class DatasetError(RuntimeError):
    """The dataset on disk does not match the expected MVTec AD layout."""


@dataclass(frozen=True)
class Sample:
    category: str
    split: str  # "train" | "test"
    image_path: Path
    label: int  # image-level: 0 = normal, 1 = defective
    defect_type: str  # "good" or the defect name
    mask_path: Path | None  # pixel-level ground truth; None for normal images

    @property
    def is_defective(self) -> bool:
        return self.label == 1


def _visible(path: Path) -> bool:
    # Skip .DS_Store and macOS "._foo.png" resource-fork files created by tar/zip.
    return not path.name.startswith(".")


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and _visible(p) and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def expected_mask_path(root: Path, category: str, defect_type: str, image_path: Path) -> Path:
    return root / category / GROUND_TRUTH_DIR / defect_type / f"{image_path.stem}{MASK_SUFFIX}{MASK_EXTENSION}"


def index_category(root: str | Path, category: str, split: str) -> list[Sample]:
    """List every sample of one category/split, in a deterministic (sorted) order.

    The train split is only ever read from ``<category>/train`` and must contain
    only ``good``; test data can therefore never enter training through here.
    """
    root = Path(root)
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    split_dir = root / category / split
    if not split_dir.is_dir():
        raise DatasetError(f"missing directory: {split_dir}. {_DOWNLOAD_HINT}")
    type_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and _visible(p))
    if not type_dirs:
        raise DatasetError(f"no sub-folders found in {split_dir}")
    if split == "train":
        extra = [d.name for d in type_dirs if d.name != GOOD]
        if extra:
            raise DatasetError(
                f"train split must contain only '{GOOD}' images, found {extra} in {split_dir}"
            )

    samples: list[Sample] = []
    for type_dir in type_dirs:
        label = 0 if type_dir.name == GOOD else 1
        for image_path in list_images(type_dir):
            mask = None if label == 0 else expected_mask_path(root, category, type_dir.name, image_path)
            samples.append(Sample(category, split, image_path, label, type_dir.name, mask))
    if not samples:
        raise DatasetError(f"no images found under {split_dir}")
    return samples


@dataclass
class ValidationReport:
    category: str
    n_images_checked: int = 0
    structural_error: str | None = None
    missing_masks: list[Path] = field(default_factory=list)
    orphan_masks: list[Path] = field(default_factory=list)  # mask with no matching test image
    corrupt_images: list[tuple[Path, str]] = field(default_factory=list)
    size_mismatches: list[tuple[Path, Path, tuple[int, int], tuple[int, int]]] = field(default_factory=list)

    @property
    def n_problems(self) -> int:
        return (
            (1 if self.structural_error else 0) + len(self.missing_masks) + len(self.orphan_masks)
            + len(self.corrupt_images) + len(self.size_mismatches)
        )

    @property
    def ok(self) -> bool:
        return self.n_problems == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_problems": self.n_problems,
            "n_images_checked": self.n_images_checked,
            "structural_error": self.structural_error,
            "missing_masks": [str(p) for p in self.missing_masks],
            "orphan_masks": [str(p) for p in self.orphan_masks],
            "corrupt_images": [{"path": str(p), "error": e} for p, e in self.corrupt_images],
            "size_mismatches": [
                {"image": str(i), "mask": str(m), "image_size": list(a), "mask_size": list(b)}
                for i, m, a, b in self.size_mismatches
            ],
        }


def find_missing_masks(samples: list[Sample]) -> list[Path]:
    return [s.mask_path for s in samples if s.mask_path is not None and not s.mask_path.is_file()]


def validate_category(root: str | Path, category: str, check_images: bool = True) -> ValidationReport:
    """Check one category for missing files, corrupted images and mask problems.

    Never raises for dataset problems; they are collected in the report.
    ``check_images=True`` fully decodes every image and mask (slower, thorough).
    """
    root = Path(root)
    report = ValidationReport(category)
    try:
        samples = index_category(root, category, "train") + index_category(root, category, "test")
    except DatasetError as exc:
        report.structural_error = str(exc)
        return report

    report.missing_masks = find_missing_masks(samples)

    # masks on disk that no test image points to
    expected = {s.mask_path for s in samples if s.mask_path is not None}
    gt_dir = root / category / GROUND_TRUTH_DIR
    if gt_dir.is_dir():
        for defect_dir in sorted(p for p in gt_dir.iterdir() if p.is_dir() and _visible(p)):
            for mask in list_images(defect_dir):
                if mask not in expected:
                    report.orphan_masks.append(mask)

    if check_images:
        for s in samples:
            report.n_images_checked += 1
            try:
                img_info = read_image_info(s.image_path, deep=True)
            except (CorruptImageError, FileNotFoundError) as exc:
                report.corrupt_images.append((s.image_path, str(exc)))
                continue
            if s.mask_path is not None and s.mask_path.is_file():
                try:
                    mask_info = read_image_info(s.mask_path, deep=True)
                except CorruptImageError as exc:
                    report.corrupt_images.append((s.mask_path, str(exc)))
                    continue
                if mask_info.size != img_info.size:
                    report.size_mismatches.append((s.image_path, s.mask_path, img_info.size, mask_info.size))
    return report
