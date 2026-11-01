"""MVTec AD data access.

Torch-free modules (importable without PyTorch):
    loading  - read/validate images and masks with Pillow
    layout   - MVTec AD folder layout, sample indexing, validation
    stats    - dataset statistics and report generation

Torch/torchvision modules (import explicitly):
    transforms, dataset, dataloader
Plotting helpers live in ``viz`` (needs matplotlib).
"""

from .layout import (
    CATEGORIES,
    FOCUS_CATEGORIES,
    DatasetError,
    Sample,
    ValidationReport,
    index_category,
    validate_category,
)
from .loading import CorruptImageError, ImageInfo, load_image, load_mask, read_image_info

__all__ = [
    "CATEGORIES",
    "FOCUS_CATEGORIES",
    "CorruptImageError",
    "DatasetError",
    "ImageInfo",
    "Sample",
    "ValidationReport",
    "index_category",
    "load_image",
    "load_mask",
    "read_image_info",
    "validate_category",
]
