"""Image / mask preprocessing (Pillow + torchvision).

Image pipeline:  PIL RGB -> Resize (bilinear, antialiased) -> ToImage + ToDtype(float32, scale) -> Normalize
Mask pipeline:   PIL L   -> Resize (nearest)               -> binary float {0, 1}

Original files are never modified; every step works on in-memory copies.
Transform classes are module-level (not lambdas) so DataLoader workers can
pickle them on macOS, where workers are started with ``spawn``.
"""

from __future__ import annotations

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 256

TRANSFORM_DOCS = {
    "image": [
        "Resize to (image_size, image_size), bilinear interpolation with antialiasing",
        "ToImage + ToDtype(float32, scale=True): PIL HxWxC uint8 [0,255] -> torch CxHxW float32 [0,1]",
        "Normalize: (x - mean) / std per channel (default: ImageNet statistics)",
    ],
    "mask": [
        "Resize to (image_size, image_size), nearest-neighbour (keeps labels binary)",
        "Binarize: any pixel > 0 becomes 1.0, else 0.0",
        "Output: 1xHxW float32 with values in {0, 1}",
    ],
}


class Binarize:
    """PIL mask -> float32 tensor of shape (1, H, W) with values in {0., 1.}."""

    def __call__(self, mask) -> torch.Tensor:
        t = T.functional.pil_to_tensor(mask)  # uint8, (1, H, W)
        return (t > 0).to(torch.float32)

    def __repr__(self) -> str:
        return "Binarize()"


def _size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    return (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)


def build_image_transform(
    image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    normalize: bool = True,
) -> T.Compose:
    steps = [
        T.Resize(_size(image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if normalize:
        steps.append(T.Normalize(mean=list(mean), std=list(std)))
    steps.append(T.ToPureTensor())  # plain torch.Tensor, not a tv_tensors.Image subclass
    return T.Compose(steps)


def build_mask_transform(image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE) -> T.Compose:
    return T.Compose([T.Resize(_size(image_size), interpolation=InterpolationMode.NEAREST), Binarize()])


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert Normalize for display. Accepts (C,H,W) or (B,C,H,W); returns values clipped to [0,1]."""
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor * s + m).clamp(0.0, 1.0)
