"""Image / mask reading with Pillow.

Nothing here ever writes to or modifies the files it reads: images are opened
read-only, fully decoded, and converted into a *new* in-memory image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class CorruptImageError(RuntimeError):
    """The file is missing a valid image payload (truncated, garbage, wrong format)."""


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str  # Pillow mode: "RGB", "L", ...
    format: str | None  # "PNG", "JPEG", ...

    @property
    def size(self) -> tuple[int, int]:
        """(width, height), Pillow convention."""
        return (self.width, self.height)


def read_image_info(path: str | Path, deep: bool = True) -> ImageInfo:
    """Return size/mode/format of an image.

    ``deep=True`` additionally verifies file structure and fully decodes the
    pixels, which is what detects truncated or corrupted files.
    ``deep=False`` only parses the header (fast).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as im:
            info = ImageInfo(im.width, im.height, im.mode, im.format)
            if deep:
                im.verify()
        if deep:
            with Image.open(path) as im:  # verify() invalidates the handle; reopen to decode
                im.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise CorruptImageError(f"{path}: {type(exc).__name__}: {exc}") from exc
    return info


def load_image(path: str | Path, mode: str = "RGB") -> Image.Image:
    """Load an image fully into memory and convert to ``mode`` (default RGB).

    Grayscale images become 3-channel RGB so every sample has the same channel
    count. The returned image is independent of the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert(mode)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise CorruptImageError(f"{path}: {type(exc).__name__}: {exc}") from exc


def load_mask(path: str | Path) -> Image.Image:
    """Load a ground-truth mask as a single-channel ("L") image, values unchanged.

    MVTec AD masks are 0 (normal pixel) / 255 (defective pixel). Binarisation
    happens in the transform so the raw values stay inspectable.
    """
    return load_image(path, mode="L")
