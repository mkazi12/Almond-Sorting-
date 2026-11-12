"""Decode dataset images without importing the desktop GUI or changing geometry."""

from pathlib import Path

from PIL import Image


class DatasetImageError(ValueError):
    """A dataset image is missing, unsupported or cannot be decoded."""


def read_image_dimensions(path: str | Path) -> tuple[int, int]:
    """Validate a full JPG/PNG decode and return its stored pixel dimensions.

    EXIF orientation is deliberately not applied: COCO boxes refer to the
    stored image coordinates. Source files and pixel values are never changed.
    """
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise DatasetImageError("Only JPG and PNG images are supported.")
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise DatasetImageError("The file contents are not JPEG or PNG.")
            image.load()
            return image.size
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise DatasetImageError(f"Cannot decode {path.name}: {error}") from error
