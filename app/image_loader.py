"""Validate and decode local JPG/PNG files using the existing Qt dependency."""

from pathlib import Path

from PySide6.QtGui import QImage, QImageReader

MAX_IMAGE_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


class ImageLoadError(ValueError):
    """An expected loading failure with a message suitable for the user."""


def _validate_positive_limit(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_image_limits(max_file_size_bytes: int, max_pixels: int) -> None:
    """Validate the configurable byte and decoded-pixel caps."""
    _validate_positive_limit(max_file_size_bytes, "Maximum image file size")
    _validate_positive_limit(max_pixels, "Maximum image pixel count")


def validate_image_extension(path: str | Path) -> None:
    """Reject filenames that do not use one of the supported image extensions."""
    if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ImageLoadError(
            "Unsupported file extension. Choose a JPG or PNG image "
            "with a .jpg, .jpeg, or .png filename."
        )


def validate_image_file_size(
    path: str | Path, max_file_size_bytes: int = MAX_IMAGE_FILE_SIZE_BYTES
) -> None:
    """Reject oversized or unreadable files before hashing or decoding them."""
    _validate_positive_limit(max_file_size_bytes, "Maximum image file size")
    try:
        file_size = Path(path).stat().st_size
    except OSError as error:
        raise ImageLoadError(
            "Could not access this image file. Choose it again and check its "
            "location and read permissions."
        ) from error
    if file_size > max_file_size_bytes:
        raise ImageLoadError(
            f"Image file is too large ({file_size} bytes; maximum is "
            f"{max_file_size_bytes} bytes). Choose a smaller JPG or PNG image."
        )


def _read_error(reader: QImageReader) -> ImageLoadError:
    """Turn a Qt read failure into recovery guidance without guessing its cause."""
    if reader.error() == QImageReader.ImageReaderError.FileNotFoundError:
        return ImageLoadError(
            "The selected file could not be found. Choose it again or select "
            "another JPG or PNG image."
        )
    if reader.error() == QImageReader.ImageReaderError.DeviceError:
        return ImageLoadError(
            "Could not access this image. Check its location and read "
            "permissions, then choose the file again."
        )
    return ImageLoadError(
        "Could not decode this image. Choose another JPG or PNG file, "
        "or re-save a readable copy and try again."
    )


def load_image(
    path: str | Path,
    *,
    max_file_size_bytes: int = MAX_IMAGE_FILE_SIZE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> QImage:
    """Return a full-resolution image, or raise ImageLoadError."""
    validate_image_limits(max_file_size_bytes, max_pixels)
    path = Path(path)
    validate_image_extension(path)
    validate_image_file_size(path, max_file_size_bytes)

    reader = QImageReader(str(path))
    # Inspect contents so renaming a BMP (for example) cannot bypass validation.
    reader.setDecideFormatFromContent(True)
    image_format = bytes(reader.format()).lower()
    if not image_format:
        raise _read_error(reader)
    if image_format not in {b"jpeg", b"jpg", b"png"}:
        raise ImageLoadError(
            "This file's contents are not JPEG or PNG. Choose a JPG or PNG "
            "image; renaming another format does not convert it."
        )

    dimensions = reader.size()
    if dimensions.isValid():
        pixel_count = dimensions.width() * dimensions.height()
        if pixel_count > max_pixels:
            raise ImageLoadError(
                f"Image dimensions are too large ({dimensions.width()} x "
                f"{dimensions.height()} pixels; maximum is {max_pixels} pixels). "
                "Choose a smaller JPG or PNG image."
            )

    image = reader.read()
    # A recognized header can still belong to a truncated or unreadable image.
    if image.isNull():
        raise _read_error(reader)
    return image
