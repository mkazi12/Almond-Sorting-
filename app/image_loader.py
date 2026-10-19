"""Validate and decode local JPG/PNG files using the existing Qt dependency."""

from pathlib import Path

from PySide6.QtGui import QImage, QImageReader


class ImageLoadError(ValueError):
    """An expected loading failure with a message suitable for the user."""


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


def load_image(path: str | Path) -> QImage:
    """Return a full-resolution image, or raise ImageLoadError.

    The suffix must be .jpg, .jpeg, or .png (case-insensitive), and the
    actual contents must be JPEG or PNG. The two need not match exactly:
    for example, genuine PNG contents in a .jpg file are accepted.

    This function does not open dialogs, scale images, or run inference.
    Callers handle cancellation before calling and display ImageLoadError
    messages for expected failures. A detected format alone is not success;
    Qt must also decode the image's pixel data.
    """
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ImageLoadError(
            "Unsupported file extension. Choose a JPG or PNG image "
            "with a .jpg, .jpeg, or .png filename."
        )

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

    image = reader.read()
    # A recognized header can still belong to a truncated or unreadable image.
    if image.isNull():
        raise _read_error(reader)
    return image
