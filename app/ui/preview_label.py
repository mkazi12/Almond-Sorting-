"""Responsive QLabel preview for a full-resolution rendered inspection image."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class ResponsiveImageLabel(QLabel):
    """Display one source image with aspect-ratio-preserving resize behavior."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._source_image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)

    @property
    def source_image(self) -> QImage | None:
        """Return the unscaled image currently used to build the preview."""
        return self._source_image

    def set_source_image(self, image: QImage | None) -> None:
        """Store a full-resolution image and refresh the displayed pixmap."""
        if image is None or image.isNull():
            self._source_image = None
            self.setPixmap(QPixmap())
            return
        self._source_image = image.copy()
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_image is None or self._source_image.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        self.setPixmap(
            QPixmap.fromImage(self._source_image).scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )