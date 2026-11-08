"""Draw validated model results on a copy of the original image."""

from collections.abc import Mapping

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from app.inference import InferenceError, InferenceResult, validate_result


def render_detections(image: QImage, result: InferenceResult, classes: Mapping[int, str]) -> QImage:
    """Return a full-resolution annotated copy; empty results leave pixels alone."""
    if image.isNull():
        raise InferenceError("Cannot render detections on an empty image.")
    validate_result(
        result,
        image_width=image.width(),
        image_height=image.height(),
        confidence_floor=result.confidence_floor,
        classes=classes,
    )
    rendered = image.copy()
    if not result.detections:
        return rendered
    # Indexed and monochrome images decode successfully but cannot be painted.
    rendered = rendered.convertToFormat(QImage.Format.Format_RGB32)
    painter = QPainter(rendered)
    if not painter.isActive():
        raise InferenceError("Could not draw the detection overlay.")
    try:
        font = painter.font()
        font.setPixelSize(12)
        painter.setFont(font)
        for detection in result.detections:
            left, top, right, bottom = detection.bbox_xyxy
            color = QColor.fromHsv((detection.category_id * 67 + 90) % 360, 210, 220)
            painter.setPen(QPen(color, 2))
            painter.drawRect(QRectF(left, top, right - left, bottom - top))
            text = f"{classes[detection.category_id]} {detection.confidence:.0%}"
            metrics = painter.fontMetrics()
            label_width = min(metrics.horizontalAdvance(text) + 8, image.width())
            label_height = min(metrics.height() + 4, image.height())
            label_left = min(max(left, 0), image.width() - label_width)
            label_top = top - label_height if top >= label_height else top
            label_top = min(max(label_top, 0), image.height() - label_height)
            label = QRectF(label_left, label_top, label_width, label_height)
            painter.fillRect(label, color)
            painter.setPen(QColor("black"))
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)
    finally:
        painter.end()
    return rendered
