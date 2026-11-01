"""Draw ground-truth boxes and class names onto a copy of a QImage.

This is not the inference overlay. It does not load a model or invent
detections. ``bbox_xywh`` is ``[x, y, width, height]`` in the original
image's pixels and may be fractional. Those values stay fractional until
the box is painted.

Rasterization treats the box as the half-open rectangle
``[x, x + width) × [y, y + height)``. Pixel column ``i`` covers
``[i, i + 1)`` and is stroked when that interval intersects the box;
rows work the same way. The stroked span is columns
``floor(x) .. ceil(x + width) - 1`` and rows
``floor(y) .. ceil(y + height) - 1``. Width and height are not rounded
on their own, so a positive area smaller than one pixel still covers the
pixel it intersects. A box with no positive area is not drawn. Each
category id uses one stable color. An empty object list leaves the copy
without boxes.
"""

import math
from collections.abc import Mapping, Sequence

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from app.dataset.records import GroundTruthObject


class PreviewError(ValueError):
    """A ground-truth preview cannot be drawn."""


# Distinct, opaque colors. category_id selects one and keeps it.
_CATEGORY_COLORS = (
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (217, 119, 6),
    (147, 51, 234),
    (8, 145, 178),
    (190, 24, 93),
    (79, 70, 229),
)

_LABEL_FONT_PX = 12
_LABEL_PADDING = 2


def category_color(category_id: int) -> QColor:
    """Return the preview color for ``category_id``."""
    red, green, blue = _CATEGORY_COLORS[category_id % len(_CATEGORY_COLORS)]
    return QColor(red, green, blue)


def label_text_color(category_id: int) -> QColor:
    """Text color that stays readable on ``category_color(category_id)``."""
    color = category_color(category_id)
    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    if luminance > 150:
        return QColor(0, 0, 0)
    return QColor(255, 255, 255)


def render_ground_truth_preview(
    image: QImage,
    objects: Sequence[GroundTruthObject],
    classes: Mapping[int, str],
) -> QImage:
    """Return a new image with ground-truth boxes and class names drawn on it.

    ``classes`` maps a category id to its display name. The input image is
    not modified. No objects means no annotation overlay.
    """
    if image.isNull():
        raise PreviewError("Cannot draw a ground-truth preview on an empty image.")
    _require_class_names(objects, classes)

    preview = image.copy()
    if preview.format() != QImage.Format.Format_RGB32:
        preview = preview.convertToFormat(QImage.Format.Format_RGB32)
    if not objects:
        return preview

    painter = QPainter(preview)
    if not painter.isActive():
        raise PreviewError("Could not draw the ground-truth preview.")
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, False)
        font = painter.font()
        font.setPixelSize(_LABEL_FONT_PX)
        painter.setFont(font)
        for obj in objects:
            _draw_object(painter, preview.width(), preview.height(), obj, classes[obj.category_id])
    finally:
        painter.end()
    return preview


def _require_class_names(
    objects: Sequence[GroundTruthObject], classes: Mapping[int, str]
) -> None:
    missing = sorted(
        {
            obj.category_id
            for obj in objects
            if not _class_name(classes, obj.category_id)
        }
    )
    if not missing:
        return
    names = ", ".join(str(category_id) for category_id in missing)
    raise PreviewError(f"No class name is configured for category {names}.")


def _class_name(classes: Mapping[int, str], category_id: int) -> str:
    name = classes.get(category_id)
    if name is None:
        return ""
    return name.strip()


def _draw_object(
    painter: QPainter,
    image_width: int,
    image_height: int,
    obj: GroundTruthObject,
    name: str,
) -> None:
    box = _raster_box(obj.bbox_xywh)
    if box is None:
        return
    color = category_color(obj.category_id)
    painter.setPen(QPen(color, 1))
    _stroke_box(painter, *box)
    _draw_label(painter, image_width, image_height, box, name, obj.category_id, color)


def _raster_box(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
    """Return the inclusive pixel span covered by a fractional COCO box.

    ``bbox`` stays in continuous pixels. The result is
    ``(left, top, pixel_width, pixel_height)`` for the half-open rectangle
    described in the module docstring, or ``None`` when the box has no
    positive area.
    """
    x, y, width, height = bbox
    if not (width > 0 and height > 0):
        return None
    left = math.floor(x)
    top = math.floor(y)
    right = math.ceil(x + width) - 1
    bottom = math.ceil(y + height) - 1
    if right < left or bottom < top:
        return None
    return left, top, right - left + 1, bottom - top + 1


def _stroke_box(painter: QPainter, x: int, y: int, width: int, height: int) -> None:
    """Stroke the inclusive pixel span of a COCO box."""
    right = x + width - 1
    bottom = y + height - 1
    painter.drawLine(x, y, right, y)
    painter.drawLine(x, bottom, right, bottom)
    painter.drawLine(x, y, x, bottom)
    painter.drawLine(right, y, right, bottom)


def _draw_label(
    painter: QPainter,
    image_width: int,
    image_height: int,
    box: tuple[int, int, int, int],
    name: str,
    category_id: int,
    color: QColor,
) -> None:
    metrics = painter.fontMetrics()
    label_width = metrics.horizontalAdvance(name) + _LABEL_PADDING * 2
    label_height = metrics.height()
    if label_width < 1 or label_height < 1 or image_width < 1 or image_height < 1:
        return
    x, y, width, height = box
    if y >= label_height:
        top = y - label_height
    elif y + height + label_height <= image_height:
        top = y + height
    else:
        top = min(max(y + 1, 0), max(image_height - label_height, 0))
    left = min(max(x, 0), max(image_width - label_width, 0))
    label_width = min(label_width, image_width - left)
    label_height = min(label_height, image_height - top)
    if label_width < 1 or label_height < 1:
        return
    rect = QRect(left, top, label_width, label_height)
    painter.fillRect(rect, color)
    painter.setPen(label_text_color(category_id))
    painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), name)
