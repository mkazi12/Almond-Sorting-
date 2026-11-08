"""Model-independent inference boundary for the RF-DETR adapter.

The adapter receives decoded RGB pixels and returns original-image pixel
coordinates. COCO category IDs are preserved; they are not storage class
indices. Model loading and prediction belong in the adapter, and drawing
belongs in the UI. There is deliberately no simulated production detector.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Protocol


class InferenceError(RuntimeError):
    """Prediction failed; this is different from a successful empty result."""


def _dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number.")
    return converted


def confidence_value(value: float) -> float:
    """Validate a prediction confidence or requested confidence floor."""
    value = _number(value, "Confidence")
    if not 0 <= value <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    return value


@dataclass(frozen=True)
class RGBImage:
    """Contiguous RGB bytes, ordered by row, with no row padding."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        _dimension(self.width, "Image width")
        _dimension(self.height, "Image height")
        if not isinstance(self.pixels, bytes):
            raise ValueError("Image pixels must be immutable RGB bytes.")
        if len(self.pixels) != self.width * self.height * 3:
            raise ValueError("RGB byte count does not match image dimensions.")


@dataclass(frozen=True)
class Detection:
    """One prediction; bbox_xyxy is [left, top, right, bottom] in pixels."""

    category_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.category_id, bool)
            or not isinstance(self.category_id, int)
            or self.category_id < 0
        ):
            raise ValueError("Category ID must be a non-negative integer.")
        object.__setattr__(self, "confidence", confidence_value(self.confidence))
        try:
            box = tuple(self.bbox_xyxy)
        except TypeError as error:
            raise ValueError("A bounding box needs four coordinates.") from error
        if len(box) != 4:
            raise ValueError("A bounding box needs four coordinates.")
        box = tuple(_number(v, "Box coordinate") for v in box)
        left, top, right, bottom = box
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("Box coordinates must be non-negative with left < right and top < bottom.")
        object.__setattr__(self, "bbox_xyxy", box)


@dataclass(frozen=True)
class InferenceResult:
    """Successful prediction, including the requested confidence floor.

    An empty detections tuple means the adapter completed successfully and
    found no objects at that floor. Adapter failures raise InferenceError.
    """

    image_width: int
    image_height: int
    confidence_floor: float
    detections: tuple[Detection, ...] = ()

    def __post_init__(self) -> None:
        _dimension(self.image_width, "Image width")
        _dimension(self.image_height, "Image height")
        object.__setattr__(self, "confidence_floor", confidence_value(self.confidence_floor))
        detections = tuple(self.detections)
        for detection in detections:
            if not isinstance(detection, Detection):
                raise ValueError("Every prediction must be a Detection record.")
            if detection.confidence < self.confidence_floor:
                raise ValueError("Detection confidence is below the recorded confidence floor.")
            if detection.bbox_xyxy[2] > self.image_width or detection.bbox_xyxy[3] > self.image_height:
                raise ValueError("Detection box exceeds the original image dimensions.")
        object.__setattr__(self, "detections", detections)


class Detector(Protocol):
    """Implemented by the checkpoint adapter in ST-05.4.

    Preprocessing and mapping predictions back to the input dimensions are
    the adapter's responsibility. Raise InferenceError when prediction fails.
    """

    def predict(self, image: RGBImage, *, confidence_floor: float) -> InferenceResult:
        ...


def validate_class_map(classes: Mapping[int, str]) -> None:
    """Require explicit integer category IDs and nonblank configured names."""
    if not isinstance(classes, Mapping) or not classes:
        raise ValueError("A detector requires a non-empty approved class map.")
    for category_id, name in classes.items():
        if isinstance(category_id, bool) or not isinstance(category_id, int) or category_id < 0:
            raise ValueError("Class-map category IDs must be non-negative integers.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Class-map category {category_id} needs a nonblank name.")


def validate_result(
    result: InferenceResult,
    *,
    image_width: int,
    image_height: int,
    confidence_floor: float,
    classes: Mapping[int, str],
) -> None:
    """Reject results for the wrong image, floor, or configured class map."""
    if not isinstance(result, InferenceResult):
        raise InferenceError("The detector returned an invalid result type.")
    if (result.image_width, result.image_height) != (image_width, image_height):
        raise InferenceError("Prediction dimensions do not match the original image.")
    if result.confidence_floor != confidence_value(confidence_floor):
        raise InferenceError("The detector did not preserve the requested confidence floor.")
    for detection in result.detections:
        name = classes.get(detection.category_id)
        if not isinstance(name, str) or not name.strip():
            raise InferenceError(f"No configured class name for category {detection.category_id}.")
