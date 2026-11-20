"""Background execution for a snapshot of one desktop analysis request."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from PySide6.QtGui import QImage

from app.inference import (
    Detector, InferenceResult, RGBImage, validate_result,
)
from app.results import AnalysisRecord, create_analysis
from app.ui.overlays import render_detections


@dataclass(frozen=True)
class AnalysisRequest:
    """Immutable identifiers and input snapshots captured when analysis starts."""

    request_id: str
    image_id: str
    image_path: Path
    model_version: str
    session_id: str
    image: QImage
    detector: Detector
    class_names: Mapping[int, str]
    confidence_floor: float


@dataclass(frozen=True)
class AnalysisCompletion:
    request: AnalysisRequest
    result: InferenceResult | None = None
    analysis: AnalysisRecord | None = None
    rendered_image: QImage | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.result is not None and self.analysis is not None


class _WorkerSignals(QObject):
    finished = Signal(object)


def rgb_pixels(image: QImage) -> RGBImage:
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    row_size = converted.width() * 3
    stride = converted.bytesPerLine()
    data = bytes(converted.constBits())
    if stride != row_size:
        data = b"".join(data[offset:offset + row_size] for offset in range(0, len(data), stride))
    return RGBImage(converted.width(), converted.height(), data)


class AnalysisWorker(QRunnable):
    def __init__(self, request: AnalysisRequest):
        super().__init__()
        self.request = request
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        request = self.request
        try:
            result = request.detector.predict(
                rgb_pixels(request.image), confidence_floor=request.confidence_floor
            )
            validate_result(
                result,
                image_width=request.image.width(),
                image_height=request.image.height(),
                confidence_floor=request.confidence_floor,
                classes=request.class_names,
            )
            analysis = create_analysis(
                request.image_path,
                result,
                request.class_names,
                model_label=request.model_version,
                expected_image_sha256=request.image_id,
            )
            rendered = render_detections(request.image, result, request.class_names)
            completion = AnalysisCompletion(request, result, analysis, rendered)
        except Exception as error:
            reason = str(error).strip() or type(error).__name__
            completion = AnalysisCompletion(
                request, error=f"{type(error).__name__}: {reason}"
            )
        self.signals.finished.emit(completion)
