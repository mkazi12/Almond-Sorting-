"""Desktop image inspection and local JSON/CSV result files."""

from collections.abc import Mapping
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.image_loader import ImageLoadError, load_image
from app.inference import (
    Detector, InferenceResult, RGBImage, confidence_value, validate_class_map, validate_result,
)
from app.results import (
    AnalysisRecord, ResultFileError, create_analysis, export_csv, load_analysis,
    save_analysis, source_image_sha256,
)
from app.ui.assets import SAMPLE_IMAGE_PATH
from app.ui.overlays import render_detections


def rgb_pixels(image: QImage) -> RGBImage:
    """Convert decoded Qt pixels to contiguous RGB; do not decode the file again."""
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    row_size = converted.width() * 3
    stride = converted.bytesPerLine()
    data = bytes(converted.constBits())
    if stride != row_size:
        data = b"".join(data[offset:offset + row_size] for offset in range(0, len(data), stride))
    return RGBImage(converted.width(), converted.height(), data)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        detector: Detector | None = None,
        class_names: Mapping[int, str] | None = None,
        confidence_floor: float = 0.25,
        model_label: str | None = None,
    ):
        super().__init__()
        if detector is not None:
            validate_class_map(class_names)
        self.detector = detector
        self.class_names = dict(class_names or {})
        self.confidence_floor = confidence_value(confidence_floor)
        self.model_label = model_label
        self.current_path: Path | None = None
        self.current_image: QImage | None = None
        self.last_result: InferenceResult | None = None
        self.current_analysis: AnalysisRecord | None = None
        self.setWindowTitle("DINOS Almond Detection")
        self.resize(800, 600)

        self.image_label = QLabel("Choose an image to begin")
        self.image_label.setObjectName("image_label")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.status_label = QLabel(
            "Choose an image to analyze." if detector is not None
            else "Model not configured. Open an image to preview it."
        )
        self.status_label.setObjectName("inference_status")
        self.status_label.setWordWrap(True)

        open_button = QPushButton("Open image")
        open_button.setObjectName("open_image_button")
        open_button.clicked.connect(self.open_image)
        sample_button = QPushButton("Load sample image")
        sample_button.setObjectName("load_sample_button")
        sample_button.clicked.connect(self.load_sample_image)
        buttons = QHBoxLayout()
        buttons.addWidget(open_button)
        buttons.addWidget(sample_button)
        open_results_button = QPushButton("Open saved results")
        open_results_button.setObjectName("open_results_button")
        open_results_button.clicked.connect(self.open_results)
        self.save_results_button = QPushButton("Save results (JSON)")
        self.save_results_button.setObjectName("save_results_button")
        self.save_results_button.clicked.connect(self.save_results)
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.setObjectName("export_csv_button")
        self.export_csv_button.clicked.connect(self.export_results_csv)
        result_buttons = QHBoxLayout()
        result_buttons.addWidget(open_results_button)
        result_buttons.addWidget(self.save_results_button)
        result_buttons.addWidget(self.export_csv_button)
        self._enable_result_actions(False)
        layout = QVBoxLayout()
        layout.addLayout(buttons)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(result_buttons)
        layout.addWidget(self.status_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if path:
            self.display_image_path(path)

    def load_sample_image(self) -> None:
        self.display_image_path(str(SAMPLE_IMAGE_PATH))

    def _enable_result_actions(self, enabled: bool) -> None:
        self.save_results_button.setEnabled(enabled)
        self.export_csv_button.setEnabled(enabled)

    def open_results(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open saved analysis", "", "Result JSON (*.json);;All files (*)"
        )
        if path:
            self.display_results_path(path)

    def display_results_path(self, path: str | Path) -> None:
        """Reopen verified source pixels and saved predictions without inference."""
        try:
            analysis = load_analysis(path)
            image = load_image(analysis.image_path)
            if source_image_sha256(analysis.image_path) != analysis.image_sha256:
                raise ResultFileError("The source image changed while the saved result was opening.")
            rendered = render_detections(image, analysis.result, analysis.class_names)
        except Exception as error:
            self.status_label.setText(f"Could not open saved results: {error}")
            return
        self.current_path = analysis.image_path
        self.current_image = image
        self.current_analysis = analysis
        self.last_result = analysis.result
        self._show_image(rendered)
        self._enable_result_actions(True)
        self.status_label.setText(
            f"Loaded saved results: {len(analysis.result.detections)} detection(s)."
            if analysis.result.detections else "Loaded saved results: no detections."
        )

    def save_results(self) -> None:
        self._write_results(csv=False)

    def export_results_csv(self) -> None:
        self._write_results(csv=True)

    def _write_results(self, *, csv: bool) -> None:
        analysis = self.current_analysis
        if analysis is None:
            return
        title = "Export detections to CSV" if csv else "Save analysis results"
        suffix = ".csv" if csv else ".json"
        file_filter = "CSV (*.csv);;All files (*)" if csv else "Result JSON (*.json);;All files (*)"
        suggested = analysis.image_path.stem + ".results" + suffix
        path, _ = QFileDialog.getSaveFileName(self, title, suggested, file_filter)
        if not path:
            return
        try:
            writer = export_csv if csv else save_analysis
            # The standard save dialog confirms an existing destination. Use
            # exactly its selected path; appending a suffix afterward could
            # overwrite a different file that was never confirmed.
            destination = writer(path, analysis, overwrite=True)
        except Exception as error:
            self.status_label.setText(f"Could not save results: {error}")
            return
        self.status_label.setText(f"Saved results: {destination}")

    def _show_image(self, image: QImage) -> None:
        self.image_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

    def display_image_path(self, path: str | Path) -> None:
        try:
            # Bind predictions to the file that is about to be decoded, rather
            # than whichever bytes exist after a possibly long model call.
            source_digest = source_image_sha256(path) if self.detector is not None else None
            image = load_image(path)
        except ResultFileError as error:
            self.status_label.setText(
                f"Could not open this image: {error}. Choose a readable JPG or PNG image."
            )
            return
        except ImageLoadError as error:
            self.status_label.setText(str(error))
            return
        self.current_path = Path(path)
        self.current_image = image
        self.last_result = None
        self.current_analysis = None
        self._enable_result_actions(False)
        self._show_image(image)
        if self.detector is None:
            self.status_label.setText("Image loaded. Model not configured; detection is unavailable.")
            return
        try:
            result = self.detector.predict(rgb_pixels(image), confidence_floor=self.confidence_floor)
            validate_result(
                result,
                image_width=image.width(),
                image_height=image.height(),
                confidence_floor=self.confidence_floor,
                classes=self.class_names,
            )
            analysis = create_analysis(
                path, result, self.class_names, model_label=self.model_label,
                expected_image_sha256=source_digest,
            )
            rendered = render_detections(image, result, self.class_names)
            self._show_image(rendered)
            self.last_result = result
            self.current_analysis = analysis
            self._enable_result_actions(True)
            self.status_label.setText(
                f"{len(result.detections)} detection(s)." if result.detections else "No detections."
            )
        except Exception as error:
            self.status_label.setText(f"Inference failed: {error}")
