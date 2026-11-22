"""Desktop image inspection and local JSON/CSV result files."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.image_loader import (
    MAX_IMAGE_FILE_SIZE_BYTES, MAX_IMAGE_PIXELS, ImageLoadError, load_image,
    validate_image_extension, validate_image_file_size, validate_image_limits,
)
from app.inference import (
    Detector, InferenceResult, confidence_value, validate_class_map,
)
from app.results import (
    AnalysisRecord, ResultFileError, export_csv, load_analysis,
    save_analysis, source_image_sha256,
)
from app.ui.analysis_worker import (
    AnalysisCompletion, AnalysisRequest, AnalysisWorker, rgb_pixels,
)
from app.ui.assets import SAMPLE_IMAGE_PATH
from app.ui.overlays import render_detections
from app.ui.preview_label import ResponsiveImageLabel


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        detector: Detector | None = None,
        class_names: Mapping[int, str] | None = None,
        confidence_floor: float = 0.25,
        model_label: str | None = None,
        max_image_file_size_bytes: int = MAX_IMAGE_FILE_SIZE_BYTES,
        max_image_pixels: int = MAX_IMAGE_PIXELS,
        session_id: str | None = None,
    ):
        super().__init__()
        validate_image_limits(max_image_file_size_bytes, max_image_pixels)
        if model_label is not None and (not isinstance(model_label, str) or not model_label.strip()):
            raise ValueError("Model version must be a nonblank string.")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("Session ID must be a nonblank string.")
        if detector is not None:
            validate_class_map(class_names)
        self.detector = detector
        self.class_names = dict(class_names or {})
        self.confidence_floor = confidence_value(confidence_floor)
        self.model_label = model_label
        self.session_id = session_id or uuid4().hex
        self.max_image_file_size_bytes = max_image_file_size_bytes
        self.max_image_pixels = max_image_pixels
        self.current_path: Path | None = None
        self.current_image_id: str | None = None
        self.current_image: QImage | None = None
        self.last_result: InferenceResult | None = None
        self.current_analysis: AnalysisRecord | None = None
        self.detection_counts: dict[int, int] | None = None
        self.active_request: AnalysisRequest | None = None
        self.last_completion: AnalysisCompletion | None = None
        self.completed_analyses: dict[str, AnalysisCompletion] = {}
        self._active_worker: AnalysisWorker | None = None
        self.setWindowTitle("DINOS Almond Detection")
        self.resize(800, 600)

        self.image_label = ResponsiveImageLabel("Choose an image to begin")
        self.image_label.setObjectName("image_label")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.status_label = QLabel(
            "Choose an image to analyze." if detector is not None
            else "Model not configured. Open an image to preview it."
        )
        self.status_label.setObjectName("inference_status")
        self.status_label.setWordWrap(True)
        self.counts_label = QLabel("Detection counts unavailable.")
        self.counts_label.setObjectName("detection_counts")
        self.counts_label.setWordWrap(True)

        open_button = QPushButton("Open image")
        open_button.setObjectName("open_image_button")
        open_button.clicked.connect(self.open_image)
        sample_button = QPushButton("Load sample image")
        sample_button.setObjectName("load_sample_button")
        sample_button.clicked.connect(self.load_sample_image)
        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setObjectName("run_analysis_button")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        self.run_analysis_button.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(open_button)
        buttons.addWidget(sample_button)
        buttons.addWidget(self.run_analysis_button)
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
        layout.addWidget(self.counts_label)
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

    def _show_analysis_counts(self, analysis: AnalysisRecord | None) -> None:
        if analysis is None:
            self.detection_counts = None
            self.counts_label.setText("Detection counts unavailable.")
            return
        self.detection_counts = dict(analysis.class_counts)
        summary = " | ".join(
            f"{analysis.class_names[category_id]}: {count}"
            for category_id, count in sorted(self.detection_counts.items())
        )
        self.counts_label.setText(f"Detection counts — {summary}")

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
        self.current_image_id = analysis.image_sha256
        self.current_image = image
        self.current_analysis = analysis
        self.last_result = analysis.result
        self._show_analysis_counts(analysis)
        self._show_image(rendered)
        self._enable_result_actions(True)
        self._update_analysis_controls()
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
        self.image_label.set_source_image(image)

    @property
    def active_model_version(self) -> str | None:
        if self.detector is None:
            return None
        version = self.model_label or getattr(self.detector, "model_label", None)
        return version if isinstance(version, str) and version.strip() else type(self.detector).__name__

    def _update_analysis_controls(self) -> None:
        self.run_analysis_button.setEnabled(
            self.detector is not None
            and self.current_image is not None
            and self.current_image_id is not None
            and self._active_worker is None
        )

    def run_analysis(self) -> None:
        if self._active_worker is not None:
            return
        model_version = self.active_model_version
        if (
            self.detector is None or self.current_path is None
            or self.current_image is None or self.current_image_id is None
            or model_version is None
        ):
            return

        request = AnalysisRequest(
            request_id=uuid4().hex,
            image_id=self.current_image_id,
            image_path=self.current_path.resolve(),
            model_version=model_version,
            session_id=self.session_id,
            image=self.current_image.copy(),
            detector=self.detector,
            class_names=MappingProxyType(dict(self.class_names)),
            confidence_floor=self.confidence_floor,
        )
        worker = AnalysisWorker(request)
        worker.signals.finished.connect(self._analysis_finished)
        self.active_request = request
        self._active_worker = worker
        self.last_result = None
        self.current_analysis = None
        self._show_analysis_counts(None)
        self._enable_result_actions(False)
        self.run_analysis_button.setEnabled(False)
        self.status_label.setText(
            f"Analysis running for {request.image_path.name} "
            f"(model {request.model_version}; session {request.session_id})."
        )
        try:
            QThreadPool.globalInstance().start(worker)
        except Exception as error:
            reason = str(error).strip() or type(error).__name__
            self._analysis_finished(
                AnalysisCompletion(request, error=f"Could not start worker: {reason}")
            )

    def _analysis_finished(self, completion: AnalysisCompletion) -> None:
        self.last_completion = completion
        self.completed_analyses[completion.request.request_id] = completion
        self.active_request = None
        self._active_worker = None

        request = completion.request
        is_current = (
            request.image_id == self.current_image_id
            and request.image_path == self.current_path
            and request.model_version == self.active_model_version
        )
        if completion.succeeded:
            if is_current:
                self.last_result = completion.result
                self.current_analysis = completion.analysis
                self._show_analysis_counts(completion.analysis)
                if completion.rendered_image is not None:
                    self._show_image(completion.rendered_image)
                self._enable_result_actions(True)
                detections = len(completion.result.detections)
                outcome = f"{detections} detection(s)" if detections else "no detections"
                self.status_label.setText(
                    f"Analysis succeeded for {request.image_path.name} "
                    f"(model {request.model_version}): {outcome}."
                )
            else:
                self.status_label.setText(
                    f"Analysis succeeded for {request.image_path.name} "
                    f"(model {request.model_version}); the current selection was not changed."
                )
        else:
            if is_current:
                self.last_result = None
                self.current_analysis = None
                self._show_analysis_counts(None)
                self._enable_result_actions(False)
            self.status_label.setText(
                f"Analysis failed for {request.image_path.name} "
                f"(model {request.model_version}): {completion.error}"
            )
        self._update_analysis_controls()

    def display_image_path(self, path: str | Path) -> None:
        try:
            validate_image_extension(path)
            validate_image_file_size(path, self.max_image_file_size_bytes)
            resolved_path = Path(path).expanduser().resolve()
            source_digest = source_image_sha256(resolved_path)
            image = load_image(
                resolved_path,
                max_file_size_bytes=self.max_image_file_size_bytes,
                max_pixels=self.max_image_pixels,
            )
            if source_image_sha256(resolved_path) != source_digest:
                raise ResultFileError("The source image changed while it was being opened.")
        except ResultFileError as error:
            self.status_label.setText(
                f"Could not open this image: {error}. Choose a readable JPG or PNG image."
            )
            return
        except ImageLoadError as error:
            self.status_label.setText(str(error))
            return
        self.current_path = resolved_path
        self.current_image_id = source_digest
        self.current_image = image
        self.last_result = None
        self.current_analysis = None
        self._show_analysis_counts(None)
        self._enable_result_actions(False)
        self._show_image(image)
        self._update_analysis_controls()
        if self.active_model_version is None:
            self.status_label.setText("Image loaded. Model not configured; detection is unavailable.")
            return
        self.status_label.setText(
            f"Image loaded. Ready to analyze with {self.active_model_version}."
        )
