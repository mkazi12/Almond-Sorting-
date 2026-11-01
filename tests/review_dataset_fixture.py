"""Development window for the authored dataset fixture.

Run from the repository root. Fixture files stay on disk until the window
closes. This script is not part of the production app.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.dataset.preview import PreviewError, render_ground_truth_preview
from app.dataset.records import objects_for_image
from app.dataset.summary import summarize
from app.image_loader import ImageLoadError, load_image

# Summary tables scroll inside this height so the preview keeps the window.
TABLE_MAX_HEIGHT = 156
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class PreviewLabel(QLabel):
    """Shows a scaled copy of an original-resolution preview image."""

    def __init__(self):
        super().__init__()
        self._source = None
        self.setObjectName("ground_truth_preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(0, 0)

    def source_image(self):
        return self._source

    def set_source(self, image):
        if image is None or image.isNull():
            self._source = None
            self.setPixmap(QPixmap())
            return
        self._source = image
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def minimumSizeHint(self):
        return QSize(0, 0)

    def _rescale(self):
        source = self._source
        if source is None or source.isNull():
            self._source = None
            pixmap = self.pixmap()
            if pixmap is not None and not pixmap.isNull():
                self.setPixmap(QPixmap())
            return
        if self.width() < 1 or self.height() < 1:
            return
        pixmap = QPixmap.fromImage(source).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(pixmap)


class DatasetReviewWindow(QMainWindow):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.summary = summarize(dataset)
        self.classes = {
            spec.category_id: spec.name
            for parsed in dataset.annotations
            for spec in parsed.classes
        }
        self.records = {image.image_id: image for image in dataset.accepted}

        self.setWindowTitle("Dataset review")
        self.resize(900, 720)

        self.summary_label = QLabel(self._totals_text())
        self.summary_label.setObjectName("dataset_summary")

        self.class_table = self._table(
            "class_counts",
            ["Class", "Annotations", "Images"],
            [
                [item.name, str(item.annotation_count), str(item.image_count)]
                for item in self.summary.classes
            ],
        )
        self.lot_table = self._table(
            "lot_counts",
            ["Lot", "Images"],
            [[item.label, str(item.image_count)] for item in self.summary.images_by_lot],
        )
        self.session_table = self._table(
            "session_counts",
            ["Session", "Images"],
            [[item.label, str(item.image_count)] for item in self.summary.images_by_session],
        )
        self.rejected_table = self._table(
            "rejected_reasons",
            ["File", "Category", "Detail"],
            [
                [item.filename, reason.category, reason.detail]
                for item in self.summary.rejected_images
                for reason in item.reasons
            ],
        )

        sections = QHBoxLayout()
        sections.setSpacing(8)
        sections.addWidget(self._section("Classes", self.class_table))
        sections.addWidget(self._section("Lots", self.lot_table))
        sections.addWidget(self._section("Sessions", self.session_table))
        sections.addWidget(self._section("Rejected files", self.rejected_table), 2)

        self.image_picker = QComboBox()
        self.image_picker.setObjectName("image_picker")
        if dataset.accepted:
            for image in dataset.accepted:
                self.image_picker.addItem(image.image_id, image.image_id)
            self.image_picker.currentIndexChanged.connect(self._show_selected_image)
        else:
            self.image_picker.addItem("No accepted images", None)
            self.image_picker.setEnabled(False)

        self.review_status = QLabel()
        self.review_status.setObjectName("review_status")
        self.review_status.setWordWrap(True)
        self.review_status.setVisible(False)

        self.image_label = PreviewLabel()

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.summary_label)
        layout.addLayout(sections)
        layout.addWidget(self.image_picker)
        layout.addWidget(self.review_status)
        layout.addWidget(self.image_label, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        if self.records:
            self._show_selected_image()
        else:
            self._show_empty_state()

    def _totals_text(self) -> str:
        summary = self.summary
        return (
            "Development fixture review. These records were not imported or validated.  "
            f"Accepted images: {summary.accepted_image_count}    "
            f"Rejected images: {summary.rejected_image_count}    "
            f"Negative images: {summary.negative_image_count}"
        )

    def _section(self, title: str, table: QTableWidget) -> QGroupBox:
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 8, 6, 6)
        box_layout.setSpacing(4)
        box_layout.addWidget(table)
        return box

    def _table(self, object_name: str, headers: list[str], rows: list[list[str]]) -> QTableWidget:
        table = QTableWidget(len(rows), len(headers))
        table.setObjectName(object_name)
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setWordWrap(True)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setStretchLastSection(True)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row_index, column_index, item)
        table.resizeRowsToContents()
        header_height = max(table.horizontalHeader().sizeHint().height(), 24)
        rows_height = sum(max(table.rowHeight(row), 22) for row in range(table.rowCount()))
        table.setFixedHeight(min(header_height + rows_height + 4, TABLE_MAX_HEIGHT))
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return table

    def _show_empty_state(self) -> None:
        self.image_label.set_source(None)
        self._set_status("No accepted images to review.")

    def _show_selected_image(self) -> None:
        image_id = self.image_picker.currentData()
        record = self.records.get(image_id)
        if record is None:
            self._show_empty_state()
            return
        try:
            image = load_image(self.dataset.resolve(record.image_path))
            objects = objects_for_image(self.dataset, record)
            preview = render_ground_truth_preview(image, objects, self.classes)
        except (ImageLoadError, PreviewError) as error:
            # Drop the previous image before the message so a later resize cannot redraw it.
            self.image_label.set_source(None)
            self._set_status(f"{record.image_id}: {error}")
            return
        self.image_label.set_source(preview)
        if objects:
            self._set_status("")
        else:
            self._set_status("Accepted negative: no ground-truth objects on this image.")

    def _set_status(self, message: str) -> None:
        self.review_status.setText(message)
        self.review_status.setVisible(bool(message))


def main() -> int:
    temporary = tempfile.TemporaryDirectory(prefix="almond-dataset-review-")
    try:
        dataset = load_dataset_fixture(Path(temporary.name))
        app = QApplication(sys.argv)
        window = DatasetReviewWindow(dataset)
        window.show()
        window.raise_()
        window.activateWindow()
        return app.exec()
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    sys.exit(main())
