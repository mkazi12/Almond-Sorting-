import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.dataset.preview import category_color
from app.dataset.records import DatasetInput, RejectedImage, RejectionReason, SourcePolicy
from tests.fixtures.load_dataset_fixture import load_dataset_fixture
from tests.review_dataset_fixture import TABLE_MAX_HEIGHT, DatasetReviewWindow


class DatasetReviewWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset = load_dataset_fixture(Path(temporary_directory.name))
        self.window = DatasetReviewWindow(self.dataset)
        self.addCleanup(self.window.close)

    def test_summary_sections_list_counts_and_reason_details(self):
        self.assertIn("Accepted images: 3", self.window.summary_label.text())
        self.assertIn("Rejected images: 1", self.window.summary_label.text())
        self.assertIn("Negative images: 1", self.window.summary_label.text())
        self.assertEqual(
            self.table_rows(self.window.class_table),
            [["A", "3", "2"], ["B", "1", "1"]],
        )
        self.assertEqual(
            self.table_rows(self.window.lot_table),
            [["lot-1", "2"], ["unknown", "1"]],
        )
        self.assertEqual(
            self.table_rows(self.window.session_table),
            [["session-1", "2"], ["session-2", "1"]],
        )
        rejected = self.table_rows(self.window.rejected_table)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(rejected[0][0], "not-a-real-capture.jpg")
        self.assertEqual(rejected[1][0], "not-a-real-capture.jpg")
        self.assertEqual(
            [row[1] for row in rejected],
            ["unreadable_image", "missing_annotation"],
        )
        self.assertIn("not produced by dataset validation", rejected[0][2])
        self.assertIn("second reason", rejected[1][2])

    def test_summary_and_selector_stay_above_the_preview(self):
        self.window.show()
        self.window.resize(900, 720)
        QApplication.processEvents()
        layout = self.window.centralWidget().layout()
        picker = self.window.image_picker.geometry()
        preview = self.window.image_label.geometry()

        self.assertLessEqual(layout.spacing(), 8)
        self.assertLessEqual(layout.contentsMargins().top(), 8)
        self.assertLess(picker.bottom(), preview.top())
        self.assertGreater(preview.height(), self.window.height() // 2)

    def test_preview_scales_up_from_the_original_render(self):
        self.window.show()
        self.window.resize(900, 720)
        QApplication.processEvents()
        label = self.window.image_label
        source = label.source_image()
        source_corner = rgb(source, 10, 10)
        self.assertEqual((source.width(), source.height()), (80, 60))
        self.assertEqual(source_corner, rgb_of(category_color(1)))

        enlarged = label.pixmap()
        self.assertIsNotNone(enlarged)
        self.assertGreater(enlarged.width(), source.width())
        self.assertGreater(enlarged.height(), source.height())
        self.assert_pixmap_fits(enlarged, label)
        self.assert_scaled_source_pixel(source, enlarged, 10, 10, source_corner)

        self.window.resize(640, 480)
        QApplication.processEvents()
        smaller = label.pixmap()
        self.assertNotEqual(smaller.size(), enlarged.size())
        self.assert_pixmap_fits(smaller, label)
        self.assert_scaled_source_pixel(source, smaller, 10, 10, source_corner)
        self.assertEqual((source.width(), source.height()), (80, 60))
        self.assertEqual(rgb(source, 10, 10), source_corner)

    def test_selecting_another_image_replaces_the_preview(self):
        first = self.window.image_label.source_image()
        self.assertEqual(rgb(first, 10, 10), rgb_of(category_color(1)))

        self.window.image_picker.setCurrentIndex(self.picker_index("fix-002"))
        second = self.window.image_label.source_image()
        self.assertIsNot(second, first)
        self.assertEqual(rgb(second, 40, 43), rgb_of(category_color(2)))
        self.assertEqual(rgb(second, 0, 0), (90, 150, 170))
        self.assertEqual(self.window.review_status.text(), "")

    def test_negative_selection_has_no_boxes_and_says_so(self):
        self.window.show()
        self.window.image_picker.setCurrentIndex(self.picker_index("fix-003"))
        QApplication.processEvents()
        source = self.window.image_label.source_image()

        self.assertEqual((source.width(), source.height()), (80, 60))
        self.assertEqual(rgb(source, 10, 10), (230, 230, 230))
        self.assertNotEqual(rgb(source, 10, 10), rgb_of(category_color(1)))
        self.assertIn("Accepted negative", self.window.review_status.text())
        self.assertIn("no ground-truth objects", self.window.review_status.text())
        self.assertTrue(self.window.review_status.isVisible())

    def test_empty_dataset_shows_an_empty_state(self):
        empty = DatasetInput(
            root=self.dataset.root,
            source_policy=SourcePolicy(require_lot=True, require_session=True),
            accepted=(),
            rejected=(),
            annotations=(),
        )
        window = DatasetReviewWindow(empty)
        self.addCleanup(window.close)
        window.show()
        QApplication.processEvents()

        self.assertIn("Accepted images: 0", window.summary_label.text())
        self.assertIn("Rejected images: 0", window.summary_label.text())
        self.assertIn("Negative images: 0", window.summary_label.text())
        self.assertEqual(window.review_status.text(), "No accepted images to review.")
        self.assertTrue(window.review_status.isVisible())
        self.assertIsNone(window.image_label.source_image())
        self.assertFalse(window.image_picker.isEnabled())
        self.assertEqual(window.image_picker.currentText(), "No accepted images")

    def test_failed_load_clears_the_previous_preview(self):
        self.show_loaded_preview()
        self.fail_second_image()

        self.assertIsNone(self.window.image_label.source_image())
        self.assertTrue(self.pixmap_is_clear(self.window.image_label))
        self.assertIn("fix-002", self.window.review_status.text())
        self.assertIn("Could not decode", self.window.review_status.text())
        self.assertTrue(self.window.review_status.isVisible())

    def test_resize_after_failed_load_does_not_restore_the_preview(self):
        self.show_loaded_preview()
        self.fail_second_image()

        self.window.resize(640, 480)
        QApplication.processEvents()

        self.assertIsNone(self.window.image_label.source_image())
        self.assertTrue(self.pixmap_is_clear(self.window.image_label))
        self.assertIn("fix-002", self.window.review_status.text())

    def test_summary_tables_scroll_inside_a_bounded_height(self):
        detail = "unreadable capture " * 8
        rejected = tuple(
            RejectedImage(
                image_path=f"images/rejected-{index:02d}.jpg",
                image_id=None,
                reasons=(RejectionReason("unreadable_image", f"{detail}{index}"),),
            )
            for index in range(30)
        )
        window = DatasetReviewWindow(replace(self.dataset, rejected=rejected))
        self.addCleanup(window.close)
        window.show()
        window.resize(900, 720)
        QApplication.processEvents()

        table = window.rejected_table
        self.assertEqual(table.rowCount(), 30)
        self.assertLessEqual(table.height(), TABLE_MAX_HEIGHT)
        self.assertLess(
            table.viewport().height(),
            sum(table.rowHeight(row) for row in range(table.rowCount())),
        )
        for column in range(table.columnCount()):
            item = table.item(0, column)
            self.assertEqual(item.toolTip(), item.text())
            self.assertGreater(len(item.text()), 0)
        self.assertGreater(len(table.item(0, 2).text()), 80)
        self.assertIn(detail.strip(), table.item(0, 2).text())
        self.assertGreater(window.image_label.height(), window.height() // 2)

    def show_loaded_preview(self):
        self.window.show()
        self.window.resize(900, 720)
        QApplication.processEvents()
        self.assertIsNotNone(self.window.image_label.source_image())
        self.assertFalse(self.pixmap_is_clear(self.window.image_label))

    def fail_second_image(self):
        record = next(image for image in self.dataset.accepted if image.image_id == "fix-002")
        self.dataset.resolve(record.image_path).write_bytes(b"not-a-readable-image")
        self.window.image_picker.setCurrentIndex(self.picker_index("fix-002"))
        QApplication.processEvents()

    def picker_index(self, image_id):
        picker = self.window.image_picker
        for index in range(picker.count()):
            if picker.itemData(index) == image_id:
                return index
        self.fail(f"No picker item for {image_id}.")

    def pixmap_is_clear(self, label):
        pixmap = label.pixmap()
        return pixmap is None or pixmap.isNull()

    def assert_scaled_source_pixel(self, source, pixmap, x, y, color):
        image = pixmap.toImage()
        center_x = int(x * pixmap.width() / source.width())
        center_y = int(y * pixmap.height() / source.height())
        span_x = max(int(pixmap.width() / source.width()), 1)
        span_y = max(int(pixmap.height() / source.height()), 1)
        for pixel_y in range(max(0, center_y - span_y), min(image.height(), center_y + span_y + 1)):
            for pixel_x in range(max(0, center_x - span_x), min(image.width(), center_x + span_x + 1)):
                if rgb(image, pixel_x, pixel_y) == color:
                    return
        self.fail(f"Rendered box pixel was not near the scaled coordinate {(center_x, center_y)}.")

    def assert_pixmap_fits(self, pixmap, label):
        self.assertLessEqual(pixmap.width(), label.width())
        self.assertLessEqual(pixmap.height(), label.height())
        self.assertTrue(pixmap.width() == label.width() or pixmap.height() == label.height())
        self.assertAlmostEqual(pixmap.width() / pixmap.height(), 80 / 60, delta=0.05)

    def table_rows(self, table):
        return [
            [table.item(row, column).text() for column in range(table.columnCount())]
            for row in range(table.rowCount())
        ]


def rgb(image, x, y):
    color = image.pixelColor(x, y)
    return (color.red(), color.green(), color.blue())


def rgb_of(color):
    return (color.red(), color.green(), color.blue())


if __name__ == "__main__":
    unittest.main()
