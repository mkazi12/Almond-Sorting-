"""Preview and adapter integration; controlled predictions exist only in tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QPushButton

from app.image_loader import load_image
from app.inference import Detection, InferenceError, InferenceResult
from app.results import create_analysis, export_csv, load_analysis, save_analysis
from app.ui.main_window import MainWindow, rgb_pixels
from app.ui.overlays import render_detections


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "image.png"
        self.image = QImage(21, 12, QImage.Format.Format_RGB32)
        self.image.fill(QColor(20, 40, 60))
        self.assertTrue(self.image.save(str(self.path), "PNG"))

    def window(self, **kwargs):
        window = MainWindow(**kwargs)
        self.addCleanup(window.close)
        return window

    def test_preview_without_model_has_no_fabricated_predictions(self):
        window = self.window()
        window.display_image_path(self.path)
        self.assertFalse(window.image_label.pixmap().isNull())
        self.assertEqual(window.current_image.size(), self.image.size())
        self.assertIsNone(window.last_result)
        self.assertIn("Model not configured", window.status_label.text())
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())

    def test_cancel_does_not_load_an_image_or_change_current_preview(self):
        window = self.window()
        window.display_image_path(self.path)
        old_path = window.current_path
        old_status = window.status_label.text()
        with patch("app.ui.main_window.QFileDialog.getOpenFileName", return_value=("", "")):
            with patch("app.ui.main_window.load_image") as loader:
                window.open_image()
        loader.assert_not_called()
        self.assertEqual(window.current_path, old_path)
        self.assertEqual(window.status_label.text(), old_status)

    def test_invalid_image_does_not_reach_detector(self):
        detector = Mock()
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path.with_name("missing.png"))
        detector.predict.assert_not_called()
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.current_image)
        self.assertIn("Choose", window.status_label.text())

    def test_decodes_once_and_passes_rgb_pixels_at_original_resolution(self):
        result = InferenceResult(21, 12, .25, (Detection(17, .9, (2, 3, 15, 10)),))
        detector = Mock()
        detector.predict.return_value = result
        window = self.window(detector=detector, class_names={17: "Almond"})
        with patch("app.ui.main_window.load_image", wraps=load_image) as loader:
            window.display_image_path(self.path)
        loader.assert_called_once_with(self.path)
        detector.predict.assert_called_once()
        image_argument = detector.predict.call_args.args[0]
        self.assertEqual((image_argument.width, image_argument.height), (21, 12))
        self.assertEqual(image_argument.pixels, bytes([20, 40, 60]) * (21 * 12))
        self.assertEqual(detector.predict.call_args.kwargs, {"confidence_floor": .25})
        self.assertIs(window.last_result, result)
        self.assertIn("1 detection", window.status_label.text())

    def test_successful_empty_result_differs_from_prediction_failure(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.assertEqual(window.status_label.text(), "No detections.")
        self.assertIsNotNone(window.last_result)
        self.assertTrue(window.save_results_button.isEnabled())
        self.assertTrue(window.export_csv_button.isEnabled())
        detector.predict.side_effect = InferenceError("Checkpoint unavailable")
        window.display_image_path(self.path)
        self.assertIn("Inference failed: Checkpoint unavailable", window.status_label.text())
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())
        self.assertFalse(window.image_label.pixmap().isNull())

    def test_adapter_result_with_unknown_category_is_not_displayed(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25, (Detection(99, .9, (2, 3, 15, 10)),))
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.assertIn("Inference failed", window.status_label.text())
        self.assertIn("99", window.status_label.text())
        self.assertIsNone(window.last_result)

    def test_qt_row_padding_is_removed_and_channels_remain_rgb(self):
        pixels = rgb_pixels(self.image)
        self.assertEqual(len(pixels.pixels), 21 * 12 * 3)
        self.assertEqual(pixels.pixels, bytes([20, 40, 60]) * (21 * 12))

    def test_rendering_preserves_source_and_original_dimensions(self):
        original = self.image.copy()
        result = InferenceResult(21, 12, .25, (Detection(17, .9, (2, 3, 15, 10)),))
        rendered = render_detections(self.image, result, {17: "Almond"})
        self.assertEqual(self.image, original)
        self.assertEqual(rendered.size(), original.size())
        self.assertNotEqual(rendered.pixelColor(2, 8), original.pixelColor(2, 8))

    def test_empty_overlay_preserves_pixels(self):
        rendered = render_detections(self.image, InferenceResult(21, 12, .25), {})
        self.assertEqual(rendered, self.image)

    def test_malformed_class_map_is_rejected_before_empty_prediction(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        for classes in ({"17": "Almond"}, {17: " "}, {17: "Almond", 99: ""}, {}):
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    self.window(detector=detector, class_names=classes)
        detector.predict.assert_not_called()

    def test_indexed_grayscale_and_monochrome_images_can_render_predictions(self):
        result = InferenceResult(21, 12, .25, (Detection(17, .9, (2, 3, 15, 10)),))
        for image_format in (QImage.Format.Format_Indexed8, QImage.Format.Format_Grayscale8,
                             QImage.Format.Format_Mono):
            with self.subTest(image_format=image_format):
                source = self.image.convertToFormat(image_format)
                original = source.copy()
                rendered = render_detections(source, result, {17: "Almond"})
                self.assertEqual(rendered.format(), QImage.Format.Format_RGB32)
                self.assertEqual(source, original)
                self.assertNotEqual(rendered.pixelColor(2, 8), source.pixelColor(2, 8))

    def test_saves_successful_analysis_and_exports_it_through_file_dialogs(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(
            21, 12, .25, (Detection(17, .9, (2, 3, 15, 10)),)
        )
        window = self.window(detector=detector, class_names={17: "Almond"}, model_label="candidate-1")
        window.display_image_path(self.path)
        json_path = self.path.with_name("results.json")
        csv_path = self.path.with_name("results.csv")
        with patch("app.ui.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(json_path), "Result JSON (*.json)")):
            window.save_results_button.click()
        saved = load_analysis(json_path)
        self.assertEqual(saved.result, window.last_result)
        self.assertEqual(saved.model_label, "candidate-1")
        self.assertEqual(saved.class_names, {17: "Almond"})
        with patch("app.ui.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(csv_path), "CSV (*.csv)")):
            with patch("app.ui.main_window.export_csv", wraps=export_csv) as exporter:
                window.export_csv_button.click()
        exporter.assert_called_once_with(str(csv_path), window.current_analysis, overwrite=True)
        self.assertTrue(csv_path.is_file())
        detector.predict.assert_called_once()

    def test_open_saved_results_uses_saved_classes_and_does_not_predict(self):
        result = InferenceResult(21, 12, .4, (Detection(99, .8, (1, 2, 18, 11)),))
        saved = create_analysis(self.path, result, {99: "Foreign material"})
        json_path = self.path.with_name("saved.json")
        save_analysis(json_path, saved)
        detector = Mock()
        detector.predict.side_effect = AssertionError("Reopening must not run inference")
        window = self.window(detector=detector, class_names={17: "Almond"})
        with patch("app.ui.main_window.QFileDialog.getOpenFileName",
                   return_value=(str(json_path), "Result JSON (*.json)")):
            window.findChild(QPushButton, "open_results_button").click()
        detector.predict.assert_not_called()
        self.assertEqual(window.last_result, result)
        self.assertEqual(window.current_analysis.class_names, {99: "Foreign material"})
        self.assertEqual(window.class_names, {17: "Almond"})
        self.assertEqual(window.confidence_floor, .25)
        self.assertIn("Loaded saved results", window.status_label.text())
        self.assertTrue(window.save_results_button.isEnabled())

    def test_failed_or_cancelled_result_open_preserves_previous_preview(self):
        window = self.window()
        window.display_image_path(self.path)
        original_path, original_image = window.current_path, window.current_image.copy()
        with patch("app.ui.main_window.QFileDialog.getOpenFileName", return_value=("", "")):
            with patch("app.ui.main_window.load_analysis") as loader:
                window.open_results()
        loader.assert_not_called()
        broken = self.path.with_name("broken.json")
        broken.write_text("{broken", encoding="utf-8")
        window.display_results_path(broken)
        self.assertIn("Could not open saved results", window.status_label.text())
        self.assertEqual(window.current_path, original_path)
        self.assertEqual(window.current_image, original_image)
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())

    def test_changed_source_cannot_reopen_saved_detections(self):
        result = InferenceResult(21, 12, .25)
        saved_path = self.path.with_name("saved.json")
        save_analysis(saved_path, create_analysis(self.path, result, {17: "Almond"}))
        self.image.fill(QColor("red"))
        self.image.save(str(self.path), "PNG")
        window = self.window()
        window.display_results_path(saved_path)
        self.assertIn("Could not open saved results", window.status_label.text())
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.current_image)

    def test_cancelled_save_keeps_results_and_does_not_write(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        status = window.status_label.text()
        analysis = window.current_analysis
        with patch("app.ui.main_window.QFileDialog.getSaveFileName", return_value=("", "")):
            with patch("app.ui.main_window.save_analysis") as writer:
                window.save_results()
        writer.assert_not_called()
        self.assertIs(window.current_analysis, analysis)
        self.assertEqual(window.status_label.text(), status)

    def test_new_preview_drops_old_analysis_and_disables_save(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.assertIsNotNone(window.current_analysis)
        window.detector = None
        window.display_image_path(self.path)
        self.assertIsNone(window.current_analysis)
        self.assertIsNone(window.last_result)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())

    def test_save_failure_and_invalid_reopen_preserve_successful_analysis(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        analysis = window.current_analysis
        original_image = window.current_image.copy()
        destination = self.path.with_name("results.json")
        with patch("app.ui.main_window.QFileDialog.getSaveFileName",
                   return_value=(str(destination), "Result JSON (*.json)")):
            with patch("app.ui.main_window.save_analysis", side_effect=OSError("Disk full")):
                window.save_results()
        self.assertIn("Could not save results", window.status_label.text())
        self.assertIs(window.current_analysis, analysis)
        destination.write_text("{broken", encoding="utf-8")
        window.display_results_path(destination)
        self.assertIn("Could not open saved results", window.status_label.text())
        self.assertIs(window.current_analysis, analysis)
        self.assertIs(window.last_result, analysis.result)
        self.assertEqual(window.current_image, original_image)
        self.assertTrue(window.save_results_button.isEnabled())
        self.assertTrue(window.export_csv_button.isEnabled())

    def test_source_changed_during_prediction_cannot_be_saved(self):
        def change_source(image, *, confidence_floor):
            # The dimensions remain the same, so identity must use file bytes.
            self.image.fill(QColor("red"))
            self.assertTrue(self.image.save(str(self.path), "PNG"))
            return InferenceResult(image.width, image.height, confidence_floor)

        detector = Mock()
        detector.predict.side_effect = change_source
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        detector.predict.assert_called_once()
        self.assertIn("changed", window.status_label.text().lower())
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())
        self.assertEqual(window.current_image.pixelColor(0, 0), QColor(20, 40, 60))


if __name__ == "__main__":
    unittest.main()
