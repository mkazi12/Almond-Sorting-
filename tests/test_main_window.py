"""Preview and adapter integration; controlled predictions exist only in tests."""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton

from app.image_loader import load_image
from app.inference import Detection, InferenceError, InferenceResult
from app.results import (
    create_analysis, export_csv, load_analysis, save_analysis, source_image_sha256,
)
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

    def wait_for_analysis(self, window):
        deadline = time.monotonic() + 5
        while window.active_request is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.005)
        self.application.processEvents()
        self.assertIsNone(window.active_request, "Analysis worker did not finish in time.")

    def run_analysis_and_wait(self, window):
        window.run_analysis_button.click()
        self.wait_for_analysis(window)

    def test_preview_without_model_has_no_fabricated_predictions(self):
        window = self.window()
        window.display_image_path(self.path)
        self.assertFalse(window.image_label.pixmap().isNull())
        self.assertEqual(window.current_image.size(), self.image.size())
        self.assertFalse(window.run_analysis_button.isEnabled())
        self.assertIsNone(window.last_result)
        self.assertIn("Model not configured", window.status_label.text())
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())

    def test_cancel_does_not_load_an_image_or_change_current_preview(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.run_analysis_and_wait(window)
        old_path = window.current_path
        old_image = window.current_image.copy()
        old_analysis = window.current_analysis
        old_status = window.status_label.text()
        with patch("app.ui.main_window.QFileDialog.getOpenFileName", return_value=("", "")):
            with patch("app.ui.main_window.load_image") as loader:
                window.open_image()
        loader.assert_not_called()
        self.assertEqual(window.current_path, old_path)
        self.assertEqual(window.current_image, old_image)
        self.assertIs(window.current_analysis, old_analysis)
        self.assertEqual(window.status_label.text(), old_status)
        self.assertTrue(window.save_results_button.isEnabled())
        self.assertEqual(detector.predict.call_count, 1)

    def test_invalid_pick_preserves_saved_analysis_and_valid_pick_recovers(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.run_analysis_and_wait(window)
        original_analysis = window.current_analysis
        original_image = window.current_image.copy()

        unsupported = self.path.with_suffix(".gif")
        unsupported.write_bytes(self.path.read_bytes())
        window.display_image_path(unsupported)
        self.assertIn("Unsupported file extension", window.status_label.text())
        self.assertEqual(window.current_path, self.path.resolve())
        self.assertEqual(window.current_image, original_image)
        self.assertIs(window.current_analysis, original_analysis)
        self.assertTrue(window.save_results_button.isEnabled())
        self.assertEqual(detector.predict.call_count, 1)

        window.display_image_path(self.path)
        self.assertEqual(window.current_path, self.path.resolve())
        self.assertIsNotNone(window.current_image)
        detector.predict.assert_called_once()
        self.run_analysis_and_wait(window)
        self.assertEqual(detector.predict.call_count, 2)

    def test_configured_file_size_limit_rejects_pick_without_prediction(self):
        detector = Mock()
        window = self.window(
            detector=detector, class_names={17: "Almond"}, max_image_file_size_bytes=1
        )
        window.display_image_path(self.path)
        self.assertIn("file is too large", window.status_label.text())
        self.assertIsNone(window.current_path)
        self.assertIsNone(window.current_image)
        detector.predict.assert_not_called()

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
        loader.assert_called_once_with(
            self.path.resolve(),
            max_file_size_bytes=window.max_image_file_size_bytes,
            max_pixels=window.max_image_pixels,
        )
        detector.predict.assert_not_called()
        self.run_analysis_and_wait(window)
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
        self.run_analysis_and_wait(window)
        self.assertIn("Analysis succeeded", window.status_label.text())
        self.assertIn("no detections", window.status_label.text())
        self.assertIsNotNone(window.last_result)
        self.assertEqual(dict(window.current_analysis.class_counts), {17: 0})
        self.assertIn("Almond: 0", window.counts_label.text())
        self.assertTrue(window.save_results_button.isEnabled())
        self.assertTrue(window.export_csv_button.isEnabled())
        detector.predict.side_effect = InferenceError("Checkpoint unavailable")
        self.run_analysis_and_wait(window)
        self.assertIn("Analysis failed", window.status_label.text())
        self.assertIn("Checkpoint unavailable", window.status_label.text())
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())
        self.assertFalse(window.image_label.pixmap().isNull())
        self.assertEqual(window.counts_label.text(), "Detection counts unavailable.")

    def test_adapter_result_with_unknown_category_is_not_displayed(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25, (Detection(99, .9, (2, 3, 15, 10)),))
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.run_analysis_and_wait(window)
        self.assertIn("Analysis failed", window.status_label.text())
        self.assertIn("99", window.status_label.text())
        self.assertIsNone(window.last_result)

    def test_running_request_is_bound_to_inputs_and_does_not_replace_new_selection(self):
        started = threading.Event()
        release = threading.Event()
        result = InferenceResult(21, 12, .25)

        def delayed_prediction(image, *, confidence_floor):
            started.set()
            release.wait(5)
            return result

        detector = Mock()
        detector.model_label = "rfdetr-nano:checkpoint@abc123"
        detector.predict.side_effect = delayed_prediction
        window = self.window(
            detector=detector,
            class_names={17: "Almond"},
            session_id="session-17",
        )
        window.display_image_path(self.path)

        window.run_analysis_button.click()
        self.assertTrue(started.wait(2), "The prediction worker did not start.")
        request = window.active_request
        self.assertIsNotNone(request)
        self.assertEqual(request.image_id, source_image_sha256(self.path))
        self.assertEqual(request.image_path, self.path.resolve())
        self.assertEqual(request.model_version, detector.model_label)
        self.assertEqual(request.session_id, "session-17")
        self.assertFalse(window.run_analysis_button.isEnabled())
        window.run_analysis_button.click()
        window.run_analysis()
        self.assertEqual(detector.predict.call_count, 1)
        self.assertIs(window.active_request, request)

        second_path = self.path.with_name("second.png")
        self.image.fill(QColor("red"))
        self.assertTrue(self.image.save(str(second_path), "PNG"))
        replacement_detector = Mock()
        replacement_detector.model_label = "rfdetr-small:other@def456"
        window.detector = replacement_detector
        window.display_image_path(second_path)
        second_image = window.current_image.copy()
        release.set()
        self.wait_for_analysis(window)

        completion = window.completed_analyses[request.request_id]
        self.assertTrue(completion.succeeded)
        self.assertEqual(completion.request.image_path, self.path.resolve())
        self.assertEqual(completion.request.image_id, request.image_id)
        self.assertEqual(completion.request.model_version, detector.model_label)
        self.assertEqual(completion.request.session_id, "session-17")
        self.assertEqual(completion.analysis.image_path, self.path.resolve())
        self.assertEqual(completion.analysis.model_label, detector.model_label)
        detector.predict.assert_called_once()
        replacement_detector.predict.assert_not_called()
        self.assertEqual(window.current_path, second_path.resolve())
        self.assertEqual(window.current_image, second_image)
        self.assertIsNone(window.current_analysis)
        self.assertTrue(window.run_analysis_button.isEnabled())
        self.assertIn("current selection was not changed", window.status_label.text())

        self.run_analysis_and_wait(window)
        replacement_detector.predict.assert_called_once()
        self.assertNotEqual(window.last_completion.request.request_id, request.request_id)
        self.assertEqual(window.last_completion.request.model_version, replacement_detector.model_label)

    def test_worker_failure_is_distinct_and_allows_retry(self):
        detector = Mock()
        detector.predict.side_effect = [
            RuntimeError("GPU memory unavailable"),
            InferenceResult(21, 12, .25),
        ]
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)

        self.run_analysis_and_wait(window)
        self.assertFalse(window.last_completion.succeeded)
        self.assertIn("Analysis failed", window.status_label.text())
        self.assertIn("GPU memory unavailable", window.status_label.text())
        self.assertTrue(window.run_analysis_button.isEnabled())

        self.run_analysis_and_wait(window)
        self.assertTrue(window.last_completion.succeeded)
        self.assertIn("Analysis succeeded", window.status_label.text())
        self.assertTrue(window.run_analysis_button.isEnabled())
        self.assertEqual(detector.predict.call_count, 2)

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

    def test_responsive_preview_scales_rendered_boxes_with_centered_margins(self):
        source = QImage(400, 100, QImage.Format.Format_RGB32)
        source.fill(QColor("white"))
        result = InferenceResult(400, 100, .25, (Detection(17, .9, (100, 20, 300, 80)),))
        rendered = render_detections(source, result, {17: "Almond"})
        window = self.window()
        window.image_label.resize(200, 200)
        window.image_label.set_source_image(rendered)
        self.application.processEvents()

        expected = QPixmap.fromImage(rendered).scaled(
            window.image_label.contentsRect().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        actual = window.image_label.pixmap()
        self.assertEqual(actual.size(), expected.size())
        self.assertEqual(actual.toImage(), expected.toImage())
        self.assertEqual(actual.width(), 200)
        self.assertEqual(actual.height(), 50)
        self.assertEqual((200 - actual.height()) // 2, 75)

        # The source box (100, 20, 300, 80) maps to (50, 10, 150, 40)
        # inside the centered 200 x 50 pixmap. The rendered pixmap comparison
        # above verifies that the box and source pixels underwent one transform.
        self.assertNotEqual(actual.toImage().pixelColor(50, 10), QColor("white"))
        self.assertNotEqual(actual.toImage().pixelColor(150, 40), QColor("white"))

    def test_responsive_preview_repeated_resizing_does_not_mutate_or_accumulate_overlays(self):
        source = QImage(120, 80, QImage.Format.Format_RGB32)
        source.fill(QColor("white"))
        result = InferenceResult(120, 80, .25, (Detection(17, .9, (20, 15, 90, 65)),))
        rendered = render_detections(source, result, {17: "Almond"})
        original_box = result.detections[0].bbox_xyxy
        window = self.window()
        for width, height in ((240, 240), (320, 160), (160, 320), (300, 200), (240, 240)):
            with self.subTest(width=width, height=height):
                window.image_label.resize(width, height)
                window.image_label.set_source_image(rendered)
                self.application.processEvents()
                self.assertEqual(window.image_label.source_image, rendered)
                self.assertEqual(result.detections[0].bbox_xyxy, original_box)
                expected = QPixmap.fromImage(rendered).scaled(
                    window.image_label.contentsRect().size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.assertEqual(window.image_label.pixmap().toImage(), expected.toImage())

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
        self.run_analysis_and_wait(window)
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

    def test_successful_analysis_is_not_reported_saved_until_manual_save(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(21, 12, .25)
        window = self.window(detector=detector, class_names={17: "Almond"})
        window.display_image_path(self.path)
        self.run_analysis_and_wait(window)
        self.assertIn("Analysis succeeded", window.status_label.text())
        self.assertNotIn("Saved results", window.status_label.text())

    def test_counts_include_zero_classes_and_clear_for_new_request(self):
        detector = Mock()
        detector.predict.return_value = InferenceResult(
            21, 12, .25, (Detection(17, .9, (2, 3, 15, 10)),)
        )
        window = self.window(
            detector=detector,
            class_names={17: "Almond", 21: "Foreign material"},
        )
        window.display_image_path(self.path)
        self.run_analysis_and_wait(window)
        self.assertIn("Almond: 1", window.counts_label.text())
        self.assertIn("Foreign material: 0", window.counts_label.text())
        window.display_image_path(self.path)
        self.assertEqual(window.counts_label.text(), "Detection counts unavailable.")
        self.assertIsNone(window.current_analysis)

    def test_open_saved_results_uses_saved_classes_and_does_not_predict(self):
        result = InferenceResult(21, 12, .4, (Detection(99, .8, (1, 2, 18, 11)),))
        saved = create_analysis(self.path, result, {99: "Foreign material"}, session_id="saved-session", model_label="saved-model")
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
        self.assertEqual(window.current_analysis.session_id, "saved-session")
        self.assertEqual(window.current_analysis.model_label, "saved-model")
        self.assertIn("Foreign material: 1", window.counts_label.text())

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
        save_analysis(saved_path, create_analysis(self.path, result, {17: "Almond"}, session_id="changed-source-session"))
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
        self.run_analysis_and_wait(window)
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
        self.run_analysis_and_wait(window)
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
        self.run_analysis_and_wait(window)
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
        self.run_analysis_and_wait(window)
        detector.predict.assert_called_once()
        self.assertIn("changed", window.status_label.text().lower())
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.current_analysis)
        self.assertFalse(window.save_results_button.isEnabled())
        self.assertFalse(window.export_csv_button.isEnabled())
        self.assertEqual(window.current_image.pixelColor(0, 0), QColor(20, 40, 60))


if __name__ == "__main__":
    unittest.main()
