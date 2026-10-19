import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from app.image_loader import load_image
from app.recorded_setup import (
    CAMERA_FPS,
    CAMERA_PROTOCOL,
    CAMERA_RESOLUTION,
    SAMPLE_MEDIA,
    SAMPLE_MEDIA_PATH,
    WORKSTATION_GPU,
    WORKSTATION_OS,
    setup_text,
)
from app.__main__ import MainWindow


class RecordedSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_required_fields_are_recorded(self):
        text = setup_text()
        self.assertIn(CAMERA_PROTOCOL, text)
        self.assertIn(CAMERA_RESOLUTION, text)
        self.assertIn(f"{CAMERA_FPS} FPS", text)
        self.assertIn(SAMPLE_MEDIA, text)
        self.assertIn(WORKSTATION_OS, text)
        self.assertIn(WORKSTATION_GPU, text)
        self.assertRegex(CAMERA_RESOLUTION, r"^\d+x\d+$")
        self.assertGreater(CAMERA_FPS, 0)

    def test_sample_media_exists_and_loads(self):
        self.assertTrue(SAMPLE_MEDIA_PATH.is_file(), f"Missing {SAMPLE_MEDIA_PATH}")
        image = load_image(SAMPLE_MEDIA_PATH)
        self.assertFalse(image.isNull())
        self.assertGreater(image.width(), 0)
        self.assertGreater(image.height(), 0)

    def test_ui_shows_recorded_setup(self):
        window = MainWindow()
        self.addCleanup(window.close)
        label = window.findChild(QLabel, "recorded_setup")
        self.assertIsNotNone(label)
        self.assertEqual(label.text(), setup_text())

    def test_load_sample_footage_uses_recorded_path(self):
        window = MainWindow()
        self.addCleanup(window.close)
        with patch.object(window, "display_image_path") as display:
            button = window.findChild(QPushButton, "load_sample_button")
            self.assertIsNotNone(button)
            button.click()
            display.assert_called_once_with(str(SAMPLE_MEDIA_PATH))


if __name__ == "__main__":
    unittest.main()
