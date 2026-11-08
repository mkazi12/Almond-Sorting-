"""The bundled sample is an actual image and opens through the preview action."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from app.image_loader import load_image
from app.ui.assets import SAMPLE_IMAGE_PATH
from app.ui.main_window import MainWindow


class SampleImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_bundled_sample_exists_and_decodes(self):
        self.assertTrue(SAMPLE_IMAGE_PATH.is_file())
        image = load_image(SAMPLE_IMAGE_PATH)
        self.assertFalse(image.isNull())
        self.assertGreater(image.width(), 0)
        self.assertGreater(image.height(), 0)

    def test_sample_button_opens_the_bundled_image(self):
        window = MainWindow()
        self.addCleanup(window.close)
        with patch.object(window, "display_image_path") as display:
            button = window.findChild(QPushButton, "load_sample_button")
            self.assertIsNotNone(button)
            self.assertEqual(button.text(), "Load sample image")
            button.click()
            display.assert_called_once_with(str(SAMPLE_IMAGE_PATH))
