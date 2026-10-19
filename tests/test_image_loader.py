import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QImage

from app.image_loader import ImageLoadError, load_image


class ImageLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.sample_image = QImage(12, 8, QImage.Format.Format_RGB32)
        self.sample_image.fill(0xFFCC9966)

    def save_image(self, filename, image_format):
        path = self.directory / filename
        self.assertTrue(
            self.sample_image.save(str(path), image_format),
            f"Could not create the synthetic {image_format} test image.",
        )
        return path

    def assert_image_load_error(self, path):
        with self.assertRaises(ImageLoadError) as caught:
            load_image(path)
        self.assertRegex(
            str(caught.exception).lower(),
            r"\b(choose|check|save|select|try)\b",
            "The error should suggest an action the user can take.",
        )

    def test_supported_images_load(self):
        cases = [
            ("sample.jpg", "JPEG"),
            ("sample.jpeg", "JPEG"),
            ("sample.png", "PNG"),
            ("uppercase.JPEG", "JPEG"),
            ("uppercase.PNG", "PNG"),
            # Either supported image format is allowed under a supported suffix.
            ("png_contents.jpg", "PNG"),
        ]
        for filename, image_format in cases:
            path = self.save_image(filename, image_format)
            for path_argument in (str(path), path):
                with self.subTest(filename=filename, path_type=type(path_argument)):
                    image = load_image(path_argument)
                    self.assertIsInstance(image, QImage)
                    self.assertFalse(image.isNull())
                    self.assertEqual((image.width(), image.height()), (12, 8))

    def test_unsupported_suffix_is_rejected(self):
        path = self.save_image("image.txt", "PNG")
        self.assert_image_load_error(path)

    def test_unsupported_image_contents_are_rejected(self):
        path = self.save_image("bmp_contents.jpg", "BMP")
        self.assert_image_load_error(path)

    def test_empty_and_non_image_files_are_rejected(self):
        cases = [
            ("empty.png", b""),
            ("not_an_image.jpg", b"This file contains text, not image pixels."),
        ]
        for filename, contents in cases:
            with self.subTest(filename=filename):
                path = self.directory / filename
                path.write_bytes(contents)
                self.assert_image_load_error(path)

    def test_truncated_png_is_rejected(self):
        original = self.save_image("complete.png", "PNG")
        truncated = self.directory / "truncated.png"
        # Keep the PNG signature and IHDR header, but remove all pixel data.
        truncated.write_bytes(original.read_bytes()[:33])
        self.assert_image_load_error(truncated)

    def test_missing_file_is_rejected(self):
        self.assert_image_load_error(self.directory / "missing.png")


if __name__ == "__main__":
    unittest.main()
