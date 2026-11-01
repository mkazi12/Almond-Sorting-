import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from app.dataset.preview import (
    PreviewError,
    category_color,
    label_text_color,
    render_ground_truth_preview,
)
from app.dataset.records import GroundTruthObject, objects_for_image
from app.image_loader import load_image
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class GroundTruthPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset = load_dataset_fixture(Path(temporary_directory.name))
        self.classes = {
            spec.category_id: spec.name for spec in self.dataset.annotations[0].classes
        }

    def test_fixture_boxes_render_at_coco_coordinates(self):
        record = self.record("fix-001")
        objects = objects_for_image(self.dataset, record)
        image = load_image(self.dataset.resolve(record.image_path))
        rendered = render_ground_truth_preview(image, objects, self.classes)
        color = rgb_of(category_color(1))

        self.assertEqual(rendered.size(), image.size())
        self.assertEqual([obj.category_id for obj in objects], [1, 1])
        for bbox in ((10, 10, 20, 15), (40, 30, 20, 15)):
            for point in box_corners(bbox):
                self.assertEqual(rgb(rendered, *point), color)
            self.assertNotEqual(rgb(rendered, bbox[0] - 1, bbox[1]), color)

    def test_classes_use_their_category_colors(self):
        record = self.record("fix-002")
        objects = objects_for_image(self.dataset, record)
        image = load_image(self.dataset.resolve(record.image_path))
        rendered = render_ground_truth_preview(image, objects, self.classes)
        by_category = {obj.category_id: obj for obj in objects}

        class_a = rgb(rendered, *bottom_center(by_category[1].bbox_xywh))
        class_b = rgb(rendered, *bottom_center(by_category[2].bbox_xywh))
        self.assertEqual(class_a, rgb_of(category_color(1)))
        self.assertEqual(class_b, rgb_of(category_color(2)))
        self.assertNotEqual(class_a, class_b)

    def test_class_labels_are_drawn(self):
        record = self.record("fix-002")
        objects = objects_for_image(self.dataset, record)
        image = load_image(self.dataset.resolve(record.image_path))
        for category_id in (1, 2):
            with self.subTest(category_id=category_id):
                selected = tuple(obj for obj in objects if obj.category_id == category_id)
                rendered = render_ground_truth_preview(image, selected, self.classes)
                text_pixels = count_color(rendered, label_text_color(category_id))
                self.assertGreater(text_pixels, 0)
                self.assertLess(text_pixels, rendered.width() * rendered.height())

    def test_box_touching_the_image_boundary_renders_on_the_edge(self):
        image = QImage(40, 30, QImage.Format.Format_RGB32)
        image.fill(0xFFB0B0B0)
        box = GroundTruthObject(1, 1, 1, (30, 20, 10, 10))
        rendered = render_ground_truth_preview(image, (box,), {1: "A"})
        color = rgb_of(category_color(1))

        self.assertEqual(rendered.size(), image.size())
        self.assertEqual(rgb(rendered, 39, 29), color)
        self.assertEqual(rgb(rendered, 30, 20), color)
        self.assertEqual(rgb(rendered, 39, 20), color)
        self.assertNotEqual(rgb(rendered, 29, 25), color)
        self.assertEqual(rgb(rendered, 0, 0), (176, 176, 176))

    def test_negative_image_has_no_annotation_overlay(self):
        record = self.record("fix-003")
        objects = objects_for_image(self.dataset, record)
        image = load_image(self.dataset.resolve(record.image_path))
        before = pixels(image)
        rendered = render_ground_truth_preview(image, objects, self.classes)

        self.assertEqual(objects, ())
        self.assertEqual(rendered.size(), image.size())
        self.assertEqual(pixels(rendered), before)
        self.assertEqual(pixels(image), before)

    def test_input_image_and_source_files_remain_unchanged(self):
        record = self.record("fix-001")
        image = load_image(self.dataset.resolve(record.image_path))
        before = pixels(image)
        files_before = self.dataset_files()
        rendered = render_ground_truth_preview(
            image, objects_for_image(self.dataset, record), self.classes
        )

        self.assertIsNot(rendered, image)
        self.assertEqual(pixels(image), before)
        self.assertNotEqual(pixels(rendered), before)
        self.assertEqual(self.dataset_files(), files_before)

    def test_fractional_and_subpixel_boxes_are_rasterized(self):
        image = QImage(40, 30, QImage.Format.Format_RGB32)
        image.fill(0xFFB0B0B0)
        background = (176, 176, 176)
        color = rgb_of(category_color(1))
        before = pixels(image)

        # width 0.4 rounds to 0 on its own and would previously be discarded.
        subpixel = GroundTruthObject(1, 1, 1, (10.4, 8.6, 0.4, 0.3))
        rendered = render_ground_truth_preview(image, (subpixel,), {1: "A"})
        self.assertEqual(rgb(rendered, 10, 8), color)
        self.assertEqual(rgb(rendered, 11, 8), background)
        self.assertEqual(rgb(rendered, 10, 7), background)

        # 10.6 + 0.6 crosses into column 11. Rounding x and width separately
        # would stroke only column 11.
        crossing = GroundTruthObject(2, 1, 1, (10.6, 4.2, 0.6, 0.7))
        rendered = render_ground_truth_preview(image, (crossing,), {1: "A"})
        self.assertEqual(rgb(rendered, 10, 4), color)
        self.assertEqual(rgb(rendered, 11, 4), color)
        self.assertEqual(rgb(rendered, 9, 4), background)
        self.assertEqual(rgb(rendered, 12, 4), background)
        self.assertEqual(rgb(rendered, 10, 3), background)

        empty = GroundTruthObject(3, 1, 1, (2.2, 2.2, 0.0, 0.4))
        rendered = render_ground_truth_preview(image, (empty,), {1: "A"})
        self.assertEqual(pixels(rendered), before)
        self.assertEqual(pixels(image), before)

    def test_missing_class_name_raises(self):
        image = QImage(8, 8, QImage.Format.Format_RGB32)
        image.fill(0xFF202020)
        box = GroundTruthObject(1, 1, 9, (1, 1, 3, 3))
        with self.assertRaises(PreviewError) as caught:
            render_ground_truth_preview(image, (box,), {1: "A"})
        self.assertIn("category 9", str(caught.exception))
        self.assertIn("class name", str(caught.exception).lower())

    def record(self, image_id):
        return next(item for item in self.dataset.accepted if item.image_id == image_id)

    def dataset_files(self):
        return {
            path.relative_to(self.dataset.root).as_posix(): path.read_bytes()
            for path in self.dataset.root.rglob("*")
            if path.is_file()
        }


def box_corners(bbox):
    x, y, width, height = bbox
    return (
        (x, y),
        (x + width - 1, y),
        (x, y + height - 1),
        (x + width - 1, y + height - 1),
    )


def bottom_center(bbox):
    x, y, width, height = (int(round(value)) for value in bbox)
    return (x + (width - 1) // 2, y + height - 1)


def rgb(image, x, y):
    color = image.pixelColor(x, y)
    return (color.red(), color.green(), color.blue())


def rgb_of(color):
    return (color.red(), color.green(), color.blue())


def pixels(image):
    return [rgb(image, x, y) for y in range(image.height()) for x in range(image.width())]


def count_color(image, color):
    expected = rgb_of(color)
    return sum(1 for y in range(image.height()) for x in range(image.width()) if rgb(image, x, y) == expected)


if __name__ == "__main__":
    unittest.main()
