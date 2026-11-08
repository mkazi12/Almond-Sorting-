"""SCRUM-43 focused tests. Categories are synthetic, not DINOS's final taxonomy."""

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.dataset.validation import (
    DatasetValidationError,
    check_image_pair,
    index_coco_document,
    require_inside_root,
    safe_relative_path,
)


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image_path = self.root / "test.png"
        Image.new("RGB", (80, 60), (160, 100, 40)).save(self.image_path)
        self.entry = {"id": 7, "file_name": "test.png", "width": 80, "height": 60}
        self.good = {"id": 9, "image_id": 7, "category_id": 1, "bbox": [10, 10, 20, 15]}
        self.mapping = {1: "A", 2: "B"}

    def check(self, annotations=None, **overrides):
        data = dict(
            image_path=self.image_path,
            image_name="test.png",
            coco_entry=self.entry,
            annotations=[self.good] if annotations is None else annotations,
            declared_category_ids=frozenset({1, 2}),
            class_mapping=self.mapping,
            allow_negative_images=False,
        )
        data.update(overrides)
        return check_image_pair(**data)

    def test_valid_image_and_fractional_edge_touching_boxes(self):
        edge = {**self.good, "bbox": [60.5, 10.25, 19.5, 49.75]}
        result = self.check([edge])
        self.assertTrue(result.accepted)
        self.assertEqual(result.image.width, 80)
        self.assertEqual(result.objects[0].bbox_xywh, (60.5, 10.25, 19.5, 49.75))

    def test_valid_jpeg(self):
        jpeg = self.root / "test.jpg"
        Image.new("RGB", (80, 60), (20, 80, 100)).save(jpeg)
        self.assertTrue(self.check(image_path=jpeg, image_name="test.jpg").accepted)

    def test_each_invalid_bbox_is_rejected_with_code_and_filename(self):
        bad_boxes = [
            [0, 0, 0, 5], [0, 0, -2, 5], [-1, 0, 3, 5], [0, -1, 2, 3],
            [79, 10, 2, 3], [10, 58, 2, 3], [10, 10, "5", 3],
            [10, 10, True, 3], [10, 10, float("nan"), 3],
            [10, 10, float("inf"), 3], [10, 10, 10 ** 1000, 3], [10, 10, 2], None,
        ]
        for box in bad_boxes:
            with self.subTest(bbox=box):
                result = self.check([{**self.good, "bbox": box}])
                self.assertFalse(result.accepted)
                self.assertEqual(result.reasons[0].category, "invalid_bbox")
                self.assertIn("test.png", result.reasons[0].detail)
                self.assertEqual(result.objects, ())

    def test_unknown_id_and_malformed_fields_are_distinct(self):
        unknown = self.check([{**self.good, "category_id": 999}])
        malformed = self.check([{**self.good, "id": "not-an-integer"}])
        undeclared = self.check([{**self.good, "category_id": 2}], declared_category_ids=frozenset({1}))
        self.assertEqual(unknown.reasons[0].category, "unknown_class_id")
        self.assertEqual(malformed.reasons[0].category, "malformed_annotation")
        self.assertEqual(undeclared.reasons[0].category, "unknown_class_id")

    def test_bad_annotation_rejects_the_whole_pair(self):
        result = self.check([self.good, {**self.good, "id": 10, "bbox": [79, 0, 2, 2]}])
        self.assertFalse(result.accepted)
        self.assertEqual(result.objects, ())
        self.assertEqual([reason.category for reason in result.reasons], ["invalid_bbox"])

    def test_missing_coco_entry_never_counts_as_negative(self):
        result = self.check([], coco_entry=None, allow_negative_images=True)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reasons[0].category, "missing_annotation")

    def test_zero_objects_follow_explicit_policy(self):
        self.assertTrue(self.check([], allow_negative_images=True).accepted)
        rejected = self.check([], allow_negative_images=False)
        self.assertEqual(rejected.reasons[0].category, "negative_image_not_allowed")

    def test_unreadable_unsupported_and_mismatched_image(self):
        corrupt = self.root / "bad.jpg"
        corrupt.write_bytes(b"not really an image")
        self.assertIn("unreadable_image", [r.category for r in self.check(image_path=corrupt, image_name="bad.jpg").reasons])
        bmp = self.root / "fake.bmp"
        Image.new("RGB", (80, 60)).save(bmp)
        result = self.check(image_path=bmp, image_name="fake.bmp")
        self.assertIn("unsupported_image", [r.category for r in result.reasons])
        changed_entry = dict(self.entry, width=81)
        self.assertEqual(self.check(coco_entry=changed_entry).reasons[0].category, "dimension_mismatch")

    def test_actual_format_is_checked_even_when_the_extension_is_supported(self):
        disguised = self.root / "disguised.png"
        Image.new("RGB", (80, 60)).save(disguised, format="BMP")
        result = self.check(image_path=disguised, image_name=disguised.name)
        self.assertFalse(result.accepted)
        self.assertIn("unreadable_image", [reason.category for reason in result.reasons])

    def test_exif_orientation_does_not_rotate_annotation_coordinates(self):
        jpeg = self.root / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6  # A viewer may rotate 90 degrees, but stored pixels are 80x60.
        Image.new("RGB", (80, 60)).save(jpeg, exif=exif)
        before = jpeg.read_bytes()
        result = self.check(image_path=jpeg, image_name=jpeg.name)
        self.assertTrue(result.accepted)
        self.assertEqual((result.image.width, result.image.height), (80, 60))
        self.assertEqual(result.objects[0].bbox_xywh, (10.0, 10.0, 20.0, 15.0))
        self.assertEqual(jpeg.read_bytes(), before)

    def test_index_joins_by_coco_image_ids(self):
        doc = {
            "images": [self.entry],
            "annotations": [self.good],
            "categories": [{"id": 1, "name": "A"}],
        }
        index = index_coco_document(doc, self.mapping)
        self.assertEqual(index.images_by_path["test.png"]["id"], 7)
        self.assertEqual(index.annotations_by_id[7], (self.good,))
        self.assertEqual([category.name for category in index.classes], ["A"])

    def test_document_errors_have_machine_readable_categories(self):
        good = {"images": [self.entry], "annotations": [self.good], "categories": [{"id": 1, "name": "A"}]}
        variations = [
            ({"images": [], "annotations": []}, "malformed_annotation_file"),
            ({**good, "images": [self.entry, self.entry]}, "duplicate_image_id"),
            ({**good, "annotations": [self.good, self.good]}, "duplicate_annotation_id"),
            ({**good, "annotations": [{**self.good, "image_id": 999}]}, "orphan_annotation"),
            ({**good, "categories": [{"id": 1, "name": "Wrong"}]}, "class_mapping_mismatch"),
            ({**good, "images": [{**self.entry, "file_name": "../../escape.png"}]}, "unsafe_path"),
        ]
        for document, code in variations:
            with self.subTest(code=code):
                with self.assertRaises(DatasetValidationError) as caught:
                    index_coco_document(document, self.mapping)
                self.assertEqual(caught.exception.category, code)

    def test_safe_paths_and_symlinks(self):
        self.assertEqual(safe_relative_path("images\\a.png"), "images/a.png")
        for value in ("../a.png", "/tmp/a.png", "C:\\a.png", "\\\\server\\x.png"):
            with self.subTest(value=value), self.assertRaises(DatasetValidationError):
                safe_relative_path(value)
        self.assertEqual(require_inside_root(self.root, "test.png"), self.image_path.resolve())
        outside = self.root.parent / f"outside_{self.root.name}.png"
        try:
            outside.write_bytes(b"data")
            link = self.root / "link.png"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable on this system")
            with self.assertRaises(DatasetValidationError) as caught:
                require_inside_root(self.root, "link.png")
            self.assertEqual(caught.exception.category, "unsafe_path")
        finally:
            outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
