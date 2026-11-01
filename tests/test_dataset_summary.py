import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.dataset.records import (
    UNKNOWN_SOURCE,
    ClassSpec,
    DatasetInput,
    GroundTruthObject,
    SourcePolicy,
)
from app.dataset.summary import SummaryError, summarize
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class DatasetSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset = load_dataset_fixture(Path(temporary_directory.name))

    def test_fixture_summary(self):
        summary = summarize(self.dataset)
        classes = {item.name: item for item in summary.classes}
        rejected = summary.rejected_images

        self.assertEqual(summary.accepted_image_count, 3)
        self.assertEqual(summary.rejected_image_count, 1)
        self.assertEqual(classes["A"].category_id, 1)
        self.assertEqual(classes["A"].annotation_count, 3)
        self.assertEqual(classes["A"].image_count, 2)
        self.assertEqual(classes["B"].category_id, 2)
        self.assertEqual(classes["B"].annotation_count, 1)
        self.assertEqual(classes["B"].image_count, 1)
        self.assertEqual(summary.negative_image_count, 1)
        self.assertEqual(
            {item.label: item.image_count for item in summary.images_by_lot},
            {"lot-1": 2, UNKNOWN_SOURCE: 1},
        )
        self.assertEqual(
            {item.label: item.image_count for item in summary.images_by_session},
            {"session-1": 2, "session-2": 1},
        )
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].filename, "not-a-real-capture.jpg")
        self.assertEqual(rejected[0].image_path, "images/not-a-real-capture.jpg")
        self.assertIsNone(rejected[0].image_id)
        self.assertEqual(
            [reason.category for reason in rejected[0].reasons],
            ["unreadable_image", "missing_annotation"],
        )
        self.assertEqual(summary.rejected_image_count, 1)

    def test_empty_input(self):
        empty = DatasetInput(
            root=self.dataset.root,
            source_policy=SourcePolicy(require_lot=True, require_session=True),
            accepted=(),
            rejected=(),
            annotations=(),
        )
        summary = summarize(empty)
        self.assertEqual(summary.accepted_image_count, 0)
        self.assertEqual(summary.rejected_image_count, 0)
        self.assertEqual(summary.negative_image_count, 0)
        self.assertEqual(summary.classes, ())
        self.assertEqual(summary.images_by_lot, ())
        self.assertEqual(summary.images_by_session, ())
        self.assertEqual(summary.rejected_images, ())

    def test_configured_class_with_no_occurrences(self):
        parsed = self.dataset.annotations[0]
        dataset = replace(
            self.dataset,
            annotations=(replace(parsed, classes=parsed.classes + (ClassSpec(3, "C"),)),),
        )
        summary = summarize(dataset)
        unused = next(item for item in summary.classes if item.category_id == 3)
        self.assertEqual(unused.name, "C")
        self.assertEqual(unused.annotation_count, 0)
        self.assertEqual(unused.image_count, 0)
        self.assertEqual(
            {item.name: item.annotation_count for item in summary.classes},
            {"A": 3, "B": 1, "C": 0},
        )

    def test_annotations_for_other_coco_images_are_not_counted(self):
        parsed = self.dataset.annotations[0]
        extra = GroundTruthObject(99, 99, 1, (0, 0, 1, 1))
        dataset = replace(self.dataset, annotations=(replace(parsed, objects=parsed.objects + (extra,)),))
        summary = summarize(dataset)
        classes = {item.category_id: item for item in summary.classes}
        self.assertEqual(classes[1].annotation_count, 3)
        self.assertEqual(classes[1].image_count, 2)

    def test_missing_or_ambiguous_associations_raise(self):
        cases = {
            "missing annotation file": self.dataset_with_missing_annotation_file(),
            "missing coco image": self.dataset_with_missing_coco_image(),
            "ambiguous coco image": self.dataset_with_duplicate_coco_image(),
            "ambiguous annotation file": self.dataset_with_duplicate_annotation_file(),
        }
        for name, dataset in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(SummaryError) as caught:
                    summarize(dataset)
                self.assertIn("fix-003" if "missing" in name else "fix-001", str(caught.exception))

    def test_category_mappings_are_explicit(self):
        conflict = self.dataset_with_conflicting_class_name()
        with self.assertRaises(SummaryError) as caught:
            summarize(conflict)
        self.assertIn("Category 1", str(caught.exception))
        self.assertIn("'A'", str(caught.exception))
        self.assertIn("'Almond'", str(caught.exception))

        unresolved = self.dataset_with_unknown_category()
        with self.assertRaises(SummaryError) as caught:
            summarize(unresolved)
        self.assertIn("category 99", str(caught.exception))
        self.assertIn("fix-001", str(caught.exception))

    def dataset_with_missing_annotation_file(self):
        accepted = tuple(
            replace(image, annotation_path=Path("annotations/missing.json"))
            if image.image_id == "fix-003"
            else image
            for image in self.dataset.accepted
        )
        return replace(self.dataset, accepted=accepted)

    def dataset_with_missing_coco_image(self):
        parsed = self.dataset.annotations[0]
        images = tuple(item for item in parsed.images if item.coco_image_id != 3)
        return replace(self.dataset, annotations=(replace(parsed, images=images),))

    def dataset_with_duplicate_coco_image(self):
        parsed = self.dataset.annotations[0]
        images = parsed.images + (parsed.images[0],)
        return replace(self.dataset, annotations=(replace(parsed, images=images),))

    def dataset_with_duplicate_annotation_file(self):
        parsed = self.dataset.annotations[0]
        return replace(self.dataset, annotations=(parsed, parsed))

    def dataset_with_conflicting_class_name(self):
        parsed = self.dataset.annotations[0]
        classes = parsed.classes + (ClassSpec(1, "Almond"),)
        return replace(self.dataset, annotations=(replace(parsed, classes=classes),))

    def dataset_with_unknown_category(self):
        parsed = self.dataset.annotations[0]
        objects = tuple(
            replace(obj, category_id=99) if obj.annotation_id == 11 else obj
            for obj in parsed.objects
        )
        return replace(self.dataset, annotations=(replace(parsed, objects=objects),))


if __name__ == "__main__":
    unittest.main()
